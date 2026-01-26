"""
Temporal Analysis Module

This module implements temporal analysis techniques for vegetation and 
contamination time series:
- Fast Fourier Transform (FFT) for frequency analysis
- Cumulative Sum (CUSUM) for change-point detection
- Wavelet analysis for multi-scale temporal patterns
- Seasonal decomposition

References:
    - Manuscript Section 2.3 (Methods - Temporal Analysis Framework)
    - Supplementary Material Section S1.5 (Temporal and Spectral Analysis)
"""

import numpy as np
from scipy import stats, signal
from scipy.fft import fft, fftfreq, ifft
from scipy.ndimage import uniform_filter1d
from typing import Dict, List, Tuple, Optional, Union
import warnings


def fft_analysis(
    time_series: np.ndarray,
    sampling_rate: float = 1.0,
    detrend: bool = True,
    window: Optional[str] = 'hann'
) -> Dict:
    """
    Perform FFT spectral analysis to identify dominant periodicities.
    
    From Supplementary Material Section S1.5 (Equation 12):
        X(k) = Σₙ x(n) · e^(-j2πkn/N)
    
    "Analysis focused on 0.1-12 cycles/year to capture both seasonal 
    and multi-year periodicities"
    
    Args:
        time_series: 1D array of temporal measurements
        sampling_rate: Samples per unit time (default 1 = monthly data)
        detrend: Whether to remove linear trend
        window: Windowing function ('hann', 'hamming', 'blackman', None)
        
    Returns:
        Dictionary containing:
            - frequencies: Frequency values
            - periods: Period values (1/frequency)
            - power: Power spectral density
            - dominant_periods: Top 5 dominant periods
            - dominant_power_pct: Percentage of total power for dominant periods
    """
    n = len(time_series)
    
    # Handle missing values
    if np.any(np.isnan(time_series)):
        time_series = np.interp(
            np.arange(n),
            np.arange(n)[~np.isnan(time_series)],
            time_series[~np.isnan(time_series)]
        )
    
    # Detrend if requested
    if detrend:
        time_series = signal.detrend(time_series)
    
    # Normalize (zero mean, unit variance)
    normalized = (time_series - np.mean(time_series)) / (np.std(time_series) + 1e-10)
    
    # Apply window function
    if window:
        if window == 'hann':
            win = np.hanning(n)
        elif window == 'hamming':
            win = np.hamming(n)
        elif window == 'blackman':
            win = np.blackman(n)
        else:
            win = np.ones(n)
        normalized = normalized * win
    
    # Compute FFT
    fft_result = fft(normalized)
    freq = fftfreq(n, d=1/sampling_rate)
    
    # Get positive frequencies only
    pos_mask = freq > 0
    power = np.abs(fft_result[pos_mask])**2
    pos_freq = freq[pos_mask]
    
    # Calculate periods (months)
    periods = 1 / pos_freq
    
    # Filter to 0.1-12 cycles/year range (as per manuscript)
    # For monthly data: 0.1-12 cycles/year = periods of 1-120 months
    valid_mask = (periods >= 1) & (periods <= 120)
    periods_valid = periods[valid_mask]
    power_valid = power[valid_mask]
    freq_valid = pos_freq[valid_mask]
    
    # Find top 5 dominant periods
    sorted_idx = np.argsort(power_valid)[::-1]
    top_n = min(5, len(sorted_idx))
    dominant_periods = periods_valid[sorted_idx[:top_n]]
    dominant_powers = power_valid[sorted_idx[:top_n]]
    
    # Calculate percentage of total power
    total_power = np.sum(power_valid)
    power_pct = (dominant_powers / total_power) * 100 if total_power > 0 else np.zeros(top_n)
    
    return {
        'frequencies': freq_valid,
        'periods': periods_valid,
        'power': power_valid,
        'dominant_periods': dominant_periods,
        'dominant_power_pct': power_pct,
        'total_power': total_power,
        'fft_complex': fft_result
    }


def cusum_analysis(
    time_series: np.ndarray,
    k: Optional[float] = None,
    h: Optional[float] = None,
    bootstrap_iterations: int = 1000
) -> Dict:
    """
    Perform CUSUM (Cumulative Sum) change-point detection.
    
    From Supplementary Material Section S1.5 (Equation 13):
        CUSUMₙ = max(0, CUSUMₙ₋₁ + xₙ - μ - k)
    
    "where k=0.5σ served as the reference value and h=4σ as the detection 
    threshold; bootstrap resampling (1,000 iterations) assessed statistical 
    significance"
    
    Args:
        time_series: 1D array of temporal measurements
        k: Reference value (default: 0.5 * std)
        h: Detection threshold (default: 4 * std)
        bootstrap_iterations: Number of bootstrap iterations (1000)
        
    Returns:
        Dictionary containing:
            - cusum: CUSUM values
            - cusum_neg: Negative CUSUM values
            - change_points: Detected change point indices
            - change_point_values: CUSUM values at change points
            - change_point_types: 'increase' or 'decrease'
            - p_values: Bootstrap p-values for change points
    """
    n = len(time_series)
    
    # Handle missing values
    if np.any(np.isnan(time_series)):
        time_series = np.interp(
            np.arange(n),
            np.arange(n)[~np.isnan(time_series)],
            time_series[~np.isnan(time_series)]
        )
    
    # Calculate parameters
    mu = np.mean(time_series)
    sigma = np.std(time_series)
    
    if k is None:
        k = 0.5 * sigma  # Reference value as per manuscript
    if h is None:
        h = 4 * sigma  # Detection threshold as per manuscript
    
    # Calculate CUSUM (positive - detects increases)
    cusum_pos = np.zeros(n)
    for i in range(1, n):
        cusum_pos[i] = max(0, cusum_pos[i-1] + (time_series[i] - mu) - k)
    
    # Calculate negative CUSUM (detects decreases)
    cusum_neg = np.zeros(n)
    for i in range(1, n):
        cusum_neg[i] = min(0, cusum_neg[i-1] + (time_series[i] - mu) + k)
    
    # Simple CUSUM (cumulative deviation from mean)
    cusum_simple = np.cumsum(time_series - mu)
    
    # Detect change points where CUSUM exceeds threshold
    change_points = []
    change_point_values = []
    change_point_types = []
    
    # Find local extrema in simple CUSUM
    for i in range(1, n - 1):
        # Local maximum (potential increase point)
        if cusum_simple[i] > cusum_simple[i-1] and cusum_simple[i] > cusum_simple[i+1]:
            if abs(cusum_simple[i]) > h:
                change_points.append(i)
                change_point_values.append(cusum_simple[i])
                change_point_types.append('increase' if cusum_simple[i] > 0 else 'decrease')
        
        # Local minimum (potential decrease point)
        elif cusum_simple[i] < cusum_simple[i-1] and cusum_simple[i] < cusum_simple[i+1]:
            if abs(cusum_simple[i]) > h:
                change_points.append(i)
                change_point_values.append(cusum_simple[i])
                change_point_types.append('decrease' if cusum_simple[i] < 0 else 'increase')
    
    # Bootstrap significance testing
    p_values = []
    if len(change_points) > 0 and bootstrap_iterations > 0:
        p_values = _bootstrap_cusum_significance(
            time_series, change_points, change_point_values, 
            bootstrap_iterations
        )
    
    return {
        'cusum': cusum_simple,
        'cusum_positive': cusum_pos,
        'cusum_negative': cusum_neg,
        'change_points': np.array(change_points),
        'change_point_values': np.array(change_point_values),
        'change_point_types': change_point_types,
        'p_values': np.array(p_values) if p_values else np.array([]),
        'threshold_h': h,
        'reference_k': k,
        'mean': mu,
        'std': sigma
    }


def _bootstrap_cusum_significance(
    time_series: np.ndarray,
    change_points: List[int],
    change_point_values: List[float],
    n_iterations: int = 1000
) -> List[float]:
    """
    Bootstrap test for CUSUM change point significance.
    
    Args:
        time_series: Original time series
        change_points: Detected change point indices
        change_point_values: CUSUM values at change points
        n_iterations: Number of bootstrap iterations
        
    Returns:
        List of p-values for each change point
    """
    n = len(time_series)
    p_values = []
    
    for cp_idx, cp_value in zip(change_points, change_point_values):
        # Count how many bootstrap samples have equal or larger CUSUM
        count = 0
        
        for _ in range(n_iterations):
            # Shuffle time series (null hypothesis: no change point)
            shuffled = np.random.permutation(time_series)
            
            # Calculate CUSUM for shuffled series
            mu = np.mean(shuffled)
            cusum_boot = np.cumsum(shuffled - mu)
            
            # Check if bootstrap max CUSUM exceeds observed
            if np.max(np.abs(cusum_boot)) >= abs(cp_value):
                count += 1
        
        p_values.append(count / n_iterations)
    
    return p_values


def wavelet_analysis(
    time_series: np.ndarray,
    scales: Optional[np.ndarray] = None,
    wavelet: str = 'morlet',
    omega0: float = 6.0
) -> Dict:
    """
    Continuous Wavelet Transform (CWT) analysis for multi-scale patterns.
    
    Identifies time-frequency localized patterns in vegetation dynamics.
    
    Args:
        time_series: 1D array of temporal measurements
        scales: Array of scales to analyze (default: 1 to n/2)
        wavelet: Wavelet type ('morlet')
        omega0: Morlet wavelet central frequency (default 6.0)
        
    Returns:
        Dictionary containing:
            - scales: Analyzed scales
            - periods: Corresponding periods
            - power: Wavelet power spectrum (2D)
            - global_power: Scale-averaged power
            - dominant_scales: Time-varying dominant scales
    """
    n = len(time_series)
    
    # Handle missing values
    if np.any(np.isnan(time_series)):
        time_series = np.interp(
            np.arange(n),
            np.arange(n)[~np.isnan(time_series)],
            time_series[~np.isnan(time_series)]
        )
    
    # Normalize
    normalized = (time_series - np.mean(time_series)) / (np.std(time_series) + 1e-10)
    
    # Default scales
    if scales is None:
        scales = np.arange(1, min(n // 2, 30))
    
    # Morlet wavelet function
    def morlet(t, s, omega0=6.0):
        t_scaled = t / s
        return np.exp(1j * omega0 * t_scaled) * np.exp(-t_scaled**2 / 2) / np.sqrt(s)
    
    # Time array centered at 0
    t = np.arange(n) - n // 2
    
    # Compute CWT
    cwt = np.zeros((len(scales), n), dtype=complex)
    
    for i, scale in enumerate(scales):
        wavelet_data = morlet(t, scale, omega0)
        # Convolution (same as correlation for symmetric wavelets)
        cwt[i] = np.convolve(normalized, wavelet_data, mode='same')
    
    # Power spectrum
    power = np.abs(cwt)**2
    
    # Global (time-averaged) wavelet spectrum
    global_power = np.mean(power, axis=1)
    
    # Dominant scale at each time point
    dominant_scales = scales[np.argmax(power, axis=0)]
    
    # Convert scales to periods
    periods = scales * (4 * np.pi) / (omega0 + np.sqrt(2 + omega0**2))
    
    return {
        'scales': scales,
        'periods': periods,
        'power': power,
        'global_power': global_power,
        'dominant_scales': dominant_scales,
        'cwt_complex': cwt
    }


def seasonal_decomposition(
    time_series: np.ndarray,
    period: int = 12
) -> Dict:
    """
    Decompose time series into trend, seasonal, and residual components.
    
    Uses classical additive decomposition:
        Y(t) = Trend(t) + Seasonal(t) + Residual(t)
    
    Args:
        time_series: 1D array of temporal measurements
        period: Seasonal period (12 for monthly data = annual cycle)
        
    Returns:
        Dictionary containing:
            - trend: Long-term trend component
            - seasonal: Seasonal component
            - residual: Irregular/residual component
            - observed: Original time series
    """
    n = len(time_series)
    
    # Handle missing values
    if np.any(np.isnan(time_series)):
        time_series = np.interp(
            np.arange(n),
            np.arange(n)[~np.isnan(time_series)],
            time_series[~np.isnan(time_series)]
        )
    
    # Calculate trend using moving average
    if n >= period:
        # Centered moving average
        trend = uniform_filter1d(time_series, size=period, mode='nearest')
    else:
        # If series is shorter than period, use simple mean
        trend = np.full(n, np.mean(time_series))
    
    # Detrended series
    detrended = time_series - trend
    
    # Calculate seasonal component (average for each position in cycle)
    seasonal = np.zeros(n)
    for i in range(period):
        indices = np.arange(i, n, period)
        if len(indices) > 0:
            seasonal_mean = np.mean(detrended[indices])
            seasonal[indices] = seasonal_mean
    
    # Residual
    residual = time_series - trend - seasonal
    
    return {
        'observed': time_series,
        'trend': trend,
        'seasonal': seasonal,
        'residual': residual,
        'period': period
    }


def recovery_phase_analysis(
    time_series: np.ndarray,
    dates: Optional[np.ndarray] = None,
    n_phases: int = 3
) -> Dict:
    """
    Identify recovery phases in vegetation time series.
    
    From manuscript: "Three recovery phases were identified:
    - Phase 1 (2019-2020): Site preparation, slope=-1.617 km²/yr
    - Phase 2 (2021-2022): Infrastructure establishment, slope=-0.120
    - Phase 3 (2022-2024): Active treatment, slope=-0.742"
    
    Args:
        time_series: Vegetation or contamination area time series
        dates: Optional date labels
        n_phases: Number of phases to identify (default 3)
        
    Returns:
        Dictionary containing phase characteristics
    """
    n = len(time_series)
    
    # Handle missing values
    if np.any(np.isnan(time_series)):
        time_series = np.interp(
            np.arange(n),
            np.arange(n)[~np.isnan(time_series)],
            time_series[~np.isnan(time_series)]
        )
    
    # Divide into phases
    segment_size = n // n_phases
    phases = []
    
    for i in range(n_phases):
        start_idx = i * segment_size
        end_idx = min((i + 1) * segment_size, n) if i < n_phases - 1 else n
        
        segment = time_series[start_idx:end_idx]
        x = np.arange(len(segment))
        
        # Linear regression for trend
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, segment)
        
        phase_info = {
            'phase_number': i + 1,
            'start_index': start_idx,
            'end_index': end_idx,
            'slope': slope,
            'slope_per_year': slope * 12,  # Convert monthly to yearly
            'intercept': intercept,
            'r_squared': r_value**2,
            'p_value': p_value,
            'mean_value': np.mean(segment),
            'std_value': np.std(segment),
            'start_value': segment[0],
            'end_value': segment[-1],
            'total_change': segment[-1] - segment[0]
        }
        
        # Add dates if provided
        if dates is not None:
            phase_info['start_date'] = dates[start_idx]
            phase_info['end_date'] = dates[end_idx - 1]
        
        phases.append(phase_info)
    
    # Overall trend
    x_all = np.arange(n)
    slope_all, intercept_all, r_all, _, _ = stats.linregress(x_all, time_series)
    
    return {
        'phases': phases,
        'overall_slope': slope_all,
        'overall_slope_per_year': slope_all * 12,
        'overall_r_squared': r_all**2,
        'total_change': time_series[-1] - time_series[0],
        'percent_change': (time_series[-1] - time_series[0]) / time_series[0] * 100 if time_series[0] != 0 else np.nan
    }


def monte_carlo_significance_test(
    observed_power: np.ndarray,
    time_series: np.ndarray,
    n_iterations: int = 1000,
    confidence_level: float = 0.95
) -> Dict:
    """
    Monte Carlo permutation test for FFT spectral peak significance.
    
    "Statistical significance of spectral peaks was assessed using Monte Carlo 
    permutation testing (n=1,000) against red noise null hypotheses"
    
    Args:
        observed_power: Observed power spectrum from FFT
        time_series: Original time series
        n_iterations: Number of Monte Carlo iterations (1000)
        confidence_level: Confidence level for significance (0.95)
        
    Returns:
        Dictionary containing:
            - significant_peaks: Boolean mask of significant peaks
            - threshold: Power threshold for significance
            - p_values: P-values for each frequency
    """
    n = len(time_series)
    n_freq = len(observed_power)
    
    # Generate red noise surrogates
    surrogate_powers = np.zeros((n_iterations, n_freq))
    
    for i in range(n_iterations):
        # Generate red noise (AR(1) process)
        ar_coef = _estimate_ar1_coefficient(time_series)
        noise = np.random.randn(n)
        surrogate = np.zeros(n)
        surrogate[0] = noise[0]
        for t in range(1, n):
            surrogate[t] = ar_coef * surrogate[t-1] + noise[t]
        
        # FFT of surrogate
        surrogate_fft = fft(surrogate - np.mean(surrogate))
        surrogate_powers[i] = np.abs(surrogate_fft[1:n_freq+1])**2
    
    # Calculate p-values (proportion of surrogates with higher power)
    p_values = np.zeros(n_freq)
    for j in range(n_freq):
        p_values[j] = np.mean(surrogate_powers[:, j] >= observed_power[j])
    
    # Determine significance threshold
    threshold_idx = int((1 - confidence_level) * n_iterations)
    sorted_max_powers = np.sort(np.max(surrogate_powers, axis=1))
    threshold = sorted_max_powers[-threshold_idx] if threshold_idx > 0 else sorted_max_powers[-1]
    
    # Identify significant peaks
    significant_peaks = observed_power > threshold
    
    return {
        'significant_peaks': significant_peaks,
        'threshold': threshold,
        'p_values': p_values,
        'confidence_level': confidence_level
    }


def _estimate_ar1_coefficient(time_series: np.ndarray) -> float:
    """Estimate AR(1) coefficient from time series."""
    n = len(time_series)
    x = time_series[:-1]
    y = time_series[1:]
    
    # Pearson correlation as AR(1) estimate
    r = np.corrcoef(x, y)[0, 1]
    
    return r if not np.isnan(r) else 0.0


def vegetation_contamination_coupling(
    vegetation_series: np.ndarray,
    contamination_series: np.ndarray
) -> Dict:
    """
    Analyze coupling between vegetation and contamination dynamics.
    
    "Vegetation-contamination correlation r=0.516"
    
    Args:
        vegetation_series: Vegetation area/index time series
        contamination_series: Contamination area/index time series
        
    Returns:
        Dictionary containing coupling metrics
    """
    n = len(vegetation_series)
    
    # Correlation analysis
    correlation, p_value = stats.pearsonr(vegetation_series, contamination_series)
    
    # Cross-correlation
    cross_corr = np.correlate(
        vegetation_series - np.mean(vegetation_series),
        contamination_series - np.mean(contamination_series),
        mode='full'
    )
    cross_corr /= n * np.std(vegetation_series) * np.std(contamination_series)
    lags = np.arange(-n + 1, n)
    
    # Find lag with maximum correlation
    max_lag_idx = np.argmax(np.abs(cross_corr))
    optimal_lag = lags[max_lag_idx]
    max_cross_corr = cross_corr[max_lag_idx]
    
    # Granger causality test (simplified)
    # Test if contamination helps predict vegetation
    granger_stats = _simplified_granger_test(vegetation_series, contamination_series)
    
    return {
        'correlation': correlation,
        'correlation_p_value': p_value,
        'cross_correlation': cross_corr,
        'lags': lags,
        'optimal_lag': optimal_lag,
        'max_cross_correlation': max_cross_corr,
        'granger_f_statistic': granger_stats['f_statistic'],
        'granger_p_value': granger_stats['p_value']
    }


def _simplified_granger_test(
    y: np.ndarray,
    x: np.ndarray,
    max_lag: int = 3
) -> Dict:
    """Simplified Granger causality test."""
    n = len(y)
    
    # Create lagged matrices
    Y = y[max_lag:]
    
    # Restricted model (only autoregressive terms)
    X_restricted = np.column_stack([y[max_lag-i:-i] for i in range(1, max_lag+1)])
    
    # Unrestricted model (autoregressive + x terms)
    X_unrestricted = np.column_stack([
        X_restricted,
        *[x[max_lag-i:-i if i > 0 else len(x)] for i in range(1, max_lag+1)]
    ])
    
    # Fit models using least squares
    try:
        # Restricted model
        beta_r = np.linalg.lstsq(X_restricted, Y, rcond=None)[0]
        residuals_r = Y - X_restricted @ beta_r
        rss_r = np.sum(residuals_r**2)
        
        # Unrestricted model
        beta_u = np.linalg.lstsq(X_unrestricted, Y, rcond=None)[0]
        residuals_u = Y - X_unrestricted @ beta_u
        rss_u = np.sum(residuals_u**2)
        
        # F-statistic
        df1 = max_lag
        df2 = n - 2 * max_lag - 1
        f_stat = ((rss_r - rss_u) / df1) / (rss_u / df2)
        p_value = 1 - stats.f.cdf(f_stat, df1, df2)
        
        return {'f_statistic': f_stat, 'p_value': p_value}
    except:
        return {'f_statistic': np.nan, 'p_value': np.nan}


if __name__ == "__main__":
    # Example usage with synthetic data
    print("Temporal Analysis Module")
    print("=" * 50)
    
    # Create synthetic time series (63 months)
    np.random.seed(42)
    n_months = 63
    t = np.arange(n_months)
    
    # Synthetic vegetation signal with seasonal pattern and trend
    trend = 0.01 * t
    seasonal = 0.3 * np.sin(2 * np.pi * t / 12)  # Annual cycle
    noise = 0.1 * np.random.randn(n_months)
    vegetation = 2.0 + trend + seasonal + noise
    
    # Synthetic contamination (decreasing)
    contamination = 150 - 0.8 * t + 5 * np.random.randn(n_months)
    
    # FFT Analysis
    print("\nFFT Analysis:")
    fft_results = fft_analysis(vegetation)
    print(f"  Dominant periods (months): {fft_results['dominant_periods'][:3]}")
    print(f"  Power percentages: {fft_results['dominant_power_pct'][:3]}")
    
    # CUSUM Analysis
    print("\nCUSUM Analysis:")
    cusum_results = cusum_analysis(vegetation)
    print(f"  Change points detected: {len(cusum_results['change_points'])}")
    if len(cusum_results['change_points']) > 0:
        print(f"  Change point indices: {cusum_results['change_points']}")
    
    # Recovery Phase Analysis
    print("\nRecovery Phase Analysis:")
    phase_results = recovery_phase_analysis(contamination)
    for phase in phase_results['phases']:
        print(f"  Phase {phase['phase_number']}: slope = {phase['slope_per_year']:.3f}/year")
    
    # Coupling Analysis
    print("\nVegetation-Contamination Coupling:")
    coupling = vegetation_contamination_coupling(vegetation, contamination)
    print(f"  Correlation: {coupling['correlation']:.3f}")
    print(f"  Optimal lag: {coupling['optimal_lag']} months")
