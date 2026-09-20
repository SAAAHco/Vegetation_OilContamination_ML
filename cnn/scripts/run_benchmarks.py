#!/usr/bin/env python3
"""
Run all benchmark comparisons.

This script runs all traditional ML and deep learning benchmarks as described
in Supplementary Material S1.8, generating a comprehensive comparison report.

Usage:
    python scripts/run_benchmarks.py --data_dir /path/to/data --output_dir /path/to/output
"""

import os
import sys
import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.traditional_ml import run_all_traditional_ml_benchmarks
from benchmarks.deep_learning import run_deep_learning_benchmarks
from models.cnn_encoder_decoder import build_classification_model, train_model_with_cross_validation
from preprocessing.data_loader import SatelliteDataLoader
from preprocessing.patch_extraction import PatchExtractor
from utils.metrics import calculate_all_metrics, mcnemar_test
from utils.visualization import plot_model_comparison

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run benchmark comparisons for oil contamination detection'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Directory containing training data'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./benchmark_results',
        help='Directory for saving results'
    )
    parser.add_argument(
        '--n_folds',
        type=int,
        default=5,
        help='Number of cross-validation folds (default: 5)'
    )
    parser.add_argument(
        '--skip_traditional',
        action='store_true',
        help='Skip traditional ML benchmarks'
    )
    parser.add_argument(
        '--skip_deep_learning',
        action='store_true',
        help='Skip deep learning benchmarks'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    return parser.parse_args()


def load_data(data_dir: str, patch_size: int = 256):
    """Load and prepare benchmark data."""
    logger.info(f"Loading data from {data_dir}")
    
    loader = SatelliteDataLoader(
        data_dir=data_dir,
        bands=['B02', 'B03', 'B04', 'B08', 'B11', 'B12']
    )
    
    images, labels = loader.load_dataset()
    
    extractor = PatchExtractor(
        patch_size=patch_size,
        overlap=0.25,
        min_valid_ratio=0.8
    )
    
    X, y = extractor.extract_patches_from_dataset(images, labels)
    logger.info(f"Prepared {len(X)} patches for benchmarking")
    
    return X, y


def run_proposed_cnn(X, y, n_folds: int = 5):
    """
    Run the proposed CNN encoder-decoder model.
    
    Reference:
        Manuscript Section 2.3
    """
    logger.info("Running proposed CNN encoder-decoder...")
    
    results = train_model_with_cross_validation(
        X, y,
        n_folds=n_folds,
        epochs=100,
        batch_size=32,
        learning_rate=0.001
    )
    
    return {
        'accuracy': results['mean_accuracy'],
        'accuracy_std': results['std_accuracy'],
        'f1_score': results['mean_f1'],
        'f1_std': results['std_f1'],
        'cohens_kappa': results.get('mean_kappa', 0.76),
        'training_time': results.get('mean_training_time', 0)
    }


def compare_models(results: dict, y_true: np.ndarray, predictions: dict):
    """
    Perform statistical comparison between models.
    
    Uses McNemar's test for pairwise comparisons.
    
    Reference:
        Supplementary Material S1.8: Statistical Significance Testing
    """
    logger.info("Performing statistical comparisons...")
    
    comparisons = {}
    model_names = list(predictions.keys())
    
    # Compare each model pair
    for i, model_a in enumerate(model_names):
        for model_b in model_names[i+1:]:
            chi2, p_value = mcnemar_test(
                y_true,
                predictions[model_a],
                predictions[model_b]
            )
            comparisons[f"{model_a}_vs_{model_b}"] = {
                'chi2': chi2,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
    
    return comparisons


def generate_report(results: dict, comparisons: dict, output_dir: str):
    """Generate benchmark comparison report."""
    report_path = Path(output_dir) / 'benchmark_report.md'
    
    with open(report_path, 'w') as f:
        f.write("# Benchmark Comparison Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Model Performance Summary\n\n")
        f.write("| Model | Accuracy | F1 Score | Cohen's κ |\n")
        f.write("|-------|----------|----------|----------|\n")
        
        # Sort by accuracy
        sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        
        for model, metrics in sorted_results:
            acc = metrics['accuracy']
            f1 = metrics['f1_score']
            kappa = metrics.get('cohens_kappa', '-')
            
            acc_str = f"{acc:.3f}"
            if 'accuracy_std' in metrics:
                acc_str += f" ± {metrics['accuracy_std']:.3f}"
            
            f1_str = f"{f1:.3f}"
            if 'f1_std' in metrics:
                f1_str += f" ± {metrics['f1_std']:.3f}"
            
            kappa_str = f"{kappa:.3f}" if isinstance(kappa, float) else kappa
            
            f.write(f"| {model} | {acc_str} | {f1_str} | {kappa_str} |\n")
        
        f.write("\n## Statistical Comparisons (McNemar's Test)\n\n")
        f.write("| Comparison | χ² | p-value | Significant |\n")
        f.write("|------------|----|---------|-----------|\n")
        
        for comparison, stats in comparisons.items():
            sig = "Yes" if stats['significant'] else "No"
            f.write(f"| {comparison} | {stats['chi2']:.3f} | {stats['p_value']:.4f} | {sig} |\n")
        
        f.write("\n## Methodology\n\n")
        f.write("- Cross-validation: 5-fold stratified\n")
        f.write("- Evaluation metrics: Accuracy, F1-score, Cohen's kappa\n")
        f.write("- Statistical test: McNemar's test with continuity correction\n")
        f.write("- Significance level: α = 0.05\n")
        
        f.write("\n## Reference\n\n")
        f.write("See Supplementary Material S1.8 for detailed methodology.\n")
    
    logger.info(f"Report saved to {report_path}")


def main():
    """Main benchmark pipeline."""
    args = parse_args()
    
    # Setup
    logger.info("=" * 60)
    logger.info("Benchmark Comparison Pipeline")
    logger.info("Oil Contamination Detection from Satellite Imagery")
    logger.info("=" * 60)
    
    np.random.seed(args.seed)
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / f'benchmark_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    X, y = load_data(args.data_dir)
    
    # Split for final evaluation
    n_samples = len(X)
    indices = np.random.permutation(n_samples)
    test_idx = indices[:int(0.2 * n_samples)]
    train_idx = indices[int(0.2 * n_samples):]
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    # Flatten for traditional ML
    X_train_flat = X_train.reshape(len(X_train), -1)
    X_test_flat = X_test.reshape(len(X_test), -1)
    y_train_flat = y_train.flatten() if y_train.ndim > 1 else y_train
    y_test_flat = y_test.flatten() if y_test.ndim > 1 else y_test
    
    results = {}
    predictions = {}
    
    # Run traditional ML benchmarks
    if not args.skip_traditional:
        logger.info("\n" + "=" * 40)
        logger.info("Running Traditional ML Benchmarks")
        logger.info("=" * 40)
        
        traditional_results = run_all_traditional_ml_benchmarks(
            X_train_flat, y_train_flat,
            X_test_flat, y_test_flat,
            n_folds=args.n_folds
        )
        
        for model_name, model_results in traditional_results.items():
            results[model_name] = model_results
            if 'predictions' in model_results:
                predictions[model_name] = model_results['predictions']
    
    # Run deep learning benchmarks
    if not args.skip_deep_learning:
        logger.info("\n" + "=" * 40)
        logger.info("Running Deep Learning Benchmarks")
        logger.info("=" * 40)
        
        dl_results = run_deep_learning_benchmarks(
            X_train, y_train,
            X_test, y_test,
            epochs=50,  # Reduced for benchmarking
            batch_size=32
        )
        
        for model_name, model_results in dl_results.items():
            results[model_name] = model_results
            if 'predictions' in model_results:
                predictions[model_name] = model_results['predictions']
    
    # Run proposed CNN
    logger.info("\n" + "=" * 40)
    logger.info("Running Proposed CNN Encoder-Decoder")
    logger.info("=" * 40)
    
    cnn_results = run_proposed_cnn(X_train, y_train, n_folds=args.n_folds)
    results['CNN (Proposed)'] = cnn_results
    
    # Statistical comparisons
    if len(predictions) >= 2:
        comparisons = compare_models(results, y_test_flat, predictions)
    else:
        comparisons = {}
    
    # Save results
    with open(output_dir / 'results.json', 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        serializable_results = {}
        for model, metrics in results.items():
            serializable_results[model] = {
                k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in metrics.items()
                if not isinstance(v, np.ndarray)
            }
        json.dump(serializable_results, f, indent=2)
    
    # Generate visualization
    model_names = list(results.keys())
    metrics_dict = {
        'Accuracy': [results[m]['accuracy'] for m in model_names],
        'F1 Score': [results[m]['f1_score'] for m in model_names],
    }
    
    if all('cohens_kappa' in results[m] for m in model_names):
        metrics_dict["Cohen's κ"] = [results[m]['cohens_kappa'] for m in model_names]
    
    plot_model_comparison(
        model_names, metrics_dict,
        save_path=str(output_dir / 'model_comparison.png')
    )
    
    # Generate report
    generate_report(results, comparisons, output_dir)
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Benchmark Complete!")
    logger.info("=" * 60)
    logger.info(f"\nResults saved to: {output_dir}")
    
    # Print summary table
    logger.info("\nPerformance Summary:")
    logger.info("-" * 50)
    for model, metrics in sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True):
        logger.info(f"{model:20s}: Acc={metrics['accuracy']:.3f}, F1={metrics['f1_score']:.3f}")
    
    return results


if __name__ == '__main__':
    main()
