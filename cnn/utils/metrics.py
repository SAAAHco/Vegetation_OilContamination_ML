"""
Evaluation metrics for oil contamination detection models.

This module provides comprehensive metrics for evaluating model performance,
including accuracy, F1-score, Cohen's kappa, and statistical significance tests.

Reference:
    Supplementary Material S1.8: Comparative Analysis
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union


def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate overall accuracy.
    
    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels
    y_pred : np.ndarray
        Predicted labels
        
    Returns
    -------
    float
        Accuracy score (0-1)
    """
    return np.mean(y_true == y_pred)


def calculate_precision(y_true: np.ndarray, y_pred: np.ndarray, 
                        average: str = 'weighted') -> float:
    """
    Calculate precision score.
    
    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels
    y_pred : np.ndarray
        Predicted labels
    average : str
        Averaging method ('micro', 'macro', 'weighted')
        
    Returns
    -------
    float
        Precision score
    """
    classes = np.unique(np.concatenate([y_true, y_pred]))
    precisions = []
    weights = []
    
    for c in classes:
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        
        if tp + fp > 0:
            precision = tp / (tp + fp)
        else:
            precision = 0.0
            
        precisions.append(precision)
        weights.append(np.sum(y_true == c))
    
    if average == 'micro':
        tp_total = sum(np.sum((y_pred == c) & (y_true == c)) for c in classes)
        fp_total = sum(np.sum((y_pred == c) & (y_true != c)) for c in classes)
        return tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    elif average == 'macro':
        return np.mean(precisions)
    else:  # weighted
        return np.average(precisions, weights=weights)


def calculate_recall(y_true: np.ndarray, y_pred: np.ndarray,
                     average: str = 'weighted') -> float:
    """
    Calculate recall score.
    
    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels
    y_pred : np.ndarray
        Predicted labels
    average : str
        Averaging method ('micro', 'macro', 'weighted')
        
    Returns
    -------
    float
        Recall score
    """
    classes = np.unique(np.concatenate([y_true, y_pred]))
    recalls = []
    weights = []
    
    for c in classes:
        tp = np.sum((y_pred == c) & (y_true == c))
        fn = np.sum((y_pred != c) & (y_true == c))
        
        if tp + fn > 0:
            recall = tp / (tp + fn)
        else:
            recall = 0.0
            
        recalls.append(recall)
        weights.append(np.sum(y_true == c))
    
    if average == 'micro':
        tp_total = sum(np.sum((y_pred == c) & (y_true == c)) for c in classes)
        fn_total = sum(np.sum((y_pred != c) & (y_true == c)) for c in classes)
        return tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    elif average == 'macro':
        return np.mean(recalls)
    else:  # weighted
        return np.average(recalls, weights=weights)


def calculate_f1_score(y_true: np.ndarray, y_pred: np.ndarray,
                       average: str = 'weighted') -> float:
    """
    Calculate F1 score.
    
    F1 = 2 * (precision * recall) / (precision + recall)
    
    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels
    y_pred : np.ndarray
        Predicted labels
    average : str
        Averaging method ('micro', 'macro', 'weighted')
        
    Returns
    -------
    float
        F1 score
    """
    precision = calculate_precision(y_true, y_pred, average)
    recall = calculate_recall(y_true, y_pred, average)
    
    if precision + recall > 0:
        return 2 * (precision * recall) / (precision + recall)
    return 0.0


def calculate_cohens_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Cohen's kappa coefficient.
    
    Kappa measures inter-rater agreement accounting for chance:
    κ = (p_o - p_e) / (1 - p_e)
    
    where p_o is observed agreement and p_e is expected agreement by chance.
    
    Reference:
        Supplementary Material S1.8, Table S3
    
    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels
    y_pred : np.ndarray
        Predicted labels
        
    Returns
    -------
    float
        Cohen's kappa coefficient (-1 to 1)
    """
    # Build confusion matrix
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(classes)
    n_samples = len(y_true)
    
    # Create class mapping
    class_to_idx = {c: i for i, c in enumerate(classes)}
    
    # Build confusion matrix
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[class_to_idx[t], class_to_idx[p]] += 1
    
    # Observed agreement
    p_o = np.trace(cm) / n_samples
    
    # Expected agreement by chance
    row_sums = cm.sum(axis=1)
    col_sums = cm.sum(axis=0)
    p_e = np.sum(row_sums * col_sums) / (n_samples ** 2)
    
    # Calculate kappa
    if p_e == 1:
        return 1.0  # Perfect agreement
    
    return (p_o - p_e) / (1 - p_e)


def calculate_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                               normalize: Optional[str] = None) -> np.ndarray:
    """
    Calculate confusion matrix.
    
    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels
    y_pred : np.ndarray
        Predicted labels
    normalize : str, optional
        Normalization method ('true', 'pred', 'all', or None)
        
    Returns
    -------
    np.ndarray
        Confusion matrix of shape (n_classes, n_classes)
    """
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(classes)
    
    # Create class mapping
    class_to_idx = {c: i for i, c in enumerate(classes)}
    
    # Build confusion matrix
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[class_to_idx[t], class_to_idx[p]] += 1
    
    # Normalize if requested
    if normalize == 'true':
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    elif normalize == 'pred':
        cm = cm.astype(float) / cm.sum(axis=0, keepdims=True)
    elif normalize == 'all':
        cm = cm.astype(float) / cm.sum()
    
    return cm


def calculate_iou(y_true: np.ndarray, y_pred: np.ndarray,
                  class_id: Optional[int] = None) -> Union[float, Dict[int, float]]:
    """
    Calculate Intersection over Union (IoU) score.
    
    IoU = TP / (TP + FP + FN)
    
    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels
    y_pred : np.ndarray
        Predicted labels
    class_id : int, optional
        Specific class to calculate IoU for. If None, returns dict for all classes.
        
    Returns
    -------
    float or dict
        IoU score(s)
    """
    classes = np.unique(np.concatenate([y_true, y_pred]))
    
    def _iou_for_class(c):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        
        if tp + fp + fn > 0:
            return tp / (tp + fp + fn)
        return 0.0
    
    if class_id is not None:
        return _iou_for_class(class_id)
    
    return {c: _iou_for_class(c) for c in classes}


def calculate_dice_coefficient(y_true: np.ndarray, y_pred: np.ndarray,
                               class_id: Optional[int] = None) -> Union[float, Dict[int, float]]:
    """
    Calculate Dice coefficient (F1 for binary segmentation).
    
    Dice = 2*TP / (2*TP + FP + FN)
    
    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels
    y_pred : np.ndarray
        Predicted labels
    class_id : int, optional
        Specific class to calculate Dice for
        
    Returns
    -------
    float or dict
        Dice coefficient(s)
    """
    classes = np.unique(np.concatenate([y_true, y_pred]))
    
    def _dice_for_class(c):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        
        if 2*tp + fp + fn > 0:
            return 2*tp / (2*tp + fp + fn)
        return 0.0
    
    if class_id is not None:
        return _dice_for_class(class_id)
    
    return {c: _dice_for_class(c) for c in classes}


def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate all standard evaluation metrics.
    
    Reference:
        Supplementary Material S1.8: Model Performance Metrics
    
    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels
    y_pred : np.ndarray
        Predicted labels
        
    Returns
    -------
    dict
        Dictionary containing all metrics
    """
    return {
        'accuracy': calculate_accuracy(y_true, y_pred),
        'precision': calculate_precision(y_true, y_pred),
        'recall': calculate_recall(y_true, y_pred),
        'f1_score': calculate_f1_score(y_true, y_pred),
        'cohens_kappa': calculate_cohens_kappa(y_true, y_pred),
        'mean_iou': np.mean(list(calculate_iou(y_true, y_pred).values())),
        'mean_dice': np.mean(list(calculate_dice_coefficient(y_true, y_pred).values()))
    }


def mcnemar_test(y_true: np.ndarray, y_pred_a: np.ndarray, 
                 y_pred_b: np.ndarray) -> Tuple[float, float]:
    """
    Perform McNemar's test to compare two classifiers.
    
    Tests whether the two classifiers have significantly different error rates.
    
    Reference:
        Supplementary Material S1.8: Statistical Significance Testing
    
    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels
    y_pred_a : np.ndarray
        Predictions from classifier A
    y_pred_b : np.ndarray
        Predictions from classifier B
        
    Returns
    -------
    tuple
        (chi2 statistic, p-value)
    """
    # Build contingency table
    # b: A correct, B incorrect
    # c: A incorrect, B correct
    correct_a = y_pred_a == y_true
    correct_b = y_pred_b == y_true
    
    b = np.sum(correct_a & ~correct_b)
    c = np.sum(~correct_a & correct_b)
    
    # McNemar's test with continuity correction
    if b + c == 0:
        return 0.0, 1.0
    
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    
    # Calculate p-value from chi-squared distribution with df=1
    from scipy import stats
    p_value = 1 - stats.chi2.cdf(chi2, df=1)
    
    return chi2, p_value


def bootstrap_confidence_interval(y_true: np.ndarray, y_pred: np.ndarray,
                                   metric_func: callable, n_iterations: int = 1000,
                                   confidence: float = 0.95) -> Tuple[float, float, float]:
    """
    Calculate bootstrap confidence interval for a metric.
    
    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels
    y_pred : np.ndarray
        Predicted labels
    metric_func : callable
        Function that takes (y_true, y_pred) and returns a float
    n_iterations : int
        Number of bootstrap iterations
    confidence : float
        Confidence level (0-1)
        
    Returns
    -------
    tuple
        (mean, lower_bound, upper_bound)
    """
    n_samples = len(y_true)
    scores = []
    
    for _ in range(n_iterations):
        # Bootstrap sample
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        score = metric_func(y_true[indices], y_pred[indices])
        scores.append(score)
    
    scores = np.array(scores)
    alpha = (1 - confidence) / 2
    
    return (
        np.mean(scores),
        np.percentile(scores, alpha * 100),
        np.percentile(scores, (1 - alpha) * 100)
    )


def calculate_class_weights(y: np.ndarray) -> Dict[int, float]:
    """
    Calculate class weights for imbalanced datasets.
    
    Uses inverse frequency weighting.
    
    Parameters
    ----------
    y : np.ndarray
        Class labels
        
    Returns
    -------
    dict
        Dictionary mapping class to weight
    """
    classes, counts = np.unique(y, return_counts=True)
    n_samples = len(y)
    n_classes = len(classes)
    
    weights = n_samples / (n_classes * counts)
    
    return {c: w for c, w in zip(classes, weights)}


def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[int, Dict[str, float]]:
    """
    Calculate metrics for each class separately.
    
    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels
    y_pred : np.ndarray
        Predicted labels
        
    Returns
    -------
    dict
        Dictionary mapping class to its metrics
    """
    classes = np.unique(np.concatenate([y_true, y_pred]))
    results = {}
    
    for c in classes:
        # Binary masks for this class
        y_true_binary = (y_true == c).astype(int)
        y_pred_binary = (y_pred == c).astype(int)
        
        tp = np.sum((y_pred_binary == 1) & (y_true_binary == 1))
        fp = np.sum((y_pred_binary == 1) & (y_true_binary == 0))
        fn = np.sum((y_pred_binary == 0) & (y_true_binary == 1))
        tn = np.sum((y_pred_binary == 0) & (y_true_binary == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        results[int(c)] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'specificity': specificity,
            'support': int(np.sum(y_true == c))
        }
    
    return results


if __name__ == '__main__':
    # Example usage
    np.random.seed(42)
    
    # Generate synthetic data
    n_samples = 1000
    n_classes = 4
    
    y_true = np.random.randint(0, n_classes, n_samples)
    # Simulate predictions with ~85% accuracy
    y_pred = y_true.copy()
    noise_idx = np.random.choice(n_samples, size=int(0.15 * n_samples), replace=False)
    y_pred[noise_idx] = np.random.randint(0, n_classes, len(noise_idx))
    
    # Calculate all metrics
    metrics = calculate_all_metrics(y_true, y_pred)
    print("Overall Metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    
    # Confusion matrix
    cm = calculate_confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix:\n{cm}")
    
    # Per-class metrics
    class_metrics = per_class_metrics(y_true, y_pred)
    print("\nPer-class Metrics:")
    for c, m in class_metrics.items():
        print(f"  Class {c}: precision={m['precision']:.3f}, recall={m['recall']:.3f}, f1={m['f1_score']:.3f}")
    
    # Bootstrap confidence interval
    mean_acc, lower, upper = bootstrap_confidence_interval(
        y_true, y_pred, calculate_accuracy, n_iterations=1000
    )
    print(f"\n95% CI for Accuracy: {mean_acc:.4f} [{lower:.4f}, {upper:.4f}]")
