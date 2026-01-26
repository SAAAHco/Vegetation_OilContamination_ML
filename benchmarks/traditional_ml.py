"""
Traditional Machine Learning Benchmark Models

This module implements the traditional ML methods used for benchmarking:
- Random Forest (RF)
- Support Vector Machine (SVM)
- Gradient Boosting (XGBoost)

These methods are compared against the custom CNN in Supplementary Material Section S1.8.

Performance Summary (from manuscript):
    Random Forest: 82.0% accuracy, F1=0.79, κ=0.62
    SVM (RBF): 79.3% accuracy, F1=0.76, κ=0.58
    XGBoost: 83.5% accuracy, F1=0.81, κ=0.65
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    cohen_kappa_score, confusion_matrix, classification_report
)
from typing import Tuple, Dict, Optional, Union
import warnings

# XGBoost import with fallback
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    warnings.warn("XGBoost not installed. Using sklearn GradientBoosting as fallback.")


def prepare_features_for_ml(
    image_patches: np.ndarray,
    flatten: bool = True
) -> np.ndarray:
    """
    Prepare image patches for traditional ML models.
    
    Traditional ML models require flattened feature vectors, not 2D images.
    
    Args:
        image_patches: Array of shape (n_samples, height, width, channels)
        flatten: Whether to flatten spatial dimensions
        
    Returns:
        Feature array suitable for ML models
    """
    if flatten:
        n_samples = image_patches.shape[0]
        # Flatten to (n_samples, height*width*channels)
        features = image_patches.reshape(n_samples, -1)
    else:
        features = image_patches
    
    return features


def build_random_forest(
    n_estimators: int = 500,
    max_depth: int = 20,
    min_samples_split: int = 5,
    random_state: int = 42,
    n_jobs: int = -1
) -> RandomForestClassifier:
    """
    Build Random Forest classifier with hyperparameters from manuscript.
    
    From Supplementary Material Section S1.8:
    "Random Forest (RF): 500 trees, max depth=20, min samples split=5; 
    achieved 82.0% accuracy, F1=0.79, κ=0.62"
    
    Args:
        n_estimators: Number of trees (500)
        max_depth: Maximum tree depth (20)
        min_samples_split: Minimum samples to split node (5)
        random_state: Random seed for reproducibility
        n_jobs: Number of parallel jobs (-1 for all cores)
        
    Returns:
        Configured RandomForestClassifier
    """
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=True,
        oob_score=True,
        random_state=random_state,
        n_jobs=n_jobs,
        class_weight='balanced'
    )
    
    return rf


def build_svm(
    kernel: str = 'rbf',
    C: float = 10.0,
    gamma: float = 0.01,
    random_state: int = 42
) -> SVC:
    """
    Build Support Vector Machine classifier with hyperparameters from manuscript.
    
    From Supplementary Material Section S1.8:
    "Support Vector Machine (SVM): RBF kernel, C=10, γ=0.01; 
    achieved 79.3% accuracy, F1=0.76, κ=0.58"
    
    Args:
        kernel: Kernel type ('rbf')
        C: Regularization parameter (10)
        gamma: Kernel coefficient (0.01)
        random_state: Random seed
        
    Returns:
        Configured SVC
    """
    svm = SVC(
        kernel=kernel,
        C=C,
        gamma=gamma,
        probability=True,
        class_weight='balanced',
        random_state=random_state
    )
    
    return svm


def build_xgboost(
    n_estimators: int = 200,
    learning_rate: float = 0.1,
    max_depth: int = 6,
    random_state: int = 42
) -> Union['XGBClassifier', GradientBoostingClassifier]:
    """
    Build XGBoost/Gradient Boosting classifier with hyperparameters from manuscript.
    
    From Supplementary Material Section S1.8:
    "Gradient Boosting (XGBoost): 200 estimators, learning rate=0.1, max depth=6; 
    achieved 83.5% accuracy, F1=0.81, κ=0.65"
    
    Args:
        n_estimators: Number of boosting rounds (200)
        learning_rate: Learning rate (0.1)
        max_depth: Maximum tree depth (6)
        random_state: Random seed
        
    Returns:
        Configured XGBClassifier or GradientBoostingClassifier
    """
    if HAS_XGBOOST:
        xgb = XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=1,
            reg_alpha=0.01,
            reg_lambda=1.0,
            random_state=random_state,
            use_label_encoder=False,
            eval_metric='logloss',
            n_jobs=-1
        )
    else:
        # Fallback to sklearn GradientBoosting
        xgb = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=0.8,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state
        )
    
    return xgb


def train_and_evaluate_model(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Model",
    scale_features: bool = True
) -> Dict:
    """
    Train and evaluate a traditional ML model.
    
    Args:
        model: sklearn-compatible classifier
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        model_name: Name for logging
        scale_features: Whether to standardize features (required for SVM)
        
    Returns:
        Dictionary with evaluation metrics
    """
    print(f"\n{'='*50}")
    print(f"Training {model_name}")
    print(f"{'='*50}")
    
    # Scale features if needed
    if scale_features:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test
        scaler = None
    
    # Train
    model.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Calculate metrics
    results = {
        'model_name': model_name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='binary'),
        'recall': recall_score(y_test, y_pred, average='binary'),
        'f1_score': f1_score(y_test, y_pred, average='binary'),
        'cohen_kappa': cohen_kappa_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'scaler': scaler,
        'model': model
    }
    
    # Print results
    print(f"\nResults for {model_name}:")
    print(f"  Accuracy: {results['accuracy']*100:.1f}%")
    print(f"  Precision: {results['precision']:.3f}")
    print(f"  Recall: {results['recall']:.3f}")
    print(f"  F1 Score: {results['f1_score']:.3f}")
    print(f"  Cohen's Kappa: {results['cohen_kappa']:.3f}")
    print(f"\nConfusion Matrix:")
    print(results['confusion_matrix'])
    
    return results


def cross_validate_model(
    model,
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    scale_features: bool = True
) -> Dict:
    """
    Perform k-fold cross-validation.
    
    "All models were trained and evaluated using identical protocols: 
    the same 2,500 image patches (256×256 pixels), 5-fold spatially-blocked 
    cross-validation"
    
    Args:
        model: sklearn-compatible classifier
        X: Features
        y: Labels
        n_folds: Number of CV folds (5)
        scale_features: Whether to standardize features
        
    Returns:
        Dictionary with CV results
    """
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    if scale_features:
        # Scale within each fold to prevent data leakage
        from sklearn.pipeline import Pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])
        model_to_evaluate = pipeline
    else:
        model_to_evaluate = model
    
    # Cross-validation scores
    cv_accuracy = cross_val_score(model_to_evaluate, X, y, cv=cv, scoring='accuracy')
    cv_f1 = cross_val_score(model_to_evaluate, X, y, cv=cv, scoring='f1')
    
    results = {
        'cv_accuracy_mean': cv_accuracy.mean(),
        'cv_accuracy_std': cv_accuracy.std(),
        'cv_f1_mean': cv_f1.mean(),
        'cv_f1_std': cv_f1.std(),
        'cv_accuracy_folds': cv_accuracy,
        'cv_f1_folds': cv_f1
    }
    
    print(f"Cross-Validation Results ({n_folds}-fold):")
    print(f"  Accuracy: {results['cv_accuracy_mean']*100:.1f}% ± {results['cv_accuracy_std']*100:.1f}%")
    print(f"  F1 Score: {results['cv_f1_mean']:.3f} ± {results['cv_f1_std']:.3f}")
    
    return results


def hyperparameter_tuning(
    model_type: str,
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5
) -> Tuple[object, Dict]:
    """
    Perform hyperparameter tuning using grid search.
    
    Args:
        model_type: 'rf', 'svm', or 'xgb'
        X: Features
        y: Labels
        n_folds: Number of CV folds
        
    Returns:
        Tuple of (best_model, best_params)
    """
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    if model_type == 'rf':
        model = RandomForestClassifier(random_state=42, n_jobs=-1)
        param_grid = {
            'n_estimators': [100, 300, 500],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    elif model_type == 'svm':
        model = SVC(random_state=42, probability=True)
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': [0.001, 0.01, 0.1, 'scale'],
            'kernel': ['rbf', 'poly']
        }
    elif model_type == 'xgb':
        if HAS_XGBOOST:
            model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
        else:
            model = GradientBoostingClassifier(random_state=42)
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 6, 9]
        }
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Grid search
    grid_search = GridSearchCV(
        model, param_grid,
        cv=cv,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_scaled, y)
    
    print(f"\nBest parameters for {model_type}:")
    print(grid_search.best_params_)
    print(f"Best CV F1 Score: {grid_search.best_score_:.3f}")
    
    return grid_search.best_estimator_, grid_search.best_params_


def run_all_traditional_ml_benchmarks(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    run_cv: bool = True
) -> Dict:
    """
    Run all traditional ML benchmark models.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        run_cv: Whether to run cross-validation
        
    Returns:
        Dictionary with results for all models
    """
    all_results = {}
    
    # Random Forest
    print("\n" + "="*60)
    print("RANDOM FOREST BENCHMARK")
    print("="*60)
    rf = build_random_forest()
    all_results['random_forest'] = train_and_evaluate_model(
        rf, X_train, y_train, X_test, y_test,
        model_name="Random Forest",
        scale_features=False  # RF doesn't require scaling
    )
    if run_cv:
        rf_cv = cross_validate_model(
            build_random_forest(),
            np.vstack([X_train, X_test]),
            np.hstack([y_train, y_test]),
            scale_features=False
        )
        all_results['random_forest'].update(rf_cv)
    
    # Support Vector Machine
    print("\n" + "="*60)
    print("SVM BENCHMARK")
    print("="*60)
    svm = build_svm()
    all_results['svm'] = train_and_evaluate_model(
        svm, X_train, y_train, X_test, y_test,
        model_name="SVM (RBF)",
        scale_features=True  # SVM requires scaling
    )
    if run_cv:
        svm_cv = cross_validate_model(
            build_svm(),
            np.vstack([X_train, X_test]),
            np.hstack([y_train, y_test]),
            scale_features=True
        )
        all_results['svm'].update(svm_cv)
    
    # XGBoost / Gradient Boosting
    print("\n" + "="*60)
    print("XGBOOST BENCHMARK")
    print("="*60)
    xgb = build_xgboost()
    model_name = "XGBoost" if HAS_XGBOOST else "Gradient Boosting"
    all_results['xgboost'] = train_and_evaluate_model(
        xgb, X_train, y_train, X_test, y_test,
        model_name=model_name,
        scale_features=False  # Tree-based models don't require scaling
    )
    if run_cv:
        xgb_cv = cross_validate_model(
            build_xgboost(),
            np.vstack([X_train, X_test]),
            np.hstack([y_train, y_test]),
            scale_features=False
        )
        all_results['xgboost'].update(xgb_cv)
    
    # Summary table
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    print(f"{'Model':<20} {'Accuracy':>10} {'F1 Score':>10} {'Kappa':>10}")
    print("-" * 52)
    for name, results in all_results.items():
        print(f"{results['model_name']:<20} {results['accuracy']*100:>9.1f}% {results['f1_score']:>10.3f} {results['cohen_kappa']:>10.3f}")
    
    return all_results


def compare_with_reference(results: Dict) -> None:
    """
    Compare results with reference values from manuscript.
    
    Reference values from Supplementary Material S1.8:
    - Random Forest: 82.0% accuracy, F1=0.79, κ=0.62
    - SVM (RBF): 79.3% accuracy, F1=0.76, κ=0.58
    - XGBoost: 83.5% accuracy, F1=0.81, κ=0.65
    """
    reference = {
        'random_forest': {'accuracy': 0.82, 'f1': 0.79, 'kappa': 0.62},
        'svm': {'accuracy': 0.793, 'f1': 0.76, 'kappa': 0.58},
        'xgboost': {'accuracy': 0.835, 'f1': 0.81, 'kappa': 0.65}
    }
    
    print("\n" + "="*60)
    print("COMPARISON WITH MANUSCRIPT REFERENCE VALUES")
    print("="*60)
    print(f"{'Model':<20} {'Metric':<10} {'Obtained':>10} {'Reference':>10} {'Diff':>10}")
    print("-" * 62)
    
    for model_name in ['random_forest', 'svm', 'xgboost']:
        if model_name in results:
            r = results[model_name]
            ref = reference[model_name]
            
            print(f"{model_name:<20} {'Accuracy':<10} {r['accuracy']*100:>9.1f}% {ref['accuracy']*100:>9.1f}% {(r['accuracy']-ref['accuracy'])*100:>+9.1f}%")
            print(f"{'':<20} {'F1 Score':<10} {r['f1_score']:>10.3f} {ref['f1']:>10.3f} {r['f1_score']-ref['f1']:>+10.3f}")
            print(f"{'':<20} {'Kappa':<10} {r['cohen_kappa']:>10.3f} {ref['kappa']:>10.3f} {r['cohen_kappa']-ref['kappa']:>+10.3f}")
            print("-" * 62)


if __name__ == "__main__":
    # Example usage with synthetic data
    print("Traditional ML Benchmarks Module")
    print("=" * 50)
    
    # Create synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 256  # Flattened patch features
    
    X = np.random.randn(n_samples, n_features)
    y = (np.random.rand(n_samples) > 0.5).astype(int)
    
    # Split data
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Run benchmarks
    results = run_all_traditional_ml_benchmarks(
        X_train, y_train, X_test, y_test,
        run_cv=True
    )
    
    # Compare with reference
    compare_with_reference(results)
