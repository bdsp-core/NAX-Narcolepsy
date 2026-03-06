import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid Qt issues

# This script includes confusion matrix generation for:
# 1. Individual cohort folds (saved in confusion_matrices_{model_name}/ subdirectories)
# 2. Aggregate confusion matrices using out-of-fold predictions (saved in aggregate_confusion_matrices/ subdirectory)

import polars as pl
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                            roc_auc_score, average_precision_score, precision_recall_curve,
                            roc_curve, confusion_matrix, classification_report)
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import os
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
import seaborn as sns
from sklearn.base import clone
import argparse
import warnings
warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output
from datetime import datetime
import json

def detect_problem_type(y):
    """
    Detect whether this is a binary or multi-class classification problem.
    
    Args:
        y: Target variable (Polars Series or array-like)
        
    Returns:
        str: 'binary' or 'multiclass'
        int: number of classes
        list: sorted list of unique classes
    """
    if hasattr(y, 'unique'):
        unique_classes = sorted(y.unique().to_list())
    else:
        unique_classes = sorted(np.unique(y))
    
    n_classes = len(unique_classes)
    
    if n_classes == 2:
        return 'binary', n_classes, unique_classes
    elif n_classes > 2:
        return 'multiclass', n_classes, unique_classes
    else:
        raise ValueError(f"Target variable has only {n_classes} unique value(s). Classification requires at least 2 classes.")
    
def get_metric_column_name(scoring_metric):
    """Map sklearn scoring metric names to our internal column names."""
    metric_mapping = {
        'average_precision': 'auprc',
        'roc_auc': 'roc_auc',
        'f1_weighted': 'f1',
        'f1_macro': 'f1',
        'f1': 'f1',
        'accuracy': 'accuracy',
        'precision_weighted': 'precision',
        'recall_weighted': 'recall'
    }
    return metric_mapping.get(scoring_metric, 'f1')  # default to f1 if not found

def calculate_adaptive_metrics(y_true, y_pred, y_proba, problem_type, average='weighted'):
    """
    Calculate metrics that adapt to binary or multi-class problems.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities
        problem_type: 'binary' or 'multiclass'
        average: Averaging strategy for multi-class metrics
        
    Returns:
        dict: Dictionary of calculated metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1': f1_score(y_true, y_pred, average=average, zero_division=0)
    }
    
    if problem_type == 'binary':
        from sklearn.metrics import average_precision_score

        # For binary classification, use the positive class probabilities
        if y_proba.ndim > 1 and y_proba.shape[1] > 1:
            y_proba_pos = y_proba[:, 1]
        else:
            y_proba_pos = y_proba
            
        metrics['roc_auc'] = roc_auc_score(y_true, y_proba_pos)
        metrics['auprc'] = average_precision_score(y_true, y_proba_pos)
    else:
        # For multi-class classification
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba, multi_class='ovr', average=average)
        except ValueError:
            # If some classes are missing in y_true, ROC AUC can't be calculated
            metrics['roc_auc'] = np.nan
            
        # AUPRC for multi-class is not straightforward, so we'll skip it or use macro average
        try:
            # Calculate macro-averaged AUPRC (average of per-class AUPRC)
            from sklearn.preprocessing import label_binarize
            from sklearn.metrics import average_precision_score
            
            # Get unique classes and binarize
            classes = sorted(np.unique(y_true))
            if len(classes) > 2:
                y_true_bin = label_binarize(y_true, classes=classes)
                auprc_scores = []
                for i in range(len(classes)):
                    if np.sum(y_true_bin[:, i]) > 0:  # Only if class exists in y_true
                        auprc_scores.append(average_precision_score(y_true_bin[:, i], y_proba[:, i]))
                metrics['auprc'] = np.mean(auprc_scores) if auprc_scores else np.nan
            else:
                metrics['auprc'] = np.nan
        except:
            metrics['auprc'] = np.nan
    
    return metrics

def build_adaptive_pipeline(X, model_class):
    """
    Build a pipeline that selectively applies scaling only to non-binary features
    """
    # Identify binary and continuous features
    binary_cols = []
    continuous_cols = [] 
    
    for col in X.columns:
        # Check if column contains only 0s and 1s
        if set(X[col].unique()).issubset({0, 1, 0.0, 1.0}):
            binary_cols.append(col)
        else:
            continuous_cols.append(col)
    
    print(f"Detected {len(binary_cols)} binary features and {len(continuous_cols)} continuous features")
    
    # Create preprocessing steps based on feature types
    if len(continuous_cols) > 0:
        # We have mixed feature types - use ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('continuous', StandardScaler(), continuous_cols),
                ('binary', FunctionTransformer(func=None), binary_cols)  # Identity transformation
            ],
            remainder='passthrough'
        )
    else:
        # All features are binary - skip scaling entirely
        preprocessor = FunctionTransformer(func=None)  # Identity transformation
        
    # Create and return the pipeline
    return Pipeline([
        ('preprocessor', preprocessor),
        ('model', model_class())
    ])

def cross_source_model_comparison(df, source_col, target_col, models_config, output_dir='output', random_state=42):
    # Detect problem type
    problem_type, n_classes, unique_classes = detect_problem_type(df[target_col])
    print(f"Detected {problem_type} classification problem with {n_classes} classes: {unique_classes}")
    
    # Get unique sources
    sources = df[source_col].unique().to_list()
    print(f"Found {len(sources)} unique sources: {sources}")
    
    results = []
    curves_data = {}  # To store curve data for plotting
    
    # For each model type
    for model_name, (model_class, param_grid) in models_config.items():
        print(f"\n===== Evaluating {model_name} =====")
        curves_data[model_name] = {}
        
        # For each source as training
        for train_source in sources:
            # Filter data for training source
            train_data = df.filter(pl.col(source_col) == train_source)
            X_train, y_train = prepare_data(train_data, target_col, source_col)
            
            # Create pipeline with current model
            pipeline = build_adaptive_pipeline(X_train, model_class)
            
            # Add 'model__' prefix to param grid keys
            model_param_grid = {f"model__{k}": v for k, v in param_grid.items()}
            
            # Train model
            print(f"\nTraining {model_name} on source: {train_source}")
            
            # Create a stratified cross-validation strategy to maintain class distribution
            from sklearn.model_selection import StratifiedKFold
            cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            # Choose scoring metric based on problem type
            scoring_metric = 'roc_auc' if problem_type == 'binary' else 'f1_weighted'
            
            grid_search = GridSearchCV(
                pipeline,
                param_grid=model_param_grid,
                cv=cv_strategy,
                scoring=scoring_metric,
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            
            print(f"Best parameters for {model_name} on {train_source}: {best_params}")
            
            # Test on each other source
            for test_source in sources:
                if test_source != train_source:
                    print(f"Testing {model_name} on source: {test_source}")
                    
                    # Filter data for test source
                    test_data = df.filter(pl.col(source_col) == test_source)
                    X_test, y_test = prepare_data(test_data, target_col, source_col)
                    
                    # Predict and evaluate
                    y_pred = best_model.predict(X_test)
                    y_proba = best_model.predict_proba(X_test)
                    
                    # Calculate adaptive metrics
                    metrics = calculate_adaptive_metrics(y_test, y_pred, y_proba, problem_type)
                    
                    # Add metadata
                    metrics.update({
                        'model': model_name,
                        'train_source': train_source,
                        'test_source': test_source,
                        'problem_type': problem_type,
                        'n_classes': n_classes,
                        'best_params': str(best_params)
                    })
                    
                    # Store curve data for plotting
                    curves_data[model_name][f"{train_source}_{test_source}"] = {
                        'y_test': y_test,
                        'y_proba': y_proba,
                        'y_pred': y_pred,
                        'problem_type': problem_type,
                        'n_classes': n_classes
                    }
                    
                    results.append(metrics)
                    
                    print(f"{model_name} {train_source} → {test_source} Performance:")
                    print(f"Accuracy: {metrics['accuracy']:.4f}")
                    print(f"Precision: {metrics['precision']:.4f}")
                    print(f"Recall: {metrics['recall']:.4f}")
                    print(f"F1 Score: {metrics['f1']:.4f}")
                    print(f"ROC AUC Score: {metrics.get('roc_auc', 'N/A')}")
                    if not np.isnan(metrics.get('auprc', np.nan)):
                        print(f"AUPRC Score: {metrics['auprc']:.4f}")
    
    # Convert results to Polars DataFrame
    results_df = pl.DataFrame(results)
    return results_df, curves_data

def prepare_data(source_df, target_col, source_col):
    """
    Prepare data for model training/testing with proper checks for data quality.
    
    Args:
        source_df: Source DataFrame
        target_col: Target column name
        source_col: Source column name
        
    Returns:
        X, y: Features and target arrays
    """
    # Check if required columns exist
    if target_col not in source_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data")
    
    if source_col not in source_df.columns:
        raise ValueError(f"Source column '{source_col}' not found in data")
    
    X = source_df.drop([target_col, source_col])
    y = source_df[target_col]
    
    # Detect problem type for quality checks
    problem_type, n_classes, unique_classes = detect_problem_type(y)
    
    # Check for missing values in features
    missing_cols = X.null_count().transpose(include_header=True).filter(pl.col('column_0')>0)['column'].to_list()
    if missing_cols:
        print(f"WARNING: Missing values detected in {len(missing_cols)} feature columns. Consider imputation strategy.")
    
    return X, y

def find_optimal_threshold(y_true, y_proba, metric='f1'):
    """
    Find the optimal threshold that maximizes a given metric.
    Only works for binary classification problems.
    
    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities for positive class
        metric: Metric to optimize ('f1', 'balanced_accuracy', 'youdens_j', 'precision_recall_product')
        
    Returns:
        optimal_threshold: The threshold that maximizes the chosen metric
        metrics_at_threshold: Dictionary of metrics at the optimal threshold
    """
    from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
    
    # Generate a range of thresholds to evaluate
    thresholds = np.linspace(0.01, 0.99, 99)
    
    # Store metrics for each threshold
    metrics = {}
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        
        # Calculate confusion matrix values
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Calculate various metrics
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        balanced_accuracy = (recall + specificity) / 2
        youdens_j = recall + specificity - 1
        precision_recall_product = precision * recall
        
        metrics[threshold] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'specificity': specificity,
            'balanced_accuracy': balanced_accuracy,
            'youdens_j': youdens_j,
            'precision_recall_product': precision_recall_product
        }
    
    # Find threshold that maximizes the chosen metric
    if metric in ['f1', 'balanced_accuracy', 'youdens_j', 'precision_recall_product']:
        optimal_threshold = max(metrics.items(), key=lambda x: x[1][metric])[0]
    else:
        raise ValueError(f"Unsupported metric: {metric}. Choose from 'f1', 'balanced_accuracy', 'youdens_j', 'precision_recall_product'")
    
    return optimal_threshold, metrics[optimal_threshold]

def train_final_production_model(df, source_col, target_col, best_model_config, 
                                output_dir='output', model_version=None, random_state=42):
    """
    Train final production model on ALL available data.
    
    This function should only be called AFTER proper cross-validation has been performed
    to select the best model and hyperparameters. No evaluation is performed here to
    avoid data leakage.
    
    Parameters:
    -----------
    df : polars.DataFrame
        Complete dataset to train on
    source_col : str or None
        Name of source column to exclude from features
    target_col : str
        Name of target column
    best_model_config : tuple
        (model_name, model_class, best_params) from validation results
    output_dir : str
        Directory to save model artifacts
    model_version : str, optional
        Version identifier for the model
    random_state : int
        Random state for reproducibility
        
    Returns:
    --------
    final_model : sklearn pipeline
        Trained production model
    model_info : dict
        Metadata about the trained model
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Unpack best model configuration
    model_name, model_class, best_params = best_model_config
    
    print(f"Training final production {model_name} model...")
    print(f"Dataset size: {len(df):,} samples")
    
    # Detect problem type and classes
    problem_type, n_classes, unique_classes = detect_problem_type(df[target_col])
    print(f"Problem type: {problem_type} classification with {n_classes} classes: {unique_classes}")
    
    # Prepare features and target
    if source_col is not None:
        X = df.drop([target_col, source_col])
        print(f"Excluded source column: {source_col}")
    else:
        X = df.drop([target_col])
    y = df[target_col]
    
    print(f"Feature columns: {X.columns}")
    print(f"Number of features: {len(X.columns)}")
    
    # Create pipeline with the best model
    print(f"Building pipeline with {model_name}...")
    pipeline = build_adaptive_pipeline(X, model_class)
    
    # Set the best parameters (add model__ prefix if not already present)
    final_params = {}
    for k, v in best_params.items():
        if not k.startswith('model__'):
            final_params[f'model__{k}'] = v
        else:
            final_params[k] = v
    
    print(f"Setting best parameters: {final_params}")
    pipeline.set_params(**final_params)
    
    # Train on ALL available data
    print("Training model on complete dataset...")
    training_start = datetime.now()
    
    # Set random state if the model supports it
    if hasattr(pipeline.named_steps['model'], 'random_state'):
        pipeline.named_steps['model'].random_state = random_state
    
    final_model = pipeline.fit(X, y)
    training_end = datetime.now()
    training_time = (training_end - training_start).total_seconds()
    
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Create comprehensive model info
    if model_version is None:
        model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    model_info = {
        # Model identification
        'model_name': model_name,
        'model_version': model_version,
        'training_timestamp': training_start.isoformat(),
        'training_duration_seconds': training_time,
        
        # Data information
        'total_samples': len(df),
        'n_features': len(X.columns),
        'feature_names': X.columns,
        'target_column': target_col,
        'source_column': source_col,
        
        # Problem characteristics
        'problem_type': problem_type,
        'n_classes': n_classes,
        'unique_classes': unique_classes.tolist() if hasattr(unique_classes, 'tolist') else list(unique_classes),
        'class_distribution': dict(zip(*y.value_counts().to_numpy().T)),
        
        # Model configuration
        'best_params': best_params,
        'final_pipeline_params': final_params,
        'random_state': random_state,
        
        # Pipeline information
        'pipeline_steps': [step for step, _ in pipeline.steps],
        'preprocessing_steps': [step for step, _ in pipeline.steps[:-1]],
        
        # Additional metadata
        'sklearn_version': None,  # You can add this if needed
        'python_version': None,   # You can add this if needed
        'notes': f'Production model trained on complete dataset after cross-validation. No evaluation performed to avoid data leakage.'
    }
    
    # Try to get sklearn version if available
    try:
        import sklearn
        model_info['sklearn_version'] = sklearn.__version__
    except:
        pass
        
    # Try to get python version if available
    try:
        import sys
        model_info['python_version'] = sys.version
    except:
        pass
    
    # Save model artifacts
    model_filename = f'production_model_{model_name}_{model_version}.pkl'
    info_filename = f'model_info_{model_name}_{model_version}.pkl'
    
    model_path = os.path.join(output_dir, model_filename)
    info_path = os.path.join(output_dir, info_filename)
    
    print(f"Saving production model to: {model_path}")
    joblib.dump(final_model, model_path)
    
    print(f"Saving model info to: {info_path}")
    joblib.dump(model_info, info_path)
    
    # Also save as JSON for easy inspection
    import json
    json_info_path = os.path.join(output_dir, f'model_info_{model_name}_{model_version}.json')
    
    # Create JSON-serializable version of model_info
    json_model_info = model_info.copy()
    # Convert any non-serializable objects to strings
    for key, value in json_model_info.items():
        try:
            json.dumps(value)
        except (TypeError, OverflowError):
            json_model_info[key] = str(value)
    
    with open(json_info_path, 'w') as f:
        json.dump(json_model_info, f, indent=2)
    
    print(f"Model info also saved as JSON: {json_info_path}")
    
    # Create a simple training summary
    print("\n" + "="*60)
    print("PRODUCTION MODEL TRAINING SUMMARY")
    print("="*60)
    print(f"Model: {model_name}")
    print(f"Version: {model_version}")
    print(f"Training samples: {len(df):,}")
    print(f"Features: {len(X.columns)}")
    print(f"Problem type: {problem_type} ({n_classes} classes)")
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Files saved to: {output_dir}")
    print("="*60)
    
    return final_model, model_info

def plot_adaptive_confusion_matrix(y_true, y_pred, model_name, output_dir='output', problem_type='binary', class_labels=None):
    """
    Plot and save confusion matrix that adapts to binary or multi-class problems.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model for plot title
        output_dir: Directory to save the plot
        problem_type: 'binary' or 'multiclass'
        class_labels: List of class labels for labeling
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Determine figure size based on number of classes
    n_classes = cm.shape[0]
    fig_size = (8, 6) if n_classes <= 3 else (max(10, n_classes * 1.5), max(8, n_classes * 1.2))
    
    plt.figure(figsize=fig_size)
    
    # Plot with seaborn
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    
    # Set labels
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title(f'Confusion Matrix - {model_name} ({problem_type})')
    
    # Set tick labels
    if class_labels is not None:
        if problem_type == 'binary':
            tick_labels = [f'{class_labels[0]} (0)', f'{class_labels[1]} (1)']
        else:
            tick_labels = [f'Class {label}' for label in class_labels]
        ax.set_xticklabels(tick_labels, rotation=45, ha='right')
        ax.set_yticklabels(tick_labels, rotation=0)
    
    # Calculate and display metrics
    total = np.sum(cm)
    accuracy = np.trace(cm) / total
    
    if problem_type == 'binary' and n_classes == 2:
        # Binary classification metrics
        tn, fp, fn, tp = cm.ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics_text = f"Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | Specificity: {specificity:.4f}"
    else:
        # Multi-class metrics (macro-averaged)
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        metrics_text = f"Accuracy: {accuracy:.4f} | Precision (macro): {precision:.4f} | Recall (macro): {recall:.4f} | F1 (macro): {f1:.4f}"
    
    # Add text with metrics below the heatmap
    plt.figtext(0.5, 0.01, metrics_text, ha="center", fontsize=10, 
                bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Adjust layout to make room for text
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, f'confusion_matrix_{model_name}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved to {os.path.join(output_dir, f'confusion_matrix_{model_name}.png')}")

def define_models_config():
    # Define models and their parameter grids
    models_config = {
        'LogisticRegression': (
            LogisticRegression,
            {
                'C': [0.01, 0.1, 1.0],
                'l1_ratio': [0.0, 0.5, 1.0],
                'solver': ['saga'],
                'class_weight': [None, 'balanced'],
                'random_state': [42],
                'max_iter': [1000]
            }
        ),
        'RandomForest': (
            RandomForestClassifier,
            {
                'n_estimators': [100, 300],
                'max_depth': [None, 20],
                'min_samples_split': [2, 5],
                'class_weight': [None, 'balanced'],
                'random_state': [42]
            }
        ),
        'GradientBoosting': (
            GradientBoostingClassifier,
            {
                'n_estimators': [100, 200],
                'learning_rate': [0.1],
                'max_depth': [5, 7],
                'subsample': [0.8],
                'random_state': [42]
            }
        ),
        'XGBoost': (
            XGBClassifier,
            {
                'n_estimators': [100, 200],
                'learning_rate': [0.1],
                'max_depth': [5, 7],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0],
                'random_state': [42]
            }
        )
    }
    
    return models_config

def plot_adaptive_model_curves(curves_data, results_df, output_dir='model_curves'):
    """
    Plot ROC and PR curves for all models, adapting to binary vs multi-class problems.
    
    Args:
        curves_data: Dictionary containing curve data for each model
        results_df: DataFrame with performance metrics
        output_dir: Directory to save plot images
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Detect problem type from first available data
    first_model = list(curves_data.keys())[0]
    first_key = list(curves_data[first_model].keys())[0]
    problem_type = curves_data[first_model][first_key]['problem_type']
    n_classes = curves_data[first_model][first_key]['n_classes']
    
    print(f"Plotting curves for {problem_type} classification with {n_classes} classes")
    
    # Find best model based on appropriate metric
    if problem_type == 'binary':
        primary_metric = 'roc_auc'
    else:
        primary_metric = 'f1'
    
    avg_metrics = results_df.group_by('model').agg(
        pl.mean(primary_metric).alias(f'avg_{primary_metric}')
    )
    best_model = avg_metrics.sort(f'avg_{primary_metric}', descending=True)[0, 'model']
    
    # Get unique train-test source combinations
    train_test_pairs = results_df.select(
        pl.col('train_source'), 
        pl.col('test_source')
    ).unique()
    
    # For each train-test source pair
    for row in train_test_pairs.iter_rows(named=True):
        train_source = row['train_source']
        test_source = row['test_source']
        pair_key = f"{train_source}_{test_source}"
        
        if problem_type == 'binary':
            # Create ROC curve plot for binary classification
            plt.figure(figsize=(10, 8))
            
            # Plot for each model
            for model_name in curves_data.keys():
                if pair_key in curves_data[model_name]:
                    data = curves_data[model_name][pair_key]
                    y_test = data['y_test']
                    y_proba = data['y_proba']
                    
                    # For binary classification, use positive class probabilities
                    y_proba_pos = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
                    
                    # Calculate ROC curve
                    fpr, tpr, _ = roc_curve(y_test, y_proba_pos)
                    auc_score = roc_auc_score(y_test, y_proba_pos)
                    
                    # Determine line style based on if it's the best model
                    if model_name == best_model:
                        plt.plot(fpr, tpr, linewidth=2, linestyle='-', 
                                 label=f"{model_name} (Best) - AUC: {auc_score:.4f}")
                    else:
                        plt.plot(fpr, tpr, linewidth=1, linestyle='--', 
                                 label=f"{model_name} - AUC: {auc_score:.4f}")
            
            # Add diagonal line (random classifier)
            plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
            
            # Configure plot
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curves: Train on {train_source}, Test on {test_source}')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            
            # Create ROC curves directory and save plot
            roc_curves_dir = os.path.join(output_dir, 'roc_curves')
            os.makedirs(roc_curves_dir, exist_ok=True)
            plt.savefig(os.path.join(roc_curves_dir, f"roc_curve_{train_source}_{test_source}.png"), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Create PR curve plot for binary classification
            plt.figure(figsize=(10, 8))
            
            # Plot for each model
            for model_name in curves_data.keys():
                if pair_key in curves_data[model_name]:
                    data = curves_data[model_name][pair_key]
                    y_test = data['y_test']
                    y_proba = data['y_proba']
                    
                    # For binary classification, use positive class probabilities
                    y_proba_pos = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
                    
                    # Calculate PR curve
                    precision, recall, _ = precision_recall_curve(y_test, y_proba_pos)
                    auprc = average_precision_score(y_test, y_proba_pos)
                    
                    # Determine line style based on if it's the best model
                    if model_name == best_model:
                        plt.plot(recall, precision, linewidth=2, linestyle='-', 
                                 label=f"{model_name} (Best) - AUPRC: {auprc:.4f}")
                    else:
                        plt.plot(recall, precision, linewidth=1, linestyle='--', 
                                 label=f"{model_name} - AUPRC: {auprc:.4f}")
            
            # Add baseline (ratio of positive samples)
            baseline = sum(y_test) / len(y_test)
            plt.axhline(y=baseline, color='r', linestyle='-', alpha=0.3, 
                        label=f'Baseline: {baseline:.4f}')
            
            # Configure plot
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Precision-Recall Curves: Train on {train_source}, Test on {test_source}')
            plt.legend(loc="best")
            plt.grid(True, alpha=0.3)
            
            # Create PRC curves directory and save plot
            prc_curves_dir = os.path.join(output_dir, 'prc_curves')
            os.makedirs(prc_curves_dir, exist_ok=True)
            plt.savefig(os.path.join(prc_curves_dir, f"pr_curve_{train_source}_{test_source}.png"), dpi=300, bbox_inches='tight')
            plt.close()
            
        else:
            # For multi-class, create ROC curves using one-vs-rest approach
            plt.figure(figsize=(12, 8))
            
            # Get class information from first model's data
            first_model_name = list(curves_data.keys())[0]
            if pair_key in curves_data[first_model_name]:
                sample_data = curves_data[first_model_name][pair_key]
                unique_classes = sorted(np.unique(sample_data['y_test']))
                
                # Plot ROC curve for each class (one-vs-rest)
                for class_idx, class_label in enumerate(unique_classes):
                    plt.subplot(2, (len(unique_classes) + 1) // 2, class_idx + 1)
                    
                    for model_name in curves_data.keys():
                        if pair_key in curves_data[model_name]:
                            data = curves_data[model_name][pair_key]
                            y_test = data['y_test']
                            y_proba = data['y_proba']
                            
                            # Create binary problem for this class vs rest
                            y_test_binary = (y_test == class_label).astype(int)
                            y_proba_class = y_proba[:, class_idx]
                            
                            # Calculate ROC curve for this class
                            try:
                                fpr, tpr, _ = roc_curve(y_test_binary, y_proba_class)
                                auc_score = roc_auc_score(y_test_binary, y_proba_class)
                                
                                # Determine line style
                                if model_name == best_model:
                                    plt.plot(fpr, tpr, linewidth=2, linestyle='-', 
                                             label=f"{model_name} - AUC: {auc_score:.3f}")
                                else:
                                    plt.plot(fpr, tpr, linewidth=1, linestyle='--', 
                                             label=f"{model_name} - AUC: {auc_score:.3f}")
                            except ValueError:
                                # Skip if this class doesn't exist in test set
                                continue
                    
                    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title(f'ROC: Class {class_label} vs Rest')
                    plt.legend(loc="lower right", fontsize=8)
                    plt.grid(True, alpha=0.3)
            
            plt.suptitle(f'Multi-class ROC Curves: Train on {train_source}, Test on {test_source}')
            plt.tight_layout()
            
            # Save plot
            plt.savefig(os.path.join(output_dir, f"multiclass_roc_{train_source}_{test_source}.png"), dpi=300, bbox_inches='tight')
            plt.close()
    
    # Create combined performance plot
    create_combined_performance_plot(results_df, output_dir, problem_type)
    
    # Create combined curves for the best model across all train/test combinations
    plot_best_model_combined_roc_curves(curves_data, results_df, output_dir)
    plot_best_model_combined_prc_curves(curves_data, results_df, output_dir)
    
    print(f"All plots saved to directory: {output_dir}")

def create_combined_performance_plot(results_df, output_dir, problem_type):
    """Create a combined performance comparison plot for all models."""
    
    model_colors = {
        'LogisticRegression': 'blue',
        'RandomForest': 'green', 
        'GradientBoosting': 'red',
        'XGBoost': 'purple',
        'SVM': 'orange',
        'LightGBM': 'brown'
    }
    
    # Average metrics per model
    if problem_type == 'binary':
        avg_model_metrics = results_df.group_by('model').agg(
            pl.mean('roc_auc').alias('avg_roc_auc'),
            pl.mean('auprc').alias('avg_auprc'),
            pl.mean('f1').alias('avg_f1'),
            pl.mean('accuracy').alias('avg_accuracy')
        )
        
        # Sort by ROC AUC
        avg_model_metrics = avg_model_metrics.sort(by='avg_roc_auc', descending=True)
        best_model = avg_model_metrics[0, 'model']
        
        # Plot summary performance
        plt.figure(figsize=(12, 8))
        
        for row in avg_model_metrics.iter_rows(named=True):
            model_name = row['model']
            x = [row['avg_roc_auc'], row['avg_auprc'], row['avg_f1'], row['avg_accuracy']]
            
            # Determine marker style based on if it's the best model
            marker_size = 100 if model_name == best_model else 60
            alpha = 1.0 if model_name == best_model else 0.7
            
            plt.scatter([1, 2, 3, 4], x, s=marker_size, label=model_name, 
                        color=model_colors.get(model_name, 'gray'), 
                        alpha=alpha, edgecolors='black')
            
            # Connect points with lines
            plt.plot([1, 2, 3, 4], x, color=model_colors.get(model_name, 'gray'), 
                     alpha=alpha, linestyle='-' if model_name == best_model else '--')
        
        plt.xticks([1, 2, 3, 4], ['ROC AUC', 'AUPRC', 'F1 Score', 'Accuracy'])
        
    else:
        # Multi-class performance plot
        avg_model_metrics = results_df.group_by('model').agg(
            pl.mean('f1').alias('avg_f1'),
            pl.mean('accuracy').alias('avg_accuracy'),
            pl.mean('precision').alias('avg_precision'),
            pl.mean('recall').alias('avg_recall')
        )
        
        # Sort by F1 score
        avg_model_metrics = avg_model_metrics.sort(by='avg_f1', descending=True)
        best_model = avg_model_metrics[0, 'model']
        
        # Plot summary performance
        plt.figure(figsize=(12, 8))
        
        for row in avg_model_metrics.iter_rows(named=True):
            model_name = row['model']
            x = [row['avg_f1'], row['avg_accuracy'], row['avg_precision'], row['avg_recall']]
            
            # Determine marker style based on if it's the best model
            marker_size = 100 if model_name == best_model else 60
            alpha = 1.0 if model_name == best_model else 0.7
            
            plt.scatter([1, 2, 3, 4], x, s=marker_size, label=model_name, 
                        color=model_colors.get(model_name, 'gray'), 
                        alpha=alpha, edgecolors='black')
            
            # Connect points with lines
            plt.plot([1, 2, 3, 4], x, color=model_colors.get(model_name, 'gray'), 
                     alpha=alpha, linestyle='-' if model_name == best_model else '--')
        
        plt.xticks([1, 2, 3, 4], ['F1 Score', 'Accuracy', 'Precision', 'Recall'])
    
    plt.ylabel('Score')
    plt.title(f'Model Performance Comparison ({problem_type.capitalize()} Classification)')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(output_dir, "model_performance_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()

def leave_one_source_out_validation(df, source_col, target_col, models_config, scoring_metric, output_dir='output', random_state=42, save_fold_models=False):
    """
    Implement true leave-one-source-out validation:
    - For each source S:
      - Train on all sources EXCEPT S
      - Test on S only
    
    Args:
        save_fold_models: If True, save individual trained models from each fold
    """
    # Detect problem type
    problem_type, n_classes, unique_classes = detect_problem_type(df[target_col])
    print(f"Leave-one-source-out validation for {problem_type} classification with {n_classes} classes: {unique_classes}")
    
    # Get unique sources
    sources = df[source_col].unique().to_list()
    print(f"Found {len(sources)} unique sources: {sources}")
    
    results = []
    curves_data = {}  # To store curve data for plotting
    aggregate_predictions = {}  # To store out-of-fold predictions for aggregate confusion matrix
    
    # For each model type
    for model_name, (model_class, param_grid) in models_config.items():
        print(f"\n===== Evaluating {model_name} =====")
        curves_data[model_name] = {}
        
        # Initialize aggregate predictions storage for this model
        aggregate_predictions[model_name] = {
            'y_true': [],
            'y_pred': [],
            'y_proba': [],
            'sources': []
        }
        
        # For each source as test set (leave one out)
        for held_out_source in sources:
            # Filter data for training (all sources except the held-out one)
            train_data = df.filter(pl.col(source_col) != held_out_source)
            X_train, y_train = prepare_data(train_data, target_col, source_col)
            
            # Filter data for testing (only the held-out source)
            test_data = df.filter(pl.col(source_col) == held_out_source)
            X_test, y_test = prepare_data(test_data, target_col, source_col)
            
            # Create pipeline with current model
            pipeline = build_adaptive_pipeline(X_train, model_class)
            
            # Add 'model__' prefix to param grid keys
            model_param_grid = {f"model__{k}": v for k, v in param_grid.items()}
            
            # Train model
            print(f"\nTraining {model_name} on all sources except: {held_out_source}")
            
            # Create a stratified cross-validation strategy to maintain class distribution
            from sklearn.model_selection import StratifiedKFold
            cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
            
            grid_search = GridSearchCV(
                pipeline,
                param_grid=model_param_grid,
                cv=cv_strategy,
                scoring=scoring_metric,
                n_jobs=-1,
                verbose=1
            )

            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_

            print(f"Best parameters for {model_name} excluding {held_out_source}: {best_params}")
            
            # Save individual fold model if requested
            if save_fold_models:
                fold_models_dir = os.path.join(output_dir, f'fold_models_{model_name}')
                os.makedirs(fold_models_dir, exist_ok=True)
                
                # Create model filename with train/test information
                model_filename = f"{model_name}_train_all_except_{held_out_source}_test_{held_out_source}.pkl"
                model_path = os.path.join(fold_models_dir, model_filename)
                
                # Create model metadata
                fold_model_info = {
                    'model_name': model_name,
                    'train_sources': f"all_except_{held_out_source}",
                    'test_source': held_out_source,
                    'training_timestamp': datetime.now().isoformat(),
                    'best_params': best_params,
                    'problem_type': problem_type,
                    'n_classes': n_classes,
                    'unique_classes': unique_classes.tolist() if hasattr(unique_classes, 'tolist') else list(unique_classes),
                    'train_samples': len(X_train),
                    'test_samples': len(X_test),
                    'n_features': len(X_train.columns),
                    'feature_names': X_train.columns,
                    'random_state': random_state,
                    'scoring_metric': scoring_metric
                }
                
                # Save the trained model
                print(f"Saving fold model to: {model_path}")
                joblib.dump(best_model, model_path)
                
                # Save model metadata as JSON
                info_filename = f"{model_name}_train_all_except_{held_out_source}_test_{held_out_source}_info.json"
                info_path = os.path.join(fold_models_dir, info_filename)
                
                with open(info_path, 'w') as f:
                    json.dump(fold_model_info, f, indent=2)
                
                print(f"Saved fold model info to: {info_path}")
            
            # Test on the held-out source
            print(f"Testing {model_name} on held-out source: {held_out_source}")
            
            # Predict and evaluate
            y_pred = best_model.predict(X_test)
            y_proba = best_model.predict_proba(X_test)
            
            # Store out-of-fold predictions for aggregate confusion matrix
            aggregate_predictions[model_name]['y_true'].extend(y_test.to_list())
            aggregate_predictions[model_name]['y_pred'].extend(y_pred.tolist() if hasattr(y_pred, 'tolist') else list(y_pred))
            aggregate_predictions[model_name]['y_proba'].extend(y_proba.tolist() if hasattr(y_proba, 'tolist') else list(y_proba))
            aggregate_predictions[model_name]['sources'].extend([held_out_source] * len(y_test))
            
            # Generate confusion matrix for this fold
            fold_cm_output_dir = os.path.join(output_dir, f'confusion_matrices_{model_name}')
            plot_adaptive_confusion_matrix(
                y_test, y_pred, f"{model_name}_fold_{held_out_source}", 
                output_dir=fold_cm_output_dir, 
                problem_type=problem_type,
                class_labels=unique_classes
            )
            
            # Calculate adaptive metrics
            metrics = calculate_adaptive_metrics(y_test, y_pred, y_proba, problem_type)
            
            # Add metadata
            metrics.update({
                'model': model_name,
                'train_sources': f"all_except_{held_out_source}",
                'test_source': held_out_source,
                'problem_type': problem_type,
                'n_classes': n_classes,
                'best_params': str(best_params)
            })
            
            # Store curve data for plotting
            curves_data[model_name][f"all_except_{held_out_source}_{held_out_source}"] = {
                'y_test': y_test,
                'y_proba': y_proba,
                'y_pred': y_pred,
                'problem_type': problem_type,
                'n_classes': n_classes
            }
            
            results.append(metrics)
            
            print(f"{model_name} trained on all except {held_out_source} → tested on {held_out_source} Performance:")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"F1 Score: {metrics['f1']:.4f}")
            if not np.isnan(metrics.get('roc_auc', np.nan)):
                print(f"ROC AUC Score: {metrics['roc_auc']:.4f}")
            if not np.isnan(metrics.get('auprc', np.nan)):
                print(f"AUPRC Score: {metrics['auprc']:.4f}")
    
    # Generate aggregate confusion matrices using out-of-fold predictions
    print(f"\n===== Generating Aggregate Confusion Matrices =====")
    
    for model_name in aggregate_predictions.keys():
        if len(aggregate_predictions[model_name]['y_true']) > 0:
            y_true_agg = np.array(aggregate_predictions[model_name]['y_true'])
            y_pred_agg = np.array(aggregate_predictions[model_name]['y_pred'])
            
            print(f"Creating aggregate confusion matrix for {model_name}")
            print(f"Total out-of-fold predictions: {len(y_true_agg)}")
            
            # Set aggregate confusion matrix output directory inside the model's confusion matrix folder
            aggregate_cm_output_dir = os.path.join(output_dir, f'confusion_matrices_{model_name}')
            
            # Generate aggregate confusion matrix
            plot_adaptive_confusion_matrix(
                y_true_agg, y_pred_agg, "aggregate", 
                output_dir=aggregate_cm_output_dir, 
                problem_type=problem_type,
                class_labels=unique_classes
            )
            
            # Calculate and print aggregate performance metrics
            agg_metrics = calculate_adaptive_metrics(
                y_true_agg, y_pred_agg, 
                np.array(aggregate_predictions[model_name]['y_proba']), 
                problem_type
            )
            
            print(f"Aggregate performance for {model_name}:")
            print(f"  Accuracy: {agg_metrics['accuracy']:.4f}")
            print(f"  Precision: {agg_metrics['precision']:.4f}")
            print(f"  Recall: {agg_metrics['recall']:.4f}")
            print(f"  F1 Score: {agg_metrics['f1']:.4f}")
            if not np.isnan(agg_metrics.get('roc_auc', np.nan)):
                print(f"  ROC AUC: {agg_metrics['roc_auc']:.4f}")
            if not np.isnan(agg_metrics.get('auprc', np.nan)):
                print(f"  AUPRC: {agg_metrics['auprc']:.4f}")
    
    # Convert results to Polars DataFrame
    results_df = pl.DataFrame(results)
    return results_df, curves_data

def plot_leave_one_out_curves(curves_data, results_df, output_dir='model_curves'):
    """
    Plot ROC and PR curves for all models, adapting to problem type.
    Adapted for leave-one-source-out validation.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Detect problem type
    first_model = list(curves_data.keys())[0]
    first_key = list(curves_data[first_model].keys())[0]
    problem_type = curves_data[first_model][first_key]['problem_type']
    
    # Find best model based on appropriate metric
    primary_metric = 'roc_auc' if problem_type == 'binary' else 'f1'
    
    avg_metrics = results_df.group_by('model').agg(
        pl.mean(primary_metric).alias(f'avg_{primary_metric}')
    )
    best_model = avg_metrics.sort(f'avg_{primary_metric}', descending=True)[0, 'model']
    
    # Get unique test sources (held-out sources)
    test_sources = results_df.select(pl.col('test_source')).unique()
    
    # For each held-out source
    for row in test_sources.iter_rows(named=True):
        held_out_source = row['test_source']
        pair_key = f"all_except_{held_out_source}_{held_out_source}"
        
        if problem_type == 'binary':
            # Create ROC curve plot
            plt.figure(figsize=(10, 8))
            
            # Plot for each model
            for model_name in curves_data.keys():
                if pair_key in curves_data[model_name]:
                    data = curves_data[model_name][pair_key]
                    y_test = data['y_test']
                    y_proba = data['y_proba']
                    
                    # For binary, use positive class probabilities
                    y_proba_pos = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
                    
                    # Calculate ROC curve
                    fpr, tpr, _ = roc_curve(y_test, y_proba_pos)
                    auc_score = roc_auc_score(y_test, y_proba_pos)
                    
                    # Determine line style based on if it's the best model
                    if model_name == best_model:
                        plt.plot(fpr, tpr, linewidth=2, linestyle='-', 
                                 label=f"{model_name} (Best) - AUC: {auc_score:.4f}")
                    else:
                        plt.plot(fpr, tpr, linewidth=1, linestyle='--', 
                                 label=f"{model_name} - AUC: {auc_score:.4f}")
            
            # Add diagonal line (random classifier)
            plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
            
            # Configure plot
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curves: Train on all except {held_out_source}, Test on {held_out_source}')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            
            # Create ROC curves directory and save plot
            roc_curves_dir = os.path.join(output_dir, 'roc_curves')
            os.makedirs(roc_curves_dir, exist_ok=True)
            plt.savefig(os.path.join(roc_curves_dir, f"roc_curve_held_out_{held_out_source}.png"), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Create PR curve plot
            plt.figure(figsize=(10, 8))
            
            # Plot for each model
            for model_name in curves_data.keys():
                if pair_key in curves_data[model_name]:
                    data = curves_data[model_name][pair_key]
                    y_test = data['y_test']
                    y_proba = data['y_proba']
                    
                    # For binary, use positive class probabilities
                    y_proba_pos = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
                    
                    # Calculate PR curve
                    precision, recall, _ = precision_recall_curve(y_test, y_proba_pos)
                    auprc = average_precision_score(y_test, y_proba_pos)
                    
                    # Determine line style based on if it's the best model
                    if model_name == best_model:
                        plt.plot(recall, precision, linewidth=2, linestyle='-', 
                                 label=f"{model_name} (Best) - AUPRC: {auprc:.4f}")
                    else:
                        plt.plot(recall, precision, linewidth=1, linestyle='--', 
                                 label=f"{model_name} - AUPRC: {auprc:.4f}")
            
            # Add baseline (ratio of positive samples)
            baseline = sum(y_test) / len(y_test)
            plt.axhline(y=baseline, color='r', linestyle='-', alpha=0.3, 
                        label=f'Baseline: {baseline:.4f}')
            
            # Configure plot
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Precision-Recall Curves: Train on all except {held_out_source}, Test on {held_out_source}')
            plt.legend(loc="best")
            plt.grid(True, alpha=0.3)
            
            # Create PRC curves directory and save plot
            prc_curves_dir = os.path.join(output_dir, 'prc_curves')
            os.makedirs(prc_curves_dir, exist_ok=True)
            plt.savefig(os.path.join(prc_curves_dir, f"pr_curve_held_out_{held_out_source}.png"), dpi=300, bbox_inches='tight')
            plt.close()
        
        else:
            # Multi-class: Create ROC curves using one-vs-rest for each class
            plt.figure(figsize=(15, 10))
            
            # Get first available data to determine classes
            first_model_name = list(curves_data.keys())[0]
            if pair_key in curves_data[first_model_name]:
                sample_data = curves_data[first_model_name][pair_key]
                unique_classes = sorted(np.unique(sample_data['y_test']))
                n_classes = len(unique_classes)
                
                # Create subplots for each class
                n_cols = min(3, n_classes)
                n_rows = (n_classes + n_cols - 1) // n_cols
                
                for class_idx, class_label in enumerate(unique_classes):
                    plt.subplot(n_rows, n_cols, class_idx + 1)
                    
                    for model_name in curves_data.keys():
                        if pair_key in curves_data[model_name]:
                            data = curves_data[model_name][pair_key]
                            y_test = data['y_test']
                            y_proba = data['y_proba']
                            
                            # Create binary problem for this class vs rest
                            y_test_binary = (y_test == class_label).astype(int)
                            
                            if class_idx < y_proba.shape[1]:
                                y_proba_class = y_proba[:, class_idx]
                                
                                try:
                                    fpr, tpr, _ = roc_curve(y_test_binary, y_proba_class)
                                    auc_score = roc_auc_score(y_test_binary, y_proba_class)
                                    
                                    # Determine line style
                                    if model_name == best_model:
                                        plt.plot(fpr, tpr, linewidth=2, linestyle='-', 
                                                 label=f"{model_name} - AUC: {auc_score:.3f}")
                                    else:
                                        plt.plot(fpr, tpr, linewidth=1, linestyle='--', 
                                                 label=f"{model_name} - AUC: {auc_score:.3f}")
                                except ValueError:
                                    # Skip if this class doesn't have enough samples
                                    continue
                    
                    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title(f'ROC: Class {class_label} vs Rest')
                    plt.legend(loc="lower right", fontsize=8)
                    plt.grid(True, alpha=0.3)
                
                plt.suptitle(f'Multi-class ROC Curves: Train on all except {held_out_source}, Test on {held_out_source}')
                plt.tight_layout()
                
                # Save plot
                plt.savefig(os.path.join(output_dir, f"multiclass_roc_held_out_{held_out_source}.png"), dpi=300, bbox_inches='tight')
                plt.close()
    
    # Create combined curves for the best model across all train/test combinations
    plot_best_model_combined_roc_curves(curves_data, results_df, output_dir)
    plot_best_model_combined_prc_curves(curves_data, results_df, output_dir)
    
    print(f"All plots saved to directory: {output_dir}")

def regular_kfold_validation(df, target_col, models_config, scoring_metric, output_dir='output', random_state=42, n_folds=5, save_fold_models=False):
    """
    Implement regular k-fold cross-validation when no source column is available.
    Adapts to binary or multi-class problems.
    
    Args:
        save_fold_models: If True, save individual trained models from each fold
    """
    # Detect problem type
    problem_type, n_classes, unique_classes = detect_problem_type(df[target_col])
    print(f"Performing {n_folds}-fold cross-validation for {problem_type} classification with {n_classes} classes")
    
    # Prepare data
    X = df.drop(target_col)
    y = df[target_col]
    
    # Create k-fold cross-validator
    from sklearn.model_selection import StratifiedKFold
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    results = []
    curves_data = {}  # To store curve data for plotting
    aggregate_predictions = {}  # To store out-of-fold predictions for aggregate confusion matrix
    
    # For each model type
    for model_name, (model_class, param_grid) in models_config.items():
        print(f"\n===== Evaluating {model_name} =====")
        curves_data[model_name] = {}
        
        # Initialize aggregate predictions storage for this model
        aggregate_predictions[model_name] = {
            'y_true': [],
            'y_pred': [],
            'y_proba': [],
            'folds': []
        }
        
        # Create pipeline with current model
        pipeline = build_adaptive_pipeline(X, model_class)
        
        # Add 'model__' prefix to param grid keys
        model_param_grid = {f"model__{k}": v for k, v in param_grid.items()}
        
        # Perform grid search with cross-validation
        grid_search = GridSearchCV(
            pipeline,
            param_grid=model_param_grid,
            cv=cv,
            scoring=scoring_metric,
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X, y)
        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_
        
        print(f"Best parameters for {model_name}: {best_params}")
        
        # Evaluate using cross-validation
        fold_idx = 0
        for train_index, test_index in cv.split(X, y):
            fold_idx += 1
            
            # Split data for this fold
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            # Train model on this fold
            fold_model = clone(best_model)
            fold_model.fit(X_train, y_train)
            
            # Save individual fold model if requested
            if save_fold_models:
                fold_models_dir = os.path.join(output_dir, f'fold_models_{model_name}')
                os.makedirs(fold_models_dir, exist_ok=True)
                
                # Create model filename with fold information
                model_filename = f"{model_name}_fold_{fold_idx}.pkl"
                model_path = os.path.join(fold_models_dir, model_filename)
                
                # Create model metadata
                fold_model_info = {
                    'model_name': model_name,
                    'fold': fold_idx,
                    'training_timestamp': datetime.now().isoformat(),
                    'best_params': best_params,
                    'problem_type': problem_type,
                    'n_classes': n_classes,
                    'unique_classes': unique_classes.tolist() if hasattr(unique_classes, 'tolist') else list(unique_classes),
                    'train_samples': len(X_train),
                    'test_samples': len(X_test),
                    'n_features': len(X_train.columns),
                    'feature_names': X_train.columns,
                    'random_state': random_state,
                    'scoring_metric': scoring_metric,
                    'n_folds': n_folds
                }
                
                # Save the trained model
                print(f"Saving fold {fold_idx} model to: {model_path}")
                joblib.dump(fold_model, model_path)
                
                # Save model metadata as JSON
                info_filename = f"{model_name}_fold_{fold_idx}_info.json"
                info_path = os.path.join(fold_models_dir, info_filename)
                
                with open(info_path, 'w') as f:
                    json.dump(fold_model_info, f, indent=2)
                
                print(f"Saved fold {fold_idx} model info to: {info_path}")
            
            # Predict and evaluate
            y_pred = fold_model.predict(X_test)
            y_proba = fold_model.predict_proba(X_test)
            
            # Store out-of-fold predictions for aggregate confusion matrix
            aggregate_predictions[model_name]['y_true'].extend(y_test.tolist() if hasattr(y_test, 'tolist') else list(y_test))
            aggregate_predictions[model_name]['y_pred'].extend(y_pred.tolist() if hasattr(y_pred, 'tolist') else list(y_pred))
            aggregate_predictions[model_name]['y_proba'].extend(y_proba.tolist() if hasattr(y_proba, 'tolist') else list(y_proba))
            aggregate_predictions[model_name]['folds'].extend([fold_idx] * len(y_test))
            
            # Generate confusion matrix for this fold
            fold_cm_output_dir = os.path.join(output_dir, f'confusion_matrices_{model_name}')
            plot_adaptive_confusion_matrix(
                y_test, y_pred, f"{model_name}_fold_{fold_idx}", 
                output_dir=fold_cm_output_dir, 
                problem_type=problem_type,
                class_labels=unique_classes
            )
            
            # Calculate adaptive metrics
            metrics = calculate_adaptive_metrics(y_test, y_pred, y_proba, problem_type)
            
            # Add metadata
            metrics.update({
                'model': model_name,
                'fold': fold_idx,
                'problem_type': problem_type,
                'n_classes': n_classes,
                'best_params': str(best_params)
            })
            
            # Store curve data for plotting
            curves_data[model_name][f"fold_{fold_idx}"] = {
                'y_test': y_test,
                'y_proba': y_proba,
                'y_pred': y_pred,
                'problem_type': problem_type,
                'n_classes': n_classes
            }
            
            results.append(metrics)
            
            print(f"{model_name} Fold {fold_idx} Performance:")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"F1 Score: {metrics['f1']:.4f}")
            if not np.isnan(metrics.get('roc_auc', np.nan)):
                print(f"ROC AUC Score: {metrics['roc_auc']:.4f}")
            if not np.isnan(metrics.get('auprc', np.nan)):
                print(f"AUPRC Score: {metrics['auprc']:.4f}")
    
    # Generate aggregate confusion matrices using out-of-fold predictions
    print(f"\n===== Generating Aggregate Confusion Matrices =====")
    
    for model_name in aggregate_predictions.keys():
        if len(aggregate_predictions[model_name]['y_true']) > 0:
            y_true_agg = np.array(aggregate_predictions[model_name]['y_true'])
            y_pred_agg = np.array(aggregate_predictions[model_name]['y_pred'])
            
            print(f"Creating aggregate confusion matrix for {model_name}")
            print(f"Total out-of-fold predictions: {len(y_true_agg)}")
            
            # Set aggregate confusion matrix output directory inside the model's confusion matrix folder
            aggregate_cm_output_dir = os.path.join(output_dir, f'confusion_matrices_{model_name}')
            
            # Generate aggregate confusion matrix
            plot_adaptive_confusion_matrix(
                y_true_agg, y_pred_agg, "aggregate", 
                output_dir=aggregate_cm_output_dir, 
                problem_type=problem_type,
                class_labels=unique_classes
            )
            
            # Calculate and print aggregate performance metrics
            agg_metrics = calculate_adaptive_metrics(
                y_true_agg, y_pred_agg, 
                np.array(aggregate_predictions[model_name]['y_proba']), 
                problem_type
            )
            
            print(f"Aggregate performance for {model_name}:")
            print(f"  Accuracy: {agg_metrics['accuracy']:.4f}")
            print(f"  Precision: {agg_metrics['precision']:.4f}")
            print(f"  Recall: {agg_metrics['recall']:.4f}")
            print(f"  F1 Score: {agg_metrics['f1']:.4f}")
            if not np.isnan(agg_metrics.get('roc_auc', np.nan)):
                print(f"  ROC AUC: {agg_metrics['roc_auc']:.4f}")
            if not np.isnan(agg_metrics.get('auprc', np.nan)):
                print(f"  AUPRC: {agg_metrics['auprc']:.4f}")
    
    # Convert results to Polars DataFrame
    results_df = pl.DataFrame(results)
    return results_df, curves_data

def plot_kfold_curves(curves_data, results_df, output_dir='model_curves'):
    """
    Plot ROC and PR curves for all models in k-fold cross-validation.
    Adapts to binary or multi-class problems.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Detect problem type
    first_model = list(curves_data.keys())[0]
    first_key = list(curves_data[first_model].keys())[0]
    problem_type = curves_data[first_model][first_key]['problem_type']
    
    # Find best model based on appropriate metric
    primary_metric = 'roc_auc' if problem_type == 'binary' else 'f1'
    
    avg_metrics = results_df.group_by('model').agg(
        pl.mean(primary_metric).alias(f'avg_{primary_metric}')
    )
    best_model = avg_metrics.sort(f'avg_{primary_metric}', descending=True)[0, 'model']
    
    # Get unique fold indices
    folds = results_df.select(pl.col('fold')).unique().to_series().sort().to_list()
    
    # For each fold, create plots based on problem type
    for fold in folds:
        fold_key = f"fold_{fold}"
        
        if problem_type == 'binary':
            # Create ROC curve plot for binary classification
            plt.figure(figsize=(10, 8))
            
            # Plot for each model
            for model_name in curves_data.keys():
                if fold_key in curves_data[model_name]:
                    data = curves_data[model_name][fold_key]
                    y_test = data['y_test']
                    y_proba = data['y_proba']
                    
                    # For binary, use positive class probabilities
                    y_proba_pos = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
                    
                    # Calculate ROC curve
                    fpr, tpr, _ = roc_curve(y_test, y_proba_pos)
                    auc_score = roc_auc_score(y_test, y_proba_pos)
                    
                    # Determine line style based on if it's the best model
                    if model_name == best_model:
                        plt.plot(fpr, tpr, linewidth=2, linestyle='-', 
                                 label=f"{model_name} (Best) - AUC: {auc_score:.4f}")
                    else:
                        plt.plot(fpr, tpr, linewidth=1, linestyle='--', 
                                 label=f"{model_name} - AUC: {auc_score:.4f}")
            
            # Add diagonal line (random classifier)
            plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
            
            # Configure plot
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curves: Fold {fold}')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            
            # Create ROC curves directory and save plot
            roc_curves_dir = os.path.join(output_dir, 'roc_curves')
            os.makedirs(roc_curves_dir, exist_ok=True)
            plt.savefig(os.path.join(roc_curves_dir, f"roc_curve_fold_{fold}.png"), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Create PR curve plot for binary classification
            plt.figure(figsize=(10, 8))
            
            # Plot for each model
            for model_name in curves_data.keys():
                if fold_key in curves_data[model_name]:
                    data = curves_data[model_name][fold_key]
                    y_test = data['y_test']
                    y_proba = data['y_proba']
                    
                    # For binary, use positive class probabilities
                    y_proba_pos = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
                    
                    # Calculate PR curve
                    precision, recall, _ = precision_recall_curve(y_test, y_proba_pos)
                    auprc = average_precision_score(y_test, y_proba_pos)
                    
                    # Determine line style based on if it's the best model
                    if model_name == best_model:
                        plt.plot(recall, precision, linewidth=2, linestyle='-', 
                                 label=f"{model_name} (Best) - AUPRC: {auprc:.4f}")
                    else:
                        plt.plot(recall, precision, linewidth=1, linestyle='--', 
                                 label=f"{model_name} - AUPRC: {auprc:.4f}")
            
            # Add baseline (ratio of positive samples)
            baseline = sum(y_test) / len(y_test)
            plt.axhline(y=baseline, color='r', linestyle='-', alpha=0.3, 
                        label=f'Baseline: {baseline:.4f}')
            
            # Configure plot
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Precision-Recall Curves: Fold {fold}')
            plt.legend(loc="best")
            plt.grid(True, alpha=0.3)
            
            # Create PRC curves directory and save plot
            prc_curves_dir = os.path.join(output_dir, 'prc_curves')
            os.makedirs(prc_curves_dir, exist_ok=True)
            plt.savefig(os.path.join(prc_curves_dir, f"pr_curve_fold_{fold}.png"), dpi=300, bbox_inches='tight')
            plt.close()
            
        else:
            # Multi-class: Create ROC curves for each class
            plt.figure(figsize=(15, 10))
            
            # Get first available data to determine classes
            first_model_name = list(curves_data.keys())[0]
            if fold_key in curves_data[first_model_name]:
                sample_data = curves_data[first_model_name][fold_key]
                unique_classes = sorted(np.unique(sample_data['y_test']))
                n_classes = len(unique_classes)
                
                # Create subplots for each class
                n_cols = min(3, n_classes)
                n_rows = (n_classes + n_cols - 1) // n_cols
                
                for class_idx, class_label in enumerate(unique_classes):
                    plt.subplot(n_rows, n_cols, class_idx + 1)
                    
                    for model_name in curves_data.keys():
                        if fold_key in curves_data[model_name]:
                            data = curves_data[model_name][fold_key]
                            y_test = data['y_test']
                            y_proba = data['y_proba']
                            
                            # Create binary problem for this class vs rest
                            y_test_binary = (y_test == class_label).astype(int)
                            
                            if class_idx < y_proba.shape[1]:
                                y_proba_class = y_proba[:, class_idx]
                                
                                try:
                                    fpr, tpr, _ = roc_curve(y_test_binary, y_proba_class)
                                    auc_score = roc_auc_score(y_test_binary, y_proba_class)
                                    
                                    # Determine line style
                                    if model_name == best_model:
                                        plt.plot(fpr, tpr, linewidth=2, linestyle='-', 
                                                 label=f"{model_name} - AUC: {auc_score:.3f}")
                                    else:
                                        plt.plot(fpr, tpr, linewidth=1, linestyle='--', 
                                                 label=f"{model_name} - AUC: {auc_score:.3f}")
                                except ValueError:
                                    # Skip if this class doesn't have enough samples
                                    continue
                    
                    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title(f'ROC: Class {class_label} vs Rest')
                    plt.legend(loc="lower right", fontsize=8)
                    plt.grid(True, alpha=0.3)
                
                plt.suptitle(f'Multi-class ROC Curves: Fold {fold}')
                plt.tight_layout()
                
                # Save plot
                plt.savefig(os.path.join(output_dir, f"multiclass_roc_fold_{fold}.png"), dpi=300, bbox_inches='tight')
                plt.close()
    
    # Create average performance plot across all folds
    plot_model_avg_performance(results_df, output_dir)
    
    # Create combined curves for the best model across all folds
    plot_best_model_combined_roc_curves(curves_data, results_df, output_dir)
    plot_best_model_combined_prc_curves(curves_data, results_df, output_dir)
    
    print(f"All plots saved to directory: {output_dir}")

def plot_model_avg_performance(results_df, output_dir):
    """Plot average performance metrics for each model across all folds."""
    
    # Detect problem type
    problem_type = results_df[0, 'problem_type'] if 'problem_type' in results_df.columns else 'binary'
    
    if problem_type == 'binary':
        # Calculate average metrics by model for binary classification
        avg_performance = results_df.group_by('model').agg(
            pl.mean('accuracy').alias('avg_accuracy'),
            pl.mean('precision').alias('avg_precision'),
            pl.mean('recall').alias('avg_recall'),
            pl.mean('f1').alias('avg_f1'),
            pl.mean('roc_auc').alias('avg_roc_auc'),
            pl.mean('auprc').alias('avg_auprc'),
            pl.std('roc_auc').alias('std_roc_auc'),
            pl.std('auprc').alias('std_auprc')
        ).sort('avg_roc_auc', descending=True)
        
        # Create a bar plot for ROC AUC and AUPRC
        plt.figure(figsize=(12, 8))
        
        models = avg_performance['model'].to_list()
        x = np.arange(len(models))
        width = 0.35
        
        # Plot ROC AUC
        roc_auc = avg_performance['avg_roc_auc'].to_list()
        roc_std = avg_performance['std_roc_auc'].to_list()
        plt.bar(x - width/2, roc_auc, width, label='ROC AUC', alpha=0.7)
        
        # Plot AUPRC
        auprc = avg_performance['avg_auprc'].to_list()
        auprc_std = avg_performance['std_auprc'].to_list()
        plt.bar(x + width/2, auprc, width, label='AUPRC', alpha=0.7)
        
        # Add error bars
        plt.errorbar(x - width/2, roc_auc, yerr=roc_std, fmt='o', color='black', capsize=5)
        plt.errorbar(x + width/2, auprc, yerr=auprc_std, fmt='o', color='black', capsize=5)
        
        # Add values on top of bars
        for i, v in enumerate(roc_auc):
            plt.text(i - width/2, v + 0.03, f"{v:.3f}", ha='center')
        
        for i, v in enumerate(auprc):
            plt.text(i + width/2, v + 0.03, f"{v:.3f}", ha='center')
            
        plt.ylabel('Score')
        plt.title('Average Model Performance Across All Folds (Binary Classification)')
        
    else:
        # Calculate average metrics by model for multi-class classification
        avg_performance = results_df.group_by('model').agg(
            pl.mean('accuracy').alias('avg_accuracy'),
            pl.mean('precision').alias('avg_precision'),
            pl.mean('recall').alias('avg_recall'),
            pl.mean('f1').alias('avg_f1'),
            pl.std('f1').alias('std_f1'),
            pl.std('accuracy').alias('std_accuracy')
        ).sort('avg_f1', descending=True)
        
        # Create a bar plot for F1 and Accuracy
        plt.figure(figsize=(12, 8))
        
        models = avg_performance['model'].to_list()
        x = np.arange(len(models))
        width = 0.35
        
        # Plot F1 Score
        f1_scores = avg_performance['avg_f1'].to_list()
        f1_std = avg_performance['std_f1'].to_list()
        plt.bar(x - width/2, f1_scores, width, label='F1 Score', alpha=0.7)
        
        # Plot Accuracy
        accuracy_scores = avg_performance['avg_accuracy'].to_list()
        accuracy_std = avg_performance['std_accuracy'].to_list()
        plt.bar(x + width/2, accuracy_scores, width, label='Accuracy', alpha=0.7)
        
        # Add error bars
        plt.errorbar(x - width/2, f1_scores, yerr=f1_std, fmt='o', color='black', capsize=5)
        plt.errorbar(x + width/2, accuracy_scores, yerr=accuracy_std, fmt='o', color='black', capsize=5)
        
        # Add values on top of bars
        for i, v in enumerate(f1_scores):
            plt.text(i - width/2, v + 0.03, f"{v:.3f}", ha='center')
        
        for i, v in enumerate(accuracy_scores):
            plt.text(i + width/2, v + 0.03, f"{v:.3f}", ha='center')
            
        plt.ylabel('Score')
        plt.title('Average Model Performance Across All Folds (Multi-class Classification)')
    
    # Configure plot
    plt.xlabel('Model')
    plt.xticks(x, models, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1.1])
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(output_dir, "avg_model_performance.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_best_model_combined_roc_curves(curves_data, results_df, output_dir='model_curves'):
    """
    Plot a single ROC curve graph showing the best performing model 
    across all train/test cohort groupings.
    
    Args:
        curves_data: Dictionary containing curve data for each model
        results_df: DataFrame with performance metrics
        output_dir: Directory to save plot images
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Detect problem type from first available data
    first_model = list(curves_data.keys())[0]
    first_key = list(curves_data[first_model].keys())[0]
    problem_type = curves_data[first_model][first_key]['problem_type']
    
    # Only proceed for binary classification
    if problem_type != 'binary':
        print("Combined ROC curves only supported for binary classification problems")
        return
    
    # Find best model based on ROC AUC
    avg_metrics = results_df.group_by('model').agg(
        pl.mean('roc_auc').alias('avg_roc_auc')
    )
    best_model = avg_metrics.sort('avg_roc_auc', descending=True)[0, 'model']
    
    print(f"Creating combined ROC curves for best model: {best_model}")
    
    # Create the plot
    plt.figure(figsize=(12, 10))
    
    # Color palette for different train/test combinations
    colors = plt.cm.Set3(np.linspace(0, 1, len(curves_data[best_model])))
    color_idx = 0
    
    # Get all train/test combinations for the best model
    for pair_key, data in curves_data[best_model].items():
        y_test = data['y_test']
        y_proba = data['y_proba']
        
        # For binary classification, use positive class probabilities
        y_proba_pos = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_proba_pos)
        auc_score = roc_auc_score(y_test, y_proba_pos)
        
        # Parse the pair key to get readable labels
        if "all_except_" in pair_key:
            # Leave-one-out format: "all_except_source_source"
            parts = pair_key.split('_')
            held_out_source = parts[-1]
            label = f"Train: all except {held_out_source}, Test: {held_out_source} (AUC: {auc_score:.3f})"
        else:
            # Regular train_test format
            parts = pair_key.split('_')
            if len(parts) >= 2:
                train_source = parts[0]
                test_source = parts[1]
                label = f"Train: {train_source}, Test: {test_source} (AUC: {auc_score:.3f})"
            else:
                label = f"{pair_key} (AUC: {auc_score:.3f})"
        
        # Plot the ROC curve
        plt.plot(fpr, tpr, color=colors[color_idx], linewidth=2, 
                 label=label)
        color_idx += 1
    
    # Add diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random Classifier')
    
    # Configure plot
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curves - Best Model ({best_model}) Across All Train/Test Combinations', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plt.tight_layout()
    roc_curves_dir = os.path.join(output_dir, 'roc_curves')
    os.makedirs(roc_curves_dir, exist_ok=True)
    plt.savefig(os.path.join(roc_curves_dir, f"best_model_combined_roc_curves.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Combined ROC curves for best model saved to {os.path.join(roc_curves_dir, 'best_model_combined_roc_curves.png')}")

def plot_best_model_combined_prc_curves(curves_data, results_df, output_dir='model_curves'):
    """
    Plot a single PRC curve graph showing the best performing model 
    across all train/test cohort groupings.
    
    Args:
        curves_data: Dictionary containing curve data for each model
        results_df: DataFrame with performance metrics
        output_dir: Directory to save plot images
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Detect problem type from first available data
    first_model = list(curves_data.keys())[0]
    first_key = list(curves_data[first_model].keys())[0]
    problem_type = curves_data[first_model][first_key]['problem_type']
    
    # Only proceed for binary classification
    if problem_type != 'binary':
        print("Combined PRC curves only supported for binary classification problems")
        return
    
    # Find best model based on AUPRC
    avg_metrics = results_df.group_by('model').agg(
        pl.mean('auprc').alias('avg_auprc')
    )
    best_model = avg_metrics.sort('avg_auprc', descending=True)[0, 'model']
    
    print(f"Creating combined PRC curves for best model: {best_model}")
    
    # Create the plot
    plt.figure(figsize=(12, 10))
    
    # Color palette for different train/test combinations
    colors = plt.cm.Set3(np.linspace(0, 1, len(curves_data[best_model])))
    color_idx = 0
    
    # Track baselines for each combination
    baselines = []
    
    # Get all train/test combinations for the best model
    for pair_key, data in curves_data[best_model].items():
        y_test = data['y_test']
        y_proba = data['y_proba']
        
        # For binary classification, use positive class probabilities
        y_proba_pos = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
        
        # Calculate PRC curve
        precision, recall, _ = precision_recall_curve(y_test, y_proba_pos)
        auprc = average_precision_score(y_test, y_proba_pos)
        
        # Calculate baseline (ratio of positive samples)
        baseline = sum(y_test) / len(y_test)
        baselines.append(baseline)
        
        # Parse the pair key to get readable labels
        if "all_except_" in pair_key:
            # Leave-one-out format: "all_except_source_source"
            parts = pair_key.split('_')
            held_out_source = parts[-1]
            label = f"Train: all except {held_out_source}, Test: {held_out_source} (AUPRC: {auprc:.3f})"
        else:
            # Regular train_test format
            parts = pair_key.split('_')
            if len(parts) >= 2:
                train_source = parts[0]
                test_source = parts[1]
                label = f"Train: {train_source}, Test: {test_source} (AUPRC: {auprc:.3f})"
            else:
                label = f"{pair_key} (AUPRC: {auprc:.3f})"
        
        # Plot the PRC curve
        plt.plot(recall, precision, color=colors[color_idx], linewidth=2, 
                 label=label)
        color_idx += 1
    
    # Add baseline lines (average baseline across all combinations)
    avg_baseline = np.mean(baselines)
    plt.axhline(y=avg_baseline, color='r', linestyle='--', alpha=0.5, 
                label=f'Average Baseline: {avg_baseline:.3f}')
    
    # Configure plot
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(f'Precision-Recall Curves - Best Model ({best_model}) Across All Train/Test Combinations', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plt.tight_layout()
    prc_curves_dir = os.path.join(output_dir, 'prc_curves')
    os.makedirs(prc_curves_dir, exist_ok=True)
    plt.savefig(os.path.join(prc_curves_dir, f"best_model_combined_prc_curves.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Combined PRC curves for best model saved to {os.path.join(prc_curves_dir, 'best_model_combined_prc_curves.png')}")

if __name__ == "__main__":
    
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Run adaptive model comparison pipeline')
    parser.add_argument('--input', required=True, help='Path to the input parquet file')
    parser.add_argument('--output_dir', required=True, help='Directory to save output files')
    parser.add_argument('--source_column', default='source', help='Name of the source column')
    parser.add_argument('--target_column', default='annot', help='Name of the target column')
    parser.add_argument('--scoring_metric', help='Name of the scoring metric')
    parser.add_argument('--drop_columns', help='Comma-separated columns to drop')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--n_folds', type=int, default=5, help='Number of folds for cross-validation')
    parser.add_argument('--skip_train_production', action='store_true', help='Skip training the final production model')
    parser.add_argument('--model_version', help='Version identifier for production model')
    parser.add_argument('--save_fold_models', action='store_true', 
                        help='Save individual models from each train/test fold. Models will be saved in fold_models_<model_name>/ subdirectories with metadata JSON files.')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set consistent random seed for reproducibility
    random_seed = args.random_seed
    np.random.seed(random_seed)
    
    # Define output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Parse columns to drop
    drop_cols = []
    if args.drop_columns:
        drop_cols = args.drop_columns.split(',')
    
    # Load the data using Polars
    df = pl.read_parquet(args.input)
    if drop_cols:
        df = df.drop(drop_cols)
    
    # Specify column names
    source_column = args.source_column
    target_column = args.target_column
    
    # Detect problem type early for user awareness
    problem_type, n_classes, unique_classes = detect_problem_type(df[target_column])
    print(f"\n===== Detected {problem_type.upper()} classification problem =====")
    print(f"Number of classes: {n_classes}")
    print(f"Class labels: {unique_classes}")
    scoring_metric = args.scoring_metric
    # Set default scoring metric if not provided
    if not scoring_metric:
        scoring_metric = 'average_precision' if problem_type == 'binary' else 'f1_weighted'
    valid_metrics = ['average_precision', 'roc_auc', 'f1_weighted', 'f1_macro', 'f1', 
                    'accuracy', 'precision_weighted', 'recall_weighted']

    if args.scoring_metric and args.scoring_metric not in valid_metrics:
        print(f"Warning: Unknown scoring metric '{args.scoring_metric}'. Valid options: {valid_metrics}")
        print("Using default based on problem type.")
        args.scoring_metric = None
    print(f"Using scoring metric: {scoring_metric}")

    # Define models to compare
    models_config = define_models_config()
    
    # Check if source column exists in the dataframe
    has_source_column = source_column in df.columns
    
    # Check data before proceeding
    print("\n===== Data Quality Check =====")
    
    # Check class distribution overall
    class_counts = df[target_column].value_counts()
    print(f"Overall class distribution:\n{class_counts}")
    
    if has_source_column:
        # Check class distribution by source
        try:
            class_by_source = df.group_by([source_column, target_column]).agg(
                pl.len().alias('count')
            ).pivot(
                index=source_column,
                on=target_column,
                values='count'
            )
            print("\nClass distribution by source:")
            print(class_by_source)
        except Exception as e:
            print(f"Could not create class distribution by source: {e}")
        
        # Run leave-one-source-out validation
        results, curves_data = leave_one_source_out_validation(
            df, source_column, target_column, models_config, 
            scoring_metric, output_dir=output_dir, random_state=random_seed,
            save_fold_models=args.save_fold_models
        )
        
        # Save detailed results
        results.write_csv(os.path.join(output_dir, "leave_one_out_results.csv"))
        
        # Create summary table by model and test source
        summary = results.group_by(['model', 'test_source']).agg(
            pl.mean('accuracy').alias('avg_accuracy'),
            pl.mean('f1').alias('avg_f1'),
            pl.mean('roc_auc').alias('avg_roc_auc') if problem_type == 'binary' else pl.lit(None).alias('avg_roc_auc'),
            pl.mean('auprc').alias('avg_auprc') if problem_type == 'binary' else pl.lit(None).alias('avg_auprc')
        ).sort(['model', 'test_source'])
        
        summary.write_csv(os.path.join(output_dir, "leave_one_out_summary.csv"))
        
        # Generate and save plots
        try:
            plot_leave_one_out_curves(curves_data, results, output_dir=output_dir)
        except Exception as e:
            print(f"Error generating leave-one-source-out plots: {e}")
    else:
        # No source column, perform regular k-fold cross-validation
        print(f"\nNo source column found. Performing regular {args.n_folds}-fold cross-validation.")
        
        # Run k-fold cross-validation
        results, curves_data = regular_kfold_validation(
            df, target_column, models_config, scoring_metric,
            output_dir=output_dir, random_state=random_seed,
            n_folds=args.n_folds, save_fold_models=args.save_fold_models
        )
        
        # Save detailed results
        results.write_csv(os.path.join(output_dir, "kfold_results.csv"))
        
        # Create summary table by model
        summary_cols = ['avg_accuracy', 'avg_f1']
        if problem_type == 'binary':
            summary_cols.extend(['avg_roc_auc', 'avg_auprc'])
        
        summary_agg = [
            pl.mean('accuracy').alias('avg_accuracy'),
            pl.mean('f1').alias('avg_f1')
        ]
        if problem_type == 'binary':
            summary_agg.extend([
                pl.mean('roc_auc').alias('avg_roc_auc'),
                pl.mean('auprc').alias('avg_auprc')
            ])
        
        summary = results.group_by(['model']).agg(summary_agg).sort(['model'])
        
        summary.write_csv(os.path.join(output_dir, "kfold_summary.csv"))
        
        # Generate and save plots
        try:
            plot_kfold_curves(curves_data, results, output_dir=output_dir)
        except Exception as e:
            print(f'Error generating k-fold plots: {e}')

    # ===== Production Model Training =====
    if args.skip_train_production:
        print("\n💡 Skipping production model training as per --skip_train_production flag")
    else:
        print("\n" + "="*70)
        print("TRAINING PRODUCTION MODEL")
        print("="*70)
        
        # Determine best model from validation results
        print("Determining best model from validation results...")
        
        # Calculate average performance metrics by model
        if problem_type == 'binary':
            avg_metrics = results.group_by('model').agg([
                pl.mean('accuracy').alias('avg_accuracy'),
                pl.mean('precision').alias('avg_precision'),
                pl.mean('recall').alias('avg_recall'),
                pl.mean('f1').alias('avg_f1'),
                pl.mean('roc_auc').alias('avg_roc_auc'),
                pl.mean('auprc').alias('avg_auprc')
            ])
        else:
            avg_metrics = results.group_by('model').agg([
                pl.mean('accuracy').alias('avg_accuracy'),
                pl.mean('precision').alias('avg_precision'),
                pl.mean('recall').alias('avg_recall'),
                pl.mean('f1').alias('avg_f1')
            ])
        
        # Sort by the scoring metric used in validation
        metric_mapping = {
            'average_precision': 'avg_auprc',
            'roc_auc': 'avg_roc_auc', 
            'f1_weighted': 'avg_f1',
            'f1_macro': 'avg_f1',
            'f1': 'avg_f1',
            'accuracy': 'avg_accuracy',
            'precision_weighted': 'avg_precision',
            'recall_weighted': 'avg_recall'
        }
        
        sort_metric = metric_mapping.get(scoring_metric, 'avg_f1')
        
        # Handle binary vs multiclass metrics
        if sort_metric in ['avg_roc_auc', 'avg_auprc'] and problem_type != 'binary':
            sort_metric = 'avg_f1'
        
        best_models = avg_metrics.sort(sort_metric, descending=True)
        best_model_name = best_models[0, 'model']
        
        print(f"Best model based on {scoring_metric}: {best_model_name}")
        print("Average performance across all validation folds:")
        best_model_metrics = best_models.filter(pl.col('model') == best_model_name)
        print(best_model_metrics)
        
        # Get the best parameters for this model
        # Find the best single result for this model to extract parameters
        model_results = results.filter(pl.col('model') == best_model_name)
        if sort_metric in model_results.columns:
            best_single_result = model_results.sort(sort_metric, descending=True)[0]
        else:
            best_single_result = model_results.sort('f1', descending=True)[0]
        
        best_params_str = best_single_result['best_params']
        
        # Parse the best parameters string back to dict
        import ast
        try:
            # Remove 'model__' prefix from parameter string if present
            cleaned_params_str = best_params_str.replace("'model__", "'")
            best_params = ast.literal_eval(cleaned_params_str)
        except (ValueError, SyntaxError) as e:
            print(f"Warning: Could not parse best parameters: {e}")
            print(f"Using default parameters for {best_model_name}")
            best_params = {}
        
        print(f"Best parameters: {best_params}")
        
        # Get model class from config
        model_class, _ = models_config[best_model_name]
        
        # Create best model configuration
        best_model_config = (best_model_name, model_class, best_params)
        
        # Train final production model
        try:
            final_model, model_info = train_final_production_model(
                df=df,
                source_col=source_column if has_source_column else None,
                target_col=target_column,
                best_model_config=best_model_config,
                output_dir=os.path.join(output_dir, 'production_models'),
                model_version=args.model_version,
                random_state=random_seed
            )
            
            print(f"\n✅ Production model training completed successfully!")
            print(f"Model saved with version: {model_info['model_version']}")
            
            # Save a final summary report
            final_summary = {
                'validation_type': 'leave_one_source_out' if has_source_column else f'{args.n_folds}_fold_cv',
                'best_model': best_model_name,
                'best_model_params': best_params,
                'scoring_metric_used': scoring_metric,
                'validation_performance': dict(best_model_metrics.to_pandas().iloc[0]),
                'production_model_info': {
                    'version': model_info['model_version'],
                    'training_samples': model_info['total_samples'],
                    'features': model_info['n_features'],
                    'training_duration': model_info['training_duration_seconds']
                }
            }
            
            # Save final summary
            import json
            summary_path = os.path.join(output_dir, 'final_model_summary.json')
            with open(summary_path, 'w') as f:
                json.dump(final_summary, f, indent=2, default=str)
            
            print(f"Final summary saved to: {summary_path}")
            
        except Exception as e:
            print(f"❌ Error during production model training: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary message about fold models
    if args.save_fold_models:
        print(f"\n📁 Individual fold models saved in fold_models_<model_name>/ subdirectories within: {output_dir}")
        print("Each model includes:")
        print("  - Trained model (.pkl file)")
        print("  - Model metadata and parameters (.json file)")
        print("  - Performance metrics for that specific fold")
    
    print(f"\n🎉 Pipeline completed! Results saved to: {output_dir}")