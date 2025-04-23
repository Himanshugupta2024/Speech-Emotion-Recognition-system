import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                            confusion_matrix, classification_report, roc_curve, auc)
from typing import Dict, List, Tuple, Union, Optional, Any

def calculate_metrics(y_true: np.ndarray, 
                     y_pred: np.ndarray, 
                     classes: List[str]) -> Dict[str, float]:
    """
    Calculate classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        classes: Class names
    
    Returns:
        Dictionary of metrics
    """
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro'),
        'recall_macro': recall_score(y_true, y_pred, average='macro'),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted'),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted'),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted')
    }
    
    # Calculate per-class metrics
    classification_rep = classification_report(y_true, y_pred, 
                                              target_names=classes, 
                                              output_dict=True)
    
    # Add per-class metrics to the dictionary
    for cls in classes:
        cls_metrics = classification_rep[cls]
        metrics[f'precision_{cls}'] = cls_metrics['precision']
        metrics[f'recall_{cls}'] = cls_metrics['recall']
        metrics[f'f1_{cls}'] = cls_metrics['f1-score']
    
    return metrics

def plot_confusion_matrix(y_true: np.ndarray, 
                         y_pred: np.ndarray, 
                         classes: List[str],
                         normalize: bool = True,
                         title: str = 'Confusion Matrix',
                         figsize: Tuple[int, int] = (10, 8),
                         save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        classes: Class names
        normalize: Whether to normalize the confusion matrix
        title: Plot title
        figsize: Figure size
        save_path: Path to save the figure
    
    Returns:
        Matplotlib figure
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize if required
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create figure
    plt.figure(figsize=figsize)
    fig = plt.figure(figsize=figsize)
    
    # Plot confusion matrix
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                cmap='Blues', xticklabels=classes, yticklabels=classes)
    
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save figure if required
    if save_path:
        plt.savefig(save_path)
    
    return fig

def plot_roc_curve(y_true: np.ndarray, 
                  y_score: np.ndarray, 
                  classes: List[str],
                  figsize: Tuple[int, int] = (10, 8),
                  save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot ROC curve for multi-class classification.
    
    Args:
        y_true: True labels (one-hot encoded)
        y_score: Predicted probabilities
        classes: Class names
        figsize: Figure size
        save_path: Path to save the figure
    
    Returns:
        Matplotlib figure
    """
    # Create figure
    fig = plt.figure(figsize=figsize)
    
    # Calculate ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    n_classes = len(classes)
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot ROC curve for each class
    for i, color, cls in zip(range(n_classes), 
                             plt.cm.rainbow(np.linspace(0, 1, n_classes)),
                             classes):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'{cls} (AUC = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    # Save figure if required
    if save_path:
        plt.savefig(save_path)
    
    return fig

def plot_training_history(history: Dict[str, List[float]],
                         figsize: Tuple[int, int] = (12, 5),
                         save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot training history.
    
    Args:
        history: Training history dictionary
        figsize: Figure size
        save_path: Path to save the figure
    
    Returns:
        Matplotlib figure
    """
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot accuracy
    ax1.plot(history['accuracy'])
    ax1.plot(history['val_accuracy'])
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot loss
    ax2.plot(history['loss'])
    ax2.plot(history['val_loss'])
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    
    # Save figure if required
    if save_path:
        plt.savefig(save_path)
    
    return fig

def evaluate_emotion_recognition(y_true: np.ndarray,
                                y_pred: np.ndarray,
                                y_score: np.ndarray,
                                emotions: List[str],
                                history: Optional[Dict[str, List[float]]] = None,
                                output_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Evaluate speech emotion recognition model.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_score: Predicted probabilities
        emotions: Emotion names
        history: Training history dictionary
        output_dir: Directory to save evaluation results
    
    Returns:
        Dictionary containing evaluation results
    """
    # Create output directory if it doesn't exist
    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred, emotions)
    
    # Plot confusion matrix
    cm_fig = plot_confusion_matrix(y_true, y_pred, emotions,
                                normalize=True,
                                title='Emotion Recognition Confusion Matrix')
    
    if output_dir:
        cm_fig.savefig(f"{output_dir}/emotion_confusion_matrix.png")
    
    # Plot ROC curve
    # Convert y_true to one-hot encoding if it's not already
    if len(y_true.shape) == 1:
        from tensorflow.keras.utils import to_categorical
        y_true_onehot = to_categorical(y_true, num_classes=len(emotions))
    else:
        y_true_onehot = y_true
    
    roc_fig = plot_roc_curve(y_true_onehot, y_score, emotions)
    
    if output_dir:
        roc_fig.savefig(f"{output_dir}/emotion_roc_curve.png")
    
    # Plot training history if provided
    if history:
        history_fig = plot_training_history(history)
        
        if output_dir:
            history_fig.savefig(f"{output_dir}/emotion_training_history.png")
    
    # Save metrics to CSV if output_dir is provided
    if output_dir:
        metrics_df = pd.DataFrame(metrics.items(), columns=['Metric', 'Value'])
        metrics_df.to_csv(f"{output_dir}/emotion_metrics.csv", index=False)
    
    # Return evaluation results
    evaluation_results = {
        'metrics': metrics,
        'figures': {
            'confusion_matrix': cm_fig,
            'roc_curve': roc_fig
        }
    }
    
    if history:
        evaluation_results['figures']['training_history'] = history_fig
    
    return evaluation_results

def evaluate_speaker_identification(y_true: np.ndarray,
                                   y_pred: np.ndarray,
                                   y_score: np.ndarray,
                                   speakers: List[str],
                                   history: Optional[Dict[str, List[float]]] = None,
                                   output_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Evaluate speaker identification model.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_score: Predicted probabilities
        speakers: Speaker names or IDs
        history: Training history dictionary
        output_dir: Directory to save evaluation results
    
    Returns:
        Dictionary containing evaluation results
    """
    # Create output directory if it doesn't exist
    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred, speakers)
    
    # Plot confusion matrix
    cm_fig = plot_confusion_matrix(y_true, y_pred, speakers,
                                normalize=True,
                                title='Speaker Identification Confusion Matrix')
    
    if output_dir:
        cm_fig.savefig(f"{output_dir}/speaker_confusion_matrix.png")
    
    # Plot ROC curve
    # Convert y_true to one-hot encoding if it's not already
    if len(y_true.shape) == 1:
        from tensorflow.keras.utils import to_categorical
        y_true_onehot = to_categorical(y_true, num_classes=len(speakers))
    else:
        y_true_onehot = y_true
    
    roc_fig = plot_roc_curve(y_true_onehot, y_score, speakers)
    
    if output_dir:
        roc_fig.savefig(f"{output_dir}/speaker_roc_curve.png")
    
    # Plot training history if provided
    if history:
        history_fig = plot_training_history(history)
        
        if output_dir:
            history_fig.savefig(f"{output_dir}/speaker_training_history.png")
    
    # Save metrics to CSV if output_dir is provided
    if output_dir:
        metrics_df = pd.DataFrame(metrics.items(), columns=['Metric', 'Value'])
        metrics_df.to_csv(f"{output_dir}/speaker_metrics.csv", index=False)
    
    # Return evaluation results
    evaluation_results = {
        'metrics': metrics,
        'figures': {
            'confusion_matrix': cm_fig,
            'roc_curve': roc_fig
        }
    }
    
    if history:
        evaluation_results['figures']['training_history'] = history_fig
    
    return evaluation_results 