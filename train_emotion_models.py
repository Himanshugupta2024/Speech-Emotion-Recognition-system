import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import argparse
import librosa
import glob
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Any, Optional
import json
import time

# Import our modules
from enhanced_feature_extraction import extract_mfcc, normalize_features
from enhanced_models import create_model

# Define the emotions
EMOTIONS = ['neutral', 'happy', 'sad', 'angry', 'fearful', 'disgusted', 'surprised', 'calm']

def load_audio_file(file_path: str, sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Load an audio file.
    
    Args:
        file_path: Path to the audio file
        sr: Sample rate
    
    Returns:
        Tuple containing the audio data and the sample rate
    """
    try:
        audio, sample_rate = librosa.load(file_path, sr=sr)
        return audio, sample_rate
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return np.array([]), sr

def extract_features_from_file(file_path: str, sr: int = 16000) -> np.ndarray:
    """
    Extract features from an audio file.
    
    Args:
        file_path: Path to the audio file
        sr: Sample rate
    
    Returns:
        Extracted features
    """
    # Load audio
    audio, _ = load_audio_file(file_path, sr)
    
    if len(audio) == 0:
        return None
    
    # Extract MFCC features
    features = extract_mfcc(audio, sr)
    
    # Pad or truncate to fixed length (3 seconds at 16kHz with hop_length=256)
    target_length = 188  # Approximately 3 seconds
    
    if features.shape[0] < target_length:
        # Pad with zeros
        padding = np.zeros((target_length - features.shape[0], features.shape[1]))
        features = np.vstack((features, padding))
    elif features.shape[0] > target_length:
        # Truncate
        features = features[:target_length, :]
    
    return features

def load_dataset(data_dir: str, sr: int = 16000) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load a dataset of audio files.
    
    Args:
        data_dir: Directory containing audio files
        sr: Sample rate
    
    Returns:
        Tuple containing features, labels, and file paths
    """
    features = []
    labels = []
    file_paths = []
    
    # Find all audio files
    audio_files = []
    for ext in ['*.wav', '*.mp3', '*.ogg']:
        audio_files.extend(glob.glob(os.path.join(data_dir, '**', ext), recursive=True))
    
    print(f"Found {len(audio_files)} audio files")
    
    # Extract features and labels
    for file_path in audio_files:
        # Extract emotion from directory name
        parent_dir = os.path.basename(os.path.dirname(file_path)).lower()
        
        # Check if parent directory is one of the emotions
        if parent_dir in EMOTIONS:
            emotion = parent_dir
        else:
            # Try to extract emotion from filename
            filename = os.path.basename(file_path).lower()
            emotion = None
            
            for e in EMOTIONS:
                if e in filename:
                    emotion = e
                    break
            
            if emotion is None:
                print(f"Warning: Could not determine emotion for {file_path}")
                continue
        
        # Extract features
        feature = extract_features_from_file(file_path, sr)
        
        if feature is not None:
            features.append(feature)
            labels.append(emotion)
            file_paths.append(file_path)
    
    if len(features) == 0:
        return np.array([]), np.array([]), []
    
    return np.array(features), np.array(labels), file_paths

def train_model(model: nn.Module,
               X_train: np.ndarray,
               y_train: np.ndarray,
               X_val: np.ndarray,
               y_val: np.ndarray,
               batch_size: int = 32,
               epochs: int = 100,
               learning_rate: float = 0.001,
               model_save_path: str = None) -> Dict[str, Any]:
    """
    Train a PyTorch model.
    
    Args:
        model: PyTorch model to train
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        batch_size: Batch size for training
        epochs: Maximum number of epochs for training
        learning_rate: Learning rate for optimizer
        model_save_path: Path to save the best model
    
    Returns:
        Dictionary containing training history
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Convert numpy arrays to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(device)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_loss = float('inf')
    
    # Training loop
    for epoch in range(epochs):
        start_time = time.time()
        
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()
        
        train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
        
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss and model_save_path:
            best_val_loss = val_loss
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved best model at epoch {epoch+1} with validation loss: {val_loss:.4f}")
        
        # Print progress
        end_time = time.time()
        epoch_time = end_time - start_time
        print(f"Epoch {epoch+1}/{epochs} - {epoch_time:.2f}s - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    return history

def evaluate_model(model: nn.Module,
                  X_test: np.ndarray,
                  y_test: np.ndarray,
                  batch_size: int = 32) -> Dict[str, Any]:
    """
    Evaluate a PyTorch model.
    
    Args:
        model: PyTorch model to evaluate
        X_test: Test features
        y_test: Test labels
        batch_size: Batch size for evaluation
    
    Returns:
        Dictionary containing evaluation metrics
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Convert numpy arrays to PyTorch tensors
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)
    
    # Create data loader
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    
    # Evaluation
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            test_total += targets.size(0)
            test_correct += (predicted == targets).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    test_loss = test_loss / test_total
    test_acc = test_correct / test_total
    
    # Calculate per-class accuracy
    class_correct = np.zeros(len(EMOTIONS))
    class_total = np.zeros(len(EMOTIONS))
    
    for i in range(len(all_targets)):
        label = all_targets[i]
        class_correct[label] += (all_predictions[i] == label)
        class_total[label] += 1
    
    class_acc = {}
    for i in range(len(EMOTIONS)):
        if class_total[i] > 0:
            class_acc[EMOTIONS[i]] = float(class_correct[i] / class_total[i])
        else:
            class_acc[EMOTIONS[i]] = 0.0
    
    return {
        'loss': test_loss,
        'accuracy': test_acc,
        'class_accuracy': class_acc,
        'predictions': np.array(all_predictions),
        'targets': np.array(all_targets)
    }

def plot_training_history(history: Dict[str, List[float]], title: str = 'Training History') -> plt.Figure:
    """
    Plot training history.
    
    Args:
        history: Dictionary containing training history
        title: Plot title
    
    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot loss
    ax1.plot(history['train_loss'], label='Train')
    ax1.plot(history['val_loss'], label='Validation')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Plot accuracy
    ax2.plot(history['train_acc'], label='Train')
    ax2.plot(history['val_acc'], label='Validation')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    fig.suptitle(title)
    plt.tight_layout()
    return fig

def plot_confusion_matrix(y_true: np.ndarray, 
                         y_pred: np.ndarray, 
                         classes: List[str],
                         title: str = 'Confusion Matrix') -> plt.Figure:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        classes: List of class names
        title: Plot title
    
    Returns:
        Matplotlib figure
    """
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot confusion matrix
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=classes, yticklabels=classes, ax=ax)
    
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    
    plt.tight_layout()
    return fig

def main():
    parser = argparse.ArgumentParser(description='Train Speech Emotion Recognition Models')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing audio files')
    parser.add_argument('--output_dir', type=str, default='models',
                        help='Directory to save models and results')
    parser.add_argument('--model_types', type=str, nargs='+', default=['cnn', 'lstm', 'cnn_lstm'],
                        choices=['cnn', 'lstm', 'cnn_lstm'],
                        help='Types of models to train')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Maximum number of epochs for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate for optimizer')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Proportion of data to use for testing')
    parser.add_argument('--val_size', type=float, default=0.2,
                        help='Proportion of training data to use for validation')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    print(f"Loading dataset from {args.data_dir}...")
    features, labels, file_paths = load_dataset(args.data_dir)
    
    if len(features) == 0:
        print("Error: No features extracted")
        return
    
    print(f"Extracted features from {len(features)} audio files")
    
    # Normalize features
    features = normalize_features(features)
    
    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    
    # Map encoded labels to emotions
    label_mapping = {i: emotion for i, emotion in enumerate(label_encoder.classes_)}
    
    # Save label mapping
    with open(os.path.join(args.output_dir, 'label_mapping.json'), 'w') as f:
        json.dump(label_mapping, f, indent=4)
    
    # Split data into train, validation, and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        features, encoded_labels, test_size=args.test_size, random_state=42, stratify=encoded_labels
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=args.val_size, random_state=42, stratify=y_train_val
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Train and evaluate models
    results = {}
    
    for model_type in args.model_types:
        print(f"\nTraining {model_type.upper()} model...")
        
        # Create model
        input_shape = X_train.shape[1:]
        num_classes = len(label_encoder.classes_)
        model = create_model(model_type, input_shape, num_classes)
        
        # Train model
        model_save_path = os.path.join(args.output_dir, f'{model_type}_model.pt')
        history = train_model(
            model, X_train, y_train, X_val, y_val, 
            batch_size=args.batch_size, 
            epochs=args.epochs, 
            learning_rate=args.learning_rate,
            model_save_path=model_save_path
        )
        
        # Plot and save training history
        history_fig = plot_training_history(history, f'{model_type.upper()} Training History')
        history_fig.savefig(os.path.join(args.output_dir, f'{model_type}_training_history.png'))
        
        # Load best model
        model.load_state_dict(torch.load(model_save_path))
        
        # Evaluate model
        print(f"Evaluating {model_type.upper()} model...")
        evaluation = evaluate_model(model, X_test, y_test, batch_size=args.batch_size)
        
        # Plot and save confusion matrix
        cm_fig = plot_confusion_matrix(
            evaluation['targets'], 
            evaluation['predictions'], 
            [label_mapping[i] for i in range(num_classes)],
            f'{model_type.upper()} Confusion Matrix'
        )
        cm_fig.savefig(os.path.join(args.output_dir, f'{model_type}_confusion_matrix.png'))
        
        # Save evaluation results
        results[model_type] = {
            'accuracy': evaluation['accuracy'],
            'loss': evaluation['loss'],
            'class_accuracy': evaluation['class_accuracy']
        }
        
        print(f"{model_type.upper()} Test Accuracy: {evaluation['accuracy']:.4f}")
        print("Per-class accuracy:")
        for emotion, acc in evaluation['class_accuracy'].items():
            print(f"  {emotion}: {acc:.4f}")
    
    # Save overall results
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    print("\nTraining completed!")
    print(f"Models and results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
