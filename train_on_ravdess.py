import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import time

# Import our modules
from enhanced_feature_extraction import extract_mfcc, normalize_features
from enhanced_models import create_model

def main():
    parser = argparse.ArgumentParser(description='Train Speech Emotion Recognition Models on RAVDESS')
    parser.add_argument('--data_dir', type=str, 
                        default='speech_emotion_speaker_id/data/raw/ravdess_by_emotion',
                        help='Directory containing organized RAVDESS dataset')
    parser.add_argument('--output_dir', type=str, default='models',
                        help='Directory to save models and results')
    parser.add_argument('--model_type', type=str, default='cnn_lstm',
                        choices=['cnn', 'lstm', 'cnn_lstm'],
                        help='Type of model to train')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate for optimizer')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Proportion of data to use for testing')
    parser.add_argument('--val_size', type=float, default=0.2,
                        help='Proportion of training data to use for validation')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading dataset from {args.data_dir}...")
    
    # Load and preprocess data
    features = []
    labels = []
    
    # Process each emotion directory
    for emotion in os.listdir(args.data_dir):
        emotion_dir = os.path.join(args.data_dir, emotion)
        if not os.path.isdir(emotion_dir):
            continue
        
        print(f"Processing {emotion} files...")
        
        # Process each audio file in the emotion directory
        for filename in os.listdir(emotion_dir):
            if not filename.endswith('.wav'):
                continue
            
            file_path = os.path.join(emotion_dir, filename)
            
            try:
                # Load audio
                import librosa
                audio, sr = librosa.load(file_path, sr=16000)
                
                # Extract MFCC features
                mfcc_features = extract_mfcc(audio, sr)
                
                # Pad or truncate to fixed length (3 seconds at 16kHz with hop_length=256)
                target_length = 188  # Approximately 3 seconds
                
                if mfcc_features.shape[0] < target_length:
                    # Pad with zeros
                    padding = np.zeros((target_length - mfcc_features.shape[0], mfcc_features.shape[1]))
                    mfcc_features = np.vstack((mfcc_features, padding))
                elif mfcc_features.shape[0] > target_length:
                    # Truncate
                    mfcc_features = mfcc_features[:target_length, :]
                
                features.append(mfcc_features)
                labels.append(emotion)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    
    if len(features) == 0:
        print("Error: No features extracted")
        return
    
    print(f"Extracted features from {len(features)} audio files")
    
    # Convert to numpy arrays
    features = np.array(features)
    labels = np.array(labels)
    
    # Normalize features
    features = normalize_features(features)
    
    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    
    # Map encoded labels to emotions
    label_mapping = {i: emotion for i, emotion in enumerate(label_encoder.classes_)}
    
    # Save label mapping
    import json
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
    
    # Create model
    input_shape = X_train.shape[1:]
    num_classes = len(label_encoder.classes_)
    model = create_model(args.model_type, input_shape, num_classes)
    
    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = model.to(device)
    
    # Convert numpy arrays to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_loss = float('inf')
    model_save_path = os.path.join(args.output_dir, f'{args.model_type}_model_ravdess.pt')
    
    print(f"\nTraining {args.model_type.upper()} model...")
    
    # Training loop
    for epoch in range(args.epochs):
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
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved best model at epoch {epoch+1} with validation loss: {val_loss:.4f}")
        
        # Print progress
        end_time = time.time()
        epoch_time = end_time - start_time
        print(f"Epoch {epoch+1}/{args.epochs} - {epoch_time:.2f}s - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.suptitle(f'{args.model_type.upper()} Training History')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f'{args.model_type}_training_history_ravdess.png'))
    
    # Load best model
    model.load_state_dict(torch.load(model_save_path))
    
    # Evaluate on test set
    model.eval()
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
    
    print(f"\nTest Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
    
    # Calculate per-class accuracy
    from sklearn.metrics import classification_report, confusion_matrix
    import seaborn as sns
    
    # Print classification report
    print("\nClassification Report:")
    class_names = [label_mapping[i] for i in range(len(label_mapping))]
    print(classification_report(all_targets, all_predictions, target_names=class_names))
    
    # Plot confusion matrix
    cm = confusion_matrix(all_targets, all_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{args.model_type.upper()} Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f'{args.model_type}_confusion_matrix_ravdess.png'))
    
    print(f"\nTraining completed! Model saved to {model_save_path}")

if __name__ == '__main__':
    main()
