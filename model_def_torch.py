import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, List, Dict, Any, Optional, Union
import os


class SimpleLSTM(nn.Module):
    """
    Simple LSTM model for speech emotion recognition.
    """
    def __init__(self, input_shape: Tuple[int, int], 
                 num_classes: int,
                 lstm_units: int = 128, 
                 dropout_rate: float = 0.3):
        """
        Args:
            input_shape: Shape of input features (time_steps, num_features)
            num_classes: Number of classes to predict
            lstm_units: Number of LSTM units
            dropout_rate: Dropout rate
        """
        super(SimpleLSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_shape[1], lstm_units, batch_first=True, return_sequences=True)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.lstm2 = nn.LSTM(lstm_units, lstm_units, batch_first=True)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(lstm_units, 64)
        self.relu = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(64, num_classes)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        x = self.fc1(x[:, -1, :])  # Take the last time step
        x = self.relu(x)
        x = self.dropout3(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


class BidirectionalLSTM(nn.Module):
    """
    Bidirectional LSTM model for speech emotion recognition.
    """
    def __init__(self, input_shape: Tuple[int, int], 
                 num_classes: int,
                 lstm_units: int = 128, 
                 dropout_rate: float = 0.3):
        """
        Args:
            input_shape: Shape of input features (time_steps, num_features)
            num_classes: Number of classes to predict
            lstm_units: Number of LSTM units
            dropout_rate: Dropout rate
        """
        super(BidirectionalLSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_shape[1], lstm_units, batch_first=True, 
                            bidirectional=True, return_sequences=True)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.lstm2 = nn.LSTM(lstm_units * 2, lstm_units, batch_first=True, 
                            bidirectional=True)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(lstm_units * 2, 64)
        self.relu = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(64, num_classes)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        x = self.fc1(x[:, -1, :])  # Take the last time step
        x = self.relu(x)
        x = self.dropout3(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


class CNN1D(nn.Module):
    """
    1D CNN model for speech emotion recognition.
    """
    def __init__(self, input_shape: Tuple[int, int], 
                 num_classes: int,
                 dropout_rate: float = 0.3):
        """
        Args:
            input_shape: Shape of input features (time_steps, num_features)
            num_classes: Number of classes to predict
            dropout_rate: Dropout rate
        """
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(input_shape[1], 64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        
        # Calculate the size after pooling
        self.flatten_size = 256 * (input_shape[0] // 8)
        
        self.fc1 = nn.Linear(self.flatten_size, 128)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, num_classes)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # Transpose to (batch, features, time)
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


class CNN2D(nn.Module):
    """
    2D CNN model for speech emotion recognition.
    """
    def __init__(self, input_shape: Tuple[int, int, int], 
                 num_classes: int,
                 dropout_rate: float = 0.3):
        """
        Args:
            input_shape: Shape of input features (time_steps, freq_bins, channels)
            num_classes: Number of classes to predict
            dropout_rate: Dropout rate
        """
        super(CNN2D, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        # Calculate the size after pooling
        time_after_pool = input_shape[0] // 8
        freq_after_pool = input_shape[1] // 8
        self.flatten_size = 128 * time_after_pool * freq_after_pool
        
        self.fc1 = nn.Linear(self.flatten_size, 128)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, num_classes)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # Add channel dimension
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


class CNNLSTM(nn.Module):
    """
    CNN-LSTM hybrid model for speech emotion recognition.
    """
    def __init__(self, input_shape: Tuple[int, int], 
                 num_classes: int,
                 lstm_units: int = 128,
                 dropout_rate: float = 0.3):
        """
        Args:
            input_shape: Shape of input features (time_steps, num_features)
            num_classes: Number of classes to predict
            lstm_units: Number of LSTM units
            dropout_rate: Dropout rate
        """
        super(CNNLSTM, self).__init__()
        self.conv1 = nn.Conv1d(input_shape[1], 64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        # Calculate the size after pooling
        self.time_after_pool = input_shape[0] // 4
        
        self.lstm = nn.LSTM(128, lstm_units, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(lstm_units * 2, 64)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # Transpose to (batch, features, time)
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # Transpose back to (batch, time, features) for LSTM
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        x = self.dropout(x)
        x = self.fc1(x[:, -1, :])  # Take the last time step
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


def create_emotion_recognition_model(input_shape: Tuple[int, ...], 
                                    num_classes: int,
                                    model_type: str = 'cnn_lstm') -> nn.Module:
    """
    Create a model for speech emotion recognition.
    
    Args:
        input_shape: Shape of input features
        num_classes: Number of emotion classes
        model_type: Type of model to create
    
    Returns:
        PyTorch model
    """
    if model_type == 'lstm':
        model = SimpleLSTM(input_shape, num_classes)
    elif model_type == 'bilstm':
        model = BidirectionalLSTM(input_shape, num_classes)
    elif model_type == 'cnn1d':
        model = CNN1D(input_shape, num_classes)
    elif model_type == 'cnn2d':
        model = CNN2D(input_shape, num_classes)
    elif model_type == 'cnn_lstm':
        model = CNNLSTM(input_shape, num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def create_speaker_identification_model(input_shape: Tuple[int, ...], 
                                       num_classes: int,
                                       model_type: str = 'cnn_lstm') -> nn.Module:
    """
    Create a model for speaker identification.
    
    Args:
        input_shape: Shape of input features
        num_classes: Number of speaker classes
        model_type: Type of model to create
    
    Returns:
        PyTorch model
    """
    # For simplicity, we use the same model architecture for both tasks
    return create_emotion_recognition_model(input_shape, num_classes, model_type)


def train_model(model: nn.Module,
               X_train: np.ndarray,
               y_train: np.ndarray,
               X_val: np.ndarray = None,
               y_val: np.ndarray = None,
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
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    if X_val is not None and y_val is not None:
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(device)
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
        if X_val is not None and y_val is not None:
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
            
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        else:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    
    # If no validation data, save the final model
    if (X_val is None or y_val is None) and model_save_path:
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        torch.save(model.state_dict(), model_save_path)
        print(f"Saved final model after {epochs} epochs")
    
    return history


def evaluate_model(model: nn.Module,
                  X_test: np.ndarray,
                  y_test: np.ndarray,
                  batch_size: int = 32) -> Dict[str, float]:
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
    
    return {
        'loss': test_loss,
        'accuracy': test_acc,
        'predictions': np.array(all_predictions),
        'targets': np.array(all_targets)
    }
