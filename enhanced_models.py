import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, List, Any, Optional

class CNNModel(nn.Module):
    """
    CNN model for speech emotion recognition.
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
        super(CNNModel, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(input_shape[1], 64, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv1d(128, 256, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Calculate the size after pooling
        self.flatten_size = 256 * (input_shape[0] // 8)
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flatten_size, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn5 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # Transpose to (batch, features, time)
        x = x.transpose(1, 2)
        
        # Convolutional layers
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.bn4(self.fc1(x)))
        x = self.dropout1(x)
        
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.dropout2(x)
        
        x = self.fc3(x)
        
        return F.softmax(x, dim=1)
    
    def get_architecture_diagram(self) -> Dict[str, Any]:
        """
        Get a diagram of the model architecture.
        
        Returns:
            Dictionary containing model architecture information
        """
        return {
            'name': 'CNN Architecture',
            'layers': [
                {'name': 'Conv1D', 'filters': 64, 'kernel_size': 5},
                {'name': 'BatchNorm1D', 'features': 64},
                {'name': 'MaxPool1D', 'kernel_size': 2},
                {'name': 'Conv1D', 'filters': 128, 'kernel_size': 5},
                {'name': 'BatchNorm1D', 'features': 128},
                {'name': 'MaxPool1D', 'kernel_size': 2},
                {'name': 'Conv1D', 'filters': 256, 'kernel_size': 5},
                {'name': 'BatchNorm1D', 'features': 256},
                {'name': 'MaxPool1D', 'kernel_size': 2},
                {'name': 'Flatten', 'output_size': self.flatten_size},
                {'name': 'Dense', 'units': 512},
                {'name': 'BatchNorm1D', 'features': 512},
                {'name': 'Dropout', 'rate': 0.3},
                {'name': 'Dense', 'units': 256},
                {'name': 'BatchNorm1D', 'features': 256},
                {'name': 'Dropout', 'rate': 0.3},
                {'name': 'Dense', 'units': 'num_classes'},
                {'name': 'Softmax', 'axis': 1}
            ]
        }


class LSTMModel(nn.Module):
    """
    LSTM model for speech emotion recognition.
    """
    def __init__(self, input_shape: Tuple[int, int], 
                 num_classes: int,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 dropout_rate: float = 0.3,
                 bidirectional: bool = True):
        """
        Args:
            input_shape: Shape of input features (time_steps, num_features)
            num_classes: Number of classes to predict
            hidden_size: Number of features in the hidden state
            num_layers: Number of recurrent layers
            dropout_rate: Dropout rate
            bidirectional: Whether to use bidirectional LSTM
        """
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_shape[1],
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size * self.num_directions, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # LSTM layers
        output, (hidden, cell) = self.lstm(x)
        
        # Get the last time step output
        if self.bidirectional:
            # Concatenate the last outputs from forward and backward passes
            hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            hidden = hidden[-1, :, :]
        
        # Fully connected layers
        x = F.relu(self.bn1(self.fc1(hidden)))
        x = self.dropout1(x)
        
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        
        x = self.fc3(x)
        
        return F.softmax(x, dim=1)
    
    def get_architecture_diagram(self) -> Dict[str, Any]:
        """
        Get a diagram of the model architecture.
        
        Returns:
            Dictionary containing model architecture information
        """
        return {
            'name': 'LSTM Architecture',
            'layers': [
                {'name': 'LSTM', 'hidden_size': self.hidden_size, 'num_layers': self.num_layers, 'bidirectional': self.bidirectional},
                {'name': 'Dense', 'units': 256},
                {'name': 'BatchNorm1D', 'features': 256},
                {'name': 'Dropout', 'rate': 0.3},
                {'name': 'Dense', 'units': 128},
                {'name': 'BatchNorm1D', 'features': 128},
                {'name': 'Dropout', 'rate': 0.3},
                {'name': 'Dense', 'units': 'num_classes'},
                {'name': 'Softmax', 'axis': 1}
            ]
        }


class CNNLSTMModel(nn.Module):
    """
    CNN-LSTM hybrid model for speech emotion recognition.
    """
    def __init__(self, input_shape: Tuple[int, int], 
                 num_classes: int,
                 hidden_size: int = 128,
                 dropout_rate: float = 0.3,
                 bidirectional: bool = True):
        """
        Args:
            input_shape: Shape of input features (time_steps, num_features)
            num_classes: Number of classes to predict
            hidden_size: Number of features in the hidden state
            dropout_rate: Dropout rate
            bidirectional: Whether to use bidirectional LSTM
        """
        super(CNNLSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(input_shape[1], 64, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Calculate the size after pooling
        self.time_after_pool = input_shape[0] // 4
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size * self.num_directions, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # Transpose to (batch, features, time)
        x = x.transpose(1, 2)
        
        # Convolutional layers
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        
        # Transpose back to (batch, time, features) for LSTM
        x = x.transpose(1, 2)
        
        # LSTM layer
        output, (hidden, cell) = self.lstm(x)
        
        # Get the last time step output
        if self.bidirectional:
            # Concatenate the last outputs from forward and backward passes
            hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            hidden = hidden[-1, :, :]
        
        # Fully connected layers
        x = F.relu(self.bn3(self.fc1(hidden)))
        x = self.dropout1(x)
        
        x = F.relu(self.bn4(self.fc2(x)))
        x = self.dropout2(x)
        
        x = self.fc3(x)
        
        return F.softmax(x, dim=1)
    
    def get_architecture_diagram(self) -> Dict[str, Any]:
        """
        Get a diagram of the model architecture.
        
        Returns:
            Dictionary containing model architecture information
        """
        return {
            'name': 'CNN+LSTM Architecture',
            'layers': [
                {'name': 'Conv1D', 'filters': 64, 'kernel_size': 5},
                {'name': 'BatchNorm1D', 'features': 64},
                {'name': 'MaxPool1D', 'kernel_size': 2},
                {'name': 'Conv1D', 'filters': 128, 'kernel_size': 5},
                {'name': 'BatchNorm1D', 'features': 128},
                {'name': 'MaxPool1D', 'kernel_size': 2},
                {'name': 'LSTM', 'hidden_size': self.hidden_size, 'bidirectional': self.bidirectional},
                {'name': 'Dense', 'units': 256},
                {'name': 'BatchNorm1D', 'features': 256},
                {'name': 'Dropout', 'rate': 0.3},
                {'name': 'Dense', 'units': 128},
                {'name': 'BatchNorm1D', 'features': 128},
                {'name': 'Dropout', 'rate': 0.3},
                {'name': 'Dense', 'units': 'num_classes'},
                {'name': 'Softmax', 'axis': 1}
            ]
        }


def create_model(model_type: str, input_shape: Tuple[int, int], num_classes: int) -> nn.Module:
    """
    Create a model based on the specified type.
    
    Args:
        model_type: Type of model to create ('cnn', 'lstm', or 'cnn_lstm')
        input_shape: Shape of input features
        num_classes: Number of classes to predict
    
    Returns:
        PyTorch model
    """
    if model_type == 'cnn':
        return CNNModel(input_shape, num_classes)
    elif model_type == 'lstm':
        return LSTMModel(input_shape, num_classes)
    elif model_type == 'cnn_lstm':
        return CNNLSTMModel(input_shape, num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
