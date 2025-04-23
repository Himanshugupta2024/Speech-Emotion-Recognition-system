import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (Input, Dense, LSTM, Conv2D, MaxPooling2D, 
                                    Flatten, Dropout, BatchNormalization, 
                                    TimeDistributed, Bidirectional, 
                                    Conv1D, MaxPooling1D, GlobalAveragePooling1D,
                                    GlobalAveragePooling2D, Activation)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from typing import Tuple, List, Dict, Any, Optional, Union


def simple_lstm_model(input_shape: Tuple[int, int], 
                      num_classes: int,
                      lstm_units: int = 128, 
                      dropout_rate: float = 0.3) -> Model:
    """
    Simple LSTM model for speech emotion recognition.
    
    Args:
        input_shape: Shape of input features (time_steps, num_features)
        num_classes: Number of classes to predict
        lstm_units: Number of LSTM units
        dropout_rate: Dropout rate
    
    Returns:
        Keras Model
    """
    model = Sequential([
        Input(shape=input_shape),
        LSTM(lstm_units, return_sequences=True),
        Dropout(dropout_rate),
        LSTM(lstm_units),
        Dropout(dropout_rate),
        Dense(64, activation='relu'),
        Dropout(dropout_rate),
        Dense(num_classes, activation='softmax')
    ])
    
    return model


def bidirectional_lstm_model(input_shape: Tuple[int, int], 
                            num_classes: int,
                            lstm_units: int = 128, 
                            dropout_rate: float = 0.3) -> Model:
    """
    Bidirectional LSTM model for speech emotion recognition.
    
    Args:
        input_shape: Shape of input features (time_steps, num_features)
        num_classes: Number of classes to predict
        lstm_units: Number of LSTM units
        dropout_rate: Dropout rate
    
    Returns:
        Keras Model
    """
    model = Sequential([
        Input(shape=input_shape),
        Bidirectional(LSTM(lstm_units, return_sequences=True)),
        Dropout(dropout_rate),
        Bidirectional(LSTM(lstm_units)),
        Dropout(dropout_rate),
        Dense(64, activation='relu'),
        Dropout(dropout_rate),
        Dense(num_classes, activation='softmax')
    ])
    
    return model


def cnn_1d_model(input_shape: Tuple[int, int], 
                num_classes: int,
                filters: List[int] = [64, 128, 256],
                kernel_sizes: List[int] = [5, 5, 5],
                dropout_rate: float = 0.3) -> Model:
    """
    1D CNN model for speech emotion recognition or speaker identification.
    
    Args:
        input_shape: Shape of input features (time_steps, num_features)
        num_classes: Number of classes to predict
        filters: List of filter sizes for each convolutional layer
        kernel_sizes: List of kernel sizes for each convolutional layer
        dropout_rate: Dropout rate
    
    Returns:
        Keras Model
    """
    model = Sequential([
        Input(shape=input_shape)
    ])
    
    # Add convolutional layers
    for i, (filter_size, kernel_size) in enumerate(zip(filters, kernel_sizes)):
        model.add(Conv1D(filters=filter_size, kernel_size=kernel_size, 
                         padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(dropout_rate))
    
    # Add global pooling and dense layers
    model.add(GlobalAveragePooling1D())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(num_classes, activation='softmax'))
    
    return model


def cnn_2d_model(input_shape: Tuple[int, int, int], 
                num_classes: int,
                filters: List[int] = [32, 64, 128],
                kernel_sizes: List[Tuple[int, int]] = [(3, 3), (3, 3), (3, 3)],
                dropout_rate: float = 0.3) -> Model:
    """
    2D CNN model for speech emotion recognition or speaker identification.
    Typically used with spectrograms.
    
    Args:
        input_shape: Shape of input features (height, width, channels)
        num_classes: Number of classes to predict
        filters: List of filter sizes for each convolutional layer
        kernel_sizes: List of kernel sizes for each convolutional layer
        dropout_rate: Dropout rate
    
    Returns:
        Keras Model
    """
    model = Sequential([
        Input(shape=input_shape)
    ])
    
    # Add convolutional layers
    for i, (filter_size, kernel_size) in enumerate(zip(filters, kernel_sizes)):
        model.add(Conv2D(filters=filter_size, kernel_size=kernel_size, 
                         padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(dropout_rate))
    
    # Add global pooling and dense layers
    model.add(GlobalAveragePooling2D())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(num_classes, activation='softmax'))
    
    return model


def cnn_lstm_model(input_shape: Tuple[int, int], 
                  num_classes: int,
                  filters: List[int] = [64, 128],
                  kernel_sizes: List[int] = [5, 5],
                  lstm_units: int = 128,
                  dropout_rate: float = 0.3) -> Model:
    """
    CNN-LSTM hybrid model for speech emotion recognition.
    
    Args:
        input_shape: Shape of input features (time_steps, num_features)
        num_classes: Number of classes to predict
        filters: List of filter sizes for each convolutional layer
        kernel_sizes: List of kernel sizes for each convolutional layer
        lstm_units: Number of LSTM units
        dropout_rate: Dropout rate
    
    Returns:
        Keras Model
    """
    model = Sequential([
        Input(shape=input_shape)
    ])
    
    # Add convolutional layers
    for i, (filter_size, kernel_size) in enumerate(zip(filters, kernel_sizes)):
        model.add(Conv1D(filters=filter_size, kernel_size=kernel_size, 
                         padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(dropout_rate))
    
    # Add LSTM layers
    model.add(Bidirectional(LSTM(lstm_units, return_sequences=True)))
    model.add(Dropout(dropout_rate))
    model.add(Bidirectional(LSTM(lstm_units)))
    model.add(Dropout(dropout_rate))
    
    # Add dense layers
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(num_classes, activation='softmax'))
    
    return model


def create_emotion_recognition_model(model_type: str,
                                    input_shape: Union[Tuple[int, int], Tuple[int, int, int]],
                                    num_emotions: int,
                                    **kwargs) -> Model:
    """
    Create a model for speech emotion recognition.
    
    Args:
        model_type: Type of model ('lstm', 'bilstm', 'cnn1d', 'cnn2d', 'cnn_lstm')
        input_shape: Shape of input features
        num_emotions: Number of emotions to predict
        **kwargs: Additional arguments for specific models
    
    Returns:
        Keras Model
    """
    if model_type == 'lstm':
        model = simple_lstm_model(input_shape, num_emotions, **kwargs)
    elif model_type == 'bilstm':
        model = bidirectional_lstm_model(input_shape, num_emotions, **kwargs)
    elif model_type == 'cnn1d':
        model = cnn_1d_model(input_shape, num_emotions, **kwargs)
    elif model_type == 'cnn2d':
        model = cnn_2d_model(input_shape, num_emotions, **kwargs)
    elif model_type == 'cnn_lstm':
        model = cnn_lstm_model(input_shape, num_emotions, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.compile(
        optimizer=Adam(learning_rate=kwargs.get('learning_rate', 0.001)),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def create_speaker_identification_model(model_type: str,
                                       input_shape: Union[Tuple[int, int], Tuple[int, int, int]],
                                       num_speakers: int,
                                       **kwargs) -> Model:
    """
    Create a model for speaker identification.
    
    Args:
        model_type: Type of model ('lstm', 'bilstm', 'cnn1d', 'cnn2d', 'cnn_lstm')
        input_shape: Shape of input features
        num_speakers: Number of speakers to identify
        **kwargs: Additional arguments for specific models
    
    Returns:
        Keras Model
    """
    # Similar architecture to emotion recognition but possibly with different hyperparameters
    if model_type == 'lstm':
        model = simple_lstm_model(input_shape, num_speakers, **kwargs)
    elif model_type == 'bilstm':
        model = bidirectional_lstm_model(input_shape, num_speakers, **kwargs)
    elif model_type == 'cnn1d':
        model = cnn_1d_model(input_shape, num_speakers, **kwargs)
    elif model_type == 'cnn2d':
        model = cnn_2d_model(input_shape, num_speakers, **kwargs)
    elif model_type == 'cnn_lstm':
        model = cnn_lstm_model(input_shape, num_speakers, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.compile(
        optimizer=Adam(learning_rate=kwargs.get('learning_rate', 0.001)),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def get_callbacks(model_path: str, patience: int = 10) -> List[tf.keras.callbacks.Callback]:
    """
    Get common callbacks for model training.
    
    Args:
        model_path: Path to save the best model
        patience: Patience for early stopping
    
    Returns:
        List of Keras callbacks
    """
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=model_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    return callbacks


def train_model(model: Model,
               X_train: np.ndarray,
               y_train: np.ndarray,
               X_val: np.ndarray,
               y_val: np.ndarray,
               model_path: str,
               batch_size: int = 32,
               epochs: int = 100,
               patience: int = 10) -> tf.keras.callbacks.History:
    """
    Train a model.
    
    Args:
        model: Keras model
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        model_path: Path to save the best model
        batch_size: Batch size
        epochs: Maximum number of epochs
        patience: Patience for early stopping
    
    Returns:
        Training history
    """
    # Get callbacks
    callbacks = get_callbacks(model_path, patience)
    
    # Train model
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    return history


def evaluate_model(model: Model,
                  X_test: np.ndarray,
                  y_test: np.ndarray,
                  batch_size: int = 32) -> Dict[str, float]:
    """
    Evaluate a model.
    
    Args:
        model: Keras model
        X_test: Test features
        y_test: Test labels
        batch_size: Batch size
    
    Returns:
        Dictionary containing evaluation metrics
    """
    # Evaluate model
    metrics = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=1)
    
    # Create dictionary of metrics
    metrics_dict = {metric: value for metric, value in zip(model.metrics_names, metrics)}
    
    return metrics_dict 