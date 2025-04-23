import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import argparse
from typing import Dict, List, Tuple, Union, Optional, Any

# Local imports
from data_loader import load_dataset, split_data
from feature_extraction import extract_features_for_all_audio, normalize_features
from model_def import (create_emotion_recognition_model, create_speaker_identification_model,
                      train_model, evaluate_model)
from evaluate import evaluate_emotion_recognition, evaluate_speaker_identification
from infer import SpeechEmotionRecognizer, SpeakerIdentifier


def train_emotion_recognition_model(data_dir: str,
                                   model_type: str = 'cnn_lstm',
                                   feature_type: str = 'mfcc',
                                   output_dir: str = 'models',
                                   batch_size: int = 32,
                                   epochs: int = 100,
                                   validation_split: float = 0.1,
                                   test_split: float = 0.2) -> Dict[str, Any]:
    """
    Train a speech emotion recognition model.
    
    Args:
        data_dir: Directory containing audio files
        model_type: Type of model to use
        feature_type: Type of feature to extract
        output_dir: Directory to save the model
        batch_size: Batch size for training
        epochs: Maximum number of epochs for training
        validation_split: Proportion of data to use for validation
        test_split: Proportion of data to use for testing
    
    Returns:
        Dictionary containing trained model, evaluation results, and other artifacts
    """
    print(f"Training speech emotion recognition model...")
    print(f"Loading data from {data_dir}...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    audio_data, metadata = load_dataset(data_dir)
    
    # Split data
    data_splits = split_data(audio_data, metadata, 
                           test_size=test_split, 
                           val_size=validation_split)
    
    # Extract emotion labels
    train_emotions = [m['emotion'] for m in data_splits['train'][1]]
    val_emotions = [m['emotion'] for m in data_splits['val'][1]]
    test_emotions = [m['emotion'] for m in data_splits['test'][1]]
    
    # Encode labels
    label_encoder = LabelEncoder()
    label_encoder.fit(train_emotions + val_emotions + test_emotions)
    
    y_train = label_encoder.transform(train_emotions)
    y_val = label_encoder.transform(val_emotions)
    y_test = label_encoder.transform(test_emotions)
    
    # Convert to one-hot encoding
    num_emotions = len(label_encoder.classes_)
    y_train_onehot = to_categorical(y_train, num_classes=num_emotions)
    y_val_onehot = to_categorical(y_val, num_classes=num_emotions)
    y_test_onehot = to_categorical(y_test, num_classes=num_emotions)
    
    # Extract features
    print(f"Extracting {feature_type} features...")
    X_train = extract_features_for_all_audio([audio for audio in data_splits['train'][0]], 
                                           feature_type=feature_type)
    X_val = extract_features_for_all_audio([audio for audio in data_splits['val'][0]], 
                                         feature_type=feature_type)
    X_test = extract_features_for_all_audio([audio for audio in data_splits['test'][0]], 
                                          feature_type=feature_type)
    
    # Normalize features (important for neural networks)
    X_train_norm, scaler = normalize_features([X_train])
    X_val_norm, _ = normalize_features([X_val], scaler)
    X_test_norm, _ = normalize_features([X_test], scaler)
    
    X_train_norm = X_train_norm[0]
    X_val_norm = X_val_norm[0]
    X_test_norm = X_test_norm[0]
    
    # Get input shape for model
    if model_type == 'cnn2d':
        # Reshape for 2D CNN
        X_train_norm = np.expand_dims(X_train_norm, axis=-1)
        X_val_norm = np.expand_dims(X_val_norm, axis=-1)
        X_test_norm = np.expand_dims(X_test_norm, axis=-1)
        input_shape = X_train_norm.shape[1:]
    else:
        # Shape for 1D models (LSTM, CNN1D, etc.)
        input_shape = X_train_norm.shape[1:]
    
    # Create model
    print(f"Creating {model_type} model for speech emotion recognition...")
    model = create_emotion_recognition_model(
        model_type=model_type,
        input_shape=input_shape,
        num_emotions=num_emotions
    )
    
    # Save model architecture summary
    with open(os.path.join(output_dir, 'emotion_model_summary.txt'), 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    
    # Train model
    print(f"Training model...")
    model_path = os.path.join(output_dir, 'emotion_model.h5')
    history = train_model(
        model=model,
        X_train=X_train_norm,
        y_train=y_train_onehot,
        X_val=X_val_norm,
        y_val=y_val_onehot,
        model_path=model_path,
        batch_size=batch_size,
        epochs=epochs
    )
    
    # Evaluate model
    print(f"Evaluating model...")
    metrics = evaluate_model(model, X_test_norm, y_test_onehot)
    
    # Get predictions for evaluation
    y_score = model.predict(X_test_norm)
    y_pred = np.argmax(y_score, axis=1)
    
    # Evaluate using our evaluation functions
    emotions = label_encoder.classes_
    evaluation_results = evaluate_emotion_recognition(
        y_true=y_test,
        y_pred=y_pred,
        y_score=y_score,
        emotions=emotions,
        history=history.history,
        output_dir=os.path.join(output_dir, 'emotion_evaluation')
    )
    
    # Save emotion classes and scaler
    np.save(os.path.join(output_dir, 'emotion_classes.npy'), emotions)
    
    # Save results
    results = {
        'model': model,
        'history': history.history,
        'metrics': metrics,
        'evaluation': evaluation_results,
        'emotions': emotions,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'model_path': model_path
    }
    
    print(f"Speech emotion recognition model trained and saved to {model_path}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    
    return results


def train_speaker_identification_model(data_dir: str,
                                      model_type: str = 'cnn_lstm',
                                      feature_type: str = 'mfcc',
                                      output_dir: str = 'models',
                                      batch_size: int = 32,
                                      epochs: int = 100,
                                      validation_split: float = 0.1,
                                      test_split: float = 0.2) -> Dict[str, Any]:
    """
    Train a speaker identification model.
    
    Args:
        data_dir: Directory containing audio files
        model_type: Type of model to use
        feature_type: Type of feature to extract
        output_dir: Directory to save the model
        batch_size: Batch size for training
        epochs: Maximum number of epochs for training
        validation_split: Proportion of data to use for validation
        test_split: Proportion of data to use for testing
    
    Returns:
        Dictionary containing trained model, evaluation results, and other artifacts
    """
    print(f"Training speaker identification model...")
    print(f"Loading data from {data_dir}...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    audio_data, metadata = load_dataset(data_dir)
    
    # Split data
    data_splits = split_data(audio_data, metadata, 
                           test_size=test_split, 
                           val_size=validation_split)
    
    # Extract speaker IDs
    train_speakers = [str(m['speaker_id']) for m in data_splits['train'][1]]
    val_speakers = [str(m['speaker_id']) for m in data_splits['val'][1]]
    test_speakers = [str(m['speaker_id']) for m in data_splits['test'][1]]
    
    # Encode labels
    label_encoder = LabelEncoder()
    label_encoder.fit(train_speakers + val_speakers + test_speakers)
    
    y_train = label_encoder.transform(train_speakers)
    y_val = label_encoder.transform(val_speakers)
    y_test = label_encoder.transform(test_speakers)
    
    # Convert to one-hot encoding
    num_speakers = len(label_encoder.classes_)
    y_train_onehot = to_categorical(y_train, num_classes=num_speakers)
    y_val_onehot = to_categorical(y_val, num_classes=num_speakers)
    y_test_onehot = to_categorical(y_test, num_classes=num_speakers)
    
    # Extract features
    print(f"Extracting {feature_type} features...")
    X_train = extract_features_for_all_audio([audio for audio in data_splits['train'][0]], 
                                           feature_type=feature_type)
    X_val = extract_features_for_all_audio([audio for audio in data_splits['val'][0]], 
                                         feature_type=feature_type)
    X_test = extract_features_for_all_audio([audio for audio in data_splits['test'][0]], 
                                          feature_type=feature_type)
    
    # Normalize features (important for neural networks)
    X_train_norm, scaler = normalize_features([X_train])
    X_val_norm, _ = normalize_features([X_val], scaler)
    X_test_norm, _ = normalize_features([X_test], scaler)
    
    X_train_norm = X_train_norm[0]
    X_val_norm = X_val_norm[0]
    X_test_norm = X_test_norm[0]
    
    # Get input shape for model
    if model_type == 'cnn2d':
        # Reshape for 2D CNN
        X_train_norm = np.expand_dims(X_train_norm, axis=-1)
        X_val_norm = np.expand_dims(X_val_norm, axis=-1)
        X_test_norm = np.expand_dims(X_test_norm, axis=-1)
        input_shape = X_train_norm.shape[1:]
    else:
        # Shape for 1D models (LSTM, CNN1D, etc.)
        input_shape = X_train_norm.shape[1:]
    
    # Create model
    print(f"Creating {model_type} model for speaker identification...")
    model = create_speaker_identification_model(
        model_type=model_type,
        input_shape=input_shape,
        num_speakers=num_speakers
    )
    
    # Save model architecture summary
    with open(os.path.join(output_dir, 'speaker_model_summary.txt'), 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    
    # Train model
    print(f"Training model...")
    model_path = os.path.join(output_dir, 'speaker_model.h5')
    history = train_model(
        model=model,
        X_train=X_train_norm,
        y_train=y_train_onehot,
        X_val=X_val_norm,
        y_val=y_val_onehot,
        model_path=model_path,
        batch_size=batch_size,
        epochs=epochs
    )
    
    # Evaluate model
    print(f"Evaluating model...")
    metrics = evaluate_model(model, X_test_norm, y_test_onehot)
    
    # Get predictions for evaluation
    y_score = model.predict(X_test_norm)
    y_pred = np.argmax(y_score, axis=1)
    
    # Evaluate using our evaluation functions
    speakers = label_encoder.classes_
    evaluation_results = evaluate_speaker_identification(
        y_true=y_test,
        y_pred=y_pred,
        y_score=y_score,
        speakers=speakers,
        history=history.history,
        output_dir=os.path.join(output_dir, 'speaker_evaluation')
    )
    
    # Save speaker classes and scaler
    np.save(os.path.join(output_dir, 'speaker_classes.npy'), speakers)
    
    # Save results
    results = {
        'model': model,
        'history': history.history,
        'metrics': metrics,
        'evaluation': evaluation_results,
        'speakers': speakers,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'model_path': model_path
    }
    
    print(f"Speaker identification model trained and saved to {model_path}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    
    return results


def load_and_test_models(emotion_model_path: str,
                        speaker_model_path: str,
                        emotions: Union[List[str], str],
                        speakers: Union[List[str], str],
                        test_audio_path: str,
                        feature_type: str = 'mfcc') -> Dict[str, Any]:
    """
    Load trained models and test on a single audio file.
    
    Args:
        emotion_model_path: Path to emotion recognition model
        speaker_model_path: Path to speaker identification model
        emotions: List of emotions or path to emotions npy file
        speakers: List of speakers or path to speakers npy file
        test_audio_path: Path to test audio file
        feature_type: Type of feature used in models
    
    Returns:
        Dictionary containing predictions
    """
    # Load emotions and speakers
    if isinstance(emotions, str) and emotions.endswith('.npy'):
        emotions = np.load(emotions, allow_pickle=True)
    
    if isinstance(speakers, str) and speakers.endswith('.npy'):
        speakers = np.load(speakers, allow_pickle=True)
    
    # Load models
    emotion_recognizer = SpeechEmotionRecognizer(
        model_path=emotion_model_path,
        emotions=emotions,
        feature_type=feature_type
    )
    
    speaker_identifier = SpeakerIdentifier(
        model_path=speaker_model_path,
        speakers=speakers,
        feature_type=feature_type
    )
    
    # Predict emotion
    emotion_probs = emotion_recognizer.predict(test_audio_path)
    predicted_emotion = max(emotion_probs.items(), key=lambda x: x[1])[0]
    
    # Predict speaker
    speaker_probs = speaker_identifier.predict(test_audio_path)
    predicted_speaker = max(speaker_probs.items(), key=lambda x: x[1])[0]
    
    # Print predictions
    print("\nEmotion Prediction:")
    for emotion, prob in emotion_probs.items():
        print(f"{emotion}: {prob:.4f}")
    print(f"Predicted Emotion: {predicted_emotion}")
    
    print("\nSpeaker Prediction:")
    for speaker, prob in speaker_probs.items():
        print(f"{speaker}: {prob:.4f}")
    print(f"Predicted Speaker: {predicted_speaker}")
    
    # Visualize predictions
    emotion_recognizer.visualize_prediction(test_audio_path)
    speaker_identifier.visualize_prediction(test_audio_path)
    
    return {
        'emotion_probs': emotion_probs,
        'predicted_emotion': predicted_emotion,
        'speaker_probs': speaker_probs,
        'predicted_speaker': predicted_speaker
    }


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Speech Emotion Recognition and Speaker Identification')
    
    # Add arguments
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train',
                      help='Mode: "train" to train models, "test" to test models')
    parser.add_argument('--data_dir', type=str, default='data/raw',
                      help='Directory containing audio files for training')
    parser.add_argument('--model_type', type=str, choices=['lstm', 'bilstm', 'cnn1d', 'cnn2d', 'cnn_lstm'],
                      default='cnn_lstm', help='Model type')
    parser.add_argument('--feature_type', type=str, choices=['mfcc', 'melspec'], 
                      default='mfcc', help='Feature type')
    parser.add_argument('--output_dir', type=str, default='models',
                      help='Directory to save models')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                      help='Maximum number of epochs for training')
    parser.add_argument('--emotion_model_path', type=str, default='models/emotion_model.h5',
                      help='Path to emotion recognition model for testing')
    parser.add_argument('--speaker_model_path', type=str, default='models/speaker_model.h5',
                      help='Path to speaker identification model for testing')
    parser.add_argument('--emotions_file', type=str, default='models/emotion_classes.npy',
                      help='Path to emotion classes file for testing')
    parser.add_argument('--speakers_file', type=str, default='models/speaker_classes.npy',
                      help='Path to speaker classes file for testing')
    parser.add_argument('--test_audio', type=str, default=None,
                      help='Path to test audio file')
    
    args = parser.parse_args()
    
    # Execute based on mode
    if args.mode == 'train':
        print("Training models...")
        
        # Train emotion recognition model
        emotion_results = train_emotion_recognition_model(
            data_dir=args.data_dir,
            model_type=args.model_type,
            feature_type=args.feature_type,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            epochs=args.epochs
        )
        
        # Train speaker identification model
        speaker_results = train_speaker_identification_model(
            data_dir=args.data_dir,
            model_type=args.model_type,
            feature_type=args.feature_type,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            epochs=args.epochs
        )
        
        print("Training complete!")
    
    elif args.mode == 'test':
        if args.test_audio is None:
            print("Error: Please provide a test audio file using --test_audio")
        else:
            print(f"Testing models on {args.test_audio}...")
            
            # Load and test models
            results = load_and_test_models(
                emotion_model_path=args.emotion_model_path,
                speaker_model_path=args.speaker_model_path,
                emotions=args.emotions_file,
                speakers=args.speakers_file,
                test_audio_path=args.test_audio,
                feature_type=args.feature_type
            )
            
            print("Testing complete!") 