import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import argparse
from typing import Dict, List, Tuple, Union, Optional, Any
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Local imports
from data_loader_sample import load_dataset, split_data
from feature_extraction_sample import extract_features_for_all_audio, normalize_features
from model_def_torch import (create_emotion_recognition_model, create_speaker_identification_model,
                           train_model, evaluate_model)


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
    audio_files, emotions, speakers = load_dataset(data_dir)

    if len(audio_files) == 0:
        print(f"No audio files found in {data_dir}")
        return {}

    print(f"Found {len(audio_files)} audio files")
    print(f"Emotions: {set(emotions)}")
    print(f"Speakers: {set(speakers)}")

    # Extract features
    print(f"Extracting {feature_type} features...")
    features = extract_features_for_all_audio(audio_files, feature_type=feature_type)

    # Normalize features
    features = normalize_features(features)

    # Encode emotion labels
    emotion_encoder = LabelEncoder()
    emotion_labels = emotion_encoder.fit_transform(emotions)
    num_emotion_classes = len(emotion_encoder.classes_)

    print(f"Number of emotion classes: {num_emotion_classes}")
    print(f"Emotion classes: {emotion_encoder.classes_}")

    # Split data into train, validation, and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, emotion_labels, test_size=test_split, random_state=42, stratify=emotion_labels
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=validation_split/(1-test_split), random_state=42, stratify=y_train
    )

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # Create model
    print(f"Creating {model_type} model...")
    input_shape = X_train.shape[1:]
    model = create_emotion_recognition_model(input_shape, num_emotion_classes, model_type)

    # Train model
    print(f"Training model...")
    model_save_path = os.path.join(output_dir, 'emotion_model.pt')
    history = train_model(
        model, X_train, y_train, X_val, y_val,
        batch_size=batch_size, epochs=epochs,
        model_save_path=model_save_path
    )

    # Evaluate model
    print(f"Evaluating model...")
    # Load best model
    model.load_state_dict(torch.load(model_save_path))
    evaluation_results = evaluate_model(model, X_test, y_test, batch_size=batch_size)

    print(f"Test accuracy: {evaluation_results['accuracy']:.4f}")

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

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'emotion_training_history.png'))

    return {
        'model': model,
        'history': history,
        'evaluation': evaluation_results,
        'emotion_encoder': emotion_encoder,
        'model_save_path': model_save_path
    }


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
    audio_files, emotions, speakers = load_dataset(data_dir)

    if len(audio_files) == 0:
        print(f"No audio files found in {data_dir}")
        return {}

    print(f"Found {len(audio_files)} audio files")
    print(f"Emotions: {set(emotions)}")
    print(f"Speakers: {set(speakers)}")

    # Extract features
    print(f"Extracting {feature_type} features...")
    features = extract_features_for_all_audio(audio_files, feature_type=feature_type)

    # Normalize features
    features = normalize_features(features)

    # Encode speaker labels
    speaker_encoder = LabelEncoder()
    speaker_labels = speaker_encoder.fit_transform(speakers)
    num_speaker_classes = len(speaker_encoder.classes_)

    print(f"Number of speaker classes: {num_speaker_classes}")
    print(f"Speaker classes: {speaker_encoder.classes_}")

    # Split data into train, validation, and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, speaker_labels, test_size=test_split, random_state=42, stratify=speaker_labels
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=validation_split/(1-test_split), random_state=42, stratify=y_train
    )

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # Create model
    print(f"Creating {model_type} model...")
    input_shape = X_train.shape[1:]
    model = create_speaker_identification_model(input_shape, num_speaker_classes, model_type)

    # Train model
    print(f"Training model...")
    model_save_path = os.path.join(output_dir, 'speaker_model.pt')
    history = train_model(
        model, X_train, y_train, X_val, y_val,
        batch_size=batch_size, epochs=epochs,
        model_save_path=model_save_path
    )

    # Evaluate model
    print(f"Evaluating model...")
    # Load best model
    model.load_state_dict(torch.load(model_save_path))
    evaluation_results = evaluate_model(model, X_test, y_test, batch_size=batch_size)

    print(f"Test accuracy: {evaluation_results['accuracy']:.4f}")

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

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'speaker_training_history.png'))

    return {
        'model': model,
        'history': history,
        'evaluation': evaluation_results,
        'speaker_encoder': speaker_encoder,
        'model_save_path': model_save_path
    }


def main():
    print("Starting main function...")
    parser = argparse.ArgumentParser(description='Speech Emotion Recognition and Speaker Identification')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'test'],
                        help='Mode: train or test')
    parser.add_argument('--data_dir', type=str, default='data/raw',
                        help='Directory containing audio files')
    parser.add_argument('--model_type', type=str, default='cnn_lstm',
                        choices=['lstm', 'bilstm', 'cnn1d', 'cnn2d', 'cnn_lstm'],
                        help='Type of model to use')
    parser.add_argument('--feature_type', type=str, default='mfcc',
                        choices=['mfcc', 'melspec'],
                        help='Type of feature to extract')
    parser.add_argument('--output_dir', type=str, default='models',
                        help='Directory to save the model')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Maximum number of epochs for training')
    parser.add_argument('--test_audio', type=str,
                        help='Path to audio file for testing')
    parser.add_argument('--emotion_model_path', type=str,
                        help='Path to emotion recognition model')
    parser.add_argument('--speaker_model_path', type=str,
                        help='Path to speaker identification model')

    args = parser.parse_args()
    print(f"Arguments parsed: mode={args.mode}, data_dir={args.data_dir}")

    if args.mode == 'train':
        # Train emotion recognition model
        emotion_results = train_emotion_recognition_model(
            args.data_dir,
            model_type=args.model_type,
            feature_type=args.feature_type,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            epochs=args.epochs
        )

        # Train speaker identification model
        speaker_results = train_speaker_identification_model(
            args.data_dir,
            model_type=args.model_type,
            feature_type=args.feature_type,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            epochs=args.epochs
        )

        print("Training completed!")
        print(f"Emotion model saved to: {emotion_results.get('model_save_path', 'N/A')}")
        print(f"Speaker model saved to: {speaker_results.get('model_save_path', 'N/A')}")

    elif args.mode == 'test':
        if not args.test_audio:
            print("Error: --test_audio is required for test mode")
            return

        if not args.emotion_model_path or not args.speaker_model_path:
            print("Error: --emotion_model_path and --speaker_model_path are required for test mode")
            return

        # Load and preprocess audio
        from data_loader import load_audio_file
        from feature_extraction import extract_features

        print(f"Loading audio file: {args.test_audio}")
        audio, sr = load_audio_file(args.test_audio)

        if len(audio) == 0:
            print(f"Error loading audio file: {args.test_audio}")
            return

        # Extract features
        features = extract_features(audio, sr, feature_type=args.feature_type)
        features = features.reshape(1, *features.shape)  # Add batch dimension

        # Load emotion model
        print(f"Loading emotion model: {args.emotion_model_path}")
        emotion_model = torch.load(args.emotion_model_path)
        emotion_model.eval()

        # Load speaker model
        print(f"Loading speaker model: {args.speaker_model_path}")
        speaker_model = torch.load(args.speaker_model_path)
        speaker_model.eval()

        # Make predictions
        with torch.no_grad():
            features_tensor = torch.tensor(features, dtype=torch.float32)
            emotion_output = emotion_model(features_tensor)
            speaker_output = speaker_model(features_tensor)

            emotion_pred = torch.argmax(emotion_output, dim=1).item()
            speaker_pred = torch.argmax(speaker_output, dim=1).item()

        # TODO: Load label encoders to map predictions to class names
        print(f"Predicted emotion: {emotion_pred}")
        print(f"Predicted speaker: {speaker_pred}")


if __name__ == '__main__':
    main()
