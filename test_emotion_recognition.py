import os
import numpy as np
import torch
import librosa
import matplotlib.pyplot as plt
import argparse
import json

# Import our modules
from enhanced_feature_extraction import extract_mfcc, normalize_features, plot_emotion_probabilities
from enhanced_models import create_model

def predict_emotion(audio_path, model_path, model_type, label_mapping_path):
    """
    Predict emotion from audio file.

    Args:
        audio_path: Path to audio file
        model_path: Path to model file
        model_type: Type of model
        label_mapping_path: Path to label mapping file

    Returns:
        Predicted emotion and probabilities
    """
    # Load audio
    audio, sr = librosa.load(audio_path, sr=16000)

    # Load label mapping
    with open(label_mapping_path, 'r') as f:
        label_mapping = json.load(f)

    # Create model
    input_shape = (188, 39)  # Fixed input shape
    num_classes = len(label_mapping)
    model = create_model(model_type, input_shape, num_classes)

    # Load model weights
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Extract features
    mfcc_features = extract_mfcc(audio, sr)

    # Pad or truncate to fixed length
    target_length = 188  # Approximately 3 seconds

    if mfcc_features.shape[0] < target_length:
        # Pad with zeros
        padding = np.zeros((target_length - mfcc_features.shape[0], mfcc_features.shape[1]))
        mfcc_features = np.vstack((mfcc_features, padding))
    elif mfcc_features.shape[0] > target_length:
        # Truncate
        mfcc_features = mfcc_features[:target_length, :]

    # Normalize features
    mfcc_features = normalize_features(np.array([mfcc_features]))[0]

    # Convert to PyTorch tensor
    mfcc_tensor = torch.tensor(mfcc_features, dtype=torch.float32).unsqueeze(0)

    # Predict emotion
    with torch.no_grad():
        outputs = model(mfcc_tensor)
        probabilities = outputs[0].numpy()

    # Get emotion label
    emotion_idx = np.argmax(probabilities)
    emotion = label_mapping[str(emotion_idx)]

    # Create emotion probabilities dictionary
    emotion_probs = {label_mapping[str(i)]: float(prob) for i, prob in enumerate(probabilities)}

    return emotion, emotion_probs

def main():
    parser = argparse.ArgumentParser(description='Test Emotion Recognition')
    parser.add_argument('--audio', type=str, required=True,
                        help='Path to audio file')
    parser.add_argument('--model_path', type=str, default='models/cnn_lstm_model_ravdess.pt',
                        help='Path to the trained model')
    parser.add_argument('--model_type', type=str, default='cnn_lstm',
                        choices=['cnn', 'lstm', 'cnn_lstm'],
                        help='Type of model')
    parser.add_argument('--label_mapping', type=str, default='models/label_mapping.json',
                        help='Path to the label mapping file')

    args = parser.parse_args()

    # Check if audio file exists
    if not os.path.exists(args.audio):
        print(f"Error: Audio file {args.audio} not found")
        return

    # Check if model file exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file {args.model_path} not found")
        return

    # Check if label mapping file exists
    if not os.path.exists(args.label_mapping):
        print(f"Error: Label mapping file {args.label_mapping} not found")
        return

    # Predict emotion
    emotion, emotion_probs = predict_emotion(args.audio, args.model_path, args.model_type, args.label_mapping)

    print(f"Predicted emotion: {emotion}")
    print("Emotion probabilities:")
    for emotion, prob in sorted(emotion_probs.items(), key=lambda x: x[1], reverse=True):
        print(f"  {emotion}: {prob:.4f}")

    # Plot emotion probabilities
    fig = plot_emotion_probabilities(emotion_probs, f"Emotion Classification ({args.model_type.upper()})")
    plt.savefig(f"emotion_probs_{os.path.basename(args.audio)}.png")
    print(f"Saved plot to emotion_probs_{os.path.basename(args.audio)}.png")

if __name__ == '__main__':
    main()
