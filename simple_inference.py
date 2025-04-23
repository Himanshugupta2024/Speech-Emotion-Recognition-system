import os
import numpy as np
import torch
import librosa
from sklearn.preprocessing import LabelEncoder
import argparse

# Import the model definition from simple_train.py
from simple_train import CNNLSTM, extract_mfcc

def predict_emotion(audio_path, model_path):
    """
    Predict emotion from an audio file.
    
    Args:
        audio_path: Path to the audio file
        model_path: Path to the trained model
    
    Returns:
        Predicted emotion
    """
    # Extract features
    features = extract_mfcc(audio_path)
    if features is None:
        return "Error extracting features"
    
    # Add batch dimension
    features = np.expand_dims(features, axis=0)
    
    # Load model
    input_shape = features.shape[1:]
    
    # Define emotion classes (same as in training)
    emotions = ['angry', 'happy', 'neutral', 'sad']
    num_classes = len(emotions)
    
    # Create model
    model = CNNLSTM(input_shape, num_classes)
    
    # Load weights
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Convert to PyTorch tensor
    features_tensor = torch.tensor(features, dtype=torch.float32)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(features_tensor)
        probabilities = outputs[0].numpy()
        predicted_idx = np.argmax(probabilities)
        predicted_emotion = emotions[predicted_idx]
    
    # Return prediction and probabilities
    result = {
        'emotion': predicted_emotion,
        'probabilities': {emotion: float(prob) for emotion, prob in zip(emotions, probabilities)}
    }
    
    return result

def main():
    parser = argparse.ArgumentParser(description='Speech Emotion Recognition Inference')
    parser.add_argument('--audio', type=str, required=True,
                        help='Path to audio file')
    parser.add_argument('--model', type=str, default='models/emotion_model_simple.pt',
                        help='Path to trained model')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.audio):
        print(f"Error: Audio file {args.audio} not found")
        return
    
    if not os.path.exists(args.model):
        print(f"Error: Model file {args.model} not found")
        return
    
    # Predict emotion
    print(f"Predicting emotion for {args.audio}...")
    result = predict_emotion(args.audio, args.model)
    
    # Print results
    print(f"\nPredicted emotion: {result['emotion']}")
    print("\nProbabilities:")
    for emotion, prob in result['probabilities'].items():
        print(f"  {emotion}: {prob:.4f}")

if __name__ == "__main__":
    main()
