import os
import numpy as np
import librosa
import tensorflow as tf
from typing import Dict, List, Tuple, Union, Optional, Any
import matplotlib.pyplot as plt
import pyaudio
import wave
import time
import sys

# Local imports
from feature_extraction import (extract_mfcc, extract_melspectrogram, 
                               extract_features_for_all_audio, pad_or_truncate)


class SpeechEmotionRecognizer:
    """
    Class for speech emotion recognition inference.
    """
    
    def __init__(self, model_path: str, 
                 emotions: List[str],
                 feature_type: str = 'mfcc',
                 sample_rate: int = 16000,
                 max_length: Optional[int] = None,
                 custom_objects: Optional[Dict] = None):
        """
        Initialize speech emotion recognizer.
        
        Args:
            model_path: Path to the trained model
            emotions: List of emotions
            feature_type: Type of feature to extract
            sample_rate: Sample rate for audio processing
            max_length: Maximum length of features (for padding/truncation)
            custom_objects: Custom objects for model loading
        """
        self.model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        self.emotions = emotions
        self.feature_type = feature_type
        self.sample_rate = sample_rate
        self.max_length = max_length
    
    def preprocess_audio(self, audio_path: str) -> np.ndarray:
        """
        Preprocess audio file for prediction.
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            Preprocessed audio features
        """
        # Load audio file
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        # Extract features
        if self.feature_type == 'mfcc':
            features = extract_mfcc(audio, sr)
        elif self.feature_type == 'melspec':
            features = extract_melspectrogram(audio, sr)
        else:
            raise ValueError(f"Unsupported feature type: {self.feature_type}")
        
        # Pad or truncate features
        if self.max_length is not None:
            features = pad_or_truncate(features, self.max_length)
        
        # Add batch dimension
        features = np.expand_dims(features, axis=0)
        
        return features
    
    def predict(self, audio_path: str) -> Dict[str, float]:
        """
        Predict emotion from audio file.
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            Dictionary of emotion probabilities
        """
        # Preprocess audio
        features = self.preprocess_audio(audio_path)
        
        # Make prediction
        predictions = self.model.predict(features)
        
        # Create dictionary of emotion probabilities
        emotion_probs = {emotion: float(prob) for emotion, prob in zip(self.emotions, predictions[0])}
        
        return emotion_probs
    
    def predict_emotion(self, audio_path: str) -> str:
        """
        Predict most likely emotion from audio file.
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            Predicted emotion
        """
        # Get emotion probabilities
        emotion_probs = self.predict(audio_path)
        
        # Get most likely emotion
        predicted_emotion = max(emotion_probs.items(), key=lambda x: x[1])[0]
        
        return predicted_emotion
    
    def visualize_prediction(self, audio_path: str, 
                            figsize: Tuple[int, int] = (10, 6),
                            show_plot: bool = True) -> plt.Figure:
        """
        Visualize emotion prediction.
        
        Args:
            audio_path: Path to audio file
            figsize: Figure size
            show_plot: Whether to show the plot
        
        Returns:
            Matplotlib figure
        """
        # Get emotion probabilities
        emotion_probs = self.predict(audio_path)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot audio waveform
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        times = np.arange(len(audio)) / sr
        ax1.plot(times, audio)
        ax1.set_title('Audio Waveform')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude')
        
        # Plot emotion probabilities
        emotions = list(emotion_probs.keys())
        probs = list(emotion_probs.values())
        ax2.bar(emotions, probs)
        ax2.set_title('Emotion Probabilities')
        ax2.set_xlabel('Emotion')
        ax2.set_ylabel('Probability')
        ax2.set_ylim([0, 1])
        
        plt.tight_layout()
        
        if show_plot:
            plt.show()
        
        return fig


class SpeakerIdentifier:
    """
    Class for speaker identification inference.
    """
    
    def __init__(self, model_path: str, 
                 speakers: List[str],
                 feature_type: str = 'mfcc',
                 sample_rate: int = 16000,
                 max_length: Optional[int] = None,
                 custom_objects: Optional[Dict] = None):
        """
        Initialize speaker identifier.
        
        Args:
            model_path: Path to the trained model
            speakers: List of speaker names or IDs
            feature_type: Type of feature to extract
            sample_rate: Sample rate for audio processing
            max_length: Maximum length of features (for padding/truncation)
            custom_objects: Custom objects for model loading
        """
        self.model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        self.speakers = speakers
        self.feature_type = feature_type
        self.sample_rate = sample_rate
        self.max_length = max_length
    
    def preprocess_audio(self, audio_path: str) -> np.ndarray:
        """
        Preprocess audio file for prediction.
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            Preprocessed audio features
        """
        # Load audio file
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        # Extract features
        if self.feature_type == 'mfcc':
            features = extract_mfcc(audio, sr)
        elif self.feature_type == 'melspec':
            features = extract_melspectrogram(audio, sr)
        else:
            raise ValueError(f"Unsupported feature type: {self.feature_type}")
        
        # Pad or truncate features
        if self.max_length is not None:
            features = pad_or_truncate(features, self.max_length)
        
        # Add batch dimension
        features = np.expand_dims(features, axis=0)
        
        return features
    
    def predict(self, audio_path: str) -> Dict[str, float]:
        """
        Predict speaker from audio file.
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            Dictionary of speaker probabilities
        """
        # Preprocess audio
        features = self.preprocess_audio(audio_path)
        
        # Make prediction
        predictions = self.model.predict(features)
        
        # Create dictionary of speaker probabilities
        speaker_probs = {speaker: float(prob) for speaker, prob in zip(self.speakers, predictions[0])}
        
        return speaker_probs
    
    def predict_speaker(self, audio_path: str) -> str:
        """
        Predict most likely speaker from audio file.
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            Predicted speaker
        """
        # Get speaker probabilities
        speaker_probs = self.predict(audio_path)
        
        # Get most likely speaker
        predicted_speaker = max(speaker_probs.items(), key=lambda x: x[1])[0]
        
        return predicted_speaker
    
    def visualize_prediction(self, audio_path: str, 
                            figsize: Tuple[int, int] = (10, 6),
                            show_plot: bool = True) -> plt.Figure:
        """
        Visualize speaker prediction.
        
        Args:
            audio_path: Path to audio file
            figsize: Figure size
            show_plot: Whether to show the plot
        
        Returns:
            Matplotlib figure
        """
        # Get speaker probabilities
        speaker_probs = self.predict(audio_path)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot audio waveform
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        times = np.arange(len(audio)) / sr
        ax1.plot(times, audio)
        ax1.set_title('Audio Waveform')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude')
        
        # Plot speaker probabilities
        speakers = list(speaker_probs.keys())
        probs = list(speaker_probs.values())
        ax2.bar(speakers, probs)
        ax2.set_title('Speaker Probabilities')
        ax2.set_xlabel('Speaker')
        ax2.set_ylabel('Probability')
        ax2.set_ylim([0, 1])
        
        plt.tight_layout()
        
        if show_plot:
            plt.show()
        
        return fig


def record_audio(output_path: str, 
                record_seconds: int = 5,
                chunk: int = 1024,
                channels: int = 1,
                rate: int = 16000) -> None:
    """
    Record audio from microphone.
    
    Args:
        output_path: Path to save the recorded audio
        record_seconds: Duration of recording in seconds
        chunk: Audio chunk size
        channels: Number of audio channels
        rate: Sample rate
    """
    # Initialize PyAudio
    p = pyaudio.PyAudio()
    
    # Open stream
    stream = p.open(format=pyaudio.paInt16,
                  channels=channels,
                  rate=rate,
                  input=True,
                  frames_per_buffer=chunk)
    
    print("Recording...")
    
    frames = []
    
    # Record audio
    for i in range(0, int(rate / chunk * record_seconds)):
        data = stream.read(chunk)
        frames.append(data)
    
    print("Recording done.")
    
    # Stop and close stream
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    # Save recorded audio to WAV file
    wf = wave.open(output_path, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))
    wf.close()
    
    print(f"Audio saved to {output_path}")


def detect_speech_in_realtime(speech_emotion_recognizer: SpeechEmotionRecognizer,
                             speaker_identifier: Optional[SpeakerIdentifier] = None,
                             record_seconds: int = 5,
                             chunk: int = 1024,
                             channels: int = 1,
                             rate: int = 16000,
                             temp_file: str = 'temp_recording.wav') -> None:
    """
    Detect speech emotion and/or speaker in real-time.
    
    Args:
        speech_emotion_recognizer: SpeechEmotionRecognizer instance
        speaker_identifier: Optional SpeakerIdentifier instance
        record_seconds: Duration of recording in seconds
        chunk: Audio chunk size
        channels: Number of audio channels
        rate: Sample rate
        temp_file: Path to save temporary audio file
    """
    try:
        while True:
            # Record audio
            record_audio(temp_file, record_seconds, chunk, channels, rate)
            
            # Predict emotion
            emotion_probs = speech_emotion_recognizer.predict(temp_file)
            predicted_emotion = max(emotion_probs.items(), key=lambda x: x[1])[0]
            
            # Print emotion prediction
            print("\nEmotion Prediction:")
            for emotion, prob in emotion_probs.items():
                print(f"{emotion}: {prob:.4f}")
            print(f"Predicted Emotion: {predicted_emotion}")
            
            # Predict speaker if speaker identifier is provided
            if speaker_identifier is not None:
                speaker_probs = speaker_identifier.predict(temp_file)
                predicted_speaker = max(speaker_probs.items(), key=lambda x: x[1])[0]
                
                # Print speaker prediction
                print("\nSpeaker Prediction:")
                for speaker, prob in speaker_probs.items():
                    print(f"{speaker}: {prob:.4f}")
                print(f"Predicted Speaker: {predicted_speaker}")
            
            # Ask user if they want to continue
            user_input = input("\nPress Enter to continue or 'q' to quit: ")
            if user_input.lower() == 'q':
                break
    
    finally:
        # Clean up
        if os.path.exists(temp_file):
            os.remove(temp_file)


if __name__ == '__main__':
    # Example usage
    # This code will run when the script is executed directly
    
    # Load emotion recognizer
    emotion_recognizer = SpeechEmotionRecognizer(
        model_path='models/emotion_model.h5',
        emotions=['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'],
        feature_type='mfcc',
        sample_rate=16000
    )
    
    # Load speaker identifier
    speaker_identifier = SpeakerIdentifier(
        model_path='models/speaker_model.h5',
        speakers=['speaker_1', 'speaker_2', 'speaker_3', 'speaker_4'],
        feature_type='mfcc',
        sample_rate=16000
    )
    
    # Process a single audio file
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
        
        # Predict emotion
        emotion_probs = emotion_recognizer.predict(audio_path)
        predicted_emotion = max(emotion_probs.items(), key=lambda x: x[1])[0]
        
        print("\nEmotion Prediction:")
        for emotion, prob in emotion_probs.items():
            print(f"{emotion}: {prob:.4f}")
        print(f"Predicted Emotion: {predicted_emotion}")
        
        # Predict speaker
        speaker_probs = speaker_identifier.predict(audio_path)
        predicted_speaker = max(speaker_probs.items(), key=lambda x: x[1])[0]
        
        print("\nSpeaker Prediction:")
        for speaker, prob in speaker_probs.items():
            print(f"{speaker}: {prob:.4f}")
        print(f"Predicted Speaker: {predicted_speaker}")
        
        # Visualize predictions
        emotion_recognizer.visualize_prediction(audio_path)
        speaker_identifier.visualize_prediction(audio_path)
    
    else:
        # Real-time detection
        print("Starting real-time detection. Press Ctrl+C to stop.")
        try:
            detect_speech_in_realtime(emotion_recognizer, speaker_identifier)
        except KeyboardInterrupt:
            print("\nExiting...") 