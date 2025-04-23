import numpy as np
import librosa
import librosa.display
from typing import Dict, List, Tuple, Union, Optional
from sklearn.preprocessing import StandardScaler

def extract_mfcc(audio: np.ndarray, 
                 sr: int = 16000, 
                 n_mfcc: int = 13,
                 n_fft: int = 512,
                 hop_length: int = 256) -> np.ndarray:
    """
    Extract Mel-frequency cepstral coefficients (MFCCs) from audio data.
    
    Args:
        audio: Audio time series
        sr: Sample rate
        n_mfcc: Number of MFCCs to return
        n_fft: Length of the FFT window
        hop_length: Number of samples between successive frames
    
    Returns:
        MFCC features
    """
    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    
    # Add delta features
    delta_mfccs = librosa.feature.delta(mfccs)
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)
    
    # Combine features
    features = np.concatenate([mfccs, delta_mfccs, delta2_mfccs])
    
    # Transpose to get time as first dimension
    features = features.T
    
    return features

def extract_melspectrogram(audio: np.ndarray,
                           sr: int = 16000,
                           n_fft: int = 512,
                           hop_length: int = 256,
                           n_mels: int = 128) -> np.ndarray:
    """
    Extract Mel-spectrogram from audio data.
    
    Args:
        audio: Audio time series
        sr: Sample rate
        n_fft: Length of the FFT window
        hop_length: Number of samples between successive frames
        n_mels: Number of Mel bands to generate
    
    Returns:
        Mel-spectrogram features
    """
    # Extract Mel-spectrogram
    melspec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    
    # Convert to log scale
    log_melspec = librosa.power_to_db(melspec, ref=np.max)
    
    # Transpose to get time as first dimension
    log_melspec = log_melspec.T
    
    return log_melspec

def extract_features(audio: np.ndarray, 
                    sr: int = 16000, 
                    feature_type: str = 'mfcc') -> np.ndarray:
    """
    Extract features from audio data.
    
    Args:
        audio: Audio time series
        sr: Sample rate
        feature_type: Type of feature to extract
    
    Returns:
        Extracted features
    """
    if feature_type == 'mfcc':
        features = extract_mfcc(audio, sr)
    elif feature_type == 'melspec':
        features = extract_melspectrogram(audio, sr)
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")
    
    return features

def extract_features_for_all_audio(audio_files: List[str], 
                                  feature_type: str = 'mfcc',
                                  sr: int = 16000) -> np.ndarray:
    """
    Extract features for all audio files.
    
    Args:
        audio_files: List of paths to audio files
        feature_type: Type of feature to extract
        sr: Sample rate
    
    Returns:
        Array of extracted features
    """
    all_features = []
    
    for file_path in audio_files:
        # Load audio
        audio, sample_rate = librosa.load(file_path, sr=sr)
        
        # Extract features
        features = extract_features(audio, sample_rate, feature_type)
        
        # Pad or truncate to fixed length (3 seconds at 16kHz with hop_length=256)
        target_length = 188  # Approximately 3 seconds
        
        if features.shape[0] < target_length:
            # Pad with zeros
            padding = np.zeros((target_length - features.shape[0], features.shape[1]))
            features = np.vstack((features, padding))
        elif features.shape[0] > target_length:
            # Truncate
            features = features[:target_length, :]
        
        all_features.append(features)
    
    return np.array(all_features)

def normalize_features(features: np.ndarray) -> np.ndarray:
    """
    Normalize features using StandardScaler.
    
    Args:
        features: Features to normalize
    
    Returns:
        Normalized features
    """
    # Reshape to 2D for scaling
    original_shape = features.shape
    features_2d = features.reshape(-1, features.shape[-1])
    
    # Normalize
    scaler = StandardScaler()
    features_2d = scaler.fit_transform(features_2d)
    
    # Reshape back to original shape
    normalized_features = features_2d.reshape(original_shape)
    
    return normalized_features
