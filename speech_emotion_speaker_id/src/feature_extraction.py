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
    # Extract mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    
    # Convert to log scale (dB)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    return log_mel_spec

def extract_chroma(audio: np.ndarray,
                   sr: int = 16000,
                   n_fft: int = 512,
                   hop_length: int = 256,
                   n_chroma: int = 12) -> np.ndarray:
    """
    Extract chromagram from audio data.
    
    Args:
        audio: Audio time series
        sr: Sample rate
        n_fft: Length of the FFT window
        hop_length: Number of samples between successive frames
        n_chroma: Number of chroma bins to produce
    
    Returns:
        Chromagram features
    """
    # Extract chromagram
    chroma = librosa.feature.chroma_stft(
        y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_chroma=n_chroma
    )
    
    return chroma

def extract_spectral_contrast(audio: np.ndarray,
                              sr: int = 16000,
                              n_fft: int = 512,
                              hop_length: int = 256,
                              n_bands: int = 6) -> np.ndarray:
    """
    Extract spectral contrast from audio data.
    
    Args:
        audio: Audio time series
        sr: Sample rate
        n_fft: Length of the FFT window
        hop_length: Number of samples between successive frames
        n_bands: Number of frequency bands
    
    Returns:
        Spectral contrast features
    """
    # Extract spectral contrast
    contrast = librosa.feature.spectral_contrast(
        y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_bands=n_bands
    )
    
    return contrast

def extract_tonnetz(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    """
    Extract tonal centroid features (tonnetz) from audio data.
    
    Args:
        audio: Audio time series
        sr: Sample rate
    
    Returns:
        Tonnetz features
    """
    # Compute harmonic/percussive source separation
    y_harmonic = librosa.effects.harmonic(audio)
    
    # Extract tonnetz
    tonnetz = librosa.feature.tonnetz(y=y_harmonic, sr=sr)
    
    return tonnetz

def extract_all_features(audio: np.ndarray, 
                         sr: int = 16000) -> Dict[str, np.ndarray]:
    """
    Extract all features from audio data.
    
    Args:
        audio: Audio time series
        sr: Sample rate
    
    Returns:
        Dictionary containing all extracted features
    """
    features = {
        'mfcc': extract_mfcc(audio, sr),
        'melspec': extract_melspectrogram(audio, sr),
        'chroma': extract_chroma(audio, sr),
        'contrast': extract_spectral_contrast(audio, sr),
        'tonnetz': extract_tonnetz(audio, sr)
    }
    
    return features

def pad_or_truncate(feature: np.ndarray, max_length: int) -> np.ndarray:
    """
    Pad or truncate feature to specified length.
    
    Args:
        feature: Feature array
        max_length: Target length
    
    Returns:
        Padded or truncated feature array
    """
    if feature.shape[1] > max_length:
        # Truncate
        return feature[:, :max_length]
    elif feature.shape[1] < max_length:
        # Pad
        pad_width = ((0, 0), (0, max_length - feature.shape[1]))
        return np.pad(feature, pad_width, mode='constant')
    else:
        return feature

def normalize_features(features: List[np.ndarray], 
                       scaler: Optional[StandardScaler] = None) -> Tuple[List[np.ndarray], StandardScaler]:
    """
    Normalize features using StandardScaler.
    
    Args:
        features: List of feature arrays
        scaler: Optional pre-fitted scaler
    
    Returns:
        Tuple of normalized features and fitted scaler
    """
    # Reshape features to 2D array for scaling
    features_flat = []
    for feature in features:
        if feature.ndim == 3:  # For batched features
            features_flat.append(feature.reshape(-1, feature.shape[-1]))
        else:  # For individual features
            features_flat.append(feature.reshape(-1, 1))
    
    features_2d = np.vstack(features_flat)
    
    # Fit or transform with scaler
    if scaler is None:
        scaler = StandardScaler()
        features_2d_scaled = scaler.fit_transform(features_2d)
    else:
        features_2d_scaled = scaler.transform(features_2d)
    
    # Reshape back to original shapes
    normalized_features = []
    start_idx = 0
    for feature in features:
        num_elements = feature.size
        feature_flat_scaled = features_2d_scaled[start_idx:start_idx + num_elements]
        normalized_feature = feature_flat_scaled.reshape(feature.shape)
        normalized_features.append(normalized_feature)
        start_idx += num_elements
    
    return normalized_features, scaler

def extract_features_for_all_audio(audio_data: List[np.ndarray], 
                                   sr: int = 16000, 
                                   feature_type: str = 'mfcc',
                                   max_length: Optional[int] = None) -> np.ndarray:
    """
    Extract specified features for all audio files in a dataset.
    
    Args:
        audio_data: List of audio time series
        sr: Sample rate
        feature_type: Type of feature to extract ('mfcc', 'melspec', 'chroma', 'contrast', 'tonnetz', or 'all')
        max_length: Optional maximum length for padding/truncating
    
    Returns:
        Array of extracted features for all audio files
    """
    all_features = []
    
    # Extract features from each audio file
    for audio in audio_data:
        if feature_type == 'mfcc':
            feature = extract_mfcc(audio, sr)
        elif feature_type == 'melspec':
            feature = extract_melspectrogram(audio, sr)
        elif feature_type == 'chroma':
            feature = extract_chroma(audio, sr)
        elif feature_type == 'contrast':
            feature = extract_spectral_contrast(audio, sr)
        elif feature_type == 'tonnetz':
            feature = extract_tonnetz(audio, sr)
        elif feature_type == 'all':
            feature_dict = extract_all_features(audio, sr)
            # Concatenate all features
            feature = np.concatenate([f for f in feature_dict.values()], axis=0)
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")
        
        # Pad or truncate if max_length is specified
        if max_length is not None:
            feature = pad_or_truncate(feature, max_length)
        
        all_features.append(feature)
    
    # If max_length wasn't specified, find the maximum length and pad all features
    if max_length is None:
        max_length = max(f.shape[1] for f in all_features)
        all_features = [pad_or_truncate(f, max_length) for f in all_features]
    
    # Stack features into a single array
    features_array = np.stack(all_features)
    
    return features_array 