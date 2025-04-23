import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union, Optional
from sklearn.preprocessing import StandardScaler

def extract_waveform(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    """
    Extract waveform from audio data.
    
    Args:
        audio: Audio time series
        sr: Sample rate
    
    Returns:
        Waveform
    """
    return audio

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
    
    return log_melspec

def extract_pitch_contour(audio: np.ndarray, 
                         sr: int = 16000,
                         hop_length: int = 256) -> np.ndarray:
    """
    Extract pitch contour from audio data.
    
    Args:
        audio: Audio time series
        sr: Sample rate
        hop_length: Number of samples between successive frames
    
    Returns:
        Pitch contour
    """
    # Extract pitch using PYIN algorithm
    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio, 
        fmin=librosa.note_to_hz('C2'), 
        fmax=librosa.note_to_hz('C7'),
        sr=sr,
        hop_length=hop_length
    )
    
    # Replace NaN values with 0
    f0 = np.nan_to_num(f0)
    
    return f0

def extract_energy_and_zcr(audio: np.ndarray, 
                          sr: int = 16000,
                          frame_length: int = 512,
                          hop_length: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract energy and zero crossing rate from audio data.
    
    Args:
        audio: Audio time series
        sr: Sample rate
        frame_length: Length of each frame
        hop_length: Number of samples between successive frames
    
    Returns:
        Tuple containing energy and zero crossing rate
    """
    # Extract energy
    energy = np.array([
        sum(abs(audio[i:i+frame_length]**2)) 
        for i in range(0, len(audio), hop_length)
    ])
    
    # Extract zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(
        audio, 
        frame_length=frame_length, 
        hop_length=hop_length
    )[0]
    
    return energy, zcr

def extract_audio_metrics(audio: np.ndarray, sr: int = 16000) -> Dict[str, float]:
    """
    Extract audio metrics from audio data.
    
    Args:
        audio: Audio time series
        sr: Sample rate
    
    Returns:
        Dictionary containing audio metrics
    """
    # Signal length in seconds
    signal_length = len(audio) / sr
    
    # Peak amplitude
    peak_amplitude = np.max(np.abs(audio))
    
    # RMS energy
    rms = np.sqrt(np.mean(audio**2))
    
    # Signal-to-noise ratio (estimated)
    signal = np.mean(librosa.feature.rms(y=audio)[0])
    noise = np.std(librosa.feature.rms(y=audio)[0])
    snr = 20 * np.log10(signal / noise) if noise > 0 else 0
    
    return {
        'signal_length': signal_length,
        'peak_amplitude': peak_amplitude,
        'rms': rms,
        'snr': snr
    }

def extract_all_features(audio: np.ndarray, sr: int = 16000) -> Dict[str, np.ndarray]:
    """
    Extract all features from audio data.
    
    Args:
        audio: Audio time series
        sr: Sample rate
    
    Returns:
        Dictionary containing all features
    """
    # Extract waveform
    waveform = extract_waveform(audio, sr)
    
    # Extract MFCC
    mfcc = extract_mfcc(audio, sr)
    
    # Extract Mel-spectrogram
    melspec = extract_melspectrogram(audio, sr)
    
    # Extract pitch contour
    pitch = extract_pitch_contour(audio, sr)
    
    # Extract energy and zero crossing rate
    energy, zcr = extract_energy_and_zcr(audio, sr)
    
    # Extract audio metrics
    metrics = extract_audio_metrics(audio, sr)
    
    return {
        'waveform': waveform,
        'mfcc': mfcc,
        'melspec': melspec,
        'pitch': pitch,
        'energy': energy,
        'zcr': zcr,
        'metrics': metrics
    }

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

def plot_waveform(audio: np.ndarray, sr: int = 16000, title: str = 'Waveform') -> plt.Figure:
    """
    Plot waveform.
    
    Args:
        audio: Audio time series
        sr: Sample rate
        title: Plot title
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    librosa.display.waveshow(audio, sr=sr, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    plt.tight_layout()
    return fig

def plot_spectrogram(melspec: np.ndarray, sr: int = 16000, hop_length: int = 256, title: str = 'Spectrogram') -> plt.Figure:
    """
    Plot spectrogram.
    
    Args:
        melspec: Mel-spectrogram
        sr: Sample rate
        hop_length: Number of samples between successive frames
        title: Plot title
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(
        melspec, 
        x_axis='time', 
        y_axis='mel', 
        sr=sr, 
        hop_length=hop_length,
        ax=ax
    )
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set_title(title)
    plt.tight_layout()
    return fig

def plot_mfcc(mfcc: np.ndarray, sr: int = 16000, hop_length: int = 256, title: str = 'MFCC') -> plt.Figure:
    """
    Plot MFCC.
    
    Args:
        mfcc: MFCC features
        sr: Sample rate
        hop_length: Number of samples between successive frames
        title: Plot title
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(
        mfcc.T, 
        x_axis='time', 
        sr=sr, 
        hop_length=hop_length,
        ax=ax
    )
    fig.colorbar(img, ax=ax)
    ax.set_title(title)
    plt.tight_layout()
    return fig

def plot_pitch_contour(pitch: np.ndarray, sr: int = 16000, hop_length: int = 256, title: str = 'Pitch Contour') -> plt.Figure:
    """
    Plot pitch contour.
    
    Args:
        pitch: Pitch contour
        sr: Sample rate
        hop_length: Number of samples between successive frames
        title: Plot title
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    times = librosa.times_like(pitch, sr=sr, hop_length=hop_length)
    ax.plot(times, pitch)
    ax.set_title(title)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    plt.tight_layout()
    return fig

def plot_energy_and_zcr(energy: np.ndarray, zcr: np.ndarray, sr: int = 16000, hop_length: int = 256, title: str = 'Energy & Zero Crossing Rate') -> plt.Figure:
    """
    Plot energy and zero crossing rate.
    
    Args:
        energy: Energy
        zcr: Zero crossing rate
        sr: Sample rate
        hop_length: Number of samples between successive frames
        title: Plot title
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    times = librosa.times_like(zcr, sr=sr, hop_length=hop_length)
    ax.plot(times, energy / np.max(energy), 'r', label='Energy')
    ax.plot(times, zcr, 'b', label='ZCR')
    ax.set_title(title)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Energy/ZCR')
    ax.legend()
    plt.tight_layout()
    return fig

def plot_emotion_probabilities(probabilities: Dict[str, float], title: str = 'Emotion Classification') -> plt.Figure:
    """
    Plot emotion probabilities.
    
    Args:
        probabilities: Dictionary containing emotion probabilities
        title: Plot title
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    emotions = list(probabilities.keys())
    probs = list(probabilities.values())
    
    # Sort by probability
    sorted_indices = np.argsort(probs)[::-1]
    emotions = [emotions[i] for i in sorted_indices]
    probs = [probs[i] for i in sorted_indices]
    
    ax.bar(emotions, probs, color='skyblue')
    ax.set_title(title)
    ax.set_xlabel('Emotion')
    ax.set_ylabel('Probability')
    ax.set_ylim(0, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig
