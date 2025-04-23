import os
import pandas as pd
import numpy as np
import librosa
import soundfile as sf
from typing import Tuple, List, Dict, Any, Optional

def load_audio_file(file_path: str, sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Load an audio file using librosa.
    
    Args:
        file_path: Path to the audio file
        sr: Target sample rate
    
    Returns:
        Tuple containing the audio data and the sample rate
    """
    try:
        audio, sample_rate = librosa.load(file_path, sr=sr)
        return audio, sample_rate
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return np.array([]), sr

def parse_sample_filename(filename: str) -> Dict[str, Any]:
    """
    Parse filename to extract metadata for our sample data.
    Format: speaker_emotion.wav
    
    Args:
        filename: Filename to parse
    
    Returns:
        Dictionary containing metadata
    """
    base_name = os.path.basename(filename)
    name_parts = os.path.splitext(base_name)[0].split('_')
    
    metadata = {}
    
    if len(name_parts) == 2:
        speaker = name_parts[0]
        emotion = name_parts[1]
        
        metadata = {
            'emotion': emotion,
            'speaker_id': speaker
        }
    else:
        # Generic fallback
        metadata = {
            'filename': base_name,
            'unknown_format': True
        }
    
    return metadata

def load_dataset(data_dir: str, sr: int = 16000) -> Tuple[List[np.ndarray], List[str], List[str]]:
    """
    Load a dataset of audio files and extract emotions and speaker IDs.
    
    Args:
        data_dir: Directory containing audio files
        sr: Target sample rate
    
    Returns:
        Tuple containing list of audio data, list of emotions, and list of speaker IDs
    """
    audio_files = []
    emotions = []
    speakers = []
    
    # Find all WAV files in the directory
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                
                # Parse metadata from filename
                metadata = parse_sample_filename(file)
                
                if 'unknown_format' not in metadata:
                    audio_files.append(file_path)
                    emotions.append(metadata['emotion'])
                    speakers.append(metadata['speaker_id'])
    
    return audio_files, emotions, speakers

def split_data(features: np.ndarray, 
              labels: np.ndarray, 
              test_size: float = 0.2, 
              validation_size: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into training, validation, and test sets.
    
    Args:
        features: Feature array
        labels: Label array
        test_size: Proportion of data to use for testing
        validation_size: Proportion of data to use for validation
    
    Returns:
        Tuple containing training, validation, and test sets
    """
    # Calculate indices for splits
    n_samples = len(features)
    indices = np.random.permutation(n_samples)
    
    test_count = int(test_size * n_samples)
    val_count = int(validation_size * n_samples)
    
    test_idx = indices[:test_count]
    val_idx = indices[test_count:test_count + val_count]
    train_idx = indices[test_count + val_count:]
    
    # Split the data
    X_train = features[train_idx]
    y_train = labels[train_idx]
    
    X_val = features[val_idx]
    y_val = labels[val_idx]
    
    X_test = features[test_idx]
    y_test = labels[test_idx]
    
    return X_train, X_val, X_test, y_train, y_val, y_test
