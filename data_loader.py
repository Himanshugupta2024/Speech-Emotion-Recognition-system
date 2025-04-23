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

def parse_filename_for_metadata(filename: str) -> Dict[str, Any]:
    """
    Parse filename to extract metadata like emotion and speaker ID.
    Adjust this function based on your dataset's filename convention.
    
    Args:
        filename: Filename to parse
    
    Returns:
        Dictionary containing metadata
    """
    # Example for RAVDESS dataset:
    # Format: modality-vocal_channel-emotion-intensity-statement-repetition-actor.wav
    # e.g., 03-01-04-01-02-01-12.wav
    
    base_name = os.path.basename(filename)
    name_parts = os.path.splitext(base_name)[0].split('-')
    
    metadata = {}
    
    # RAVDESS specific parsing - adjust for your dataset
    if len(name_parts) == 7:
        modality = int(name_parts[0])
        vocal_channel = int(name_parts[1])
        emotion_code = int(name_parts[2])
        intensity = int(name_parts[3])
        statement = int(name_parts[4])
        repetition = int(name_parts[5])
        actor_id = int(name_parts[6])
        
        # RAVDESS emotion codes
        emotion_map = {
            1: 'neutral',
            2: 'calm',
            3: 'happy',
            4: 'sad',
            5: 'angry',
            6: 'fear',
            7: 'disgust',
            8: 'surprise'
        }
        
        metadata = {
            'emotion': emotion_map.get(emotion_code, 'unknown'),
            'emotion_code': emotion_code,
            'speaker_id': actor_id,
            'intensity': intensity,
            'statement': statement,
            'repetition': repetition
        }
    else:
        # Generic fallback - extract what we can from the filename
        metadata = {
            'filename': base_name,
            'unknown_format': True
        }
    
    return metadata

def load_dataset(data_dir: str, 
                 metadata_file: Optional[str] = None, 
                 sr: int = 16000) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
    """
    Load a dataset of audio files and their metadata.
    
    Args:
        data_dir: Directory containing audio files
        metadata_file: Optional path to metadata CSV file
        sr: Target sample rate
    
    Returns:
        Tuple containing list of audio data and list of metadata dictionaries
    """
    audio_data = []
    metadata_list = []
    
    # If metadata file is provided, load it
    metadata_df = None
    if metadata_file and os.path.exists(metadata_file):
        metadata_df = pd.read_csv(metadata_file)
    
    # Get all audio files in directory
    audio_files = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith(('.wav', '.mp3', '.flac')):
                audio_files.append(os.path.join(root, file))
    
    # Load each audio file
    for file_path in audio_files:
        audio, _ = load_audio_file(file_path, sr=sr)
        
        if len(audio) > 0:  # Check if audio was loaded successfully
            audio_data.append(audio)
            
            # If metadata dataframe exists, try to find metadata for this file
            metadata = {}
            if metadata_df is not None:
                file_name = os.path.basename(file_path)
                file_metadata = metadata_df[metadata_df['file_name'] == file_name]
                
                if not file_metadata.empty:
                    metadata = file_metadata.iloc[0].to_dict()
                else:
                    # If not found in metadata file, try to parse from filename
                    metadata = parse_filename_for_metadata(file_path)
            else:
                # No metadata file, parse from filename
                metadata = parse_filename_for_metadata(file_path)
            
            metadata['file_path'] = file_path
            metadata_list.append(metadata)
    
    return audio_data, metadata_list

def split_data(audio_data: List[np.ndarray], 
               metadata: List[Dict[str, Any]], 
               test_size: float = 0.2, 
               val_size: float = 0.1,
               random_state: int = 42) -> Dict[str, Tuple[List[np.ndarray], List[Dict[str, Any]]]]:
    """
    Split data into training, validation, and test sets.
    
    Args:
        audio_data: List of audio data arrays
        metadata: List of metadata dictionaries
        test_size: Proportion of data to use for test set
        val_size: Proportion of data to use for validation set
        random_state: Random seed
    
    Returns:
        Dictionary containing train, val, and test splits
    """
    from sklearn.model_selection import train_test_split
    
    # First split out test set
    train_audio, test_audio, train_meta, test_meta = train_test_split(
        audio_data, metadata, test_size=test_size, random_state=random_state, 
        stratify=[m['emotion'] for m in metadata] if all('emotion' in m for m in metadata) else None
    )
    
    # Then split training into train and validation
    if val_size > 0:
        val_size_adjusted = val_size / (1 - test_size)  # Adjust val_size based on remaining data
        train_audio, val_audio, train_meta, val_meta = train_test_split(
            train_audio, train_meta, test_size=val_size_adjusted, random_state=random_state,
            stratify=[m['emotion'] for m in train_meta] if all('emotion' in m for m in train_meta) else None
        )
        
        return {
            'train': (train_audio, train_meta),
            'val': (val_audio, val_meta),
            'test': (test_audio, test_meta)
        }
    
    return {
        'train': (train_audio, train_meta),
        'test': (test_audio, test_meta)
    } 