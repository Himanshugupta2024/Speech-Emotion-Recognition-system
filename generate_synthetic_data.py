import os
import numpy as np
import soundfile as sf
import argparse
from typing import List, Dict, Any

# Define the emotions
EMOTIONS = ['neutral', 'happy', 'sad', 'angry', 'fearful', 'disgusted', 'surprised', 'calm']

def generate_tone(frequency: float, 
                 duration: float, 
                 sample_rate: int = 16000, 
                 amplitude: float = 0.5) -> np.ndarray:
    """
    Generate a pure tone.
    
    Args:
        frequency: Frequency of the tone in Hz
        duration: Duration of the tone in seconds
        sample_rate: Sample rate in Hz
        amplitude: Amplitude of the tone
    
    Returns:
        Audio data
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    tone = amplitude * np.sin(2 * np.pi * frequency * t)
    return tone

def apply_emotion_characteristics(audio: np.ndarray, 
                                 emotion: str, 
                                 sample_rate: int = 16000) -> np.ndarray:
    """
    Apply emotion characteristics to audio data.
    
    Args:
        audio: Audio data
        emotion: Emotion to apply
        sample_rate: Sample rate in Hz
    
    Returns:
        Modified audio data
    """
    # Create a copy of the audio
    modified_audio = audio.copy()
    
    # Apply emotion-specific modifications
    if emotion == 'neutral':
        # No modification
        pass
    
    elif emotion == 'happy':
        # Increase amplitude and add higher frequencies
        modified_audio *= 1.2
        t = np.linspace(0, len(audio) / sample_rate, len(audio), endpoint=False)
        higher_freq = 0.3 * np.sin(2 * np.pi * 880 * t)
        modified_audio += higher_freq
        
    elif emotion == 'sad':
        # Decrease amplitude and add lower frequencies
        modified_audio *= 0.8
        t = np.linspace(0, len(audio) / sample_rate, len(audio), endpoint=False)
        lower_freq = 0.3 * np.sin(2 * np.pi * 220 * t)
        modified_audio = 0.7 * modified_audio + 0.3 * lower_freq
        
    elif emotion == 'angry':
        # Add distortion and increase amplitude
        modified_audio *= 1.5
        modified_audio = np.clip(modified_audio, -0.9, 0.9)
        noise = 0.1 * np.random.normal(0, 1, len(audio))
        modified_audio += noise
        
    elif emotion == 'fearful':
        # Add tremolo effect
        t = np.linspace(0, len(audio) / sample_rate, len(audio), endpoint=False)
        tremolo = 0.2 * np.sin(2 * np.pi * 5 * t) + 0.8
        modified_audio *= tremolo
        
    elif emotion == 'disgusted':
        # Add noise and distortion
        noise = 0.15 * np.random.normal(0, 1, len(audio))
        modified_audio += noise
        modified_audio = np.clip(modified_audio, -0.8, 0.8)
        
    elif emotion == 'surprised':
        # Add sudden amplitude changes
        t = np.linspace(0, len(audio) / sample_rate, len(audio), endpoint=False)
        surprise = 0.3 * np.sin(2 * np.pi * 2 * t) + 0.7
        modified_audio *= surprise
        modified_audio *= 1.2
        
    elif emotion == 'calm':
        # Smooth the audio
        modified_audio *= 0.9
        window_size = 100
        window = np.ones(window_size) / window_size
        smoothed = np.convolve(modified_audio, window, mode='same')
        modified_audio = 0.7 * modified_audio + 0.3 * smoothed
    
    # Normalize
    modified_audio = modified_audio / np.max(np.abs(modified_audio))
    
    return modified_audio

def generate_synthetic_speech(duration: float = 3.0, 
                             sample_rate: int = 16000) -> np.ndarray:
    """
    Generate synthetic speech-like audio.
    
    Args:
        duration: Duration of the audio in seconds
        sample_rate: Sample rate in Hz
    
    Returns:
        Audio data
    """
    # Generate a base tone
    base_freq = 110  # Hz (A2 note)
    base_tone = generate_tone(base_freq, duration, sample_rate)
    
    # Add harmonics
    harmonics = []
    for i in range(2, 6):
        harmonic = generate_tone(base_freq * i, duration, sample_rate, amplitude=0.5/i)
        harmonics.append(harmonic)
    
    # Add formants (simulate vowels)
    formants = []
    formant_freqs = [500, 1500, 2500]  # Typical formant frequencies
    for freq in formant_freqs:
        formant = generate_tone(freq, duration, sample_rate, amplitude=0.3)
        formants.append(formant)
    
    # Combine all components
    speech = base_tone
    for harmonic in harmonics:
        speech += harmonic
    for formant in formants:
        speech += formant
    
    # Add amplitude modulation (simulate syllables)
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    syllable_rate = 4  # syllables per second
    syllable_mod = 0.5 * np.sin(2 * np.pi * syllable_rate * t) + 0.5
    speech *= syllable_mod
    
    # Add some noise
    noise = 0.05 * np.random.normal(0, 1, len(speech))
    speech += noise
    
    # Normalize
    speech = speech / np.max(np.abs(speech))
    
    return speech

def generate_dataset(output_dir: str, 
                    num_samples_per_emotion: int = 10, 
                    duration: float = 3.0,
                    sample_rate: int = 16000) -> None:
    """
    Generate a synthetic dataset.
    
    Args:
        output_dir: Directory to save the dataset
        num_samples_per_emotion: Number of samples to generate per emotion
        duration: Duration of each sample in seconds
        sample_rate: Sample rate in Hz
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a directory for each emotion
    for emotion in EMOTIONS:
        emotion_dir = os.path.join(output_dir, emotion)
        os.makedirs(emotion_dir, exist_ok=True)
        
        # Generate samples for this emotion
        for i in range(num_samples_per_emotion):
            # Generate base speech
            speech = generate_synthetic_speech(duration, sample_rate)
            
            # Apply emotion characteristics
            speech = apply_emotion_characteristics(speech, emotion, sample_rate)
            
            # Save to file
            file_path = os.path.join(emotion_dir, f"{emotion}_{i+1:03d}.wav")
            sf.write(file_path, speech, sample_rate)
            
            print(f"Generated {file_path}")
    
    print(f"Generated {num_samples_per_emotion * len(EMOTIONS)} samples in {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Generate Synthetic Speech Emotion Dataset')
    parser.add_argument('--output_dir', type=str, default='synthetic_data',
                        help='Directory to save the dataset')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of samples to generate per emotion')
    parser.add_argument('--duration', type=float, default=3.0,
                        help='Duration of each sample in seconds')
    parser.add_argument('--sample_rate', type=int, default=16000,
                        help='Sample rate in Hz')
    
    args = parser.parse_args()
    
    generate_dataset(
        args.output_dir, 
        args.num_samples, 
        args.duration, 
        args.sample_rate
    )

if __name__ == "__main__":
    main()
