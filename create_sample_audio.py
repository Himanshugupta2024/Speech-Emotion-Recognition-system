import os
import numpy as np
import soundfile as sf

def create_sample_audio():
    """
    Create a sample audio file for testing.
    """
    # Create directory for sample data
    os.makedirs('sample_data', exist_ok=True)
    
    # Create a simple sine wave
    sample_rate = 16000
    duration = 3  # seconds
    frequency = 440  # Hz (A4 note)
    
    # Generate time array
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # Generate sine wave
    audio = 0.5 * np.sin(2 * np.pi * frequency * t)
    
    # Add some variation to simulate speech
    audio += 0.3 * np.sin(2 * np.pi * 220 * t)  # Add lower frequency
    audio += 0.1 * np.sin(2 * np.pi * 880 * t)  # Add higher frequency
    
    # Add some noise
    audio += 0.05 * np.random.normal(0, 1, len(audio))
    
    # Normalize
    audio = audio / np.max(np.abs(audio))
    
    # Create a few sample files with different "emotions"
    emotions = ['happy', 'sad', 'angry', 'neutral']
    speakers = ['speaker1', 'speaker2']
    
    for emotion in emotions:
        for speaker in speakers:
            # Modify the audio slightly for each emotion and speaker
            if emotion == 'happy':
                modified_audio = audio * 1.2
                modified_audio = np.clip(modified_audio, -1, 1)
            elif emotion == 'sad':
                modified_audio = audio * 0.8
            elif emotion == 'angry':
                modified_audio = audio * 1.5
                modified_audio = np.clip(modified_audio, -1, 1)
            else:  # neutral
                modified_audio = audio
            
            # Add speaker variation
            if speaker == 'speaker1':
                # Shift frequency slightly
                t_modified = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
                speaker_audio = 0.5 * np.sin(2 * np.pi * 450 * t_modified)
                speaker_audio += 0.3 * np.sin(2 * np.pi * 225 * t_modified)
                speaker_audio += 0.1 * np.sin(2 * np.pi * 900 * t_modified)
                speaker_audio += 0.05 * np.random.normal(0, 1, len(speaker_audio))
                speaker_audio = speaker_audio / np.max(np.abs(speaker_audio))
                
                modified_audio = 0.7 * modified_audio + 0.3 * speaker_audio
            
            # Create filename
            filename = f"{speaker}_{emotion}.wav"
            filepath = os.path.join('sample_data', filename)
            
            # Save audio file
            sf.write(filepath, modified_audio, sample_rate)
            print(f"Created {filepath}")
    
    print("\nSample audio files created successfully!")
    print("Files are located in the 'sample_data' directory.")

if __name__ == "__main__":
    create_sample_audio()
