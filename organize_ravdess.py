import os
import shutil
import argparse

def organize_ravdess_by_emotion(ravdess_dir, output_dir):
    """
    Organize RAVDESS dataset by emotion.
    
    Args:
        ravdess_dir: Directory containing RAVDESS dataset
        output_dir: Directory to save organized dataset
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # RAVDESS emotion labels
    emotion_labels = {
        '01': 'neutral',
        '02': 'calm',
        '03': 'happy',
        '04': 'sad',
        '05': 'angry',
        '06': 'fearful',
        '07': 'disgusted',
        '08': 'surprised'
    }
    
    # Create directories for each emotion
    for emotion in emotion_labels.values():
        os.makedirs(os.path.join(output_dir, emotion), exist_ok=True)
    
    # Process each actor directory
    for actor_dir in os.listdir(ravdess_dir):
        if not actor_dir.startswith('Actor_'):
            continue
        
        actor_path = os.path.join(ravdess_dir, actor_dir)
        if not os.path.isdir(actor_path):
            continue
        
        # Process each audio file
        for filename in os.listdir(actor_path):
            if not filename.endswith('.wav'):
                continue
            
            # Parse filename
            # Format: modality-vocal_channel-emotion-intensity-statement-repetition-actor.wav
            parts = filename.split('-')
            if len(parts) != 7:
                print(f"Skipping file with unexpected format: {filename}")
                continue
            
            emotion_code = parts[2]
            if emotion_code not in emotion_labels:
                print(f"Skipping file with unknown emotion code: {filename}")
                continue
            
            emotion = emotion_labels[emotion_code]
            
            # Copy file to emotion directory
            src_path = os.path.join(actor_path, filename)
            dst_path = os.path.join(output_dir, emotion, filename)
            shutil.copy2(src_path, dst_path)
            print(f"Copied {filename} to {emotion} directory")
    
    # Count files in each emotion directory
    for emotion in emotion_labels.values():
        emotion_dir = os.path.join(output_dir, emotion)
        file_count = len([f for f in os.listdir(emotion_dir) if f.endswith('.wav')])
        print(f"{emotion}: {file_count} files")

def main():
    parser = argparse.ArgumentParser(description='Organize RAVDESS dataset by emotion')
    parser.add_argument('--ravdess_dir', type=str, 
                        default='speech_emotion_speaker_id/data/raw/ravdess',
                        help='Directory containing RAVDESS dataset')
    parser.add_argument('--output_dir', type=str, 
                        default='speech_emotion_speaker_id/data/raw/ravdess_by_emotion',
                        help='Directory to save organized dataset')
    
    args = parser.parse_args()
    
    organize_ravdess_by_emotion(args.ravdess_dir, args.output_dir)

if __name__ == '__main__':
    main()
