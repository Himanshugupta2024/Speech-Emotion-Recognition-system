import os
import argparse
import subprocess
import time

def run_command(command):
    """
    Run a command and print its output.
    
    Args:
        command: Command to run
    """
    print(f"Running: {command}")
    process = subprocess.Popen(
        command, 
        shell=True, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, 
        universal_newlines=True
    )
    
    for line in process.stdout:
        print(line.strip())
    
    process.wait()
    
    if process.returncode != 0:
        print(f"Command failed with return code {process.returncode}")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Run Speech Emotion Recognition Pipeline')
    parser.add_argument('--data_dir', type=str, 
                        default='speech_emotion_speaker_id/data/raw/ravdess',
                        help='Directory containing RAVDESS dataset')
    parser.add_argument('--output_dir', type=str, default='models',
                        help='Directory to save models and results')
    parser.add_argument('--organize_data', action='store_true',
                        help='Organize RAVDESS dataset by emotion')
    parser.add_argument('--train_models', action='store_true',
                        help='Train models')
    parser.add_argument('--model_types', type=str, nargs='+', default=['cnn', 'lstm', 'cnn_lstm'],
                        choices=['cnn', 'lstm', 'cnn_lstm'],
                        help='Types of models to train')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs for training')
    parser.add_argument('--test_audio', type=str, default=None,
                        help='Path to audio file to test')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Organize RAVDESS dataset by emotion
    if args.organize_data:
        print("\n=== Organizing RAVDESS Dataset ===\n")
        command = f"python organize_ravdess.py --ravdess_dir {args.data_dir}"
        if not run_command(command):
            return
    
    # Train models
    if args.train_models:
        for model_type in args.model_types:
            print(f"\n=== Training {model_type.upper()} Model ===\n")
            command = f"python train_on_ravdess.py --model_type {model_type} --epochs {args.epochs} --output_dir {args.output_dir}"
            if not run_command(command):
                return
    
    # Test model on audio file
    if args.test_audio:
        for model_type in args.model_types:
            print(f"\n=== Testing {model_type.upper()} Model ===\n")
            model_path = os.path.join(args.output_dir, f"{model_type}_model_ravdess.pt")
            label_mapping = os.path.join(args.output_dir, "label_mapping.json")
            
            if os.path.exists(model_path) and os.path.exists(label_mapping):
                command = f"python test_emotion_recognition.py --audio {args.test_audio} --model_path {model_path} --model_type {model_type} --label_mapping {label_mapping}"
                if not run_command(command):
                    return
            else:
                print(f"Warning: Model file {model_path} or label mapping file {label_mapping} not found")
    
    print("\n=== Pipeline Completed ===\n")

if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
