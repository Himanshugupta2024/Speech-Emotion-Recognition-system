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
    parser.add_argument('--data_dir', type=str, default='synthetic_data',
                        help='Directory containing audio files or where to generate synthetic data')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Directory to save output files')
    parser.add_argument('--models_dir', type=str, default='models',
                        help='Directory to save trained models')
    parser.add_argument('--generate_data', action='store_true',
                        help='Generate synthetic data')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of samples to generate per emotion')
    parser.add_argument('--train_models', action='store_true',
                        help='Train models')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs for training')
    parser.add_argument('--analyze_audio', type=str, default=None,
                        help='Path to audio file to analyze')
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.models_dir, exist_ok=True)
    
    # Generate synthetic data
    if args.generate_data:
        print("\n=== Generating Synthetic Data ===\n")
        command = f"python generate_synthetic_data.py --output_dir {args.data_dir} --num_samples {args.num_samples}"
        if not run_command(command):
            return
    
    # Train models
    if args.train_models:
        print("\n=== Training Models ===\n")
        command = f"python train_emotion_models.py --data_dir {args.data_dir} --output_dir {args.models_dir} --epochs {args.epochs}"
        if not run_command(command):
            return
    
    # Analyze audio
    if args.analyze_audio:
        print("\n=== Analyzing Audio ===\n")
        command = (
            f"python emotion_analysis_dashboard.py --audio {args.analyze_audio} "
            f"--cnn_model {args.models_dir}/cnn_model.pt "
            f"--lstm_model {args.models_dir}/lstm_model.pt "
            f"--cnn_lstm_model {args.models_dir}/cnn_lstm_model.pt "
            f"--output_dir {args.output_dir}"
        )
        if not run_command(command):
            return
    
    print("\n=== Pipeline Completed ===\n")
    
    if args.analyze_audio:
        print(f"Analysis dashboard saved to {args.output_dir}/dashboard.png")
        print(f"Open {args.output_dir}/dashboard.png to view the results")

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
