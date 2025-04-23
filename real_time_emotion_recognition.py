import os
import numpy as np
import torch
import librosa
import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt
import argparse
import json
import time
from threading import Thread
from queue import Queue
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
matplotlib.use('TkAgg')

# Import our modules
from enhanced_feature_extraction import extract_mfcc, normalize_features
from enhanced_models import create_model

# Global variables
recording = False
audio_queue = Queue()
emotion_queue = Queue()
stop_threads = False

def record_audio(sample_rate=16000, chunk_size=16000, queue=None):
    """
    Record audio in chunks and put them in a queue.
    
    Args:
        sample_rate: Sample rate
        chunk_size: Chunk size
        queue: Queue to put audio chunks
    """
    global recording, stop_threads
    
    while not stop_threads:
        if recording:
            # Record audio
            audio_chunk = sd.rec(chunk_size, samplerate=sample_rate, channels=1, dtype='float32')
            sd.wait()
            
            # Put audio chunk in queue
            if queue is not None:
                queue.put(audio_chunk.flatten())
        else:
            time.sleep(0.1)

def process_audio(model, model_type, label_mapping, sample_rate=16000, queue=None, emotion_queue=None):
    """
    Process audio chunks from queue and predict emotions.
    
    Args:
        model: PyTorch model
        model_type: Type of model
        label_mapping: Mapping from indices to emotion labels
        sample_rate: Sample rate
        queue: Queue to get audio chunks
        emotion_queue: Queue to put emotion predictions
    """
    global stop_threads
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    while not stop_threads:
        if not queue.empty():
            # Get audio chunk
            audio_chunk = queue.get()
            
            # Extract features
            mfcc_features = extract_mfcc(audio_chunk, sample_rate)
            
            # Pad or truncate to fixed length
            target_length = 188  # Approximately 3 seconds
            
            if mfcc_features.shape[0] < target_length:
                # Pad with zeros
                padding = np.zeros((target_length - mfcc_features.shape[0], mfcc_features.shape[1]))
                mfcc_features = np.vstack((mfcc_features, padding))
            elif mfcc_features.shape[0] > target_length:
                # Truncate
                mfcc_features = mfcc_features[:target_length, :]
            
            # Normalize features
            mfcc_features = normalize_features(np.array([mfcc_features]))[0]
            
            # Convert to PyTorch tensor
            mfcc_tensor = torch.tensor(mfcc_features, dtype=torch.float32).unsqueeze(0).to(device)
            
            # Predict emotion
            with torch.no_grad():
                outputs = model(mfcc_tensor)
                probabilities = outputs[0].cpu().numpy()
            
            # Get emotion label
            emotion_idx = np.argmax(probabilities)
            emotion = label_mapping[str(emotion_idx)]
            
            # Create emotion probabilities dictionary
            emotion_probs = {label_mapping[str(i)]: float(prob) for i, prob in enumerate(probabilities)}
            
            # Put emotion prediction in queue
            if emotion_queue is not None:
                emotion_queue.put(emotion_probs)
        else:
            time.sleep(0.1)

def update_gui(root, emotion_label, emotion_probs_figure, emotion_queue=None):
    """
    Update GUI with emotion predictions.
    
    Args:
        root: Tkinter root
        emotion_label: Tkinter label to display emotion
        emotion_probs_figure: Matplotlib figure to display emotion probabilities
        emotion_queue: Queue to get emotion predictions
    """
    global stop_threads
    
    if not stop_threads and not emotion_queue.empty():
        # Get emotion prediction
        emotion_probs = emotion_queue.get()
        
        # Get primary emotion
        primary_emotion = max(emotion_probs, key=emotion_probs.get)
        primary_prob = emotion_probs[primary_emotion]
        
        # Update emotion label
        emotion_label.config(text=f"Emotion: {primary_emotion.capitalize()} ({primary_prob:.2f})")
        
        # Update emotion probabilities figure
        ax = emotion_probs_figure.axes[0]
        ax.clear()
        
        # Sort emotions by probability
        sorted_emotions = sorted(emotion_probs.items(), key=lambda x: x[1], reverse=True)
        emotions = [e.capitalize() for e, _ in sorted_emotions]
        probs = [p for _, p in sorted_emotions]
        
        # Plot bar chart
        bars = ax.bar(emotions, probs, color='skyblue')
        
        # Highlight primary emotion
        bars[0].set_color('orange')
        
        ax.set_ylim(0, 1)
        ax.set_ylabel('Probability')
        ax.set_title('Emotion Probabilities')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Redraw canvas
        emotion_probs_figure.canvas.draw()
    
    # Schedule next update
    root.after(100, update_gui, root, emotion_label, emotion_probs_figure, emotion_queue)

def create_gui(model, model_type, label_mapping):
    """
    Create GUI for real-time emotion recognition.
    
    Args:
        model: PyTorch model
        model_type: Type of model
        label_mapping: Mapping from indices to emotion labels
    """
    global recording, stop_threads
    
    # Create root window
    root = tk.Tk()
    root.title("Real-time Emotion Recognition")
    root.geometry("800x600")
    
    # Create main frame
    main_frame = ttk.Frame(root, padding=10)
    main_frame.pack(fill=tk.BOTH, expand=True)
    
    # Create title label
    title_label = ttk.Label(main_frame, text="Real-time Emotion Recognition", font=("Arial", 16))
    title_label.pack(pady=10)
    
    # Create model info label
    model_info_label = ttk.Label(main_frame, text=f"Model: {model_type.upper()}")
    model_info_label.pack(pady=5)
    
    # Create emotion label
    emotion_label = ttk.Label(main_frame, text="Emotion: None", font=("Arial", 14))
    emotion_label.pack(pady=10)
    
    # Create emotion probabilities figure
    emotion_probs_figure = plt.Figure(figsize=(8, 4), dpi=100)
    ax = emotion_probs_figure.add_subplot(111)
    ax.set_title('Emotion Probabilities')
    ax.set_ylabel('Probability')
    ax.set_ylim(0, 1)
    
    # Create canvas for emotion probabilities figure
    canvas = FigureCanvasTkAgg(emotion_probs_figure, master=main_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=10)
    
    # Create button frame
    button_frame = ttk.Frame(main_frame)
    button_frame.pack(pady=10)
    
    # Create start button
    def start_recording():
        global recording
        recording = True
        start_button.config(state=tk.DISABLED)
        stop_button.config(state=tk.NORMAL)
    
    start_button = ttk.Button(button_frame, text="Start Recording", command=start_recording)
    start_button.pack(side=tk.LEFT, padx=5)
    
    # Create stop button
    def stop_recording():
        global recording
        recording = False
        start_button.config(state=tk.NORMAL)
        stop_button.config(state=tk.DISABLED)
    
    stop_button = ttk.Button(button_frame, text="Stop Recording", command=stop_recording, state=tk.DISABLED)
    stop_button.pack(side=tk.LEFT, padx=5)
    
    # Create exit button
    def exit_app():
        global stop_threads
        stop_threads = True
        root.quit()
        root.destroy()
    
    exit_button = ttk.Button(button_frame, text="Exit", command=exit_app)
    exit_button.pack(side=tk.LEFT, padx=5)
    
    # Start audio recording thread
    audio_thread = Thread(target=record_audio, args=(16000, 16000, audio_queue))
    audio_thread.daemon = True
    audio_thread.start()
    
    # Start audio processing thread
    processing_thread = Thread(target=process_audio, args=(model, model_type, label_mapping, 16000, audio_queue, emotion_queue))
    processing_thread.daemon = True
    processing_thread.start()
    
    # Start GUI update
    root.after(100, update_gui, root, emotion_label, emotion_probs_figure, emotion_queue)
    
    # Handle window close
    root.protocol("WM_DELETE_WINDOW", exit_app)
    
    # Start main loop
    root.mainloop()

def main():
    parser = argparse.ArgumentParser(description='Real-time Emotion Recognition')
    parser.add_argument('--model_path', type=str, default='models/cnn_lstm_model_ravdess.pt',
                        help='Path to the trained model')
    parser.add_argument('--model_type', type=str, default='cnn_lstm',
                        choices=['cnn', 'lstm', 'cnn_lstm'],
                        help='Type of model')
    parser.add_argument('--label_mapping', type=str, default='models/label_mapping.json',
                        help='Path to the label mapping file')
    
    args = parser.parse_args()
    
    # Check if model file exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file {args.model_path} not found")
        return
    
    # Check if label mapping file exists
    if not os.path.exists(args.label_mapping):
        print(f"Error: Label mapping file {args.label_mapping} not found")
        return
    
    # Load label mapping
    with open(args.label_mapping, 'r') as f:
        label_mapping = json.load(f)
    
    # Create model
    input_shape = (188, 39)  # Fixed input shape
    num_classes = len(label_mapping)
    model = create_model(args.model_type, input_shape, num_classes)
    
    # Load model weights
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    
    print(f"Loaded {args.model_type.upper()} model from {args.model_path}")
    print(f"Emotions: {', '.join(label_mapping.values())}")
    
    # Create GUI
    create_gui(model, args.model_type, label_mapping)

if __name__ == '__main__':
    main()
