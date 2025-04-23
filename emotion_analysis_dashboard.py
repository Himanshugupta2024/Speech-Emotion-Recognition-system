import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import librosa
import argparse
from typing import Dict, List, Tuple, Any
import json
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec

# Import our modules
from enhanced_feature_extraction import (
    extract_all_features, 
    plot_waveform, 
    plot_spectrogram, 
    plot_mfcc, 
    plot_pitch_contour, 
    plot_energy_and_zcr, 
    plot_emotion_probabilities
)
from enhanced_models import create_model

# Define the emotions
EMOTIONS = ['neutral', 'happy', 'sad', 'angry', 'fearful', 'disgusted', 'surprised', 'calm']

def load_audio(file_path: str, sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Load an audio file.
    
    Args:
        file_path: Path to the audio file
        sr: Sample rate
    
    Returns:
        Tuple containing the audio data and the sample rate
    """
    try:
        audio, sample_rate = librosa.load(file_path, sr=sr)
        return audio, sample_rate
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return np.array([]), sr

def predict_emotion(audio: np.ndarray, 
                   sr: int, 
                   model_path: str, 
                   model_type: str = 'cnn_lstm') -> Dict[str, float]:
    """
    Predict emotion from audio data.
    
    Args:
        audio: Audio data
        sr: Sample rate
        model_path: Path to the model
        model_type: Type of model
    
    Returns:
        Dictionary containing emotion probabilities
    """
    # Extract features
    features = extract_all_features(audio, sr)
    mfcc_features = features['mfcc']
    
    # Add batch dimension
    mfcc_features = np.expand_dims(mfcc_features, axis=0)
    
    # Create model
    input_shape = mfcc_features.shape[1:]
    num_classes = len(EMOTIONS)
    model = create_model(model_type, input_shape, num_classes)
    
    # Load model weights
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Convert to PyTorch tensor
    mfcc_tensor = torch.tensor(mfcc_features, dtype=torch.float32)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(mfcc_tensor)
        probabilities = outputs[0].numpy()
    
    # Create dictionary of emotion probabilities
    emotion_probs = {emotion: float(prob) for emotion, prob in zip(EMOTIONS, probabilities)}
    
    return emotion_probs

def create_feature_extraction_pipeline_diagram() -> plt.Figure:
    """
    Create a diagram of the feature extraction pipeline.
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Define the stages
    stages = ['Raw Audio', 'Preprocessing', 'Feature Extraction', 'Feature Vector', 'Neural Network', 'Emotion Prediction']
    
    # Define the features
    features = ['Waveform', 'Mel-Spectrogram', 'MFCC', 'Pitch Contour', 'Energy & ZCR']
    
    # Define colors
    colors = ['#4B79BF', '#8F4CBF', '#BF4C4C', '#4CBF8F', '#BF8F4C']
    
    # Create Sankey diagram
    from matplotlib.sankey import Sankey
    
    sankey = Sankey(ax=ax, scale=0.1, offset=0.2, head_angle=120, margin=0.4)
    
    # Add flows
    sankey.add(
        flows=[1, -0.2, -0.2, -0.2, -0.2, -0.2],
        labels=[stages[0], features[0], features[1], features[2], features[3], features[4]],
        orientations=[0, -1, -1, -1, -1, -1],
        pathlengths=[0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
        facecolor=colors[0]
    )
    
    # Add more flows
    sankey.add(
        flows=[1, -1],
        labels=[stages[1], stages[2]],
        orientations=[0, 0],
        prior=0,
        connect=(1, 0),
        pathlengths=[0.25, 0.25],
        facecolor=colors[1]
    )
    
    sankey.add(
        flows=[1, -1],
        labels=[stages[3], stages[4]],
        orientations=[0, 0],
        prior=1,
        connect=(1, 0),
        pathlengths=[0.25, 0.25],
        facecolor=colors[2]
    )
    
    sankey.add(
        flows=[1, -1],
        labels=['', stages[5]],
        orientations=[0, 0],
        prior=2,
        connect=(1, 0),
        pathlengths=[0.25, 0.25],
        facecolor=colors[3]
    )
    
    sankey.finish()
    
    ax.set_title('Feature Extraction Pipeline', fontsize=16)
    ax.axis('off')
    
    plt.tight_layout()
    return fig

def create_model_architecture_diagram(model_type: str) -> plt.Figure:
    """
    Create a diagram of the model architecture.
    
    Args:
        model_type: Type of model ('cnn', 'lstm', or 'cnn_lstm')
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create a dummy model to get the architecture
    input_shape = (188, 39)  # Example shape
    num_classes = len(EMOTIONS)
    model = create_model(model_type, input_shape, num_classes)
    
    # Get the architecture diagram
    architecture = model.get_architecture_diagram()
    
    # Define colors for different layer types
    colors = {
        'Conv1D': '#4B79BF',
        'BatchNorm1D': '#8F4CBF',
        'MaxPool1D': '#BF4C4C',
        'LSTM': '#4CBF8F',
        'Dense': '#BF8F4C',
        'Dropout': '#BFBF4C',
        'Flatten': '#4CBFBF',
        'Softmax': '#BF4C8F'
    }
    
    # Create a horizontal bar for each layer
    layers = architecture['layers']
    y_positions = np.arange(len(layers))
    
    for i, layer in enumerate(layers):
        layer_name = layer['name']
        color = colors.get(layer_name, '#CCCCCC')
        
        # Create layer description
        if layer_name == 'Conv1D':
            desc = f"{layer_name} ({layer['filters']} filters, {layer['kernel_size']}x1)"
        elif layer_name == 'BatchNorm1D':
            desc = f"{layer_name} ({layer['features']} features)"
        elif layer_name == 'MaxPool1D':
            desc = f"{layer_name} ({layer['kernel_size']}x1)"
        elif layer_name == 'LSTM':
            if 'bidirectional' in layer and layer['bidirectional']:
                desc = f"Bidirectional {layer_name} ({layer['hidden_size']} units)"
            else:
                desc = f"{layer_name} ({layer['hidden_size']} units)"
        elif layer_name == 'Dense':
            desc = f"{layer_name} ({layer['units']} units)"
        elif layer_name == 'Dropout':
            desc = f"{layer_name} (rate={layer['rate']})"
        elif layer_name == 'Flatten':
            desc = f"{layer_name} ({layer['output_size']} units)"
        else:
            desc = layer_name
        
        # Draw the bar
        ax.barh(y_positions[i], 1, color=color)
        
        # Add text
        ax.text(0.5, y_positions[i], desc, ha='center', va='center', color='white', fontweight='bold')
    
    # Set labels and title
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlim(0, 1)
    ax.set_title(architecture['name'], fontsize=16)
    
    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    plt.tight_layout()
    return fig

def create_dashboard(audio_path: str, 
                    model_paths: Dict[str, str],
                    output_dir: str = 'output') -> None:
    """
    Create a comprehensive analysis dashboard.
    
    Args:
        audio_path: Path to the audio file
        model_paths: Dictionary containing paths to models for each model type
        output_dir: Directory to save the dashboard
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load audio
    audio, sr = load_audio(audio_path)
    
    if len(audio) == 0:
        print(f"Error: Could not load audio file {audio_path}")
        return
    
    # Extract features
    features = extract_all_features(audio, sr)
    
    # Create visualizations
    waveform_fig = plot_waveform(audio, sr)
    spectrogram_fig = plot_spectrogram(features['melspec'], sr)
    mfcc_fig = plot_mfcc(features['mfcc'], sr)
    pitch_fig = plot_pitch_contour(features['pitch'], sr)
    energy_zcr_fig = plot_energy_and_zcr(features['energy'], features['zcr'], sr)
    
    # Save visualizations
    waveform_fig.savefig(os.path.join(output_dir, 'waveform.png'))
    spectrogram_fig.savefig(os.path.join(output_dir, 'spectrogram.png'))
    mfcc_fig.savefig(os.path.join(output_dir, 'mfcc.png'))
    pitch_fig.savefig(os.path.join(output_dir, 'pitch_contour.png'))
    energy_zcr_fig.savefig(os.path.join(output_dir, 'energy_zcr.png'))
    
    # Create feature extraction pipeline diagram
    pipeline_fig = create_feature_extraction_pipeline_diagram()
    pipeline_fig.savefig(os.path.join(output_dir, 'feature_extraction_pipeline.png'))
    
    # Create model architecture diagrams
    for model_type in model_paths.keys():
        architecture_fig = create_model_architecture_diagram(model_type)
        architecture_fig.savefig(os.path.join(output_dir, f'{model_type}_architecture.png'))
    
    # Predict emotions using each model
    emotion_predictions = {}
    for model_type, model_path in model_paths.items():
        if os.path.exists(model_path):
            try:
                emotion_probs = predict_emotion(audio, sr, model_path, model_type)
                emotion_predictions[model_type] = emotion_probs
                
                # Create and save emotion probability plot
                emotion_fig = plot_emotion_probabilities(emotion_probs, f'Emotion Classification ({model_type.upper()})')
                emotion_fig.savefig(os.path.join(output_dir, f'emotion_probs_{model_type}.png'))
            except Exception as e:
                print(f"Error predicting emotions with {model_type} model: {e}")
        else:
            print(f"Warning: Model file {model_path} not found")
    
    # Create a combined dashboard
    create_combined_dashboard(
        audio_path=audio_path,
        features=features,
        emotion_predictions=emotion_predictions,
        output_path=os.path.join(output_dir, 'dashboard.png')
    )
    
    # Save audio metrics as JSON
    with open(os.path.join(output_dir, 'audio_metrics.json'), 'w') as f:
        json.dump(features['metrics'], f, indent=4)
    
    # Save emotion predictions as JSON
    with open(os.path.join(output_dir, 'emotion_predictions.json'), 'w') as f:
        json.dump(emotion_predictions, f, indent=4)
    
    print(f"Dashboard created in {output_dir}")

def create_combined_dashboard(audio_path: str,
                             features: Dict[str, Any],
                             emotion_predictions: Dict[str, Dict[str, float]],
                             output_path: str) -> None:
    """
    Create a combined dashboard with all visualizations.
    
    Args:
        audio_path: Path to the audio file
        features: Dictionary containing extracted features
        emotion_predictions: Dictionary containing emotion predictions for each model
        output_path: Path to save the dashboard
    """
    # Create figure
    fig = plt.figure(figsize=(20, 24))
    gs = gridspec.GridSpec(6, 2, figure=fig)
    
    # Add title
    fig.suptitle(f"Speech Emotion Analysis: {os.path.basename(audio_path)}", fontsize=24)
    
    # Waveform
    ax1 = fig.add_subplot(gs[0, 0])
    librosa.display.waveshow(features['waveform'], sr=16000, ax=ax1)
    ax1.set_title('Waveform', fontsize=16)
    
    # Spectrogram
    ax2 = fig.add_subplot(gs[0, 1])
    img = librosa.display.specshow(
        features['melspec'], 
        x_axis='time', 
        y_axis='mel', 
        sr=16000, 
        ax=ax2
    )
    fig.colorbar(img, ax=ax2, format='%+2.0f dB')
    ax2.set_title('Mel-Spectrogram', fontsize=16)
    
    # MFCC
    ax3 = fig.add_subplot(gs[1, 0])
    img = librosa.display.specshow(
        features['mfcc'].T, 
        x_axis='time', 
        sr=16000, 
        ax=ax3
    )
    fig.colorbar(img, ax=ax3)
    ax3.set_title('MFCC', fontsize=16)
    
    # Pitch Contour
    ax4 = fig.add_subplot(gs[1, 1])
    times = librosa.times_like(features['pitch'], sr=16000)
    ax4.plot(times, features['pitch'])
    ax4.set_title('Pitch Contour', fontsize=16)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Frequency (Hz)')
    
    # Energy & ZCR
    ax5 = fig.add_subplot(gs[2, 0])
    times = librosa.times_like(features['zcr'], sr=16000)
    ax5.plot(times, features['energy'] / np.max(features['energy']), 'r', label='Energy')
    ax5.plot(times, features['zcr'], 'b', label='ZCR')
    ax5.set_title('Energy & Zero Crossing Rate', fontsize=16)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Energy/ZCR')
    ax5.legend()
    
    # Audio Metrics
    ax6 = fig.add_subplot(gs[2, 1])
    metrics = features['metrics']
    ax6.axis('off')
    ax6.text(0.5, 0.8, 'Audio Metrics', fontsize=16, ha='center', weight='bold')
    ax6.text(0.5, 0.6, f"Signal Length: {metrics['signal_length']:.4f} sec", fontsize=12, ha='center')
    ax6.text(0.5, 0.5, f"Peak Amplitude: {metrics['peak_amplitude']:.4f}", fontsize=12, ha='center')
    ax6.text(0.5, 0.4, f"RMS Energy: {metrics['rms']:.4f}", fontsize=12, ha='center')
    ax6.text(0.5, 0.3, f"Signal-to-Noise Ratio: {metrics['snr']:.4f} dB", fontsize=12, ha='center')
    
    # Emotion Predictions
    row = 3
    col = 0
    
    for model_type, emotion_probs in emotion_predictions.items():
        ax = fig.add_subplot(gs[row, col])
        
        # Sort emotions by probability
        emotions = list(emotion_probs.keys())
        probs = list(emotion_probs.values())
        sorted_indices = np.argsort(probs)[::-1]
        emotions = [emotions[i] for i in sorted_indices]
        probs = [probs[i] for i in sorted_indices]
        
        # Plot bar chart
        ax.bar(emotions, probs, color='skyblue')
        ax.set_title(f'Emotion Classification ({model_type.upper()})', fontsize=16)
        ax.set_xlabel('Emotion')
        ax.set_ylabel('Probability')
        ax.set_ylim(0, 1)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Highlight primary emotion
        primary_emotion = emotions[0]
        primary_prob = probs[0]
        ax.text(0.5, 0.9, f"Primary Emotion: {primary_emotion} ({primary_prob:.4f})", 
                transform=ax.transAxes, ha='center', fontsize=14, 
                bbox=dict(facecolor='white', alpha=0.8))
        
        # Move to next position
        col += 1
        if col > 1:
            col = 0
            row += 1
    
    # Feature Extraction Pipeline
    pipeline_ax = fig.add_subplot(gs[5, :])
    pipeline_ax.axis('off')
    pipeline_ax.set_title('Feature Extraction Pipeline', fontsize=16)
    
    # Create a simplified pipeline diagram
    stages = ['Raw Audio', 'Preprocessing', 'Feature\nExtraction', 'Feature\nVector', 'Neural\nNetwork', 'Emotion\nPrediction']
    stage_positions = np.linspace(0.1, 0.9, len(stages))
    
    for i, (stage, pos) in enumerate(zip(stages, stage_positions)):
        # Draw box
        pipeline_ax.add_patch(plt.Rectangle((pos-0.05, 0.4), 0.1, 0.2, 
                                          facecolor=f'C{i}', alpha=0.7))
        # Add text
        pipeline_ax.text(pos, 0.5, stage, ha='center', va='center', 
                       fontsize=12, fontweight='bold')
        
        # Add arrow
        if i < len(stages) - 1:
            pipeline_ax.arrow(pos+0.05, 0.5, stage_positions[i+1]-stage_positions[i]-0.1, 0,
                            head_width=0.02, head_length=0.02, fc='black', ec='black')
    
    # Add features
    features_list = ['Waveform', 'Mel-Spectrogram', 'MFCC', 'Pitch Contour', 'Energy & ZCR']
    feature_pos = stage_positions[2]  # Position at Feature Extraction
    
    for i, feature in enumerate(features_list):
        y_pos = 0.3 - i * 0.05
        pipeline_ax.text(feature_pos, y_pos, feature, ha='center', va='center', 
                       fontsize=10, fontweight='bold', 
                       bbox=dict(facecolor=f'C{i+3}', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description='Speech Emotion Analysis Dashboard')
    parser.add_argument('--audio', type=str, required=True,
                        help='Path to audio file')
    parser.add_argument('--cnn_model', type=str, default='models/cnn_model.pt',
                        help='Path to CNN model')
    parser.add_argument('--lstm_model', type=str, default='models/lstm_model.pt',
                        help='Path to LSTM model')
    parser.add_argument('--cnn_lstm_model', type=str, default='models/cnn_lstm_model.pt',
                        help='Path to CNN-LSTM model')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Directory to save the dashboard')
    
    args = parser.parse_args()
    
    # Check if audio file exists
    if not os.path.exists(args.audio):
        print(f"Error: Audio file {args.audio} not found")
        return
    
    # Create model paths dictionary
    model_paths = {
        'cnn': args.cnn_model,
        'lstm': args.lstm_model,
        'cnn_lstm': args.cnn_lstm_model
    }
    
    # Create dashboard
    create_dashboard(args.audio, model_paths, args.output_dir)

if __name__ == "__main__":
    main()
