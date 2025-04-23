import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from enhanced_models import create_model

def create_architecture_diagram(model_type, output_path):
    """
    Create a diagram of the model architecture.
    
    Args:
        model_type: Type of model ('cnn', 'lstm', or 'cnn_lstm')
        output_path: Path to save the diagram
    """
    # Create a dummy model to get the architecture
    input_shape = (188, 39)  # Example shape
    num_classes = 8  # 8 emotions
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
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
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
        rect = patches.Rectangle((0.1, y_positions[i] - 0.4), 0.8, 0.8, linewidth=1, edgecolor='black', facecolor=color)
        ax.add_patch(rect)
        
        # Add text
        ax.text(0.5, y_positions[i], desc, ha='center', va='center', color='white', fontweight='bold')
    
    # Set labels and title
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, len(layers) - 0.5)
    ax.set_title(architecture['name'], fontsize=16)
    
    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved architecture diagram to {output_path}")

def create_feature_extraction_pipeline_diagram(output_path):
    """
    Create a diagram of the feature extraction pipeline.
    
    Args:
        output_path: Path to save the diagram
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Define the stages
    stages = ['Raw Audio', 'Preprocessing', 'Feature Extraction', 'Feature Vector', 'Neural Network', 'Emotion Prediction']
    
    # Define the features
    features = ['Waveform', 'Mel-Spectrogram', 'MFCC', 'Pitch Contour', 'Energy & ZCR']
    
    # Define colors
    colors = ['#4B79BF', '#8F4CBF', '#BF4C4C', '#4CBF8F', '#BF8F4C']
    
    # Create a simplified pipeline diagram
    stage_positions = np.linspace(0.1, 0.9, len(stages))
    
    for i, (stage, pos) in enumerate(zip(stages, stage_positions)):
        # Draw box
        rect = patches.Rectangle((pos-0.05, 0.4), 0.1, 0.2, 
                              facecolor=f'C{i}', alpha=0.7, edgecolor='black')
        ax.add_patch(rect)
        # Add text
        ax.text(pos, 0.5, stage, ha='center', va='center', 
               fontsize=12, fontweight='bold')
        
        # Add arrow
        if i < len(stages) - 1:
            ax.arrow(pos+0.05, 0.5, stage_positions[i+1]-stage_positions[i]-0.1, 0,
                    head_width=0.02, head_length=0.02, fc='black', ec='black')
    
    # Add features
    feature_pos = stage_positions[2]  # Position at Feature Extraction
    
    for i, feature in enumerate(features):
        y_pos = 0.3 - i * 0.05
        ax.text(feature_pos, y_pos, feature, ha='center', va='center', 
               fontsize=10, fontweight='bold', 
               bbox=dict(facecolor=f'C{i+3}', alpha=0.5, boxstyle='round,pad=0.3'))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('Feature Extraction Pipeline', fontsize=16)
    ax.axis('off')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved feature extraction pipeline diagram to {output_path}")

def main():
    # Create output directory
    os.makedirs('static/img', exist_ok=True)
    
    # Generate architecture diagrams
    create_architecture_diagram('cnn', 'static/img/cnn_architecture.png')
    create_architecture_diagram('lstm', 'static/img/lstm_architecture.png')
    create_architecture_diagram('cnn_lstm', 'static/img/hybrid_architecture.png')
    
    # Generate feature extraction pipeline diagram
    create_feature_extraction_pipeline_diagram('static/img/feature_extraction_pipeline.png')
    
    print("Generated all architecture diagrams")

if __name__ == '__main__':
    main()
