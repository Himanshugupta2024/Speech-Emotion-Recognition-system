import matplotlib.pyplot as plt
import numpy as np
import os

# Create output directory if it doesn't exist
os.makedirs(os.path.dirname(__file__), exist_ok=True)

# Feature Extraction Pipeline diagram
def create_feature_extraction_pipeline():
    plt.figure(figsize=(12, 6))
    
    # Create a simple flow diagram using arrows
    plt.plot([0, 1], [0.5, 0.5], 'b-', linewidth=5)
    plt.plot([1, 2], [0.5, 0.5], 'b-', linewidth=5)
    plt.plot([2, 3], [0.5, 0.5], 'b-', linewidth=5)
    plt.plot([3, 4], [0.5, 0.5], 'b-', linewidth=5)
    
    # Add arrow heads
    plt.arrow(0.9, 0.5, 0.1, 0, head_width=0.05, head_length=0.05, fc='b', ec='b')
    plt.arrow(1.9, 0.5, 0.1, 0, head_width=0.05, head_length=0.05, fc='b', ec='b')
    plt.arrow(2.9, 0.5, 0.1, 0, head_width=0.05, head_length=0.05, fc='b', ec='b')
    plt.arrow(3.9, 0.5, 0.1, 0, head_width=0.05, head_length=0.05, fc='b', ec='b')
    
    # Add boxes for stages
    plt.gca().add_patch(plt.Rectangle((0, 0.3), 0.5, 0.4, fill=True, color='royalblue', alpha=0.7))
    plt.gca().add_patch(plt.Rectangle((1, 0.3), 0.5, 0.4, fill=True, color='purple', alpha=0.7))
    plt.gca().add_patch(plt.Rectangle((2, 0.3), 0.5, 0.4, fill=True, color='green', alpha=0.7))
    plt.gca().add_patch(plt.Rectangle((3, 0.3), 0.5, 0.4, fill=True, color='teal', alpha=0.7))
    plt.gca().add_patch(plt.Rectangle((4, 0.3), 0.5, 0.4, fill=True, color='crimson', alpha=0.7))
    
    # Add labels
    plt.text(0.25, 0.5, 'Raw Audio', ha='center', va='center', color='white', fontsize=10)
    plt.text(1.25, 0.5, 'Preprocessing', ha='center', va='center', color='white', fontsize=10)
    plt.text(2.25, 0.5, 'Feature Vector', ha='center', va='center', color='white', fontsize=10)
    plt.text(3.25, 0.5, 'Neural Network', ha='center', va='center', color='white', fontsize=10)
    plt.text(4.25, 0.5, 'Prediction', ha='center', va='center', color='white', fontsize=10)
    
    # Add feature extraction branches
    plt.gca().add_patch(plt.Rectangle((1, 0.8), 0.1, 0.1, fill=True, color='pink', alpha=0.7))
    plt.gca().add_patch(plt.Rectangle((1, 0.95), 0.1, 0.1, fill=True, color='orange', alpha=0.7))
    plt.gca().add_patch(plt.Rectangle((1, 1.1), 0.1, 0.1, fill=True, color='yellow', alpha=0.7))
    plt.gca().add_patch(plt.Rectangle((1, 1.25), 0.1, 0.1, fill=True, color='lightgreen', alpha=0.7))
    plt.gca().add_patch(plt.Rectangle((1, 1.4), 0.1, 0.1, fill=True, color='lightblue', alpha=0.7))
    
    # Connect branches
    plt.plot([1.05, 1.05, 2.25], [0.7, 0.85, 0.85], 'k-', linewidth=1)
    plt.plot([1.05, 1.05, 2.25], [0.7, 1.0, 1.0], 'k-', linewidth=1)
    plt.plot([1.05, 1.05, 2.25], [0.7, 1.15, 1.15], 'k-', linewidth=1)
    plt.plot([1.05, 1.05, 2.25], [0.7, 1.3, 1.3], 'k-', linewidth=1)
    plt.plot([1.05, 1.05, 2.25], [0.7, 1.45, 1.45], 'k-', linewidth=1)
    
    # Add branch labels
    plt.text(1.2, 0.85, 'Waveform', ha='left', va='center', fontsize=8)
    plt.text(1.2, 1.0, 'Mel Spectrogram', ha='left', va='center', fontsize=8)
    plt.text(1.2, 1.15, 'MFCC', ha='left', va='center', fontsize=8)
    plt.text(1.2, 1.3, 'Pitch Contour', ha='left', va='center', fontsize=8)
    plt.text(1.2, 1.45, 'Energy & ZCR', ha='left', va='center', fontsize=8)
    
    # Remove axes
    plt.axis('off')
    plt.xlim(-0.5, 5)
    plt.ylim(0, 1.7)
    plt.title('Feature Extraction Pipeline')
    
    # Save diagram
    plt.savefig(os.path.join(os.path.dirname(__file__), 'feature_extraction_pipeline.png'), dpi=150, bbox_inches='tight')
    plt.close()

# CNN Architecture diagram
def create_cnn_architecture():
    plt.figure(figsize=(10, 4))
    
    # Create a simple colormap for visualization
    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0, 1, 10))
    
    # Create layers representation
    layers = np.ones((10, 5))
    plt.imshow(layers.T, aspect='auto', cmap=cmap)
    
    # Add layer labels
    plt.text(9, 2.5, 'Output', ha='center', va='center', color='white', fontsize=12)
    plt.text(7, 2.5, 'Softmax', ha='center', va='center', color='white', fontsize=12)
    
    # Remove axes
    plt.axis('off')
    plt.title('CNN Architecture')
    
    # Save diagram
    plt.savefig(os.path.join(os.path.dirname(__file__), 'cnn_architecture.png'), dpi=150, bbox_inches='tight')
    plt.close()

# LSTM Architecture diagram
def create_lstm_architecture():
    plt.figure(figsize=(10, 4))
    
    # Create a simple colormap for visualization
    cmap = plt.cm.magma
    colors = cmap(np.linspace(0, 1, 10))
    
    # Create layers representation
    layers = np.ones((10, 5))
    plt.imshow(layers.T, aspect='auto', cmap=cmap)
    
    # Add layer labels
    plt.text(9, 2.5, 'Output', ha='center', va='center', color='white', fontsize=12)
    plt.text(7, 2.5, 'Dense 3', ha='center', va='center', color='white', fontsize=12)
    
    # Remove axes
    plt.axis('off')
    plt.title('LSTM Architecture')
    
    # Save diagram
    plt.savefig(os.path.join(os.path.dirname(__file__), 'lstm_architecture.png'), dpi=150, bbox_inches='tight')
    plt.close()

# CNN+LSTM Architecture diagram
def create_hybrid_architecture():
    plt.figure(figsize=(10, 4))
    
    # Create a simple colormap for visualization
    cmap = plt.cm.plasma
    colors = cmap(np.linspace(0, 1, 10))
    
    # Create layers representation
    layers = np.ones((10, 5))
    plt.imshow(layers.T, aspect='auto', cmap=cmap)
    
    # Add layer labels
    plt.text(9, 2.5, 'Output', ha='center', va='center', color='white', fontsize=12)
    plt.text(7, 2.5, 'Softmax', ha='center', va='center', color='white', fontsize=12)
    
    # Remove axes
    plt.axis('off')
    plt.title('CNN+LSTM Architecture')
    
    # Save diagram
    plt.savefig(os.path.join(os.path.dirname(__file__), 'hybrid_architecture.png'), dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    print("Generating architecture diagrams...")
    create_feature_extraction_pipeline()
    create_cnn_architecture()
    create_lstm_architecture()
    create_hybrid_architecture()
    print("Done!") 