import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import pickle
from typing import Dict, List, Tuple, Any, Optional
import json
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

class SpeakerIdentificationModel(nn.Module):
    """
    Speaker identification model.
    """
    def __init__(self, input_shape: Tuple[int, int], num_speakers: int):
        """
        Args:
            input_shape: Shape of input features (time_steps, num_features)
            num_speakers: Number of speakers to identify
        """
        super(SpeakerIdentificationModel, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(input_shape[1], 64, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Calculate the size after pooling
        self.time_after_pool = input_shape[0] // 4
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(128, num_speakers)
        
    def forward(self, x):
        # Transpose to (batch, features, time)
        x = x.transpose(1, 2)
        
        # Convolutional layers
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        
        # Transpose back to (batch, time, features) for LSTM
        x = x.transpose(1, 2)
        
        # LSTM layer
        output, (hidden, cell) = self.lstm(x)
        
        # Get the last time step output from both directions
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        
        # Fully connected layers
        x = F.relu(self.bn3(self.fc1(hidden)))
        x = self.dropout1(x)
        
        x = self.fc2(x)
        
        return F.softmax(x, dim=1)

class SpeakerIdentifier:
    """
    Speaker identification system.
    """
    def __init__(self, model_path: Optional[str] = None, label_mapping_path: Optional[str] = None):
        """
        Args:
            model_path: Path to the trained model
            label_mapping_path: Path to the label mapping file
        """
        self.model = None
        self.label_mapping = {}
        self.speakers = []
        
        # Create model directory
        os.makedirs('models', exist_ok=True)
        
        # Default paths
        self.model_path = model_path or 'models/speaker_identification_model.pt'
        self.label_mapping_path = label_mapping_path or 'models/speaker_mapping.json'
        self.speaker_data_path = 'models/speaker_data.pkl'
        
        # Load model and label mapping if they exist
        self.load_model()
        self.load_label_mapping()
        self.load_speaker_data()
    
    def load_model(self):
        """
        Load the trained model.
        """
        if os.path.exists(self.model_path):
            try:
                # Create a dummy model with the correct number of speakers
                num_speakers = len(self.label_mapping) if self.label_mapping else 1
                input_shape = (188, 39)  # Fixed input shape
                self.model = SpeakerIdentificationModel(input_shape, num_speakers)
                
                # Load model weights
                self.model.load_state_dict(torch.load(self.model_path))
                self.model.eval()
                
                print(f"Loaded speaker identification model from {self.model_path}")
                return True
            except Exception as e:
                print(f"Error loading speaker identification model: {e}")
                return False
        else:
            print(f"Speaker identification model not found at {self.model_path}")
            return False
    
    def load_label_mapping(self):
        """
        Load the label mapping.
        """
        if os.path.exists(self.label_mapping_path):
            try:
                with open(self.label_mapping_path, 'r') as f:
                    self.label_mapping = json.load(f)
                
                print(f"Loaded speaker mapping from {self.label_mapping_path}")
                return True
            except Exception as e:
                print(f"Error loading speaker mapping: {e}")
                return False
        else:
            print(f"Speaker mapping not found at {self.label_mapping_path}")
            return False
    
    def load_speaker_data(self):
        """
        Load speaker data.
        """
        if os.path.exists(self.speaker_data_path):
            try:
                with open(self.speaker_data_path, 'rb') as f:
                    self.speakers = pickle.load(f)
                
                print(f"Loaded speaker data from {self.speaker_data_path}")
                return True
            except Exception as e:
                print(f"Error loading speaker data: {e}")
                return False
        else:
            print(f"Speaker data not found at {self.speaker_data_path}")
            return False
    
    def save_model(self):
        """
        Save the trained model.
        """
        if self.model is not None:
            try:
                torch.save(self.model.state_dict(), self.model_path)
                print(f"Saved speaker identification model to {self.model_path}")
                return True
            except Exception as e:
                print(f"Error saving speaker identification model: {e}")
                return False
        else:
            print("No model to save")
            return False
    
    def save_label_mapping(self):
        """
        Save the label mapping.
        """
        try:
            with open(self.label_mapping_path, 'w') as f:
                json.dump(self.label_mapping, f, indent=4)
            
            print(f"Saved speaker mapping to {self.label_mapping_path}")
            return True
        except Exception as e:
            print(f"Error saving speaker mapping: {e}")
            return False
    
    def save_speaker_data(self):
        """
        Save speaker data.
        """
        try:
            with open(self.speaker_data_path, 'wb') as f:
                pickle.dump(self.speakers, f)
            
            print(f"Saved speaker data to {self.speaker_data_path}")
            return True
        except Exception as e:
            print(f"Error saving speaker data: {e}")
            return False
    
    def extract_features(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        """
        Extract features from audio data.
        
        Args:
            audio: Audio data
            sr: Sample rate
        
        Returns:
            Extracted features
        """
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=39)
        
        # Transpose to (time, features)
        mfcc = mfcc.T
        
        # Pad or truncate to fixed length
        target_length = 188  # Approximately 3 seconds
        
        if mfcc.shape[0] < target_length:
            # Pad with zeros
            padding = np.zeros((target_length - mfcc.shape[0], mfcc.shape[1]))
            mfcc = np.vstack((mfcc, padding))
        elif mfcc.shape[0] > target_length:
            # Truncate
            mfcc = mfcc[:target_length, :]
        
        # Normalize features
        mfcc = (mfcc - np.mean(mfcc, axis=0)) / (np.std(mfcc, axis=0) + 1e-10)
        
        return mfcc
    
    def register_speaker(self, audio: np.ndarray, sr: int, speaker_name: str) -> bool:
        """
        Register a new speaker.
        
        Args:
            audio: Audio data
            sr: Sample rate
            speaker_name: Name of the speaker
        
        Returns:
            True if registration was successful, False otherwise
        """
        try:
            # Extract features
            features = self.extract_features(audio, sr)
            
            # Add speaker to list
            speaker_id = len(self.speakers)
            self.speakers.append({
                'id': speaker_id,
                'name': speaker_name,
                'features': features
            })
            
            # Update label mapping
            self.label_mapping[str(speaker_id)] = speaker_name
            
            # Save speaker data and label mapping
            self.save_speaker_data()
            self.save_label_mapping()
            
            # Retrain model if we have at least 2 speakers
            if len(self.speakers) >= 2:
                self.train_model()
            
            return True
        except Exception as e:
            print(f"Error registering speaker: {e}")
            return False
    
    def train_model(self) -> bool:
        """
        Train the speaker identification model.
        
        Returns:
            True if training was successful, False otherwise
        """
        try:
            # Prepare training data
            X = np.array([speaker['features'] for speaker in self.speakers])
            y = np.array([speaker['id'] for speaker in self.speakers])
            
            # Create model
            input_shape = X[0].shape
            num_speakers = len(self.speakers)
            self.model = SpeakerIdentificationModel(input_shape, num_speakers)
            
            # Convert to PyTorch tensors
            X_tensor = torch.tensor(X, dtype=torch.float32)
            y_tensor = torch.tensor(y, dtype=torch.long)
            
            # Define loss function and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            
            # Train model
            self.model.train()
            
            # Simple training loop (for demonstration)
            num_epochs = 50
            batch_size = 1  # Since we might have very few speakers
            
            for epoch in range(num_epochs):
                # Forward pass
                outputs = self.model(X_tensor)
                loss = criterion(outputs, y_tensor)
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
            
            # Save model
            self.save_model()
            
            return True
        except Exception as e:
            print(f"Error training speaker identification model: {e}")
            return False
    
    def identify_speaker(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Identify the speaker from audio data.
        
        Args:
            audio: Audio data
            sr: Sample rate
        
        Returns:
            Dictionary containing the identified speaker and confidence
        """
        if self.model is None or len(self.speakers) < 2:
            return {
                'speaker': 'unknown',
                'confidence': 0.0,
                'probabilities': {}
            }
        
        try:
            # Extract features
            features = self.extract_features(audio, sr)
            
            # Add batch dimension
            features = np.expand_dims(features, axis=0)
            
            # Convert to PyTorch tensor
            features_tensor = torch.tensor(features, dtype=torch.float32)
            
            # Predict speaker
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(features_tensor)
                probabilities = outputs[0].numpy()
            
            # Get speaker ID
            speaker_idx = np.argmax(probabilities)
            confidence = probabilities[speaker_idx]
            
            # Get speaker name
            speaker_name = self.label_mapping.get(str(speaker_idx), 'unknown')
            
            # Create probabilities dictionary
            speaker_probs = {self.label_mapping.get(str(i), f'Speaker {i}'): float(prob) 
                           for i, prob in enumerate(probabilities)}
            
            return {
                'speaker': speaker_name,
                'confidence': float(confidence),
                'probabilities': speaker_probs
            }
        except Exception as e:
            print(f"Error identifying speaker: {e}")
            return {
                'speaker': 'unknown',
                'confidence': 0.0,
                'probabilities': {}
            }
    
    def delete_speaker(self, speaker_id: int) -> bool:
        """
        Delete a speaker.
        
        Args:
            speaker_id: ID of the speaker to delete
        
        Returns:
            True if deletion was successful, False otherwise
        """
        try:
            # Find speaker index
            speaker_index = None
            for i, speaker in enumerate(self.speakers):
                if speaker['id'] == speaker_id:
                    speaker_index = i
                    break
            
            if speaker_index is None:
                print(f"Speaker with ID {speaker_id} not found")
                return False
            
            # Remove speaker from list
            self.speakers.pop(speaker_index)
            
            # Update label mapping
            if str(speaker_id) in self.label_mapping:
                del self.label_mapping[str(speaker_id)]
            
            # Save speaker data and label mapping
            self.save_speaker_data()
            self.save_label_mapping()
            
            # Retrain model if we have at least 2 speakers
            if len(self.speakers) >= 2:
                self.train_model()
            else:
                # Reset model if we have less than 2 speakers
                self.model = None
                if os.path.exists(self.model_path):
                    os.remove(self.model_path)
            
            return True
        except Exception as e:
            print(f"Error deleting speaker: {e}")
            return False
    
    def plot_speaker_probabilities(self, probabilities: Dict[str, float], title: str = 'Speaker Identification') -> plt.Figure:
        """
        Plot speaker probabilities.
        
        Args:
            probabilities: Dictionary containing speaker probabilities
            title: Plot title
        
        Returns:
            Matplotlib figure
        """
        # Sort probabilities
        sorted_items = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        speakers = [item[0] for item in sorted_items]
        probs = [item[1] for item in sorted_items]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot bar chart
        bars = ax.bar(speakers, probs, color='skyblue')
        
        # Highlight highest probability
        if len(bars) > 0:
            bars[0].set_color('orange')
        
        # Set labels and title
        ax.set_ylabel('Probability')
        ax.set_title(title)
        ax.set_ylim(0, 1)
        
        # Rotate x-axis labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        return fig

# Example usage
if __name__ == '__main__':
    # Create speaker identifier
    speaker_identifier = SpeakerIdentifier()
    
    # Register a speaker
    audio, sr = librosa.load('sample_data/speaker1_happy.wav', sr=16000)
    speaker_identifier.register_speaker(audio, sr, 'Speaker 1')
    
    # Register another speaker
    audio, sr = librosa.load('sample_data/speaker2_happy.wav', sr=16000)
    speaker_identifier.register_speaker(audio, sr, 'Speaker 2')
    
    # Identify speaker
    audio, sr = librosa.load('sample_data/speaker1_angry.wav', sr=16000)
    result = speaker_identifier.identify_speaker(audio, sr)
    
    print(f"Identified speaker: {result['speaker']} with confidence {result['confidence']:.4f}")
    
    # Plot speaker probabilities
    fig = speaker_identifier.plot_speaker_probabilities(result['probabilities'])
    plt.show()
