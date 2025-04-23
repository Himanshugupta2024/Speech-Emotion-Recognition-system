import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import librosa
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Define a simple CNN-LSTM model
class CNNLSTM(nn.Module):
    def __init__(self, input_shape, num_classes, lstm_units=64, dropout_rate=0.3):
        super(CNNLSTM, self).__init__()
        self.conv1 = nn.Conv1d(input_shape[1], 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        # Calculate the size after pooling
        self.time_after_pool = input_shape[0] // 4

        self.lstm = nn.LSTM(64, lstm_units, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(lstm_units, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Transpose to (batch, features, time)
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # Transpose back to (batch, time, features) for LSTM
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        x = self.dropout(x[:, -1, :])  # Take the last time step
        x = self.fc(x)
        x = self.softmax(x)
        return x

# Function to extract MFCC features
def extract_mfcc(audio_path, sr=16000, n_mfcc=13):
    try:
        audio, _ = librosa.load(audio_path, sr=sr)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)

        # Add delta features
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)

        # Combine features
        features = np.concatenate([mfccs, delta_mfccs, delta2_mfccs])

        # Transpose to get time as first dimension
        features = features.T

        # Pad or truncate to fixed length (3 seconds)
        target_length = 188  # Approximately 3 seconds
        if features.shape[0] < target_length:
            padding = np.zeros((target_length - features.shape[0], features.shape[1]))
            features = np.vstack((features, padding))
        elif features.shape[0] > target_length:
            features = features[:target_length, :]

        return features
    except Exception as e:
        print(f"Error extracting features from {audio_path}: {e}")
        return None

# Function to load dataset
def load_sample_dataset(data_dir):
    audio_files = []
    emotions = []
    speakers = []

    # Find all WAV files in the directory
    for file in os.listdir(data_dir):
        if file.endswith('.wav'):
            file_path = os.path.join(data_dir, file)

            # Parse metadata from filename (format: speaker_emotion.wav)
            parts = os.path.splitext(file)[0].split('_')
            if len(parts) == 2:
                speaker, emotion = parts

                audio_files.append(file_path)
                emotions.append(emotion)
                speakers.append(speaker)

    return audio_files, emotions, speakers

# Main function
def main():
    print("Starting simple training script...")

    # Parameters
    data_dir = "sample_data"
    output_dir = "models"
    batch_size = 2
    epochs = 50

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    print(f"Loading data from {data_dir}...")
    audio_files, emotions, speakers = load_sample_dataset(data_dir)

    if len(audio_files) == 0:
        print(f"No audio files found in {data_dir}")
        return

    print(f"Found {len(audio_files)} audio files")
    print(f"Emotions: {set(emotions)}")
    print(f"Speakers: {set(speakers)}")

    # Extract features
    print("Extracting MFCC features...")
    features = []
    for file_path in audio_files:
        feature = extract_mfcc(file_path)
        if feature is not None:
            features.append(feature)

    if len(features) == 0:
        print("No features extracted")
        return

    features = np.array(features)
    print(f"Feature shape: {features.shape}")

    # Encode emotion labels
    emotion_encoder = LabelEncoder()
    emotion_labels = emotion_encoder.fit_transform(emotions[:len(features)])
    num_emotion_classes = len(emotion_encoder.classes_)

    print(f"Number of emotion classes: {num_emotion_classes}")
    print(f"Emotion classes: {emotion_encoder.classes_}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, emotion_labels, test_size=0.2, random_state=42
    )

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # Create model
    print("Creating CNN-LSTM model...")
    input_shape = X_train.shape[1:]
    model = CNNLSTM(input_shape, num_emotion_classes)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }

    # Training loop
    print("Training model...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()

        train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        # Evaluation
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                test_total += targets.size(0)
                test_correct += (predicted == targets).sum().item()

        test_loss = test_loss / test_total
        test_acc = test_correct / test_total
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    # Save model
    model_save_path = os.path.join(output_dir, 'emotion_model_simple.pt')
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    # Plot training history
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['test_loss'], label='Test')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['test_acc'], label='Test')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history_simple.png'))
    print(f"Training history plot saved to {os.path.join(output_dir, 'training_history_simple.png')}")

if __name__ == "__main__":
    main()
