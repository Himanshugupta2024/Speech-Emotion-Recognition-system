# Speech Emotion Recognition System

This is a comprehensive speech emotion recognition system that can analyze audio recordings to detect emotions using various neural network architectures (CNN, LSTM, or CNN-LSTM hybrid models).

## Features

- **Multiple Emotion Detection**: Identifies 8 emotions (Neutral, Happy, Sad, Angry, Fearful, Disgusted, Surprised, Calm)
- **Rich Feature Extraction**: Extracts waveform, spectrogram, MFCC, pitch contour, energy, and zero-crossing rate
- **Multiple Model Architectures**: Supports CNN, LSTM, and CNN-LSTM hybrid models
- **Comprehensive Visualization**: Provides detailed visualizations of audio features and model predictions
- **Synthetic Data Generation**: Includes tools to generate synthetic emotional speech data for testing

## System Architecture

The system consists of the following components:

1. **Feature Extraction Pipeline**: Extracts various audio features from raw audio
2. **Model Architectures**: Implements CNN, LSTM, and CNN-LSTM hybrid models
3. **Training Pipeline**: Trains models on emotional speech data
4. **Analysis Dashboard**: Visualizes audio features and emotion predictions

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/speech-emotion-recognition.git
   cd speech-emotion-recognition
   ```

2. Create and activate a virtual environment:
   ```
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # macOS/Linux
   python -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```
   pip install torch numpy librosa matplotlib scikit-learn pandas soundfile seaborn
   ```

## Usage

### Generating Synthetic Data

To generate synthetic emotional speech data for testing:

```bash
python generate_synthetic_data.py --output_dir synthetic_data --num_samples 10
```

This will create a dataset with 10 samples for each of the 8 emotions.

### Training Models

To train the models on your dataset:

```bash
python train_emotion_models.py --data_dir synthetic_data --output_dir models --epochs 50
```

This will train CNN, LSTM, and CNN-LSTM models on the dataset and save the models, training history plots, and confusion matrices in the `models` directory.

### Creating Analysis Dashboard

To analyze an audio file and create a comprehensive dashboard:

```bash
python emotion_analysis_dashboard.py --audio path/to/audio.wav --output_dir output
```

This will create a dashboard with visualizations of audio features and emotion predictions in the `output` directory.

## Model Architectures

### CNN Architecture

The CNN model consists of multiple convolutional layers followed by fully connected layers:

1. Conv1D (64 filters, kernel size 5)
2. BatchNorm1D
3. MaxPool1D (kernel size 2)
4. Conv1D (128 filters, kernel size 5)
5. BatchNorm1D
6. MaxPool1D (kernel size 2)
7. Conv1D (256 filters, kernel size 5)
8. BatchNorm1D
9. MaxPool1D (kernel size 2)
10. Flatten
11. Dense (512 units)
12. BatchNorm1D
13. Dropout (0.3)
14. Dense (256 units)
15. BatchNorm1D
16. Dropout (0.3)
17. Dense (num_classes units)
18. Softmax

### LSTM Architecture

The LSTM model consists of bidirectional LSTM layers followed by fully connected layers:

1. Bidirectional LSTM (128 units)
2. Dense (256 units)
3. BatchNorm1D
4. Dropout (0.3)
5. Dense (128 units)
6. BatchNorm1D
7. Dropout (0.3)
8. Dense (num_classes units)
9. Softmax

### CNN+LSTM Architecture

The CNN-LSTM hybrid model combines convolutional layers for feature extraction with LSTM layers for temporal modeling:

1. Conv1D (64 filters, kernel size 5)
2. BatchNorm1D
3. MaxPool1D (kernel size 2)
4. Conv1D (128 filters, kernel size 5)
5. BatchNorm1D
6. MaxPool1D (kernel size 2)
7. Bidirectional LSTM (128 units)
8. Dense (256 units)
9. BatchNorm1D
10. Dropout (0.3)
11. Dense (128 units)
12. BatchNorm1D
13. Dropout (0.3)
14. Dense (num_classes units)
15. Softmax

## Feature Extraction

The system extracts the following features from audio:

1. **Waveform**: Raw audio signal
2. **Mel-Spectrogram**: Time-frequency representation with mel scaling
3. **MFCC**: Mel-frequency cepstral coefficients
4. **Pitch Contour**: Fundamental frequency over time
5. **Energy**: Signal energy over time
6. **Zero Crossing Rate**: Rate at which the signal changes sign

## Dashboard

The analysis dashboard includes:

1. **Waveform Visualization**: Shows the raw audio signal
2. **Spectrogram Visualization**: Shows the mel-spectrogram
3. **MFCC Visualization**: Shows the MFCC features
4. **Pitch Contour Visualization**: Shows the pitch contour
5. **Energy & ZCR Visualization**: Shows energy and zero crossing rate
6. **Audio Metrics**: Shows signal length, peak amplitude, RMS energy, and SNR
7. **Emotion Predictions**: Shows emotion probabilities for each model
8. **Feature Extraction Pipeline**: Shows the feature extraction process
9. **Model Architecture**: Shows the architecture of the models

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The project structure is inspired by best practices in machine learning project organization
- Thanks to the creators of PyTorch, Librosa, and other open-source libraries used in this project
