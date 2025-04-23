# Speech Emotion Recognition & Speaker Identification

This project implements Speech Emotion Recognition (SER) and Speaker Identification (SID) using deep learning techniques. The system can analyze audio recordings to detect emotions and identify speakers using various neural network architectures (LSTMs, CNNs, or hybrid models).

## Features

- **Speech Emotion Recognition (SER)**: Identify emotions in speech like happy, sad, angry, etc.
- **Speaker Identification (SID)**: Identify who is speaking from a set of known speakers
- **Multiple Model Types**: Supports LSTM, Bidirectional LSTM, 1D CNN, 2D CNN, and CNN-LSTM hybrid models
- **Audio Feature Extraction**: Extract MFCC, Mel-spectrograms, and other audio features
- **Real-time Detection**: Process audio from your microphone in real-time
- **Training & Evaluation**: Comprehensive training pipeline with model evaluation tools
- **Visualization Tools**: Visualize waveforms, spectrograms, and prediction probabilities
- **Web Interface**: Record audio directly in the browser and see detailed analysis visualizations

## Project Structure

```
speech_emotion_speaker_id/
├── data/
│   ├── raw/                # Raw audio datasets
│   └── processed/          # Preprocessed features
├── models/                 # Saved model files
├── notebooks/              # Jupyter notebooks for exploration
├── src/
│   ├── __init__.py         # Makes src a module
│   ├── data_loader.py      # Load and preprocess audio data
│   ├── feature_extraction.py # Extract audio features
│   ├── model_def.py        # Define model architectures
│   ├── evaluate.py         # Evaluate model performance
│   ├── infer.py            # Run predictions on new audio
│   └── main.py             # Main script for training and testing
├── static/                 # Static files for web interface
│   ├── css/                # CSS stylesheets
│   ├── js/                 # JavaScript files
│   └── img/                # Image assets
├── templates/              # HTML templates for web interface
├── app.py                  # Flask web application
├── requirements.txt        # List of dependencies
└── README.md               # Project overview and instructions
```

## Getting Started

### Prerequisites

- Python 3.9 or newer
- Pip package manager

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/speech_emotion_speaker_id.git
   cd speech_emotion_speaker_id
   ```

2. Create and activate a virtual environment:
   ```
   # Windows
   python -m venv speech_proj
   speech_proj\Scripts\activate

   # macOS/Linux
   python -m venv speech_proj
   source speech_proj/bin/activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Preparing Data

The system works with several popular emotion and speaker datasets:

1. **RAVDESS**: Download from [Zenodo](https://zenodo.org/record/1188976)
2. **TESS**: Download from [University of Toronto](https://tspace.library.utoronto.ca/handle/1807/24487)
3. **EMO-DB**: Download from [EMO-DB](http://emodb.bilderbar.info/download/)
4. **VoxCeleb**: For speaker identification, a subset can be downloaded from [VoxCeleb](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/)

After downloading, place the audio files in the `data/raw` directory.

### Training Models

To train both SER and SID models:

```bash
python src/main.py --mode train --data_dir data/raw --model_type cnn_lstm --feature_type mfcc --output_dir models
```

Optional arguments:
- `--model_type`: Choose from 'lstm', 'bilstm', 'cnn1d', 'cnn2d', 'cnn_lstm'
- `--feature_type`: Choose from 'mfcc', 'melspec'
- `--batch_size`: Batch size for training (default: 32)
- `--epochs`: Maximum number of epochs (default: 100)

### Testing Models

To test the trained models on a specific audio file:

```bash
python src/main.py --mode test --test_audio path/to/audio/file.wav --emotion_model_path models/emotion_model.h5 --speaker_model_path models/speaker_model.h5
```

### Real-time Detection

To detect emotions and identify speakers in real-time using your microphone:

```bash
python src/infer.py
```

### Web Interface

To run the web interface for audio recording and analysis:

```bash
python app.py
```

This will start a Flask web server at `http://localhost:5000`. Open this URL in your browser to:

1. Record audio directly from your microphone
2. Upload audio files for analysis
3. View detailed visualizations including:
   - Waveform
   - Spectrogram
   - MFCC features
   - Pitch contour
   - Energy & Zero Crossing Rate
4. See emotion predictions and audio metrics
5. Download analysis reports

## Model Architectures

The project implements the following neural network architectures:

1. **LSTM**: Simple LSTM for sequential audio data
2. **Bidirectional LSTM**: Improved LSTM with bidirectional processing
3. **1D CNN**: Convolutional neural network for 1D audio features
4. **2D CNN**: Convolutional network for 2D spectrograms
5. **CNN-LSTM Hybrid**: Combines CNN for feature extraction and LSTM for temporal patterns

## Performance Metrics

Models are evaluated using:
- Accuracy, Precision, Recall, F1-score
- Confusion matrices
- ROC curves with AUC scores
- Training/validation loss and accuracy curves

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The project structure is inspired by best practices in machine learning project organization
- Thanks to the creators of RAVDESS, TESS, EMO-DB, and VoxCeleb datasets 