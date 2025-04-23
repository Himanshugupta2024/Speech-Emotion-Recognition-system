import os
import numpy as np
import json
import librosa
import tempfile
import base64
from flask import Flask, render_template, request, jsonify, send_from_directory
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
# import tensorflow as tf  # Commented out to avoid dependency
from io import BytesIO

# Local imports - commented out to avoid dependency issues
# from feature_extraction import (extract_mfcc, extract_melspectrogram, extract_chroma,
#                               extract_spectral_contrast, extract_tonnetz)
# from infer import SpeechEmotionRecognizer, SpeakerIdentifier

app = Flask(__name__, static_folder='static')

# Global variables for models
emotion_recognizer = None
speaker_identifier = None

# Mock feature extraction functions
def extract_mfcc(audio, sr):
    """Mock MFCC extraction function"""
    # Create a dummy MFCC representation
    return np.random.rand(39, int(len(audio)/512))

def load_models(emotion_model_path=None, speaker_model_path=None):
    """Load the trained models - now just sets up mock data."""
    global emotion_recognizer, speaker_identifier
    
    # We're not actually loading models in this demo version
    print("Mock model setup - no actual models loaded.")

def analyze_audio(audio_path):
    """Analyze audio and generate visualizations."""
    results = {}
    
    # Load audio
    audio, sr = librosa.load(audio_path, sr=16000)
    
    # Calculate signal length and metrics
    signal_length = len(audio) / sr
    peak_amplitude = np.max(np.abs(audio))
    signal_to_noise = 10 * np.log10(np.mean(audio**2) / np.var(audio - np.mean(audio)))
    
    # Store audio metrics
    results['audio_metrics'] = {
        'signal_length': f"{signal_length:.4f} sec",
        'peak_amplitude': f"{peak_amplitude:.4f}",
        'signal_to_noise': f"{signal_to_noise:.4f} dB"
    }
    
    # Generate waveform plot
    plt.figure(figsize=(10, 3))
    times = np.arange(len(audio)) / sr
    plt.plot(times, audio, color='blue')
    plt.title('Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    waveform_img = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    results['waveform'] = waveform_img
    
    # Generate spectrogram
    plt.figure(figsize=(10, 3))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    spectrogram_img = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    results['spectrogram'] = spectrogram_img
    
    # Generate MFCC plot
    mfccs = extract_mfcc(audio, sr)
    plt.figure(figsize=(10, 3))
    plt.imshow(mfccs, aspect='auto', origin='lower')
    plt.colorbar()
    plt.title('MFCC')
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    mfcc_img = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    results['mfcc'] = mfcc_img
    
    # Generate pitch contour
    plt.figure(figsize=(10, 3))
    try:
        # Use a simple pitch estimation
        f0, voiced_flag, voiced_probs = librosa.pyin(audio, fmin=librosa.note_to_hz('C2'), 
                                                    fmax=librosa.note_to_hz('C7'),
                                                    sr=sr)
        # Handle NaN values in f0
        f0 = np.nan_to_num(f0)
        # Make sure times and f0 have the same length
        plot_times = times[:len(f0)]
        plt.plot(plot_times, f0, color='blue')
    except Exception as e:
        print(f"Error generating pitch contour: {str(e)}")
        # Fallback to a simple frequency representation
        S = np.abs(librosa.stft(audio))
        freq = librosa.fft_frequencies(sr=sr)
        t = librosa.times_like(S, sr=sr)
        S_dB = librosa.amplitude_to_db(S, ref=np.max)
        S_rolled = np.rollaxis(S_dB, 0, 2)
        strongest_freq = freq[np.argmax(S, axis=0)]
        plt.plot(t, strongest_freq, color='blue')
        
    plt.title('Pitch Contour')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    pitch_img = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    results['pitch_contour'] = pitch_img
    
    # Generate Energy & Zero Crossing Rate
    plt.figure(figsize=(10, 3))
    # Ensure consistent sizes for energy and time arrays
    frame_length = 1024
    hop_length = 1024
    energy = np.array([sum(abs(audio[i:i+frame_length]**2)) 
                      for i in range(0, len(audio) - frame_length, hop_length)])
    energy_times = np.arange(len(energy)) * (hop_length / sr)
    
    # Extract zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(
        audio, 
        frame_length=frame_length, 
        hop_length=hop_length
    )[0]
    
    # Ensure zcr and energy have the same length
    min_len = min(len(energy), len(zcr))
    energy = energy[:min_len]
    zcr = zcr[:min_len]
    energy_times = energy_times[:min_len]
    
    plt.plot(energy_times, energy / np.max(energy), 'r', label='Energy')
    plt.plot(energy_times, zcr, 'b', label='Zero Crossing Rate')
    plt.title('Energy & Zero Crossing Rate')
    plt.xlabel('Time (s)')
    plt.ylabel('Energy/ZCR')
    plt.legend()
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    energy_zcr_img = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    results['energy_zcr'] = energy_zcr_img
    
    # Analyze audio features for emotion detection
    emotions = ['neutral', 'happy', 'sad', 'angry', 'fearful', 'disgusted', 'surprised', 'calm']
    
    # Calculate key audio features
    rms = np.sqrt(np.mean(audio**2))
    zcr_mean = np.mean(zcr)
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)[0])
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0])
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr)[0])
    
    # Print debug info
    print(f"Audio features - RMS: {rms:.4f}, ZCR: {zcr_mean:.4f}, Centroid: {spectral_centroid:.1f}, Bandwidth: {spectral_bandwidth:.1f}")
    
    # Enhanced heuristic to determine emotion based on audio features
    # These are simplified rules that could be replaced with actual ML model predictions
    
    # Default values
    emotion = 'neutral'
    confidence = 0.60
    
    # Detect each emotion based on audio characteristics
    if rms > 0.1 and zcr_mean > 0.15 and spectral_centroid > 2500:
        emotion = 'angry'
        confidence = 0.82
    elif rms > 0.07 and spectral_centroid > 2000 and spectral_bandwidth > 2000:
        emotion = 'happy'
        confidence = 0.78
    elif rms < 0.03 and spectral_centroid < 1500 and spectral_bandwidth < 1500:
        emotion = 'sad'
        confidence = 0.88
    elif zcr_mean > 0.12 and spectral_centroid > 2000 and spectral_rolloff > 3000:
        emotion = 'surprised'
        confidence = 0.74
    elif rms < 0.04 and zcr_mean < 0.1 and spectral_centroid < 1800:
        emotion = 'fearful'
        confidence = 0.67
    elif rms > 0.06 and zcr_mean > 0.12 and spectral_rolloff < 2000:
        emotion = 'disgusted'
        confidence = 0.63
    elif rms < 0.05 and zcr_mean < 0.08 and spectral_bandwidth < 1700:
        emotion = 'calm'
        confidence = 0.81
    else:
        emotion = 'neutral'
        confidence = 0.73
    
    # Explicitly set emotion data in results
    results['emotion'] = {
        'predicted': emotion,
        'confidence': f"{confidence:.4f}"
    }
    
    # Create emotion analysis visualization
    plt.figure(figsize=(10, 3))
    # Simulate probabilities for each emotion
    if emotion == 'neutral':
        probs = [0.73, 0.05, 0.04, 0.06, 0.03, 0.02, 0.01, 0.06]
    elif emotion == 'happy':
        probs = [0.08, 0.78, 0.01, 0.03, 0.01, 0.02, 0.03, 0.04]
    elif emotion == 'sad':
        probs = [0.05, 0.01, 0.88, 0.01, 0.02, 0.01, 0.01, 0.01]
    elif emotion == 'angry':
        probs = [0.03, 0.02, 0.01, 0.82, 0.04, 0.05, 0.02, 0.01]
    elif emotion == 'fearful':
        probs = [0.07, 0.01, 0.06, 0.04, 0.67, 0.08, 0.06, 0.01]
    elif emotion == 'disgusted':
        probs = [0.06, 0.03, 0.02, 0.12, 0.08, 0.63, 0.04, 0.02]
    elif emotion == 'surprised':
        probs = [0.04, 0.07, 0.01, 0.02, 0.05, 0.02, 0.74, 0.05]
    else:  # calm
        probs = [0.09, 0.02, 0.03, 0.01, 0.01, 0.01, 0.02, 0.81]
    
    colors = ['#3498db', '#2ecc71', '#9b59b6', '#e74c3c', '#f39c12', '#1abc9c', '#d35400', '#7f8c8d']
    plt.bar(emotions, probs, color=colors)
    plt.title('Emotion Classification')
    plt.ylabel('Probability')
    plt.ylim([0, 1])
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    emotion_analysis_img = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    results['emotion_analysis'] = emotion_analysis_img
    
    print(f"Detected emotion: {emotion} with confidence {confidence:.4f}")
    
    # Set speaker data
    results['speaker'] = {
        'predicted': 'unknown',
        'confidence': '0.0'
    }
    
    return results

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Process the uploaded audio file."""
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio']
    
    # Save the file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp:
        audio_file.save(temp.name)
        temp_path = temp.name
    
    try:
        # Analyze the audio
        results = analyze_audio(temp_path)
        
        # Ensure emotion data is set
        if 'emotion' not in results:
            results['emotion'] = {
                'predicted': 'neutral',
                'confidence': '0.5000'
            }
        
        # Debug: Print results
        print("Response being sent:", results)
        
        # Clean up the temporary file
        os.unlink(temp_path)
        
        return jsonify(results)
    except Exception as e:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        
        print(f"Error during analysis: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/generate_report', methods=['POST'])
def generate_report():
    """Generate a PDF report with the analysis results."""
    # This would be implemented with a PDF generation library like ReportLab or WeasyPrint
    return jsonify({'message': 'Report generation not implemented yet'})

if __name__ == '__main__':
    # Call the mock model loading function
    load_models()
    
    # Run the app
    app.run(debug=True) 