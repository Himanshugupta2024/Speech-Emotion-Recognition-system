// Global variables
let mediaRecorder;
let audioChunks = [];
let audioBlob;
let audioUrl;
let recordingInProgress = false;
let speakerList = [];
let speakerModal; // Add Bootstrap modal reference

// DOM Elements
document.addEventListener('DOMContentLoaded', function() {
    // Initialize Bootstrap modal
    const modalElement = document.getElementById('speakerModal');
    if (modalElement) {
        if (typeof bootstrap !== 'undefined') {
            speakerModal = new bootstrap.Modal(modalElement);
        } else {
            console.error('Bootstrap is not loaded. Modal functionality will not work.');
        }
    }
    
    // Audio recording elements
    const startRecordingBtn = document.getElementById('startRecording');
    const stopRecordingBtn = document.getElementById('stopRecording');
    const audioPlayback = document.getElementById('audioPlayback');
    const audioFileUpload = document.getElementById('audioFileUpload');
    const analyzeBtn = document.getElementById('analyzeBtn');
    
    // Analysis result elements
    const waveformImg = document.getElementById('waveformImg');
    const spectrogramImg = document.getElementById('spectrogramImg');
    const emotionAnalysisImg = document.getElementById('emotionAnalysisImg');
    const mfccImg = document.getElementById('mfccImg');
    const pitchImg = document.getElementById('pitchImg');
    const energyImg = document.getElementById('energyImg');
    
    // Audio metrics elements
    const primaryEmotionDisplay = document.getElementById('primaryEmotionDisplay');
    const signalLength = document.getElementById('signalLength');
    const peakAmplitude = document.getElementById('peakAmplitude');
    const signalToNoise = document.getElementById('signalToNoise');
    
    // Speaker elements
    const speakerDisplay = document.getElementById('speakerDisplay');
    const speakerName = document.getElementById('speakerName');
    const registerSpeaker = document.getElementById('registerSpeaker');
    const speakerList = document.getElementById('speakerList');
    const manageSpeakers = document.getElementById('manageSpeakers');
    
    // Event listeners
    if (startRecordingBtn) {
        startRecordingBtn.addEventListener('click', startRecording);
    }
    
    if (stopRecordingBtn) {
        stopRecordingBtn.addEventListener('click', stopRecording);
    }
    
    if (audioFileUpload) {
        audioFileUpload.addEventListener('change', handleFileUpload);
    }
    
    if (analyzeBtn) {
        analyzeBtn.addEventListener('click', analyzeAudio);
    }
    
    if (registerSpeaker) {
        registerSpeaker.addEventListener('click', registerNewSpeaker);
    }
    
    if (manageSpeakers) {
        manageSpeakers.addEventListener('click', manageSpeakersModal);
    }
    
    // Load speakers if any
    loadSavedSpeakers();
});

// Recording functions
async function startRecording() {
    if (recordingInProgress) return;
    
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];
        
        mediaRecorder.ondataavailable = e => {
            audioChunks.push(e.data);
        };
        
        mediaRecorder.onstop = processRecordedAudio;
        
        mediaRecorder.start();
        recordingInProgress = true;
        
        // Update UI
        document.getElementById('startRecording').disabled = true;
        document.getElementById('stopRecording').disabled = false;
        document.getElementById('recordingStatus').classList.remove('d-none');
        
        // Start recording timer
        startRecordingTimer();
    } catch (err) {
        console.error('Error accessing microphone:', err);
        alert('Error accessing microphone. Please check your permissions and try again.');
    }
}

function stopRecording() {
    if (!recordingInProgress) return;
    
    mediaRecorder.stop();
    recordingInProgress = false;
    
    // Update UI
    document.getElementById('startRecording').disabled = false;
    document.getElementById('stopRecording').disabled = true;
    document.getElementById('recordingStatus').classList.add('d-none');
    
    // Reset timer
    stopRecordingTimer();
}

function processRecordedAudio() {
    audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
    audioUrl = URL.createObjectURL(audioBlob);
    
    // Update audio player
    const audioPlayback = document.getElementById('audioPlayback');
    audioPlayback.src = audioUrl;
    audioPlayback.classList.remove('d-none');
    
    // Enable analyze button
    document.getElementById('analyzeBtn').disabled = false;
}

function handleFileUpload(e) {
    const file = e.target.files[0];
    if (!file) return;
    
    const reader = new FileReader();
    reader.onload = function(e) {
        const arrayBuffer = e.target.result;
        const blob = new Blob([arrayBuffer], { type: file.type });
        audioBlob = blob;
        audioUrl = URL.createObjectURL(blob);
        
        // Update audio player
        const audioPlayback = document.getElementById('audioPlayback');
        audioPlayback.src = audioUrl;
        audioPlayback.classList.remove('d-none');
        
        // Enable analyze button
        document.getElementById('analyzeBtn').disabled = false;
    };
    reader.readAsArrayBuffer(file);
}

// Recording timer
let recordingTimerInterval;
let recordingSeconds = 0;

function startRecordingTimer() {
    recordingSeconds = 0;
    updateRecordingTimer();
    recordingTimerInterval = setInterval(updateRecordingTimer, 1000);
}

function updateRecordingTimer() {
    recordingSeconds++;
    const minutes = Math.floor(recordingSeconds / 60);
    const seconds = recordingSeconds % 60;
    document.getElementById('recordingTimer').textContent = 
        `${minutes}:${seconds.toString().padStart(2, '0')}`;
}

function stopRecordingTimer() {
    clearInterval(recordingTimerInterval);
}

// Analysis functions
function analyzeAudio() {
    if (!audioBlob) {
        alert('Please record or upload audio first.');
        return;
    }
    
    // Show loading indicator if available
    const loadingSpinner = document.getElementById('loadingSpinner');
    if (loadingSpinner) {
        loadingSpinner.classList.remove('d-none');
    }
    
    // Prepare form data
    const formData = new FormData();
    formData.append('audio', audioBlob, 'recording.wav');
    
    // Get selected speaker if any
    const speakerSelect = document.getElementById('speakerList');
    if (speakerSelect && speakerSelect.value !== "" && speakerSelect.value !== "Select a speaker") {
        formData.append('speaker_id', speakerSelect.value);
    }
    
    // Send to server
    fetch('/analyze', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        displayResults(data);
        
        // Always show results container after processing
        const resultsContainer = document.getElementById('resultsContainer');
        if (resultsContainer) {
            resultsContainer.classList.remove('d-none');
        }
        
        // Hide loading spinner
        if (loadingSpinner) {
            loadingSpinner.classList.add('d-none');
        }
    })
    .catch(error => {
        console.error('Error analyzing audio:', error);
        alert('An error occurred while analyzing the audio. Please try again.');
        
        // Hide loading spinner on error
        if (loadingSpinner) {
            loadingSpinner.classList.add('d-none');
        }
    });
}

function displayResults(data) {
    console.log("Received data:", data); // Debug log to see exact server response
    
    // Display waveform
    if (data.waveform) {
        const waveformImg = document.getElementById('waveformImg');
        if (waveformImg) {
            waveformImg.src = "data:image/png;base64," + data.waveform;
        }
    }
    
    // Display spectrogram
    if (data.spectrogram) {
        const spectrogramImg = document.getElementById('spectrogramImg');
        if (spectrogramImg) {
            spectrogramImg.src = "data:image/png;base64," + data.spectrogram;
        }
    }
    
    // Display emotion analysis
    if (data.emotion_analysis) {
        const emotionImg = document.getElementById('emotionAnalysisImg');
        if (emotionImg) {
            emotionImg.src = "data:image/png;base64," + data.emotion_analysis;
        }
    }
    
    // Display MFCC
    if (data.mfcc) {
        const mfccImg = document.getElementById('mfccImg');
        if (mfccImg) {
            mfccImg.src = "data:image/png;base64," + data.mfcc;
        }
    }
    
    // Display pitch contour
    if (data.pitch_contour) {
        const pitchImg = document.getElementById('pitchImg');
        if (pitchImg) {
            pitchImg.src = "data:image/png;base64," + data.pitch_contour;
        }
    }
    
    // Display energy & ZCR
    if (data.energy_zcr) {
        const energyImg = document.getElementById('energyImg');
        if (energyImg) {
            energyImg.src = "data:image/png;base64," + data.energy_zcr;
        }
    }
    
    // Display primary emotion
    const primaryEmotionDisplay = document.getElementById('primaryEmotionDisplay');
    if (primaryEmotionDisplay) {
        const emotionSpan = primaryEmotionDisplay.querySelector('span');
        if (emotionSpan) {
            // Try to get emotion from various possible formats
            let detectedEmotion = "UNKNOWN";
            let confidence = "";
            
            if (data.emotion && data.emotion.predicted) {
                detectedEmotion = data.emotion.predicted.toUpperCase();
                confidence = data.emotion.confidence || "";
            } else if (data.primary_emotion) {
                detectedEmotion = data.primary_emotion.toUpperCase();
            }
            
            // Set the emotion text
            emotionSpan.textContent = detectedEmotion;
            if (confidence) {
                emotionSpan.title = `Confidence: ${confidence}`;
            }
        }
    }
    
    // Display audio metrics
    if (data.audio_metrics) {
        const signalLengthEl = document.getElementById('signalLength');
        const peakAmplitudeEl = document.getElementById('peakAmplitude');
        const signalToNoiseEl = document.getElementById('signalToNoise');
        
        if (signalLengthEl) signalLengthEl.textContent = data.audio_metrics.signal_length;
        if (peakAmplitudeEl) peakAmplitudeEl.textContent = data.audio_metrics.peak_amplitude;
        if (signalToNoiseEl) signalToNoiseEl.textContent = data.audio_metrics.signal_to_noise;
    }
    
    // Display speaker if identified
    const speakerDisplay = document.getElementById('speakerDisplay');
    if (speakerDisplay) {
        const speakerSpan = speakerDisplay.querySelector('span');
        if (speakerSpan) {
            if (data.speaker && data.speaker.predicted) {
                speakerSpan.textContent = data.speaker.predicted === 'unknown' 
                    ? 'Unknown Speaker' 
                    : data.speaker.predicted;
                if (data.speaker.confidence) {
                    speakerSpan.title = `Confidence: ${data.speaker.confidence}`;
                }
            } else if (data.identified_speaker) {
                speakerSpan.textContent = data.identified_speaker;
            }
        }
    }
}

// Speaker management
function loadSavedSpeakers() {
    const savedSpeakers = localStorage.getItem('speakersList');
    if (savedSpeakers) {
        speakerList = JSON.parse(savedSpeakers);
        updateSpeakerDropdown();
    }
}

function updateSpeakerDropdown() {
    const speakerSelect = document.getElementById('speakerList');
    if (!speakerSelect) return;
    
    // Clear current options
    speakerSelect.innerHTML = '';
    
    // Add default option
    const defaultOption = document.createElement('option');
    defaultOption.value = "";
    defaultOption.textContent = "Select a speaker";
    speakerSelect.appendChild(defaultOption);
    
    // Add speakers
    speakerList.forEach(speaker => {
        const option = document.createElement('option');
        option.value = speaker.id;
        option.textContent = speaker.name;
        speakerSelect.appendChild(option);
    });
}

function registerNewSpeaker() {
    const speakerNameInput = document.getElementById('speakerName');
    const speakerName = speakerNameInput.value.trim();
    
    if (!speakerName) {
        alert('Please enter a speaker name');
        return;
    }
    
    if (!audioBlob) {
        alert('Please record or upload audio first');
        return;
    }
    
    // For demo purposes, we'll just add to localStorage
    // In a real app, this would send to the server
    const newSpeaker = {
        id: Date.now().toString(),
        name: speakerName
    };
    
    speakerList.push(newSpeaker);
    localStorage.setItem('speakersList', JSON.stringify(speakerList));
    
    // Update dropdown
    updateSpeakerDropdown();
    
    // Clear input
    speakerNameInput.value = '';
    
    alert('Speaker registered successfully!');
}

function manageSpeakersModal() {
    const modalSpeakerList = document.getElementById('modalSpeakerList');
    if (!modalSpeakerList) return;
    
    // Clear current list
    modalSpeakerList.innerHTML = '';
    
    if (speakerList.length === 0) {
        const listItem = document.createElement('li');
        listItem.className = 'list-group-item';
        listItem.textContent = 'No speakers registered';
        modalSpeakerList.appendChild(listItem);
    } else {
        speakerList.forEach((speaker, index) => {
            const listItem = document.createElement('li');
            listItem.className = 'list-group-item d-flex justify-content-between align-items-center';
            
            const nameSpan = document.createElement('span');
            nameSpan.textContent = speaker.name;
            listItem.appendChild(nameSpan);
            
            const deleteBtn = document.createElement('button');
            deleteBtn.className = 'btn btn-sm btn-danger';
            deleteBtn.innerHTML = '&times;'; // × symbol
            deleteBtn.onclick = function() { deleteSpeaker(index); };
            listItem.appendChild(deleteBtn);
            
            modalSpeakerList.appendChild(listItem);
        });
    }
    
    // Show the Bootstrap modal
    if (speakerModal) {
        speakerModal.show();
    } else if (typeof bootstrap !== 'undefined') {
        // Try to create the modal on demand if not already created
        const modalElement = document.getElementById('speakerModal');
        if (modalElement) {
            const modal = new bootstrap.Modal(modalElement);
            modal.show();
        } else {
            console.error('Modal element not found');
        }
    } else {
        alert('Speaker management modal could not be shown. Bootstrap may not be loaded.');
    }
}

function deleteSpeaker(index) {
    if (confirm('Are you sure you want to delete this speaker?')) {
        speakerList.splice(index, 1);
        localStorage.setItem('speakersList', JSON.stringify(speakerList));
        updateSpeakerDropdown();
        manageSpeakersModal(); // Refresh the modal
    }
}

// Download report
document.getElementById('downloadReport')?.addEventListener('click', function() {
    if (!audioBlob) {
        alert('Please analyze audio first.');
        return;
    }
    
    // In a real app, this would generate a report on the server
    alert('Report generation would happen here in a complete app.');
}); 