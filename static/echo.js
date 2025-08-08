// Voice Agents Project - Echo Bot
// JavaScript for voice recording and playback functionality

console.log('🎙️ Echo Bot Loaded!');

let mediaRecorder;
let recordedChunks = [];
let stream;

// Initialize Echo Bot when page loads
document.addEventListener('DOMContentLoaded', function() {
    initializeEchoBot();
});

function initializeEchoBot() {
    console.log('🔊 Echo Bot initialized for Day 4!');
    
    const startBtn = document.getElementById('startRecordBtn');
    const stopBtn = document.getElementById('stopRecordBtn');
    const statusDiv = document.getElementById('recordingStatus');
    
    if (startBtn && stopBtn && statusDiv) {
        startBtn.addEventListener('click', startRecording);
        stopBtn.addEventListener('click', stopRecording);
        
        // Check for browser support
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            statusDiv.textContent = '❌ Your browser does not support audio recording. Please use Chrome, Firefox, or Edge.';
            statusDiv.className = 'recording-status error';
            startBtn.disabled = true;
            return;
        }
        
        // Check MediaRecorder support
        if (!window.MediaRecorder) {
            statusDiv.textContent = '❌ MediaRecorder not supported. Please update your browser.';
            statusDiv.className = 'recording-status error';
            startBtn.disabled = true;
            return;
        }
        
        // Check for HTTPS (required for getUserMedia in most browsers)
        if (location.protocol !== 'https:' && location.hostname !== 'localhost' && location.hostname !== '127.0.0.1') {
            statusDiv.textContent = '⚠️ HTTPS required for microphone access. Use localhost for testing.';
            statusDiv.className = 'recording-status error';
        } else {
            statusDiv.textContent = '🎤 Ready to record! Click "Start Recording" to begin.';
            statusDiv.className = 'recording-status ready';
        }
        
        // Log supported MIME types for debugging
        const supportedTypes = [
            'audio/webm;codecs=opus',
            'audio/webm',
            'audio/mp4',
            'audio/wav',
            'audio/ogg'
        ];
        
        console.log('📋 Supported audio formats:');
        supportedTypes.forEach(type => {
            console.log(`${type}: ${MediaRecorder.isTypeSupported(type)}`);
        });
    }
}

async function startRecording() {
    console.log('🔴 Starting recording...');
    
    const startBtn = document.getElementById('startRecordBtn');
    const stopBtn = document.getElementById('stopRecordBtn');
    const statusDiv = document.getElementById('recordingStatus');
    const playbackSection = document.getElementById('playbackSection');
    
    try {
        // Request microphone access with detailed error handling
        console.log('🎤 Requesting microphone access...');
        
        stream = await navigator.mediaDevices.getUserMedia({ 
            audio: {
                echoCancellation: false,
                noiseSuppression: false,
                autoGainControl: false
            } 
        });
        
        console.log('✅ Microphone access granted');
        
        // Update status to show microphone is working
        statusDiv.textContent = '🎤 Microphone connected! Preparing to record...';
        statusDiv.className = 'recording-status ready';
        
        // Reset recorded chunks
        recordedChunks = [];
        
        // Create MediaRecorder with fallback MIME types
        let options = { mimeType: 'audio/webm;codecs=opus' };
        
        // Check for supported MIME types and use fallbacks
        if (!MediaRecorder.isTypeSupported('audio/webm;codecs=opus')) {
            if (MediaRecorder.isTypeSupported('audio/webm')) {
                options = { mimeType: 'audio/webm' };
            } else if (MediaRecorder.isTypeSupported('audio/mp4')) {
                options = { mimeType: 'audio/mp4' };
            } else if (MediaRecorder.isTypeSupported('audio/wav')) {
                options = { mimeType: 'audio/wav' };
            } else {
                // Use default (no mimeType specified)
                options = {};
            }
        }
        
        console.log('Using MediaRecorder options:', options);
        
        // Create MediaRecorder
        mediaRecorder = new MediaRecorder(stream, options);
        
        // Handle data available event
        mediaRecorder.ondataavailable = function(event) {
            if (event.data.size > 0) {
                recordedChunks.push(event.data);
                console.log('📊 Recording chunk received:', event.data.size, 'bytes');
            }
        };
        
        // Handle recording stop event
        mediaRecorder.onstop = function() {
            console.log('⏹️ Recording stopped');
            processRecording();
            
            // Stop all tracks to release microphone
            stream.getTracks().forEach(track => track.stop());
        };
        
        // Start recording
        mediaRecorder.start(1000); // Collect data every second
        
        // Update UI
        startBtn.disabled = true;
        stopBtn.disabled = false;
        statusDiv.textContent = '🔴 Recording... Click "Stop Recording" when done.';
        statusDiv.className = 'recording-status recording';
        playbackSection.style.display = 'none';
        
        console.log('✅ Recording started successfully');
        
    } catch (error) {
        console.error('❌ Error starting recording:', error);
        
        let errorMessage = '';
        
        if (error.name === 'NotAllowedError') {
            errorMessage = '❌ Microphone access denied. Please allow microphone access and try again.';
        } else if (error.name === 'NotFoundError') {
            errorMessage = '❌ No microphone found. Please connect a microphone and try again.';
        } else if (error.name === 'NotSupportedError') {
            errorMessage = '❌ Audio recording not supported in this browser. Try Chrome, Firefox, or Edge.';
        } else if (error.name === 'AbortError') {
            errorMessage = '❌ Recording was aborted. Please try again.';
        } else {
            errorMessage = `❌ Error: ${error.message}. Please check your microphone settings.`;
        }
        
        statusDiv.textContent = errorMessage;
        statusDiv.className = 'recording-status error';
        
        // Reset button states
        startBtn.disabled = false;
        stopBtn.disabled = true;
    }
}

function stopRecording() {
    console.log('⏹️ Stopping recording...');
    
    const startBtn = document.getElementById('startRecordBtn');
    const stopBtn = document.getElementById('stopRecordBtn');
    const statusDiv = document.getElementById('recordingStatus');
    
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
        
        // Update UI immediately
        statusDiv.textContent = '🔄 Processing recording...';
        statusDiv.className = 'recording-status';
        startBtn.disabled = false;
        stopBtn.disabled = true;
    }
}

async function processRecording() {
    console.log('🔄 Processing recording...', recordedChunks.length, 'chunks');
    
    const statusDiv = document.getElementById('recordingStatus');
    const playbackSection = document.getElementById('playbackSection');
    const playbackAudio = document.getElementById('playbackAudio');
    const recordingInfo = document.getElementById('recordingInfo');
    const downloadBtn = document.getElementById('downloadBtn');
    
    if (recordedChunks.length === 0) {
        statusDiv.textContent = '❌ No audio data recorded. Please try again.';
        statusDiv.className = 'recording-status error';
        return;
    }
    
    // Create blob from recorded chunks
    const blob = new Blob(recordedChunks, { type: mediaRecorder.mimeType || 'audio/webm' });
    const audioUrl = URL.createObjectURL(blob);
    
    console.log('✅ Audio blob created:', blob.size, 'bytes', 'Type:', blob.type);
    
    // Set up playback
    playbackAudio.src = audioUrl;
    playbackAudio.load();
    
    // Update recording info with actual format
    const size = (blob.size / 1024).toFixed(2);
    const format = blob.type || 'Unknown format';
    recordingInfo.innerHTML = `
        <strong>Recording Details:</strong><br>
        📏 Size: ${size} KB<br>
        🕐 Format: ${format}<br>
        🎤 Ready for playback!
    `;
    
    // Set up download functionality with proper file extension
    downloadBtn.onclick = function() {
        const link = document.createElement('a');
        link.href = audioUrl;
        
        // Determine file extension based on MIME type
        let extension = '.webm';
        if (format.includes('mp4')) extension = '.mp4';
        else if (format.includes('wav')) extension = '.wav';
        else if (format.includes('ogg')) extension = '.ogg';
        
        link.download = `echo-bot-recording-${new Date().toISOString().slice(0, 19)}${extension}`;
        link.click();
    };
    
    // Show playback section
    playbackSection.style.display = 'block';
    
    // Update status
    statusDiv.textContent = '✅ Recording complete! Processing with Murf AI...';
    statusDiv.className = 'recording-status uploading';
    
    // Day 7: Echo Bot v2 - Transcribe and replay with Murf voice
    await echoWithMurfVoice(blob);
    
    // Auto-play will happen after Murf audio is ready
}

// Day 7: Echo Bot v2 - Transcribe and replay with Murf voice
async function echoWithMurfVoice(audioBlob) {
    const statusDiv = document.getElementById('recordingStatus');
    const playbackAudio = document.getElementById('playbackAudio');
    const recordingInfo = document.getElementById('recordingInfo');
    const transcriptionSection = document.getElementById('transcriptionSection') || createTranscriptionSection();
    const transcriptionStatus = document.getElementById('transcriptionStatus') || createTranscriptionStatusDiv();
    
    try {
        console.log('🎯 Day 7: Starting Echo Bot v2 with Murf voice...');
        
        // Update status
        statusDiv.textContent = '🎯 Transcribing and generating Murf voice...';
        statusDiv.className = 'recording-status uploading';
        
        transcriptionStatus.textContent = '🎤 Processing your voice with AI...';
        transcriptionStatus.className = 'transcription-status uploading';
        
        // Show transcription section
        transcriptionSection.style.display = 'block';
        
        const formData = new FormData();
        formData.append('audio', audioBlob, 'recording.webm');

        const response = await fetch('/tts/echo', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (response.ok && result.success) {
            console.log('✅ Day 7: Echo Bot v2 successful!', result);
            
            // Update status
            statusDiv.textContent = '✅ Echo Bot v2 complete! Listen to your Murf voice below.';
            statusDiv.className = 'recording-status success';
            
            transcriptionStatus.textContent = '✅ Transcription and voice generation completed!';
            transcriptionStatus.className = 'transcription-status success';
            
            // Display what was transcribed
            const transcriptionText = document.getElementById('transcriptionText');
            if (transcriptionText) {
                transcriptionText.innerHTML = `
                    <div class="transcription-result">
                        <h4>📝 What you said:</h4>
                        <p>"${result.original_text}"</p>
                    </div>
                `;
            }
            
            // Update recording info
            recordingInfo.innerHTML = `
                <strong>Echo Bot v2 (Day 7):</strong><br>
                🎤 Original: Your voice<br>
                🤖 Murf Voice: ${result.voice_id}<br>
                📝 Transcribed: ${result.original_text.substring(0, 50)}...<br>
                🎯 Status: Ready to play!
            `;
            
            // Set the Murf-generated audio URL
            playbackAudio.src = result.audio_url;
            playbackAudio.load();
            
            // Auto-play the Murf audio
            setTimeout(() => {
                playbackAudio.play().catch(error => {
                    console.log('Autoplay prevented:', error.message);
                    statusDiv.textContent = '✅ Echo Bot v2 ready! Click play to hear your voice in Murf AI.';
                });
            }, 1000);
            
        } else {
            throw new Error(result.detail || 'Echo Bot v2 failed');
        }

    } catch (error) {
        console.error('❌ Day 7 Echo Bot v2 error:', error);
        statusDiv.textContent = `❌ Echo Bot v2 failed: ${error.message}`;
        statusDiv.className = 'recording-status error';
        
        transcriptionStatus.textContent = `❌ Processing failed: ${error.message}`;
        transcriptionStatus.className = 'transcription-status error';
    }
}

// Day 5: Upload audio to server
async function uploadAudioToServer(audioBlob) {
    const statusDiv = document.getElementById('recordingStatus');
    const uploadStatus = document.getElementById('uploadStatus') || createUploadStatusDiv();
    
    try {
        uploadStatus.textContent = '📤 Uploading audio to server...';
        uploadStatus.className = 'upload-status uploading';
        
        const formData = new FormData();
        formData.append('audio', audioBlob, 'recording.webm');

        const response = await fetch('/upload-audio', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (response.ok && result.success) {
            uploadStatus.textContent = `✅ Upload successful! File: ${result.filename} (${result.size_mb}MB)`;
            uploadStatus.className = 'upload-status success';
            console.log('Upload details:', result);
        } else {
            throw new Error(result.detail || 'Upload failed');
        }

    } catch (error) {
        console.error('Upload error:', error);
        uploadStatus.textContent = `❌ Upload failed: ${error.message}`;
        uploadStatus.className = 'upload-status error';
    }
}

// Day 6: Transcribe audio
async function transcribeAudio(audioBlob) {
    const transcriptionStatus = document.getElementById('transcriptionStatus') || createTranscriptionStatusDiv();
    const transcriptionSection = document.getElementById('transcriptionSection') || createTranscriptionSection();
    
    try {
        transcriptionStatus.textContent = '🎯 Transcribing audio...';
        transcriptionStatus.className = 'transcription-status uploading';
        
        const formData = new FormData();
        formData.append('audio', audioBlob, 'recording.webm');

        const response = await fetch('/transcribe/file', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (response.ok && result.success) {
            displayTranscription(result);
            transcriptionStatus.textContent = '✅ Transcription completed successfully!';
            transcriptionStatus.className = 'transcription-status success';
        } else {
            throw new Error(result.detail || 'Transcription failed');
        }

    } catch (error) {
        console.error('Transcription error:', error);
        transcriptionStatus.textContent = `❌ Transcription failed: ${error.message}`;
        transcriptionStatus.className = 'transcription-status error';
    }
}

// Display transcription results
function displayTranscription(result) {
    const transcriptionText = document.getElementById('transcriptionText') || createTranscriptionTextDiv();
    const transcriptionDetails = document.getElementById('transcriptionDetails') || createTranscriptionDetailsDiv();
    
    // Show transcription section
    const transcriptionSection = document.getElementById('transcriptionSection');
    if (transcriptionSection) {
        transcriptionSection.style.display = 'block';
    }
    
    // Display the transcribed text
    if (result.transcript && result.transcript.trim()) {
        transcriptionText.innerHTML = `<strong>Transcript:</strong><br>"${result.transcript}"`;
    } else {
        transcriptionText.innerHTML = '<strong>Transcript:</strong><br><em>No speech detected in the recording.</em>';
    }
    
    // Display transcription details
    transcriptionDetails.innerHTML = `
        <strong>Details:</strong><br>
        📊 Confidence: ${Math.round((result.confidence || 0.95) * 100)}%<br>
        ⏱️ Duration: ${result.audio_duration || 'N/A'}s<br>
        📝 Words: ${result.words_count || 0}<br>
        🔥 Status: ${result.status || 'completed'}
    `;
}

// Helper function to create upload status div if it doesn't exist
function createUploadStatusDiv() {
    const uploadStatus = document.createElement('div');
    uploadStatus.id = 'uploadStatus';
    uploadStatus.className = 'upload-status';
    
    const playbackSection = document.getElementById('playbackSection');
    if (playbackSection) {
        playbackSection.appendChild(uploadStatus);
    }
    
    return uploadStatus;
}

// Helper function to create transcription status div if it doesn't exist
function createTranscriptionStatusDiv() {
    const transcriptionStatus = document.createElement('div');
    transcriptionStatus.id = 'transcriptionStatus';
    transcriptionStatus.className = 'transcription-status';
    
    const playbackSection = document.getElementById('playbackSection');
    if (playbackSection) {
        playbackSection.appendChild(transcriptionStatus);
    }
    
    return transcriptionStatus;
}

// Helper function to create transcription section if it doesn't exist
function createTranscriptionSection() {
    const transcriptionSection = document.createElement('div');
    transcriptionSection.id = 'transcriptionSection';
    transcriptionSection.style.display = 'none';
    transcriptionSection.innerHTML = `
        <h3>📝 Transcription Results</h3>
        <div id="transcriptionText" class="transcription-text"></div>
        <div id="transcriptionDetails" class="transcription-details"></div>
    `;
    
    const playbackSection = document.getElementById('playbackSection');
    if (playbackSection) {
        playbackSection.appendChild(transcriptionSection);
    }
    
    return transcriptionSection;
}

// Helper function to create transcription text div if it doesn't exist
function createTranscriptionTextDiv() {
    const transcriptionText = document.createElement('div');
    transcriptionText.id = 'transcriptionText';
    transcriptionText.className = 'transcription-text';
    
    return transcriptionText;
}

// Helper function to create transcription details div if it doesn't exist
function createTranscriptionDetailsDiv() {
    const transcriptionDetails = document.createElement('div');
    transcriptionDetails.id = 'transcriptionDetails';
    transcriptionDetails.className = 'transcription-details';
    
    return transcriptionDetails;
}

// Add visual feedback for microphone access
function showMicrophoneAccess(granted) {
    const statusDiv = document.getElementById('recordingStatus');
    if (granted) {
        statusDiv.textContent = '🎤 Microphone access granted! Ready to record.';
        statusDiv.className = 'recording-status ready';
    } else {
        statusDiv.textContent = '❌ Microphone access denied. Please enable microphone permissions.';
        statusDiv.className = 'recording-status error';
    }
}
