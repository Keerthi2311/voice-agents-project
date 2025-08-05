// Voice Agents Project - Day 3
// JavaScript for TTS audio playback functionality

console.log('🎤 Voice Agents Project Day 3 - TTS Audio Playback Loaded!');

// Initialize the app when page loads
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

function initializeApp() {
    console.log('✅ Day 3 Complete: FastAPI + TTS + Audio Playback working together!');
    
    // Update speed display when range changes
    const speedRange = document.getElementById('speedRange');
    const speedValue = document.getElementById('speedValue');
    
    if (speedRange && speedValue) {
        speedRange.addEventListener('input', function() {
            speedValue.textContent = this.value + 'x';
        });
    }
}

// NEW: Main TTS function for Day 3
async function generateSpeech() {
    const textInput = document.getElementById('textInput');
    const voiceSelect = document.getElementById('voiceSelect');
    const speedRange = document.getElementById('speedRange');
    const generateBtn = document.getElementById('generateBtn');
    const messageDiv = document.getElementById('message');
    const audioSection = document.getElementById('audioSection');
    const audioPlayer = document.getElementById('audioPlayer');
    const audioInfo = document.getElementById('audioInfo');
    
    // Get input values
    const text = textInput.value.trim();
    const voiceId = voiceSelect.value;
    const speed = parseFloat(speedRange.value);
    
    // Validate input
    if (!text) {
        showMessage('Please enter some text to convert to speech!', 'error');
        textInput.focus();
        return;
    }
    
    if (text.length > 1000) {
        showMessage('Text is too long! Please keep it under 1000 characters.', 'error');
        return;
    }
    
    try {
        // Show loading state
        generateBtn.disabled = true;
        generateBtn.innerHTML = '<span class="loading"></span>Generating Speech...';
        showMessage('Converting text to speech...', 'info');
        audioSection.classList.remove('show');
        
        // Prepare the request
        const requestData = {
            text: text,
            voice_id: voiceId,
            speed: speed
        };
        
        console.log('TTS Request:', requestData);
        
        // Make API call to our TTS endpoint
        const response = await fetch('/api/tts/generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestData)
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        console.log('TTS Response:', result);
        
        if (result.success && result.audio_url) {
            // Success! Show the audio player
            showMessage('Speech generated successfully! 🎉', 'success');
            
            // Set up audio player
            audioPlayer.src = result.audio_url;
            audioPlayer.load(); // Reload the audio element
            
            // Show audio section
            audioSection.classList.add('show');
            
            // Update audio info
            audioInfo.innerHTML = `
                <p><strong>Text:</strong> "${result.text}"</p>
                <p><strong>Voice:</strong> ${result.voice_id}</p>
                <p><strong>Audio URL:</strong> <a href="${result.audio_url}" target="_blank">Open Audio</a></p>
            `;
            
            // Auto-play the audio (with user gesture requirement)
            try {
                await audioPlayer.play();
                console.log('Audio started playing automatically');
            } catch (autoplayError) {
                console.log('Autoplay prevented by browser, user needs to click play');
                showMessage('Audio ready! Click the play button to listen. 🔊', 'success');
            }
            
        } else {
            // Handle API success but no audio URL
            showMessage(result.message || 'TTS generation completed, but no audio URL received.', 'error');
            console.error('TTS API returned success but no audio URL:', result);
        }
        
    } catch (error) {
        console.error('TTS Error:', error);
        showMessage(`Error generating speech: ${error.message}`, 'error');
        audioSection.classList.remove('show');
    } finally {
        // Reset button state
        generateBtn.disabled = false;
        generateBtn.innerHTML = '🎙️ Generate Speech';
    }
}

// Helper function to show messages
function showMessage(message, type = 'info') {
    const messageDiv = document.getElementById('message');
    if (!messageDiv) return;
    
    messageDiv.textContent = message;
    messageDiv.className = `message ${type}`;
    messageDiv.style.display = 'block';
    
    // Auto-hide info messages after 5 seconds
    if (type === 'info') {
        setTimeout(() => {
            messageDiv.style.display = 'none';
        }, 5000);
    }
}

// Day 1/2 functions - keeping for backward compatibility
async function testAPI() {
    try {
        const response = await fetch('/api/hello');
        const data = await response.json();
        
        const responseDiv = document.getElementById('response');
        responseDiv.style.display = 'block';
        responseDiv.innerHTML = `
            <strong>API Response:</strong><br>
            ${JSON.stringify(data, null, 2)}
        `;
        
        console.log('API Response:', data);
    } catch (error) {
        const responseDiv = document.getElementById('response');
        responseDiv.style.display = 'block';
        responseDiv.innerHTML = `<strong>Error:</strong> ${error.message}`;
        console.error('API Error:', error);
    }
}

function showMessage_old() {
    alert('JavaScript is working! 🎉');
    console.log('JavaScript test button clicked');
}

// Keep the old function name for backward compatibility
function showMessage() {
    showMessage_old();
}

// Add some sample text suggestions
function loadSampleText(sampleNumber) {
    const textInput = document.getElementById('textInput');
    const samples = [
        "Hello! This is Day 3 of the 30 Days Voice Agents challenge. I'm testing the text-to-speech functionality!",
        "Welcome to the future of voice technology. This AI-powered system can convert any text into natural-sounding speech.",
        "The quick brown fox jumps over the lazy dog. This sentence contains every letter in the English alphabet!",
        "Good morning! Today is a beautiful day to learn about artificial intelligence and voice synthesis technology."
    ];
    
    if (textInput && samples[sampleNumber]) {
        textInput.value = samples[sampleNumber];
        textInput.focus();
    }
}

// Add keyboard shortcut for generating speech (Ctrl/Cmd + Enter)
document.addEventListener('keydown', function(event) {
    if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
        event.preventDefault();
        generateSpeech();
    }
});

// ==========================================
// DAY 4: ECHO BOT FUNCTIONALITY
// ==========================================

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
            statusDiv.textContent = '❌ Your browser does not support audio recording.';
            statusDiv.className = 'recording-status error';
            startBtn.disabled = true;
        } else {
            statusDiv.textContent = '🎤 Ready to record! Click "Start Recording" to begin.';
            statusDiv.className = 'recording-status ready';
        }
    }
}

async function startRecording() {
    console.log('🔴 Starting recording...');
    
    const startBtn = document.getElementById('startRecordBtn');
    const stopBtn = document.getElementById('stopRecordBtn');
    const statusDiv = document.getElementById('recordingStatus');
    const playbackSection = document.getElementById('playbackSection');
    
    try {
        // Request microphone access
        stream = await navigator.mediaDevices.getUserMedia({ 
            audio: {
                echoCancellation: false,
                noiseSuppression: false,
                autoGainControl: false
            } 
        });
        
        // Reset recorded chunks
        recordedChunks = [];
        
        // Create MediaRecorder
        mediaRecorder = new MediaRecorder(stream, {
            mimeType: 'audio/webm;codecs=opus'
        });
        
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
        statusDiv.textContent = `❌ Error: ${error.message}. Please allow microphone access.`;
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

function processRecording() {
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
    const blob = new Blob(recordedChunks, { type: 'audio/webm' });
    const audioUrl = URL.createObjectURL(blob);
    
    console.log('✅ Audio blob created:', blob.size, 'bytes');
    
    // Set up playback
    playbackAudio.src = audioUrl;
    playbackAudio.load();
    
    // Update recording info
    const duration = calculateDuration(blob);
    const size = (blob.size / 1024).toFixed(2);
    recordingInfo.innerHTML = `
        <strong>Recording Details:</strong><br>
        📏 Size: ${size} KB<br>
        🕐 Format: WebM (Opus)<br>
        🎤 Ready for playback!
    `;
    
    // Set up download functionality
    downloadBtn.onclick = function() {
        const link = document.createElement('a');
        link.href = audioUrl;
        link.download = `echo-bot-recording-${new Date().toISOString().slice(0, 19)}.webm`;
        link.click();
    };
    
    // Show playback section
    playbackSection.style.display = 'block';
    
    // Update status
    statusDiv.textContent = '✅ Recording complete! You can now play it back below.';
    statusDiv.className = 'recording-status ready';
    
    // Auto-play the recording (if browser allows)
    setTimeout(() => {
        playbackAudio.play().catch(error => {
            console.log('Autoplay prevented:', error.message);
        });
    }, 500);
}

function calculateDuration(blob) {
    // This is an approximation - actual duration detection would require decoding
    // For WebM/Opus, roughly estimate based on file size
    return 'Unknown';
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

// Log success message
console.log('%c🎤 Welcome to Voice Agents Project Day 3!', 'color: #4CAF50; font-size: 18px; font-weight: bold;');
console.log('%cNew Feature: TTS Audio Playback! 🔊', 'color: #45a049; font-size: 14px;');
console.log('%cPress Ctrl+Enter to generate speech quickly!', 'color: #666; font-size: 12px;');
