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
    statusDiv.textContent = '✅ Recording complete! You can now play it back below.';
    statusDiv.className = 'recording-status ready';
    
    // Auto-play the recording (if browser allows)
    setTimeout(() => {
        playbackAudio.play().catch(error => {
            console.log('Autoplay prevented:', error.message);
        });
    }, 500);
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
