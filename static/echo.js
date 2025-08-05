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
