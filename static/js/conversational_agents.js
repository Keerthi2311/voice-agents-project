// Day 12: Enhanced Conversational Agent JavaScript
// Single record button with state management and animations

console.log('🎙️ Day 12: Enhanced Conversational Agent Loaded!');

let mediaRecorder;
let recordedChunks = [];
let stream;
let currentSessionId;
let isRecording = false;
let isProcessing = false;

// Recording states
const RecordingState = {
    IDLE: 'idle',
    RECORDING: 'recording',
    PROCESSING: 'processing',
    PLAYING: 'playing',
    ERROR: 'error'
};

let currentState = RecordingState.IDLE;

// Initialize the enhanced conversational agent
document.addEventListener('DOMContentLoaded', function() {
    initializeConversationalAgent();
    generateSessionId();
});

function initializeConversationalAgent() {
    console.log('🚀 Day 12: Initializing Enhanced Conversational Agent...');
    
    const recordBtn = document.getElementById('recordBtn');
    const statusDiv = document.getElementById('conversationStatus');
    
    if (recordBtn && statusDiv) {
        // Single button handler that manages all states
        recordBtn.addEventListener('click', handleRecordButtonClick);
        
        // Check for browser support
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            updateStatus('❌ Your browser does not support audio recording. Please use Chrome, Firefox, or Edge.', 'error');
            recordBtn.disabled = true;
            return;
        }
        
        // Check MediaRecorder support
        if (!window.MediaRecorder) {
            updateStatus('❌ MediaRecorder not supported. Please update your browser.', 'error');
            recordBtn.disabled = true;
            return;
        }
        
        // Initial state
        updateButtonState(RecordingState.IDLE);
        updateStatus('🎤 Ready to chat! Click the microphone to start recording.', 'ready');
        
        console.log('✅ Enhanced Conversational Agent initialized successfully');
    }
}

function generateSessionId() {
    // Generate a unique session ID for this conversation
    currentSessionId = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    console.log(`📱 Generated session ID: ${currentSessionId}`);
}

async function handleRecordButtonClick() {
    console.log(`🔘 Record button clicked - Current state: ${currentState}`);
    
    switch (currentState) {
        case RecordingState.IDLE:
            await startRecording();
            break;
        case RecordingState.RECORDING:
            await stopRecording();
            break;
        case RecordingState.PROCESSING:
            // Button disabled during processing
            console.log('⏳ Currently processing, please wait...');
            break;
        case RecordingState.PLAYING:
            // Stop audio playback and return to idle
            stopAudioPlayback();
            updateButtonState(RecordingState.IDLE);
            updateStatus('🎤 Ready for your next message!', 'ready');
            break;
        case RecordingState.ERROR:
            // Reset to idle state
            updateButtonState(RecordingState.IDLE);
            updateStatus('🎤 Ready to try again!', 'ready');
            break;
    }
}

async function startRecording() {
    console.log('🔴 Starting enhanced recording...');
    
    try {
        updateButtonState(RecordingState.RECORDING);
        updateStatus('🔴 Recording... Click again to stop.', 'recording');
        
        // Request microphone access with enhanced settings
        stream = await navigator.mediaDevices.getUserMedia({ 
            audio: {
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true,
                sampleRate: 44100
            } 
        });
        
        // Reset recorded chunks
        recordedChunks = [];
        
        // Create MediaRecorder with best available format
        let options = { mimeType: 'audio/webm;codecs=opus' };
        
        if (!MediaRecorder.isTypeSupported('audio/webm;codecs=opus')) {
            if (MediaRecorder.isTypeSupported('audio/webm')) {
                options = { mimeType: 'audio/webm' };
            } else if (MediaRecorder.isTypeSupported('audio/mp4')) {
                options = { mimeType: 'audio/mp4' };
            } else {
                options = {};
            }
        }
        
        mediaRecorder = new MediaRecorder(stream, options);
        
        // Handle data available
        mediaRecorder.ondataavailable = function(event) {
            if (event.data.size > 0) {
                recordedChunks.push(event.data);
                console.log('🎵 Recording chunk received:', event.data.size, 'bytes');
            }
        };
        
        // Handle recording stop
        mediaRecorder.onstop = function() {
            console.log('⏹️ Recording stopped, processing...');
            processRecording();
            
            // Stop all tracks to release microphone
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
            }
        };
        
        // Start recording
        mediaRecorder.start(1000);
        isRecording = true;
        
        console.log('✅ Enhanced recording started successfully');
        
    } catch (error) {
        console.error('❌ Enhanced recording start failed:', error);
        handleRecordingError(error);
    }
}

async function stopRecording() {
    console.log('⏹️ Stopping enhanced recording...');
    
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        updateButtonState(RecordingState.PROCESSING);
        updateStatus('🔄 Processing your message...', 'processing');
        
        mediaRecorder.stop();
        isRecording = false;
    }
}

async function processRecording() {
    console.log('🔄 Processing enhanced recording...', recordedChunks.length, 'chunks');
    
    if (recordedChunks.length === 0) {
        updateStatus('❌ No audio recorded. Please try again.', 'error');
        updateButtonState(RecordingState.ERROR);
        return;
    }
    
    try {
        // Create blob from recorded chunks
        const blob = new Blob(recordedChunks, { 
            type: mediaRecorder.mimeType || 'audio/webm' 
        });
        
        console.log('📦 Audio blob created:', blob.size, 'bytes');
        
        // Update status for upload
        updateStatus('📤 Sending to AI assistant...', 'processing');
        
        // Create FormData and upload to conversational agent
        const formData = new FormData();
        formData.append('audio_file', blob, 'recording.webm');
        
        // Send to enhanced conversational agent endpoint
        const response = await fetch(`/agent/chat/${currentSessionId}`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`Server error: ${response.status} - ${response.statusText}`);
        }
        
        const result = await response.json();
        console.log('🤖 Enhanced agent response:', result);
        
        if (result.success) {
            // Display conversation results
            displayConversationResult(result);
            
            // Auto-play audio response if available
            if (result.audio_url && result.auto_play) {
                await playAudioResponse(result.audio_url);
            } else {
                // No audio or auto-play disabled
                updateButtonState(RecordingState.IDLE);
                updateStatus('✅ Response ready! Click mic for next message.', 'ready');
            }
        } else {
            throw new Error(result.message || 'Conversation processing failed');
        }
        
    } catch (error) {
        console.error('❌ Enhanced processing failed:', error);
        updateStatus(`❌ Error: ${error.message}`, 'error');
        updateButtonState(RecordingState.ERROR);
    }
}

function displayConversationResult(result) {
    console.log('📺 Displaying enhanced conversation result');
    
    const conversationHistory = document.getElementById('conversationHistory');
    const transcriptionDiv = document.getElementById('transcriptionResult');
    const responseDiv = document.getElementById('responseResult');
    
    if (conversationHistory) {
        conversationHistory.style.display = 'block';
    }
    
    // Display what user said
    if (transcriptionDiv) {
        transcriptionDiv.innerHTML = `
            <div class="user-message">
                <strong>🗣️ You said:</strong>
                <p>"${result.transcribed_text}"</p>
            </div>
        `;
    }
    
    // Display AI response
    if (responseDiv) {
        responseDiv.innerHTML = `
            <div class="ai-response">
                <strong>🤖 AI Assistant:</strong>
                <p>"${result.llm_response}"</p>
                ${result.error_context ? `<small class="error-note">⚠️ ${result.pipeline_status}</small>` : ''}
            </div>
        `;
    }
    
    // Update conversation stats if element exists
    const statsDiv = document.getElementById('conversationStats');
    if (statsDiv) {
        statsDiv.innerHTML = `
            <div class="conversation-stats">
                <span>💬 Messages: ${result.chat_history_length}</span>
                <span>🎯 Model: ${result.model}</span>
                <span>🔊 Voice: ${result.voice}</span>
                ${result.day ? `<span>📅 Day: ${result.day}</span>` : ''}
            </div>
        `;
    }
}

async function playAudioResponse(audioUrl) {
    console.log('🔊 Playing enhanced audio response');
    
    try {
        updateButtonState(RecordingState.PLAYING);
        updateStatus('🔊 Playing AI response... Click to stop.', 'playing');
        
        // Create or get hidden audio element
        let audioPlayer = document.getElementById('hiddenAudioPlayer');
        if (!audioPlayer) {
            audioPlayer = document.createElement('audio');
            audioPlayer.id = 'hiddenAudioPlayer';
            audioPlayer.style.display = 'none';
            document.body.appendChild(audioPlayer);
        }
        
        // Set up audio events
        audioPlayer.onended = function() {
            console.log('🔊 Audio playback completed');
            updateButtonState(RecordingState.IDLE);
            updateStatus('✅ Response played! Ready for your next message.', 'ready');
        };
        
        audioPlayer.onerror = function(error) {
            console.error('🔊 Audio playback failed:', error);
            updateButtonState(RecordingState.IDLE);
            updateStatus('⚠️ Audio playback failed, but response is ready above.', 'ready');
        };
        
        // Load and play audio
        audioPlayer.src = audioUrl;
        audioPlayer.load();
        
        try {
            await audioPlayer.play();
            console.log('✅ Enhanced audio playback started');
        } catch (playError) {
            console.log('🔇 Autoplay prevented, user interaction required');
            updateButtonState(RecordingState.IDLE);
            updateStatus('🔇 Audio ready! Enable autoplay or click play manually.', 'ready');
        }
        
    } catch (error) {
        console.error('🔊 Enhanced audio setup failed:', error);
        updateButtonState(RecordingState.IDLE);
        updateStatus('⚠️ Audio unavailable, but response is shown above.', 'ready');
    }
}

function stopAudioPlayback() {
    const audioPlayer = document.getElementById('hiddenAudioPlayer');
    if (audioPlayer) {
        audioPlayer.pause();
        audioPlayer.currentTime = 0;
    }
}

function updateButtonState(state) {
    currentState = state;
    const recordBtn = document.getElementById('recordBtn');
    if (!recordBtn) return;
    
    // Remove all state classes
    recordBtn.classList.remove('idle', 'recording', 'processing', 'playing', 'error');
    
    // Add current state class
    recordBtn.classList.add(state);
    
    // Update button content and properties
    switch (state) {
        case RecordingState.IDLE:
            recordBtn.innerHTML = '<span class="record-icon">🎤</span><span class="record-text">Start Recording</span>';
            recordBtn.disabled = false;
            break;
        case RecordingState.RECORDING:
            recordBtn.innerHTML = '<span class="record-icon recording-pulse">🔴</span><span class="record-text">Stop Recording</span>';
            recordBtn.disabled = false;
            break;
        case RecordingState.PROCESSING:
            recordBtn.innerHTML = '<span class="record-icon processing-spin">⏳</span><span class="record-text">Processing...</span>';
            recordBtn.disabled = true;
            break;
        case RecordingState.PLAYING:
            recordBtn.innerHTML = '<span class="record-icon">🔊</span><span class="record-text">Playing Response</span>';
            recordBtn.disabled = false;
            break;
        case RecordingState.ERROR:
            recordBtn.innerHTML = '<span class="record-icon">⚠️</span><span class="record-text">Try Again</span>';
            recordBtn.disabled = false;
            break;
    }
    
    console.log(`🔘 Button state updated to: ${state}`);
}

function updateStatus(message, type = 'info') {
    const statusDiv = document.getElementById('conversationStatus');
    if (!statusDiv) return;
    
    statusDiv.textContent = message;
    statusDiv.className = `conversation-status ${type}`;
    
    // Add animation class for visual feedback
    statusDiv.classList.add('status-update');
    setTimeout(() => {
        statusDiv.classList.remove('status-update');
    }, 300);
    
    console.log(`📢 Status updated: ${message} (${type})`);
}

function handleRecordingError(error) {
    console.error('🚨 Enhanced recording error:', error);
    
    let errorMessage = '';
    
    if (error.name === 'NotAllowedError') {
        errorMessage = '❌ Microphone access denied. Please allow microphone access and try again.';
    } else if (error.name === 'NotFoundError') {
        errorMessage = '❌ No microphone found. Please connect a microphone and try again.';
    } else if (error.name === 'NotSupportedError') {
        errorMessage = '❌ Audio recording not supported. Try Chrome, Firefox, or Edge.';
    } else if (error.name === 'AbortError') {
        errorMessage = '❌ Recording was aborted. Please try again.';
    } else {
        errorMessage = `❌ Recording error: ${error.message}`;
    }
    
    updateStatus(errorMessage, 'error');
    updateButtonState(RecordingState.ERROR);
}

// Utility function to clear conversation history
async function clearConversationHistory() {
    try {
        const response = await fetch(`/agent/chat/${currentSessionId}/history`, {
            method: 'DELETE'
        });
        
        if (response.ok) {
            console.log('🗑️ Conversation history cleared');
            
            // Clear UI elements
            const conversationHistory = document.getElementById('conversationHistory');
            if (conversationHistory) {
                conversationHistory.style.display = 'none';
            }
            
            const transcriptionDiv = document.getElementById('transcriptionResult');
            if (transcriptionDiv) {
                transcriptionDiv.innerHTML = '';
            }
            
            const responseDiv = document.getElementById('responseResult');
            if (responseDiv) {
                responseDiv.innerHTML = '';
            }
            
            updateStatus('🗑️ Conversation cleared! Ready to start fresh.', 'ready');
            updateButtonState(RecordingState.IDLE);
            
            // Generate new session ID
            generateSessionId();
        }
    } catch (error) {
        console.error('🗑️ Failed to clear history:', error);
        updateStatus('⚠️ Failed to clear history, but you can continue chatting.', 'ready');
    }
}

// Add keyboard shortcuts for enhanced UX
document.addEventListener('keydown', function(event) {
    // Spacebar to toggle recording (when not typing in input fields)
    if (event.code === 'Space' && !['INPUT', 'TEXTAREA'].includes(event.target.tagName)) {
        event.preventDefault();
        if (currentState === RecordingState.IDLE || currentState === RecordingState.RECORDING) {
            handleRecordButtonClick();
        }
    }
    
    // Escape key to stop/reset
    if (event.key === 'Escape') {
        if (currentState === RecordingState.RECORDING) {
            stopRecording();
        } else if (currentState === RecordingState.PLAYING) {
            stopAudioPlayback();
            updateButtonState(RecordingState.IDLE);
            updateStatus('🎤 Ready for your next message!', 'ready');
        }
    }
    
    // Ctrl/Cmd + Delete to clear conversation
    if ((event.ctrlKey || event.metaKey) && event.key === 'Delete') {
        event.preventDefault();
        clearConversationHistory();
    }
});

// Add visual feedback for microphone access
document.addEventListener('DOMContentLoaded', function() {
    // Check for HTTPS requirement
    if (location.protocol !== 'https:' && location.hostname !== 'localhost' && location.hostname !== '127.0.0.1') {
        updateStatus('⚠️ HTTPS required for microphone access. Use localhost for development.', 'error');
        const recordBtn = document.getElementById('recordBtn');
        if (recordBtn) {
            recordBtn.disabled = true;
        }
    }
});

// Log success message
console.log('%c🎙️ Day 12: Enhanced Conversational Agent Ready!', 'color: #4CAF50; font-size: 18px; font-weight: bold;');
console.log('%c✨ New Features: Single Record Button, Auto-play, Smooth Animations!', 'color: #45a049; font-size: 14px;');
console.log('%c🎹 Shortcuts: Spacebar (record), Escape (stop), Ctrl+Delete (clear)', 'color: #666; font-size: 12px;');