// Voice Agents Project - TTS Generator
// JavaScript for text-to-speech functionality

console.log('🔊 TTS Generator Loaded!');

// Initialize the TTS app when page loads
document.addEventListener('DOMContentLoaded', function() {
    initializeTTS();
});

function initializeTTS() {
    console.log('✅ TTS Generator initialized!');
    
    // Update speed display when range changes
    const speedRange = document.getElementById('speedRange');
    const speedValue = document.getElementById('speedValue');
    
    if (speedRange && speedValue) {
        speedRange.addEventListener('input', function() {
            speedValue.textContent = this.value + 'x';
        });
    }
}

// Main TTS function
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

// Add keyboard shortcut for generating speech (Ctrl/Cmd + Enter)
document.addEventListener('keydown', function(event) {
    if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
        event.preventDefault();
        generateSpeech();
    }
});

// Add some sample text suggestions
function loadSampleText(sampleNumber) {
    const textInput = document.getElementById('textInput');
    const samples = [
        "Hello! This is the Voice Agents TTS generator. I'm testing the text-to-speech functionality!",
        "Welcome to the future of voice technology. This AI-powered system can convert any text into natural-sounding speech.",
        "The quick brown fox jumps over the lazy dog. This sentence contains every letter in the English alphabet!",
        "Good morning! Today is a beautiful day to learn about artificial intelligence and voice synthesis technology."
    ];
    
    if (textInput && samples[sampleNumber]) {
        textInput.value = samples[sampleNumber];
        textInput.focus();
    }
}
