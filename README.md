# 🎙️ Enhanced Voice Chat AI - 30 Days Voice Agents Challenge

A cutting-edge conversational AI interface with real-time voice interactions, smart UI animations, and comprehensive error handling - built during the 30 Days Voice Agents Challenge.

This isn't just another voice assistant - it's a **premium conversational AI** experience with:

- ✨ **Revolutionary Single-Button Interface** - Smart state-aware record button that adapts to your conversation flow
- 🎨 **Cinematic UI Design** - Glassmorphism effects, animated particles, and smooth state transitions
- 🧠 **Intelligent Fallback System** - Graceful degradation with contextual responses when services are unavailable  
- 📱 **Mobile-First Responsive** - Optimized for all devices with touch-friendly interactions
- 🔊 **Auto-Play Audio Responses** - Seamless voice conversations without manual intervention
- ⌨️ **Power User Shortcuts** - Keyboard controls for efficient interaction

## 🚀 Live Demo Experience

### Enhanced UI Features (Day 12 Revamp)

**🎤 Smart Record Button States:**
- **Idle**: Elegant blue gradient with subtle pulse rings
- **Recording**: Dramatic red gradient with energetic pulse animations  
- **Processing**: Smooth spinning loader with teal colors
- **Playing**: Gentle green gradient indicating response playback
- **Error**: Clear red warning state with recovery options

**💫 Visual Polish:**
- Animated background particles that respond to interaction
- Glassmorphism design with backdrop blur effects
- Smooth micro-animations and hover states
- Chat-style message bubbles with user/AI distinction
- Progressive enhancement for accessibility

## 🏗️ Technical Architecture

### Core Pipeline
```
Audio Input → STT (AssemblyAI) → LLM (Gemini) → TTS (Murf) → Audio Output
     ↓              ↓               ↓            ↓           ↓
Validation → Transcription → Context+History → Synthesis → Auto-play
```

### Technology Stack
- **Backend**: FastAPI (Python) with async/await patterns
- **Frontend**: Vanilla JavaScript with modern Web APIs
- **AI Services**: 
  - 🎯 **AssemblyAI** for speech-to-text
  - 🧠 **Google Gemini 1.5 Flash** for conversational AI
  - 🗣️ **Murf AI** for text-to-speech synthesis
- **Storage**: In-memory session management (20 message history)
- **Audio**: MediaRecorder API with optimal codec selection

### Smart Error Handling
- **3-tier fallback system** for each service (STT, LLM, TTS)
- **Contextual demo responses** when APIs are unavailable
- **Exponential backoff retry** with circuit breaker patterns
- **Graceful degradation** maintaining conversation flow

## 🎯 Key Features Breakdown

### Day 1-6: Foundation
- ✅ FastAPI server setup with health checks
- ✅ REST TTS API with Murf integration
- ✅ Interactive audio playback system
- ✅ MediaRecorder echo bot implementation
- ✅ File upload with validation
- ✅ AssemblyAI speech transcription

### Day 7-12: Advanced Features
- ✅ **Conversational Agent Pipeline** - Complete STT→LLM→TTS flow
- ✅ **Session Management** - Persistent chat history per conversation
- ✅ **Enhanced Error Handling** - Robust fallbacks and user feedback
- ✅ **Premium UI Overhaul** - Single button interface with animations
- ✅ **Auto-Play Audio** - Seamless voice responses
- ✅ **Mobile Optimization** - Touch-friendly responsive design

## 🛠️ Quick Start Guide

### Prerequisites
```bash
# Required
Python 3.8+ 
Modern browser (Chrome/Firefox/Safari/Edge)
Microphone access permissions

# Optional (for full functionality)
AssemblyAI API key
Google Gemini API key  
Murf AI API key
```

### Installation

1. **Clone and Setup**
   ```bash
   git clone <your-repo>
   cd voice-agents-project
   
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

2. **Environment Configuration**
   ```bash
   # Create .env file with your API keys
   cp .env.example .env
   
   # Edit .env with your keys:
   ASSEMBLYAI_API_KEY=your_assemblyai_key_here
   GEMINI_API_KEY=your_gemini_key_here
   MURF_API_KEY=your_murf_key_here
   ```

3. **Launch Application**
   ```bash
   # Option 1: Direct launch
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   
   # Option 2: Use startup script
   ./start.sh
   
   # Option 3: Alternative port if 8000 is busy
   uvicorn main:app --host 0.0.0.0 --port 8002 --reload
   ```

4. **Access Interface**
   ```
   🌐 Open: http://localhost:8000
   🎤 Grant microphone permissions when prompted
   ✨ Start chatting with your AI assistant!
   ```

## 📋 API Reference

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Main enhanced chat interface |
| `GET` | `/health` | Server status and version info |
| `POST` | `/agent/chat/{session_id}` | **Enhanced conversational pipeline** |
| `GET` | `/agent/chat/{session_id}/history` | Retrieve conversation history |
| `DELETE` | `/agent/chat/{session_id}/history` | Clear session history |
| `GET` | `/ui/status` | UI features and capabilities |

### Enhanced Chat Pipeline (`/agent/chat/{session_id}`)

**Request:**
```javascript
FormData: {
  audio_file: Blob  // WebM/WAV audio recording
}
```

**Response:**
```json
{
  "success": true,
  "session_id": "conv_1703123456_abc123",
  "transcribed_text": "Hello, how are you today?",
  "llm_response": "I'm doing great! How can I help you?",
  "audio_url": "https://murf-audio-url.mp3",
  "model": "gemini-1.5-flash-enhanced",
  "voice": "en-US-ken (Murf AI)",
  "chat_history_length": 2,
  "day": 12,
  "ui_version": "enhanced",
  "auto_play": true,
  "pipeline_status": "full_success",
  "ui_features": {
    "single_record_button": true,
    "auto_play_audio": true,
    "smooth_animations": true
  }
}
```

## 🎨 UI Enhancement Details

### Single Record Button Logic
```javascript
const RecordingState = {
    IDLE: 'idle',        // Ready to record
    RECORDING: 'recording',  // Currently recording
    PROCESSING: 'processing', // Sending to AI
    PLAYING: 'playing',   // AI response playing
    ERROR: 'error'       // Error state, click to retry
};
```

### Keyboard Shortcuts
- `Spacebar` - Start/stop recording
- `Escape` - Cancel current operation  
- `Ctrl+Delete` - Clear conversation history

### Responsive Breakpoints
- **Desktop** (>768px): Full sidebar + main content
- **Tablet** (768px): Collapsible sidebar
- **Mobile** (<480px): Single column with optimized button sizing

## 🔧 Configuration Options

### Audio Settings
```javascript
// MediaRecorder configuration
{
  audio: {
    echoCancellation: true,
    noiseSuppression: true, 
    autoGainControl: true,
    sampleRate: 44100
  }
}
```

### API Timeout & Retry Settings
```python
MAX_RETRIES = 3
RETRY_DELAY = 1.0  # seconds
AUDIO_SIZE_LIMIT = 10 * 1024 * 1024  # 10MB
CHAT_HISTORY_LIMIT = 20  # messages
```

## 🎯 Performance Benchmarks

- **Audio Processing**: < 2s for typical 10-second recordings
- **STT Response**: < 5s via AssemblyAI
- **LLM Generation**: < 3s via Gemini 1.5 Flash  
- **TTS Synthesis**: < 4s via Murf AI
- **Total Pipeline**: < 15s end-to-end
- **Memory Usage**: ~50MB with 20-message history
- **Mobile Performance**: 60fps animations on modern devices

## 🐛 Troubleshooting

### Common Issues

**🎤 Microphone Not Working**
```bash
# Check HTTPS requirement
# Chrome: Ensure localhost or HTTPS
# Check browser permissions in Settings > Privacy > Microphone
```

**🔌 Port Already in Use**
```bash
# Find and kill existing processes
lsof -ti:8000 | xargs kill -9
# Or use different port
uvicorn main:app --port 8002 --reload
```

**🔑 API Keys Missing**
```bash
# App will work in demo mode without keys
# Add real keys to .env for full functionality
ASSEMBLYAI_API_KEY=your_key_here
```

### Debug Mode
```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🌟 What's Next?

### Planned Enhancements (Days 13-30)
- 🌍 **Multi-language support** with automatic detection
- 👥 **Multi-user sessions** with voice identification
- 🎵 **Background noise filtering** with advanced audio processing  
- 🧠 **RAG integration** with document context
- 📊 **Analytics dashboard** with conversation insights
- 🎭 **Voice cloning** with custom personas
- 🔗 **API integrations** with calendars, email, etc.

## 🤝 Contributing

This project was built as part of the **30 Days Voice Agents Challenge**. While it's primarily a learning exercise, contributions and improvements are welcome!

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Format code
black main.py
```

## 📜 License & Credits

**Built with ❤️ during the 30 Days Voice Agents Challenge**

- **AssemblyAI** - Speech-to-text transcription
- **Google Gemini** - Conversational AI responses
- **Murf AI** - Text-to-speech synthesis
- **FastAPI** - Modern Python web framework
- **Inspiration** - Modern chat interfaces and voice UX patterns

---

