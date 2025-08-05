# 🎤 Voice Agents Project - 30 Days Challenge

## Day 3: TTS Audio Playback

Welcome to the Voice Agents Project! This is Day 3 of the 30 Days of Voice Agents challenge.

**🔗 GitHub Repository**: https://github.com/Keerthi2311/voice-agents-project

### 🚀 Project Overview

This project demonstrates a complete full-stack setup with:
- **Backend**: FastAPI (Python) - High-performance web framework
- **Frontend**: HTML5 + Vanilla JavaScript - Responsive and modern UI
- **APIs**: RESTful endpoints for voice agent functionality
- **TTS Integration**: Murf AI Text-to-Speech API integration
- **🆕 Audio Playback**: Client-side audio player with TTS integration

### 📁 Project Structure

```
voice-agents-project/
├── main.py                 # FastAPI application with TTS
├── templates/
│   └── index.html          # Main HTML page
├── static/
│   └── app.js             # JavaScript functionality
├── requirements.txt        # Python dependencies
├── .env                   # Environment variables (API keys)
├── .gitignore            # Git ignore rules
└── README.md              # This file
```

### 🛠️ Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Keerthi2311/voice-agents-project.git
   cd voice-agents-project
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables**:
   ```bash
   # Copy .env file and add your Murf API key
   cp .env .env.local
   # Edit .env.local and add your MURF_API_KEY
   ```

4. **Run the Server**:
   ```bash
   python main.py
   ```
   Or using uvicorn directly:
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8001
   ```

5. **Access the Application**:
   - Main page: `http://localhost:8001`
   - API Documentation: `http://localhost:8001/docs`

### 🔗 API Endpoints

#### Day 1 Endpoints:
- `GET /` - Serves the main HTML page
- `GET /api/hello` - Sample API endpoint
- `GET /api/voice-status` - Voice agent status endpoint

#### 🆕 Day 2 Endpoints (TTS):
- `POST /api/tts/generate` - Generate speech from text using Murf AI
- `GET /api/tts/voices` - Get list of available TTS voices
- `GET /docs` - Interactive API documentation (Swagger UI)

### ✨ Features

#### 🆕 Day 3 Features:
- **Interactive TTS Interface**: User-friendly text input and voice controls
- **Audio Playback**: HTML5 audio player with generated speech
- **Voice Selection**: Multiple voice options (Davis, Jenny, Alice, Jack)
- **Speed Control**: Adjustable speech speed (0.5x to 2.0x)
- **Real-time Feedback**: Loading states and success/error messages
- **Keyboard Shortcuts**: Ctrl+Enter to generate speech quickly
- **Input Validation**: Text length and content validation

#### Day 2 Features:
- **REST TTS API**: Convert text to speech using Murf AI
- **Multiple Voices**: Support for different voice options
- **Secure API Keys**: Environment variable configuration
- **Interactive API Docs**: FastAPI's built-in Swagger UI
- **Error Handling**: Comprehensive error responses

#### Day 1 Features:
- **Modern UI**: Clean, responsive design
- **Interactive Elements**: API testing buttons
- **FastAPI Backend**: High-performance Python server

### 🎯 Day 3 Objectives ✅

- [x] Create text input field on HTML page
- [x] Add submit button for TTS generation
- [x] Connect frontend to TTS API endpoint
- [x] Implement HTML5 audio player
- [x] Add voice selection controls
- [x] Include speech speed adjustment
- [x] Add loading states and user feedback
- [x] Implement error handling on frontend

### 🎯 Day 2 Objectives ✅

- [x] Create TTS endpoint that accepts text input
- [x] Integrate with Murf AI REST TTS API  
- [x] Return audio URL from generated speech
- [x] Implement secure API key management
- [x] Add comprehensive error handling
- [x] Create interactive API documentation
- [x] Support multiple voice options

### 🎯 Day 1 Objectives ✅

- [x] Initialize Python backend using FastAPI
- [x] Create basic index.html file  
- [x] Create corresponding JavaScript file
- [x] Serve HTML page from Python server
- [x] Add interactive API testing functionality
- [x] Implement modern, responsive design

### 🔮 Next Steps (Coming Days)

- Voice recognition integration (STT)
- Real-time voice streaming
- Voice command processing
- Advanced AI voice agents
- WebSocket voice interactions
- Voice conversation flows

### 🧪 Testing the Complete TTS Experience

1. **Using the Web Interface** (Day 3 - NEW):
   - Go to `http://localhost:8001`
   - Enter text in the text area
   - Select a voice (Davis, Jenny, Alice, or Jack)
   - Adjust speed if desired
   - Click "Generate Speech"
   - Listen to the generated audio!

2. **Using FastAPI Docs** (Day 2):
   - Go to `http://localhost:8001/docs`
   - Find the `POST /api/tts/generate` endpoint
   - Click "Try it out"
   - Enter your text and test the API

3. **Using curl** (Day 2):
   ```bash
   curl -X POST "http://localhost:8001/api/tts/generate" \
        -H "Content-Type: application/json" \
        -d '{"text": "Hello, this is my first TTS test!", "voice_id": "en-US-davis"}'
   ```

4. **Example Response**:
   ```json
   {
     "success": true,
     "audio_url": "https://example.com/generated-audio.mp3",
     "message": "TTS generation successful!",
     "text": "Hello, this is my first TTS test!",
     "voice_id": "en-US-davis",
     "day": 2
   }
   ```

### 📸 Screenshot

Take a screenshot of the running application and post on LinkedIn with:
- Project description
- Day 1 completion
- #30DaysOfVoiceAgents hashtag

## Day 4: Echo Bot with Voice Recording & Playback 🎙️

**Goal**: Build an Echo Bot that records your voice and plays it back to you

**Application Structure**: Multi-page web application with navigation
- 🏠 **Home Page** (`/`) - Landing page with feature overview
- 🔊 **TTS Generator** (`/tts`) - Text-to-speech functionality  
- 🎙️ **Echo Bot** (`/echo`) - Voice recording and playback

**Features Implemented**:
- 🎤 Browser-based voice recording using MediaRecorder API
- ⏺️ Start/Stop recording controls with visual feedback
- 🔊 Instant audio playback of recorded voice
- 💾 Download functionality for recorded audio
- 🎨 Professional multi-page UI with navigation
- 📱 Real-time microphone access and permission handling
- 🔄 Audio processing with WebM/Opus format support

**Technical Implementation**:
- **Frontend**: HTML5 MediaRecorder API, Web Audio API, Multi-page SPA
- **Backend**: FastAPI with multiple route handlers
- **Audio Format**: WebM with Opus codec
- **Recording Features**: Echo cancellation control, gain control
- **Playbook**: HTML5 audio element with immediate feedback
- **UI/UX**: Animated recording indicators, status messages, navigation
- **Browser Support**: Modern browsers with microphone access

**Key Code Components**:
```python
# FastAPI Multi-page Routes
@app.get("/")
async def read_root(): # Home page
@app.get("/tts") 
async def tts_page(): # TTS Generator
@app.get("/echo")
async def echo_page(): # Echo Bot
```

```html
<!-- Navigation Component -->
<nav class="navbar">
    <h1>🎤 Voice Agents</h1>
    <div class="nav-links">
        <a href="/">🏠 Home</a>
        <a href="/tts">🔊 TTS Generator</a>
        <a href="/echo">🎙️ Echo Bot</a>
    </div>
</nav>
```

```javascript
// MediaRecorder API Implementation
const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
const mediaRecorder = new MediaRecorder(stream);
mediaRecorder.start();
// Process and playback recorded audio
```

**Learning Outcomes**:
- Multi-page web application architecture
- Browser audio API integration  
- Real-time audio recording and processing
- User permission handling for microphone access
- Audio blob creation and URL generation
- Cross-browser compatibility considerations
- Navigation and UX design patterns

---

**Challenge**: 30 Days of Voice Agents  
**Day**: 4/30  
**Status**: ✅ Complete  
**Framework**: FastAPI + HTML/JS + Murf AI TTS + MediaRecorder API
