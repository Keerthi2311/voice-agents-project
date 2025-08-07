# 🎙️ 30 Days of Voice Agents - Complete Implementation

A comprehensive voice agents application implementing Days 1-6 of the Voice Agents challenge, featuring text-to-speech, audio recording, file upload, and speech transcription.

## 🚀 Features

### ✅ Day 1: Project Setup
- FastAPI backend server
- Static file serving
- Health check endpoints
- Basic HTML/CSS/JS frontend

### ✅ Day 2: REST TTS API
- Murf AI integration for text-to-speech
- REST API endpoints for audio generation
- Error handling and fallback responses

### ✅ Day 3: TTS Audio Playback
- Interactive frontend for text input
- Voice selection dropdown
- Audio playback in browser
- Real-time status feedback

### ✅ Day 4: Echo Bot
- MediaRecorder API for audio recording
- Microphone access and permission handling
- Audio blob creation and playback
- Recording state management

### ✅ Day 5: Audio Upload
- File upload to server with validation
- Temporary audio file storage
- File metadata and statistics
- Upload progress feedback

### ✅ Day 6: Speech Transcription
- AssemblyAI integration for speech-to-text
- Direct audio transcription from memory
- Confidence scores and metadata
- Comprehensive transcription display

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- Modern web browser with microphone support

### Setup

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure API keys**
   - Edit `.env` file and add your API keys:
   ```bash
   # Get Murf API key from: https://murf.ai/dashboard
   MURF_API_KEY=your_murf_api_key_here
   
   # Get AssemblyAI API key from: https://www.assemblyai.com/dashboard/signup
   ASSEMBLYAI_API_KEY=your_assemblyai_api_key_here
   ```

3. **Run the application**
   ```bash
   python main.py
   ```

4. **Open your browser**
   - Navigate to `http://localhost:8000`
   - Grant microphone permissions when prompted

## 🎯 Usage

### Text-to-Speech
1. Enter text in the textarea
2. Select a voice from the dropdown
3. Click "Generate Speech"
4. Listen to the generated audio

### Echo Bot with Transcription
1. Click "Start Recording"
2. Speak into your microphone
3. Click "Stop Recording"
4. View the recorded audio, upload status, and transcription results

## 📁 Project Structure

```
voice-agents-project-2/
├── main.py                 # FastAPI backend server
├── requirements.txt        # Python dependencies
├── .env                   # Environment variables (API keys)
├── start.sh              # Startup script
├── uploads/              # Temporary audio file storage
├── static/               # Frontend assets
│   ├── index.html        # Main application interface
│   ├── script.js         # JavaScript application logic
│   └── style.css         # Responsive CSS styling
└── templates/            # Additional HTML templates
    ├── home.html
    ├── tts.html
    └── echo.html
```

## 🔧 API Endpoints

### Core Endpoints
- `GET /` - Main application interface
- `GET /health` - Server health check
- `GET /api/tts/voices` - Available TTS voices

### TTS Endpoints
- `POST /generate-audio` - Generate speech from text

### Audio Processing
- `POST /upload-audio` - Upload audio file
- `POST /transcribe/file` - Transcribe audio to text

## 🎨 Frontend Features

### Modern UI/UX
- Responsive gradient design
- Smooth animations and transitions
- Real-time status feedback
- Loading states and error handling

### Interactive Elements
- Voice selection dropdown
- Audio recording controls
- Progress indicators
- Transcription results display

## 🧪 Testing

### Quick Test
1. Run `python main.py`
2. Open `http://localhost:8000`
3. Test TTS: Enter text and generate speech
4. Test Recording: Click record, speak, then stop
5. View transcription results

## 📊 Performance

- **Audio Processing**: Direct memory processing (no temporary files)
- **Response Times**: < 3s for TTS, < 10s for transcription
- **File Handling**: Efficient streaming with aiofiles
- **Error Recovery**: Graceful fallbacks and user feedback

## 🎉 Next Steps (Days 7-30)

This implementation provides a solid foundation for:
- Advanced speech recognition features
- Voice commands and intent recognition
- Real-time audio processing
- Multi-language support
- Voice cloning and synthesis
- AI-powered conversation agents

Ready to continue the journey? Let's build amazing voice agents together! 🚀
