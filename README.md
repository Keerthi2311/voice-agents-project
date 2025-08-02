# 🎤 Voice Agents Project - 30 Days Challenge

## Day 1: Project Setup

Welcome to the Voice Agents Project! This is Day 1 of the 30 Days of Voice Agents challenge.

**🔗 GitHub Repository**: https://github.com/Keerthi2311/voice-agents-project

### 🚀 Project Overview

This project demonstrates a complete full-stack setup with:
- **Backend**: FastAPI (Python) - High-performance web framework
- **Frontend**: HTML5 + Vanilla JavaScript - Responsive and modern UI
- **APIs**: RESTful endpoints for voice agent functionality

### 📁 Project Structure

```
voice-agents-project/
├── main.py                 # FastAPI application
├── templates/
│   └── index.html          # Main HTML page
├── static/
│   └── app.js             # JavaScript functionality
├── requirements.txt        # Python dependencies
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

3. **Run the Server**:
   ```bash
   python main.py
   ```
   Or using uvicorn directly:
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8001
   ```

4. **Access the Application**:
   Open your browser and navigate to: `http://localhost:8001`

### 🔗 API Endpoints

- `GET /` - Serves the main HTML page
- `GET /api/hello` - Sample API endpoint
- `GET /api/voice-status` - Voice agent status endpoint

### ✨ Features

- **Modern UI**: Glassmorphism design with gradient backgrounds
- **Interactive Elements**: Hover effects and animations
- **API Testing**: Built-in buttons to test backend endpoints
- **Responsive Design**: Works on desktop and mobile devices
- **Real-time Feedback**: Status messages and JSON response display

### 🎯 Day 1 Objectives ✅

- [x] Initialize Python backend using FastAPI
- [x] Create basic index.html file
- [x] Create corresponding JavaScript file
- [x] Serve HTML page from Python server
- [x] Add interactive API testing functionality
- [x] Implement modern, responsive design

### 🔮 Next Steps (Coming Days)

- Voice recognition integration
- Text-to-speech functionality
- Voice command processing
- Advanced AI voice agents
- Real-time voice interactions

### 📸 Screenshot

Take a screenshot of the running application and post on LinkedIn with:
- Project description
- Day 1 completion
- #30DaysOfVoiceAgents hashtag

---

**Challenge**: 30 Days of Voice Agents  
**Day**: 1/30  
**Status**: ✅ Complete  
**Framework**: FastAPI + HTML/JS
