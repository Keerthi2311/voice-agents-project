#!/bin/bash

# 30 Days Voice Agents - Startup Script
echo "�️ Starting Voice Agents Application..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install/update dependencies
echo "📋 Installing dependencies..."
pip install -r requirements.txt

# Check for environment file
if [ ! -f ".env" ]; then
    echo "⚠️  Warning: .env file not found. Creating template..."
    echo "MURF_API_KEY=your_murf_api_key_here" > .env
    echo "ASSEMBLYAI_API_KEY=your_assemblyai_api_key_here" >> .env
    echo "📝 Please edit .env file with your API keys"
fi

# Create uploads directory if it doesn't exist
mkdir -p uploads

# Start the application
echo "🚀 Starting FastAPI server..."
echo "📱 Open http://localhost:8000 in your browser"
echo "🎯 Features available: TTS, Recording, Upload, Transcription"
echo ""
python main.py
