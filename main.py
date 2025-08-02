from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import os

app = FastAPI(title="Voice Agents Project ", version="1.0.0")

# Mount static files directory
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main HTML page"""
    with open("templates/index.html", "r") as file:
        html_content = file.read()
    return HTMLResponse(content=html_content)

@app.get("/api/hello")
async def hello_api():
    """Sample API endpoint"""
    return {"message": "Hello from Voice Agents Backend!", "day": 1, "status": "success"}

@app.get("/api/voice-status")
async def voice_status():
    """API endpoint for voice agent status"""
    return {
        "voice_agent": "initialized",
        "features": ["text-to-speech", "speech-to-text", "voice-commands"],
        "day": 1,
        "challenge": "30 Days of Voice Agents"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
