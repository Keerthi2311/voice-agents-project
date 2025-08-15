from pydantic import BaseModel
from typing import Optional

class ChatRequest(BaseModel):
    session_id: str
    # audio_file will be handled as UploadFile in FastAPI, not in schema

class ChatResponse(BaseModel):
    success: bool
    session_id: str
    transcribed_text: Optional[str]
    llm_response: Optional[str]
    audio_url: Optional[str]
    model: Optional[str]
    voice: Optional[str]
    chat_history_length: Optional[int]
    message: Optional[str]
    day: Optional[int]
    ui_version: Optional[str]
    auto_play: Optional[bool]
    error_handling_active: Optional[bool]
    pipeline_status: Optional[str]
    error_context: Optional[str]
    pipeline_steps: Optional[list]
    ui_features: Optional[dict]
    critical_error: Optional[bool]
    error_message: Optional[str]
    error_type: Optional[str]

class HistoryResponse(BaseModel):
    session_id: str
    history: list
    message_count: int
    day: int

class ClearHistoryResponse(BaseModel):
    session_id: str
    message: str
    day: int
