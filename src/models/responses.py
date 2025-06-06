from pydantic import BaseModel
from typing import List, Optional, Dict, Any


class KeyframeResponse(BaseModel):
    filename: str
    timecode: str


class CroppedKeyframeResponse(BaseModel):
    filename: str
    class_name: str
    original_class_name: str
    bbox: List[float]


class VibeAnalysis(BaseModel):
    id: str
    name: str
    confidence: float


class ContentAnalysis(BaseModel):
    audio_transcription: Optional[str] = None
    clothing_description: str
    vibes: List[VibeAnalysis]


class UploadVideoResponse(BaseModel):
    video_id: str
    original_filename: str
    scenes_detected: int
    keyframes_generated: int
    keyframes: List[Dict[str, Any]]
    keyframes_url: str
    cropped_keyframes_generated: int
    cropped_keyframes: List[Dict[str, Any]]
    cropped_keyframes_url: str
    masked_keyframes_generated: int
    masked_keyframes: List[Dict[str, Any]]
    masked_keyframes_url: str
    content_analysis: Optional[ContentAnalysis] = None


class KeyframeListItem(BaseModel):
    filename: str
    frame_number: int
    url: str


class KeyframeListResponse(BaseModel):
    video_id: str
    keyframes: List[KeyframeListItem]


class CroppedKeyframeListItem(BaseModel):
    filename: str
    url: str


class CroppedKeyframeListResponse(BaseModel):
    video_id: str
    cropped_keyframes: List[CroppedKeyframeListItem]
