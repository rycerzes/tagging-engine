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
    file_hash: str
    cached: bool = False
    cached_from_video_id: Optional[str] = None
    scenes_detected: int
    keyframes_generated: int
    keyframes: List[Dict[str, str]]
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
    keyframes: List[Dict[str, str]]


class CroppedKeyframeListItem(BaseModel):
    filename: str
    url: str


class CroppedKeyframeListResponse(BaseModel):
    video_id: str
    cropped_keyframes: List[Dict[str, Any]]


class FashionMatch(BaseModel):
    product_id: str
    score: float
    payload: Dict[str, Any]


class CropMatch(BaseModel):
    crop_id: str
    filename: str
    class_name: str
    original_class_name: str
    bbox: List[float]
    fashion_matches: List[FashionMatch]


class VideoQueryResponse(BaseModel):
    video_id: str
    total_crops: int
    crop_matches: List[CropMatch]


class SimpleFashionMatch(BaseModel):
    product_id: str
    score: float
    product_name: str
    title: str
    description: str
    product_type: str
    price: str
    tags: str
    collections: str


class SimpleCropMatch(BaseModel):
    crop_id: str
    filename: str
    class_name: str
    original_class_name: str
    fashion_matches: List[SimpleFashionMatch]


class CombinedVideoResponse(BaseModel):
    video_id: str
    original_filename: str
    file_hash: str
    cached: bool = False
    cached_from_video_id: Optional[str] = None
    scenes_detected: int
    keyframes_generated: int
    cropped_keyframes_generated: int
    masked_keyframes_generated: int
    content_analysis: Optional[ContentAnalysis] = None
    total_crops_with_matches: int
    crop_matches: List[SimpleCropMatch]