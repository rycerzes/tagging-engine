from typing import List
from pydantic import BaseModel


class KeyframeResponse(BaseModel):
    filename: str
    timecode: str


class CroppedKeyframeResponse(BaseModel):
    filename: str
    class_name: str
    original_class_name: str
    bbox: List[float]


class UploadVideoResponse(BaseModel):
    video_id: str
    original_filename: str
    scenes_detected: int
    keyframes_generated: int
    keyframes: List[KeyframeResponse]
    keyframes_url: str
    cropped_keyframes_generated: int
    cropped_keyframes: List[CroppedKeyframeResponse]
    cropped_keyframes_url: str


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
