import torch
from pathlib import Path

# Directory configurations
UPLOAD_DIR = Path("/tmp/uploads")
KEYFRAMES_DIR = Path("/tmp/keyframes")
CROPPED_KEYFRAMES_DIR = Path("/tmp/keyframes-cropped")

# Grounding DINO configuration
GROUNDING_MODEL = "IDEA-Research/grounding-dino-tiny"
TEXT_PROMPT = "watch. topwear. bottomwear. shoes. headgear."
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Detection thresholds
BOX_THRESHOLD = 0.4
TEXT_THRESHOLD = 0.3

# Ensure directories exist
UPLOAD_DIR.mkdir(exist_ok=True)
KEYFRAMES_DIR.mkdir(exist_ok=True)
CROPPED_KEYFRAMES_DIR.mkdir(exist_ok=True)
