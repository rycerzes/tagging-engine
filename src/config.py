import torch
from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Directory configurations
UPLOAD_DIR = Path("/tmp/uploads")
KEYFRAMES_DIR = Path("/tmp/keyframes")
CROPPED_KEYFRAMES_DIR = Path("/tmp/keyframes-cropped")
MASKED_KEYFRAMES_DIR = Path("/tmp/keyframes-masked")

# Grounding DINO configuration
GROUNDING_MODEL = "IDEA-Research/grounding-dino-tiny"

# SAM2 configuration
SAM2_MODEL = "facebook/sam2.1-hiera-base-plus"

TEXT_PROMPT = "watch. topwear. bottomwear. shoes. headgear."

# Detection thresholds
BOX_THRESHOLD = 0.4
TEXT_THRESHOLD = 0.3

# Ensure directories exist
UPLOAD_DIR.mkdir(exist_ok=True)
KEYFRAMES_DIR.mkdir(exist_ok=True)
CROPPED_KEYFRAMES_DIR.mkdir(exist_ok=True)
MASKED_KEYFRAMES_DIR.mkdir(exist_ok=True)
