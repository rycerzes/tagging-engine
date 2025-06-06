import torch
import os
from pathlib import Path
import dotenv

dotenv.load_dotenv()

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

# Gemini configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.5-flash-preview-05-20"
USE_GEMINI_FOR_TEXT_PROMPT = False

# Default text prompt for object detection
# This can be overridden by Gemini if USE_GEMINI_FOR_TEXT_PROMPT is true
TEXT_PROMPT = """wristwear. topwear. bottomwear. footwear. 
headgear. accessories. bag. outerwear."""

# Detection thresholds
BOX_THRESHOLD = 0.4
TEXT_THRESHOLD = 0.3

# Ensure directories exist
UPLOAD_DIR.mkdir(exist_ok=True)
KEYFRAMES_DIR.mkdir(exist_ok=True)
CROPPED_KEYFRAMES_DIR.mkdir(exist_ok=True)
MASKED_KEYFRAMES_DIR.mkdir(exist_ok=True)
