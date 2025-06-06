import google.generativeai as genai
import time
from pathlib import Path
import logging

from ..config import GEMINI_API_KEY, GEMINI_MODEL, TEXT_PROMPT

logger = logging.getLogger(__name__)


class GeminiService:
    def __init__(self):
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY environment variable is required")

        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel(GEMINI_MODEL)

    def generate_text_prompt(self, video_path: Path) -> str:
        """Generate a text prompt for object detection based on video content."""
        try:
            logger.info(f"Uploading video to Gemini: {video_path}")
            video_file = genai.upload_file(str(video_path))

            logger.info("Waiting for video processing...")
            video_file = genai.get_file(video_file.name)
            while video_file.state.name == "PROCESSING":
                logger.info("Video still processing...")
                time.sleep(2)
                video_file = genai.get_file(video_file.name)

            if video_file.state.name == "FAILED":
                logger.error("Video processing failed")
                return TEXT_PROMPT

            system_prompt = """
            Analyze this video and identify the main objects that should be detected for fashion/style tagging.
            Focus on wearable items, accessories, and fashion-related objects that are clearly visible throughout the video.
            
            Return a concise list of object categories separated by periods, suitable for object detection.
            Examples: "watch. sunglasses. hat. shirt. jacket. pants. shoes. bag. necklace. ring."
            
            Keep it focused on the most prominent and relevant items visible in the video.
            Use simple, clear object names that an object detection model would understand.
            Only include objects that are actually visible and prominent in the video.
            """

            response = self.model.generate_content([system_prompt, video_file])

            generated_prompt = response.text.strip()

            if generated_prompt and len(generated_prompt) > 5:
                if not generated_prompt.endswith("."):
                    generated_prompt += "."

                logger.info(f"Generated text prompt: {generated_prompt}")

                # Clean up the uploaded file
                genai.delete_file(video_file.name)

                return generated_prompt
            else:
                logger.warning("Generated prompt is too short, using default")
                genai.delete_file(video_file.name)
                return TEXT_PROMPT

        except Exception as e:
            logger.error(f"Error generating text prompt with Gemini: {str(e)}")
            return TEXT_PROMPT
