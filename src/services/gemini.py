import google.generativeai as genai
import time
from pathlib import Path
import logging
from PIL import Image
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor

from ..config import GEMINI_API_KEY, GEMINI_MODEL, TEXT_PROMPT, VIBES_LIST

logger = logging.getLogger(__name__)


class GeminiService:
    def __init__(self):
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY environment variable is required")

        logger.info(f"Loading Gemini model: {GEMINI_MODEL}")
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel(GEMINI_MODEL)
        self.executor = ThreadPoolExecutor(max_workers=2)
        logger.info("Gemini service initialized successfully")

    def generate_text_prompt_from_video(self, video_path: Path) -> str:
        """Generate a text prompt for object detection based on video content."""
        logger.info("Starting Gemini text prompt generation from video")
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

                logger.info(f"Generated text prompt from video: {generated_prompt}")

                # Clean up the uploaded file
                genai.delete_file(video_file.name)

                return generated_prompt
            else:
                logger.warning("Generated prompt is too short, using default")
                genai.delete_file(video_file.name)
                return TEXT_PROMPT

        except Exception as e:
            logger.error(
                f"Error generating text prompt from video with Gemini: {str(e)}"
            )
            return TEXT_PROMPT

    def generate_text_prompt_from_image(self, image_path: Path) -> str:
        """Generate a text prompt for object detection based on image content."""
        logger.info("Starting Gemini text prompt generation from image")
        try:
            logger.info(f"Processing image with Gemini: {image_path}")

            # Load and prepare image
            image = Image.open(image_path)

            system_prompt = """
            Analyze this image and identify the main objects that should be detected for fashion/style tagging.
            Focus on wearable items, accessories, and fashion-related objects that are clearly visible in the image.
            
            Return a concise list of object categories separated by periods, suitable for object detection.
            Examples: "watch. sunglasses. hat. shirt. jacket. pants. shoes. bag. necklace. ring."
            
            Keep it focused on the most prominent and relevant items visible in the image.
            Use simple, clear object names that an object detection model would understand.
            Only include objects that are actually visible and prominent in the image.
            """

            response = self.model.generate_content([system_prompt, image])

            generated_prompt = response.text.strip()

            if generated_prompt and len(generated_prompt) > 5:
                if not generated_prompt.endswith("."):
                    generated_prompt += "."

                logger.info(f"Generated text prompt from image: {generated_prompt}")
                return generated_prompt
            else:
                logger.warning("Generated prompt is too short, using default")
                return TEXT_PROMPT

        except Exception as e:
            logger.error(
                f"Error generating text prompt from image with Gemini: {str(e)}"
            )
            return TEXT_PROMPT

    async def analyze_content_async(
        self, file_path: Path, is_video: bool = True
    ) -> dict:
        """Asynchronously analyze video/image content for transcription, clothing description, and vibes."""
        logger.info(
            f"Starting async Gemini content analysis for {'video' if is_video else 'image'}"
        )

        loop = asyncio.get_event_loop()

        try:
            if is_video:
                result = await loop.run_in_executor(
                    self.executor, self._analyze_video_content, file_path
                )
            else:
                result = await loop.run_in_executor(
                    self.executor, self._analyze_image_content, file_path
                )
            return result
        except Exception as e:
            logger.error(f"Error analyzing content with Gemini: {str(e)}")
            return {
                "audio_transcription": None,
                "clothing_description": "Unable to analyze content",
                "vibes": [],
            }

    def analyze_content(self, file_path: Path, is_video: bool = True) -> dict:
        """Synchronous wrapper for backward compatibility."""
        try:
            if is_video:
                return self._analyze_video_content(file_path)
            else:
                return self._analyze_image_content(file_path)
        except Exception as e:
            logger.error(f"Error analyzing content with Gemini: {str(e)}")
            return {
                "audio_transcription": None,
                "clothing_description": "Unable to analyze content",
                "vibes": [],
            }

    def _analyze_video_content(self, video_path: Path) -> dict:
        """Analyze video content for transcription, clothing description, and vibes."""
        logger.info(f"Uploading video for content analysis: {video_path}")

        video_file = None
        try:
            video_file = genai.upload_file(str(video_path))
            logger.info(f"Video uploaded successfully with ID: {video_file.name}")

            logger.info("Waiting for video processing...")
            max_wait_time = 300  # 5 minutes max wait
            wait_time = 0

            while wait_time < max_wait_time:
                try:
                    video_file = genai.get_file(video_file.name)
                    if video_file.state.name == "ACTIVE":
                        logger.info("Video processing completed successfully")
                        break
                    elif video_file.state.name == "FAILED":
                        logger.error("Video processing failed")
                        return {
                            "audio_transcription": None,
                            "clothing_description": "Video processing failed",
                            "vibes": [],
                        }
                    elif video_file.state.name == "PROCESSING":
                        logger.info(f"Video still processing... ({wait_time}s elapsed)")
                        time.sleep(5)
                        wait_time += 5
                    else:
                        logger.warning(f"Unknown video state: {video_file.state.name}")
                        time.sleep(5)
                        wait_time += 5
                except Exception as e:
                    logger.error(f"Error checking video status: {str(e)}")
                    time.sleep(5)
                    wait_time += 5

            if wait_time >= max_wait_time:
                logger.error("Video processing timeout")
                return {
                    "audio_transcription": None,
                    "clothing_description": "Video processing timeout",
                    "vibes": [],
                }

            vibes_json = json.dumps(VIBES_LIST, indent=2)

            system_prompt = f"""
            Analyze this video and provide a comprehensive analysis in JSON format.

            Available vibes to choose from:
            {vibes_json}

            Please return a JSON response with the following structure:
            {{
                "audio_transcription": "transcribed text from audio (or null if no clear speech)",
                "clothing_description": "detailed description of clothing and fashion items visible",
                "vibes": [
                    {{
                        "id": "vibe_id_from_list",
                        "name": "Vibe Name",
                        "confidence": 0.85
                    }}
                ]
            }}

            Rules:
            1. For audio_transcription: Only include if there's clear speech. Return null if unclear or no speech.
            2. For clothing_description: Describe visible clothing, accessories, and fashion items in detail.
            3. For vibes: Select 1-3 vibes that best match the style/aesthetic. Include confidence scores (0.0-1.0).
            4. Only use vibe IDs from the provided list.
            5. Return valid JSON only, no additional text or formatting.
            """

            response = self.model.generate_content([system_prompt, video_file])
            response_text = response.text.strip()

            # Try to extract JSON from response if it's wrapped in markdown or other text
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.rfind("```")
                response_text = response_text[json_start:json_end].strip()

            # Parse JSON response
            result = json.loads(response_text)
            logger.info("Successfully analyzed video content")

            return result

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Gemini JSON response: {str(e)}")
            logger.error(
                f"Raw response: {response.text if 'response' in locals() else 'No response'}"
            )
            return {
                "audio_transcription": None,
                "clothing_description": "Failed to parse analysis response",
                "vibes": [],
            }
        except Exception as e:
            logger.error(f"Error in video content analysis: {str(e)}")
            return {
                "audio_transcription": None,
                "clothing_description": "Error analyzing video content",
                "vibes": [],
            }
        finally:
            # Clean up the uploaded file
            if video_file:
                try:
                    genai.delete_file(video_file.name)
                    logger.info("Successfully cleaned up uploaded video file")
                except Exception as e:
                    logger.warning(f"Failed to clean up video file: {str(e)}")

    def _analyze_image_content(self, image_path: Path) -> dict:
        """Analyze image content for clothing description and vibes."""
        logger.info(f"Processing image for content analysis: {image_path}")

        try:
            image = Image.open(image_path)
            vibes_json = json.dumps(VIBES_LIST, indent=2)

            system_prompt = f"""
            Analyze this image and provide a comprehensive analysis in JSON format.

            Available vibes to choose from:
            {vibes_json}

            Please return a JSON response with the following structure:
            {{
                "audio_transcription": null,
                "clothing_description": "detailed description of clothing and fashion items visible",
                "vibes": [
                    {{
                        "id": "vibe_id_from_list",
                        "name": "Vibe Name",
                        "confidence": 0.85
                    }}
                ]
            }}

            Rules:
            1. For audio_transcription: Always null for images.
            2. For clothing_description: Describe visible clothing, accessories, and fashion items in detail.
            3. For vibes: Select 1-3 vibes that best match the style/aesthetic. Include confidence scores (0.0-1.0).
            4. Only use vibe IDs from the provided list.
            5. Return valid JSON only, no additional text or formatting.
            """

            response = self.model.generate_content([system_prompt, image])
            response_text = response.text.strip()

            # Try to extract JSON from response if it's wrapped in markdown or other text
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.rfind("```")
                response_text = response_text[json_start:json_end].strip()

            result = json.loads(response_text)
            logger.info("Successfully analyzed image content")
            return result

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Gemini JSON response: {str(e)}")
            logger.error(
                f"Raw response: {response.text if 'response' in locals() else 'No response'}"
            )
            return {
                "audio_transcription": None,
                "clothing_description": "Failed to parse analysis response",
                "vibes": [],
            }
        except Exception as e:
            logger.error(f"Error analyzing image content: {str(e)}")
            return {
                "audio_transcription": None,
                "clothing_description": "Error analyzing image content",
                "vibes": [],
            }

    # Keep backward compatibility
    def generate_text_prompt(self, video_path: Path) -> str:
        """Legacy method - defaults to video processing"""
        return self.generate_text_prompt_from_video(video_path)
