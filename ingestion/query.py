import numpy as np
import argparse
import json
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from qdrant_client import QdrantClient
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from urllib.parse import urlparse
import os

# Load environment variables
load_dotenv()


class FashionQuerier:
    def __init__(self, qdrant_url: str = None, qdrant_collection_name: str = None):
        # Load configuration from environment variables
        self.qdrant_url = qdrant_url or os.getenv("QDRANT_URL")
        self.collection_name = qdrant_collection_name or os.getenv("QDRANT_COLLECTION_NAME")

        if not self.qdrant_url:
            raise ValueError("QDRANT_URL must be set in environment variables or passed as parameter")
        if not self.collection_name:
            raise ValueError("QDRANT_COLLECTION_NAME must be set in environment variables or passed as parameter")

        # Parse Qdrant URL to extract host and port
        parsed_url = urlparse(self.qdrant_url)
        qdrant_host = parsed_url.hostname
        qdrant_port = parsed_url.port or 6333

        print(f"Connecting to Qdrant at {self.qdrant_url} (collection: {self.collection_name})")

        # Check for CUDA availability and configure device
        self.device = self._setup_device()
        print(f"Using device: {self.device}")

        # Initialize FashionCLIP model and processor from Hugging Face
        model_name = "patrickjohncyh/fashion-clip"
        print(f"Loading {model_name} model...")
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name)

        # Move model to GPU if available
        if torch.cuda.is_available():
            self.model = self.model.to(self.device)
            print(f"FashionCLIP model moved to {self.device}")

        self.client = QdrantClient(host=qdrant_host, port=qdrant_port)

    def _setup_device(self):
        """Setup and return the best available device"""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"CUDA available! Using GPU: {gpu_name} ({gpu_memory:.1f} GB)")

            # Clear cache to start fresh
            torch.cuda.empty_cache()
            return device
        else:
            print("CUDA not available. Using CPU (this will be slower)")
            return torch.device("cpu")

    def _generate_text_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text query using FashionCLIP with GPU acceleration"""
        try:
            # Ensure text is within token limits (consistent with vectorize.py)
            text_tokens = text.split()
            if len(text_tokens) > 40:
                text = " ".join(text_tokens[:40])

            # Preprocess text
            inputs = self.processor(text=[text], return_tensors="pt", padding=True)

            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate embedding
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
                # Normalize features
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            text_embedding = text_features.cpu().numpy()[0]

            # Clear GPU cache after processing
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            return text_embedding
        except Exception as e:
            print(f"Error generating text embedding: {e}")
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            return None

    def _generate_image_embedding(self, image_path: str) -> np.ndarray:
        """Generate embedding for image query using FashionCLIP with GPU acceleration"""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert("RGB")

            # Preprocess image
            inputs = self.processor(images=[image], return_tensors="pt", padding=True)

            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate embedding
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                # Normalize features
                image_features = image_features / image_features.norm(
                    dim=-1, keepdim=True
                )

            image_embedding = image_features.cpu().numpy()[0]

            # Clear GPU cache after processing
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            return image_embedding
        except Exception as e:
            print(f"Error generating image embedding: {e}")
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            return None

    def search_by_text(
        self, query: str, limit: int = 10, score_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Search products using text description"""
        query_embedding = self._generate_text_embedding(query)

        if query_embedding is None:
            print(f"Failed to generate embedding for text query: {query}")
            return []

        search_results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding.tolist(),
            limit=limit,
            score_threshold=score_threshold,
            with_payload=True,
        )

        results = []
        for result in search_results.points:
            results.append(
                {
                    "product_id": result.payload["product_id"],
                    "title": result.payload["title"],
                    "description": result.payload["description"],
                    "product_type": result.payload["product_type"],
                    "price": result.payload["price"],
                    "tags": result.payload["tags"],
                    "score": result.score,
                    "folder_path": result.payload["folder_path"],
                }
            )

        return results

    def search_by_image(
        self, image_path: str, limit: int = 10, score_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Search products using image"""
        query_embedding = self._generate_image_embedding(image_path)

        if query_embedding is None:
            print(f"Failed to generate embedding for image: {image_path}")
            return []

        search_results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding.tolist(),
            limit=limit,
            score_threshold=score_threshold,
            with_payload=True,
        )

        results = []
        for result in search_results.points:
            results.append(
                {
                    "product_id": result.payload["product_id"],
                    "title": result.payload["title"],
                    "description": result.payload["description"],
                    "product_type": result.payload["product_type"],
                    "price": result.payload["price"],
                    "tags": result.payload["tags"],
                    "score": result.score,
                    "folder_path": result.payload["folder_path"],
                }
            )

        return results

    def search_by_multimodal(
        self,
        text_query: Optional[str] = None,
        image_path: Optional[str] = None,
        limit: int = 10,
        score_threshold: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """Search using both text and image (combined query)"""
        embeddings = []

        if text_query:
            text_embedding = self._generate_text_embedding(text_query)
            if text_embedding is not None:
                embeddings.append(text_embedding)
            else:
                print(f"Failed to generate text embedding for: {text_query}")

        if image_path:
            image_embedding = self._generate_image_embedding(image_path)
            if image_embedding is not None:
                embeddings.append(image_embedding)
            else:
                print(f"Failed to generate image embedding for: {image_path}")

        if not embeddings:
            raise ValueError("Failed to generate embeddings for provided inputs")

        # Average the embeddings if both are provided
        combined_embedding = np.mean(embeddings, axis=0)
        combined_embedding = combined_embedding / np.linalg.norm(combined_embedding)

        search_results = self.client.query_points(
            collection_name=self.collection_name,
            query=combined_embedding.tolist(),
            limit=limit,
            score_threshold=score_threshold,
            with_payload=True,
        )

        results = []
        for result in search_results.points:
            results.append(
                {
                    "product_id": result.payload["product_id"],
                    "title": result.payload["title"],
                    "description": result.payload["description"],
                    "product_type": result.payload["product_type"],
                    "price": result.payload["price"],
                    "tags": result.payload["tags"],
                    "score": result.score,
                    "folder_path": result.payload["folder_path"],
                }
            )

        return results

    def get_product_by_id(self, product_id: str) -> Optional[Dict[str, Any]]:
        """Get specific product by ID"""
        search_results = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter={
                "must": [{"key": "product_id", "match": {"value": product_id}}]
            },
            limit=1,
        )

        if search_results[0]:
            result = search_results[0][0]
            return {
                "product_id": result.payload["product_id"],
                "title": result.payload["title"],
                "description": result.payload["description"],
                "product_type": result.payload["product_type"],
                "price": result.payload["price"],
                "tags": result.payload["tags"],
                "collections": result.payload["collections"],
                "folder_path": result.payload["folder_path"],
            }

        return None


def format_results(results: List[Dict[str, Any]], format_type: str = "text") -> str:
    """Format search results as text or JSON"""
    if format_type.lower() == "json":
        return json.dumps(results, indent=2)
    else:
        # Text format
        output = []
        output.append(f"Found {len(results)} results:\n")

        for i, result in enumerate(results, 1):
            output.append(f"{i}. {result['title']}")
            output.append(f"   Product ID: {result['product_id']}")
            output.append(f"   Type: {result['product_type']}")
            output.append(f"   Price: {result['price']}")
            output.append(f"   Score: {result['score']:.3f}")
            if result["description"]:
                desc = (
                    result["description"][:100] + "..."
                    if len(result["description"]) > 100
                    else result["description"]
                )
                output.append(f"   Description: {desc}")
            output.append("")

        return "\n".join(output)


def main():
    """CLI interface for fashion product search"""
    parser = argparse.ArgumentParser(
        description="Search fashion products using FashionCLIP"
    )
    parser.add_argument("--text", type=str, help="Text query for search")
    parser.add_argument("--image", type=str, help="Image path for search")
    parser.add_argument(
        "--limit", type=int, default=10, help="Maximum number of results (default: 10)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Score threshold (default: 0.5)"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )
    # Remove hardcoded host/port args since we're using env vars
    parser.add_argument(
        "--qdrant-url",
        type=str,
        help="Qdrant URL (overrides QDRANT_URL env var)",
    )
    parser.add_argument(
        "--collection",
        type=str,
        help="Collection name (overrides QDRANT_COLLECTION_NAME env var)",
    )

    args = parser.parse_args()

    if not args.text and not args.image:
        print("Error: Please provide either --text or --image query")
        return

    querier = FashionQuerier(
        qdrant_url=args.qdrant_url,
        qdrant_collection_name=args.collection
    )

    try:
        if args.text and args.image:
            # Multimodal search
            results = querier.search_by_multimodal(
                text_query=args.text,
                image_path=args.image,
                limit=args.limit,
                score_threshold=args.threshold,
            )
        elif args.text:
            # Text search
            results = querier.search_by_text(
                query=args.text, limit=args.limit, score_threshold=args.threshold
            )
        elif args.image:
            # Image search
            results = querier.search_by_image(
                image_path=args.image, limit=args.limit, score_threshold=args.threshold
            )

        # Format and print results
        formatted_output = format_results(results, args.format)
        print(formatted_output)

    except Exception as e:
        print(f"Error during search: {e}")


if __name__ == "__main__":
    main()
