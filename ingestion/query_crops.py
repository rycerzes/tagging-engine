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


class CropQuerier:
    def __init__(self, qdrant_url: str = None, qdrant_collection_name: str = None):
        # Load configuration from environment variables
        self.qdrant_url = qdrant_url or os.getenv("QDRANT_URL")
        self.collection_name = qdrant_collection_name or os.getenv(
            "QDRANT_COLLECTION_NAME"
        )

        if not self.qdrant_url:
            raise ValueError(
                "QDRANT_URL must be set in environment variables or passed as parameter"
            )
        if not self.collection_name:
            raise ValueError(
                "QDRANT_COLLECTION_NAME must be set in environment variables or passed as parameter"
            )

        # Parse Qdrant URL to extract host and port
        parsed_url = urlparse(self.qdrant_url)
        qdrant_host = parsed_url.hostname
        qdrant_port = parsed_url.port or 6333

        print(
            f"Connecting to Qdrant at {self.qdrant_url} (collection: {self.collection_name})"
        )

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
            # Ensure text is within token limits
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

    def search_crops_by_text(
        self, query: str, limit: int = 10, score_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Search crop embeddings using text description"""
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
                    "crop_id": result.id,
                    "product_id": result.payload["product_id"],
                    "product_name": result.payload["product_name"],
                    "class_name": result.payload["class_name"],
                    "title": result.payload["title"],
                    "description": result.payload["description"],
                    "product_type": result.payload["product_type"],
                    "price": result.payload["price"],
                    "tags": result.payload["tags"],
                    "collections": result.payload["collections"],
                    "file_path": result.payload["file_path"],
                    "filename": result.payload["filename"],
                    "score": result.score,
                }
            )

        return results

    def search_crops_by_image(
        self, image_path: str, limit: int = 10, score_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Search crop embeddings using image"""
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
                    "crop_id": result.id,
                    "product_id": result.payload["product_id"],
                    "product_name": result.payload["product_name"],
                    "class_name": result.payload["class_name"],
                    "title": result.payload["title"],
                    "description": result.payload["description"],
                    "product_type": result.payload["product_type"],
                    "price": result.payload["price"],
                    "tags": result.payload["tags"],
                    "collections": result.payload["collections"],
                    "file_path": result.payload["file_path"],
                    "filename": result.payload["filename"],
                    "score": result.score,
                }
            )

        return results

    def search_crops_by_multimodal(
        self,
        text_query: Optional[str] = None,
        image_path: Optional[str] = None,
        limit: int = 10,
        score_threshold: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """Search crop embeddings using both text and image (combined query)"""
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
                    "crop_id": result.id,
                    "product_id": result.payload["product_id"],
                    "product_name": result.payload["product_name"],
                    "class_name": result.payload["class_name"],
                    "title": result.payload["title"],
                    "description": result.payload["description"],
                    "product_type": result.payload["product_type"],
                    "price": result.payload["price"],
                    "tags": result.payload["tags"],
                    "collections": result.payload["collections"],
                    "file_path": result.payload["file_path"],
                    "filename": result.payload["filename"],
                    "score": result.score,
                }
            )

        return results

    def get_crop_by_id(self, crop_id: str) -> Optional[Dict[str, Any]]:
        """Get specific crop by ID"""
        search_results = self.client.retrieve(
            collection_name=self.collection_name,
            ids=[crop_id],
            with_payload=True,
        )

        if search_results:
            result = search_results[0]
            return {
                "crop_id": result.id,
                "product_id": result.payload["product_id"],
                "product_name": result.payload["product_name"],
                "class_name": result.payload["class_name"],
                "title": result.payload["title"],
                "description": result.payload["description"],
                "product_type": result.payload["product_type"],
                "price": result.payload["price"],
                "tags": result.payload["tags"],
                "collections": result.payload["collections"],
                "file_path": result.payload["file_path"],
                "filename": result.payload["filename"],
            }

        return None

    def search_crops_by_class(
        self, class_name: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search crops by specific class name (e.g., 'shirt', 'dress', etc.)"""
        search_results = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter={
                "must": [{"key": "class_name", "match": {"value": class_name}}]
            },
            limit=limit,
            with_payload=True,
        )

        results = []
        for result in search_results[0]:
            results.append(
                {
                    "crop_id": result.id,
                    "product_id": result.payload["product_id"],
                    "product_name": result.payload["product_name"],
                    "class_name": result.payload["class_name"],
                    "title": result.payload["title"],
                    "description": result.payload["description"],
                    "product_type": result.payload["product_type"],
                    "price": result.payload["price"],
                    "tags": result.payload["tags"],
                    "collections": result.payload["collections"],
                    "file_path": result.payload["file_path"],
                    "filename": result.payload["filename"],
                }
            )

        return results

    def search_crops_by_product(
        self, product_id: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get all crops for a specific product"""
        search_results = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter={
                "must": [{"key": "product_id", "match": {"value": product_id}}]
            },
            limit=limit,
            with_payload=True,
        )

        results = []
        for result in search_results[0]:
            results.append(
                {
                    "crop_id": result.id,
                    "product_id": result.payload["product_id"],
                    "product_name": result.payload["product_name"],
                    "class_name": result.payload["class_name"],
                    "title": result.payload["title"],
                    "description": result.payload["description"],
                    "product_type": result.payload["product_type"],
                    "price": result.payload["price"],
                    "tags": result.payload["tags"],
                    "collections": result.payload["collections"],
                    "file_path": result.payload["file_path"],
                    "filename": result.payload["filename"],
                }
            )

        return results

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the crop collection"""
        try:
            collection_info = self.client.get_collection(self.collection_name)

            # Get class distribution
            class_counts = {}
            scroll_results = self.client.scroll(
                collection_name=self.collection_name,
                limit=10000,  # Get a sample to calculate stats
                with_payload=True,
            )

            for result in scroll_results[0]:
                class_name = result.payload.get("class_name", "unknown")
                class_counts[class_name] = class_counts.get(class_name, 0) + 1

            return {
                "collection_name": self.collection_name,
                "total_crops": collection_info.points_count,
                "vector_size": 512,
                "class_distribution": class_counts,
                "status": str(collection_info.status),
            }
        except Exception as e:
            print(f"Error getting collection stats: {e}")
            return {}


def format_crop_results(
    results: List[Dict[str, Any]], format_type: str = "text"
) -> str:
    """Format crop search results as text or JSON"""
    if format_type.lower() == "json":
        return json.dumps(results, indent=2)
    else:
        # Text format
        output = []
        output.append(f"Found {len(results)} crop results:\n")

        for i, result in enumerate(results, 1):
            output.append(f"{i}. [{result['class_name']}] {result['title']}")
            output.append(f"   Crop ID: {result['crop_id']}")
            output.append(
                f"   Product: {result['product_name']} ({result['product_id']})"
            )
            output.append(f"   File: {result['filename']}")
            output.append(f"   Score: {result.get('score', 'N/A')}")
            if result["description"]:
                desc = (
                    result["description"][:80] + "..."
                    if len(result["description"]) > 80
                    else result["description"]
                )
                output.append(f"   Description: {desc}")
            output.append("")

        return "\n".join(output)


def main():
    """CLI interface for crop search"""
    parser = argparse.ArgumentParser(
        description="Search fashion product crops using FashionCLIP"
    )
    parser.add_argument("--text", type=str, help="Text query for search")
    parser.add_argument("--image", type=str, help="Image path for search")
    parser.add_argument("--class-name", type=str, help="Search by specific class name")
    parser.add_argument("--product-id", type=str, help="Get all crops for a product")
    parser.add_argument("--crop-id", type=str, help="Get specific crop by ID")
    parser.add_argument(
        "--stats", action="store_true", help="Show collection statistics"
    )
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

    querier = CropQuerier(
        qdrant_url=args.qdrant_url, qdrant_collection_name=args.collection
    )

    try:
        if args.stats:
            # Show collection statistics
            stats = querier.get_collection_stats()
            if stats:
                print("Collection Statistics:")
                print(f"  Collection: {stats['collection_name']}")
                print(f"  Total crops: {stats['total_crops']}")
                print(f"  Vector size: {stats['vector_size']}")
                print(f"  Status: {stats['status']}")
                print("\nClass distribution:")
                for class_name, count in sorted(
                    stats["class_distribution"].items(),
                    key=lambda x: x[1],
                    reverse=True,
                ):
                    print(f"  {class_name}: {count}")
            return

        if args.crop_id:
            # Get specific crop
            result = querier.get_crop_by_id(args.crop_id)
            if result:
                formatted_output = format_crop_results([result], args.format)
                print(formatted_output)
            else:
                print(f"Crop not found: {args.crop_id}")
            return

        if args.product_id:
            # Get all crops for product
            results = querier.search_crops_by_product(args.product_id, args.limit)
        elif args.class_name:
            # Search by class name
            results = querier.search_crops_by_class(args.class_name, args.limit)
        elif args.text and args.image:
            # Multimodal search
            results = querier.search_crops_by_multimodal(
                text_query=args.text,
                image_path=args.image,
                limit=args.limit,
                score_threshold=args.threshold,
            )
        elif args.text:
            # Text search
            results = querier.search_crops_by_text(
                query=args.text, limit=args.limit, score_threshold=args.threshold
            )
        elif args.image:
            # Image search
            results = querier.search_crops_by_image(
                image_path=args.image, limit=args.limit, score_threshold=args.threshold
            )
        else:
            print(
                "Error: Please provide a search query (--text, --image, --class-name, --product-id, --crop-id, or --stats)"
            )
            return

        # Format and print results
        formatted_output = format_crop_results(results, args.format)
        print(formatted_output)

    except Exception as e:
        print(f"Error during search: {e}")


if __name__ == "__main__":
    main()
