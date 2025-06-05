import os
import json
import glob
import numpy as np
import argparse
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from typing import List, Dict, Any
from dotenv import load_dotenv
from urllib.parse import urlparse

# Load environment variables
load_dotenv()


class FashionVectorizer:
    def __init__(
        self,
        data_path: str = "/root/flickd-ai/tagging-engine/data/raw/downloaded_images",
        qdrant_url: str = None,
        qdrant_collection_name: str = None,
    ):
        self.data_path = data_path

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
        self._setup_collection()

    def _setup_device(self):
        """Setup and return the best available device"""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"CUDA available! Using GPU: {gpu_name} ({gpu_memory:.1f} GB)")
            print(f"Available GPUs: {gpu_count}")

            # Clear cache to start fresh
            torch.cuda.empty_cache()
            return device
        else:
            print("CUDA not available. Using CPU (this will be slower)")
            return torch.device("cpu")

    def _setup_collection(self):
        """Initialize Qdrant collection"""
        collections = self.client.get_collections().collections
        collection_exists = any(c.name == self.collection_name for c in collections)

        if not collection_exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=512, distance=Distance.COSINE),
            )
            print(f"Created collection: {self.collection_name}")

    def _generate_image_embeddings(self, image_paths: List[str]) -> np.ndarray:
        """Generate embeddings for multiple images using FashionCLIP with GPU acceleration"""
        try:
            # Use larger batch size for GPU, smaller for CPU
            batch_size = 64 if self.device.type == "cuda" else 16
            print(f"Processing {len(image_paths)} images with batch size {batch_size}")

            # Load and preprocess images
            images = []
            for img_path in image_paths:
                try:
                    image = Image.open(img_path).convert("RGB")
                    images.append(image)
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
                    continue

            if not images:
                print("No valid images to process")
                return None

            # Process images in batches
            all_embeddings = []
            for i in range(0, len(images), batch_size):
                batch_images = images[i : i + batch_size]

                # Preprocess batch
                inputs = self.processor(
                    images=batch_images, return_tensors="pt", padding=True
                )

                # Move inputs to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Generate embeddings
                with torch.no_grad():
                    image_features = self.model.get_image_features(**inputs)
                    # Normalize features
                    image_features = image_features / image_features.norm(
                        dim=-1, keepdim=True
                    )

                all_embeddings.append(image_features.cpu().numpy())

                # Clear GPU cache after each batch
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

            # Combine all embeddings
            embeddings = np.concatenate(all_embeddings, axis=0)

            return embeddings
        except Exception as e:
            print(f"Error processing images: {e}")
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            return None

    def _generate_text_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text using FashionCLIP with GPU acceleration"""
        try:
            # Ensure text is within token limits (77 tokens max, keeping under 40 as recommended)
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

            embedding = text_features.cpu().numpy()[0]

            # Clear GPU cache after processing
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            return embedding
        except Exception as e:
            print(f"Error processing text: {e}")
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            return None

    def _create_smart_text_description(self, product_info: Dict[str, Any]) -> str:
        """Create optimized text description that fits within FashionCLIP token limits (77 tokens max)"""
        # Extract key information in order of importance
        title = product_info.get("title", "")
        description = product_info.get("description", "")
        product_type = product_info.get("product_type", "")

        # Parse tags to extract most relevant attributes
        tags = product_info.get("product_tags", "")
        tag_parts = [tag.strip() for tag in tags.split(",") if tag.strip()]

        # Extract key attributes from tags
        color = next(
            (
                tag.split(":")[1].strip()
                for tag in tag_parts
                if tag.startswith("Colour:")
            ),
            "",
        )
        fabric = next(
            (
                tag.split(":")[1].strip()
                for tag in tag_parts
                if tag.startswith("Fabric:")
            ),
            "",
        )
        gender = next(
            (
                tag.split(":")[1].strip()
                for tag in tag_parts
                if tag.startswith("Gender:")
            ),
            "",
        )

        # Build description strategically, keeping it under 40 tokens as recommended
        parts = []

        # Start with product type for fashion context
        if product_type:
            parts.append(product_type)

        # Add color if available (very important for fashion)
        if color:
            parts.append(f"{color}")

        # Add fabric/material if available
        if fabric:
            parts.append(f"{fabric}")

        # Add gender context if available
        if gender and gender.lower() != "unisex":
            parts.append(f"for {gender}")

        # Add title if space allows
        if title:
            current_text = " ".join(parts)
            title_words = title.split()
            remaining_tokens = 35 - len(current_text.split())  # Keep buffer for safety

            if remaining_tokens > 0:
                truncated_title = " ".join(title_words[:remaining_tokens])
                parts.append(truncated_title)

        # If we still have space and description exists, add some of it
        current_text = " ".join(parts)
        if description and len(current_text.split()) < 30:
            desc_words = description.split()
            remaining_tokens = 35 - len(current_text.split())
            if remaining_tokens > 0:
                truncated_desc = " ".join(desc_words[:remaining_tokens])
                parts.append(truncated_desc)

        final_text = " ".join(parts)

        # Final safety check
        if len(final_text.split()) > 40:
            words = final_text.split()
            final_text = " ".join(words[:40])

        return final_text if final_text else title or "fashion item"

    def process_product_folder(self, folder_path: str) -> Dict[str, Any]:
        """Process a single product folder"""
        product_info_path = os.path.join(folder_path, "product_info.json")

        if not os.path.exists(product_info_path):
            print(f"No product_info.json found in {folder_path}")
            return None

        # Load product info
        with open(product_info_path, "r") as f:
            product_info = json.load(f)

        # Find all image files
        image_files = glob.glob(os.path.join(folder_path, "*.jpg"))
        image_files.extend(glob.glob(os.path.join(folder_path, "*.png")))
        image_files = [f for f in image_files if not f.endswith("product_info.json")]

        if not image_files:
            print(f"No images found in {folder_path}")
            return None

        # Generate image embeddings using FashionCLIP batch processing
        image_embeddings = self._generate_image_embeddings(image_files)

        # Create optimized text description
        text_description = self._create_smart_text_description(product_info)
        text_embedding = self._generate_text_embedding(text_description)

        if image_embeddings is None or text_embedding is None:
            print(f"Failed to generate embeddings for {folder_path}")
            return None

        # Use the average of image embeddings for multi-image products
        if len(image_embeddings.shape) > 1:
            combined_image_embedding = np.mean(image_embeddings, axis=0)
        else:
            combined_image_embedding = image_embeddings

        # Combine image and text embeddings (average them)
        combined_embedding = (combined_image_embedding + text_embedding.flatten()) / 2
        # Normalize the final combined embedding
        combined_embedding = combined_embedding / np.linalg.norm(combined_embedding)

        return {
            "embedding": combined_embedding,
            "metadata": {
                "product_id": product_info.get("product_id", "unknown"),
                "title": product_info.get("title", ""),
                "description": product_info.get("description", ""),
                "product_type": product_info.get("product_type", ""),
                "price": product_info.get("price_display_amount", ""),
                "tags": product_info.get("product_tags", ""),
                "collections": product_info.get("product_collections", ""),
                "image_count": len(image_files),
                "folder_path": folder_path,
                "text_description": text_description,  # Store the processed text for debugging
            },
        }

    def vectorize_all_products(self, max_products: int = None):
        """Process all product folders and store in Qdrant"""
        product_folders = [
            d
            for d in os.listdir(self.data_path)
            if os.path.isdir(os.path.join(self.data_path, d))
            and d.startswith("product_")
        ]

        points = []
        processed_count = 0
        skipped_count = 0
        folder_index = 0

        # Monitor GPU memory if using CUDA
        if self.device.type == "cuda":
            print(
                f"Initial GPU memory - Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB, "
                f"Cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB"
            )

        # Continue processing until we reach max_products or run out of folders
        while folder_index < len(product_folders) and (
            max_products is None or processed_count < max_products
        ):
            folder = product_folders[folder_index]
            folder_path = os.path.join(self.data_path, folder)
            folder_index += 1

            # Check if product_info.json exists
            product_info_path = os.path.join(folder_path, "product_info.json")
            if not os.path.exists(product_info_path):
                print(f"Skipping {folder} - no product_info.json found")
                skipped_count += 1
                continue

            # Check if folder has images before processing
            image_files = glob.glob(os.path.join(folder_path, "*.jpg"))
            image_files.extend(glob.glob(os.path.join(folder_path, "*.png")))
            image_files = [
                f for f in image_files if not f.endswith("product_info.json")
            ]

            if not image_files:
                print(f"Skipping {folder} - no images found")
                skipped_count += 1
                continue

            print(f"Processing {folder}...")

            result = self.process_product_folder(folder_path)
            if result:
                point = PointStruct(
                    id=int(result["metadata"]["product_id"]),
                    vector=result["embedding"].tolist(),
                    payload=result["metadata"],
                )
                points.append(point)
                processed_count += 1

                if max_products is not None:
                    print(f"Vectorized {processed_count}/{max_products} products")

                # Batch insert every 100 products
                if len(points) >= 100:
                    self.client.upsert(
                        collection_name=self.collection_name, points=points
                    )
                    points = []
                    print(f"Inserted batch, total processed: {processed_count}")

                    # Show GPU memory usage if using CUDA
                    if self.device.type == "cuda":
                        allocated = torch.cuda.memory_allocated() / 1024**3
                        cached = torch.cuda.memory_reserved() / 1024**3
                        print(
                            f"GPU memory - Allocated: {allocated:.2f} GB, Cached: {cached:.2f} GB"
                        )
            else:
                print(f"Failed to process {folder}")
                skipped_count += 1

        # Insert remaining points
        if points:
            self.client.upsert(collection_name=self.collection_name, points=points)

        if max_products is not None:
            print(
                f"Vectorization complete! Successfully vectorized {processed_count} products (limit: {max_products}), skipped {skipped_count} folders."
            )
        else:
            print(
                f"Vectorization complete! Processed {processed_count} products, skipped {skipped_count} folders."
            )

        # Final GPU memory report
        if self.device.type == "cuda":
            allocated = torch.cuda.memory_allocated() / 1024**3
            cached = torch.cuda.memory_reserved() / 1024**3
            print(
                f"Final GPU memory - Allocated: {allocated:.2f} GB, Cached: {cached:.2f} GB"
            )

    def search_products(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for products using text query"""
        try:
            # Generate embedding for the query
            query_embedding = self._generate_text_embedding(query)
            if query_embedding is None:
                print(f"Failed to generate embedding for query: {query}")
                return []

            # Search in Qdrant
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                limit=top_k,
                with_payload=True,
            )

            # Format results
            results = []
            for result in search_results:
                results.append(
                    {
                        "product_id": result.payload.get("product_id"),
                        "title": result.payload.get("title"),
                        "description": result.payload.get("description"),
                        "product_type": result.payload.get("product_type"),
                        "price": result.payload.get("price"),
                        "score": result.score,
                        "folder_path": result.payload.get("folder_path"),
                    }
                )

            return results

        except Exception as e:
            print(f"Error searching products: {e}")
            return []

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection"""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "status": str(collection_info.status),
                "vectors_count": collection_info.vectors_count,
                "points_count": collection_info.points_count,
            }
        except Exception as e:
            print(f"Error getting collection info: {e}")
            return {}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Vectorize fashion products using FashionCLIP"
    )
    parser.add_argument(
        "--max", type=int, help="Maximum number of products to vectorize"
    )
    args = parser.parse_args()

    vectorizer = FashionVectorizer()
    vectorizer.vectorize_all_products(max_products=args.max)
