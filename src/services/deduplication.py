import numpy as np
import faiss
from typing import List, Dict, Any, Tuple
from PIL import Image
from pathlib import Path
import torch
from transformers import CLIPProcessor, CLIPModel
import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from urllib.parse import urlparse
import hashlib
from dotenv import load_dotenv
import asyncio

from ..config import DEVICE

# Load environment variables
load_dotenv()


class FaissDeduplicationService:
    def __init__(self, similarity_threshold: float = 0.85):  # Much lower threshold
        self.similarity_threshold = similarity_threshold
        self._processor = None
        self._model = None
        self.model_name = "patrickjohncyh/fashion-clip"

        # Qdrant configuration
        self.qdrant_url = os.getenv("QDRANT_URL")
        self.fashion_products_collection = os.getenv("QDRANT_COLLECTION_NAME")

        if not self.qdrant_url:
            raise ValueError("QDRANT_URL must be set in environment variables")

        # Parse Qdrant URL to extract host and port
        parsed_url = urlparse(self.qdrant_url)
        qdrant_host = parsed_url.hostname
        qdrant_port = parsed_url.port or 6333

        self.client = QdrantClient(host=qdrant_host, port=qdrant_port)

    def _setup_collection(self, collection_name: str):
        """Initialize Qdrant collection - delete and recreate if it exists"""
        try:
            collections = self.client.get_collections().collections
            collection_exists = any(c.name == collection_name for c in collections)

            if collection_exists:
                print(f"Collection {collection_name} exists, deleting...")
                self.client.delete_collection(collection_name=collection_name)
                print(f"Deleted collection: {collection_name}")

            # Create new collection
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=512, distance=Distance.COSINE),
            )
            print(f"Created new Qdrant collection: {collection_name}")
        except Exception as e:
            print(f"Error setting up Qdrant collection: {e}")
            raise

    def _generate_crop_id(self, filename: str, video_id: str) -> str:
        """Generate unique ID for crop"""
        combined = f"{video_id}_{filename}"
        return hashlib.md5(combined.encode()).hexdigest()

    def _load_model(self):
        """Lazy load the Fashion-CLIP model"""
        if self._processor is None or self._model is None:
            print(f"Loading Fashion-CLIP model: {self.model_name}")
            self._processor = CLIPProcessor.from_pretrained(self.model_name)
            self._model = CLIPModel.from_pretrained(self.model_name).to(DEVICE)
            self._model.eval()

    def _get_image_embedding(self, image_path: Path) -> np.ndarray:
        """Extract CLIP image embedding"""
        image = Image.open(image_path).convert("RGB")

        inputs = self._processor(images=image, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            image_features = self._model.get_image_features(**inputs)
            # Normalize the features
            image_features = image_features / image_features.norm(
                p=2, dim=-1, keepdim=True
            )

        return image_features.cpu().numpy().flatten()

    def _query_fashion_products(
        self, embedding: np.ndarray, limit: int = 2
    ) -> List[Dict[str, Any]]:
        """Query similar products from fashion_products collection"""
        try:
            search_results = self.client.query_points(
                collection_name=self.fashion_products_collection,
                query=embedding.tolist(),
                limit=limit,
                score_threshold=0.75,
                with_payload=True,
            )

            results = []
            for result in search_results.points:
                results.append(
                    {
                        "product_id": result.id,
                        "score": result.score,
                        "payload": result.payload,
                    }
                )

            return results
        except Exception as e:
            print(f"Error querying fashion_products collection: {e}")
            return []

    def deduplicate_crops(
        self,
        cropped_dir: Path,
        cropped_files: List[Dict[str, Any]],
        video_id: str = None,
    ) -> Tuple[List[Dict[str, Any]], List[str], List[PointStruct]]:
        """
        Deduplicate cropped images using FAISS similarity search

        Returns:
            - List of unique cropped files (deduplicated)
            - List of filenames that were removed as duplicates
        """
        if not cropped_files:
            return cropped_files, []

        if not video_id:
            raise ValueError("video_id is required for collection naming")

        # Use video_id as collection name
        collection_name = video_id
        self._setup_collection(collection_name)

        self._load_model()

        print(f"Starting deduplication of {len(cropped_files)} cropped images...")

        # Group by class name first to deduplicate within same categories more aggressively
        class_groups = {}
        for i, crop_info in enumerate(cropped_files):
            class_name = crop_info.get("original_class_name", "unknown")
            if class_name not in class_groups:
                class_groups[class_name] = []
            class_groups[class_name].append((i, crop_info))

        # Extract embeddings for all images
        embeddings = []
        valid_files = []
        original_indices = []

        for crop_info in cropped_files:
            crop_path = cropped_dir / crop_info["filename"]
            if crop_path.exists():
                try:
                    embedding = self._get_image_embedding(crop_path)
                    embeddings.append(embedding)
                    valid_files.append(crop_info)
                    original_indices.append(len(valid_files) - 1)
                except Exception as e:
                    print(f"Failed to process {crop_info['filename']}: {e}")
                    continue

        if len(embeddings) == 0:
            return [], [], []

        # Convert to numpy array and normalize
        embeddings = np.array(embeddings).astype(np.float32)
        faiss.normalize_L2(embeddings)

        # Build FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        index.add(embeddings)

        # Find duplicates with more aggressive thresholds
        similarities, indices = index.search(embeddings, len(embeddings))

        # Track which images to keep and which to remove
        to_remove = set()
        removed_files = []

        # First pass: remove very similar images within same class (higher threshold)
        class_threshold = 0.90
        for class_name, class_items in class_groups.items():
            if len(class_items) <= 1:
                continue

            class_indices = [
                item[0] for item in class_items if item[0] < len(valid_files)
            ]

            for i in class_indices:
                if i in to_remove:
                    continue

                for j in range(
                    1, min(10, len(similarities[i]))
                ):  # Check top 10 similar
                    similar_idx = indices[i][j]
                    if (
                        similar_idx in class_indices
                        and similar_idx not in to_remove
                        and similarities[i][j] >= class_threshold
                        and similar_idx > i
                    ):
                        to_remove.add(similar_idx)
                        removed_files.append(valid_files[similar_idx]["filename"])

        # Second pass: remove similar images across all classes (lower threshold)
        for i in range(len(valid_files)):
            if i in to_remove:
                continue

            # Find similar images (excluding self)
            for j in range(1, min(5, len(similarities[i]))):  # Check top 5 similar
                if similarities[i][j] >= self.similarity_threshold:
                    similar_idx = indices[i][j]
                    if similar_idx not in to_remove and similar_idx > i:
                        to_remove.add(similar_idx)
                        removed_files.append(valid_files[similar_idx]["filename"])

        # Remove duplicate files from disk
        for filename in removed_files:
            file_path = cropped_dir / filename
            if file_path.exists():
                os.remove(file_path)
                print(f"Removed duplicate: {filename}")

        # Process unique files - prepare for Qdrant storage
        unique_files = []
        points_to_store = []

        for i in range(len(valid_files)):
            if i in to_remove:
                continue

            file_info = valid_files[i]
            embedding = embeddings[i]

            # Generate crop ID
            crop_id = self._generate_crop_id(file_info["filename"], video_id)
            file_info["crop_id"] = crop_id

            # Prepare point for Qdrant storage
            metadata = {
                "filename": file_info["filename"],
                "class_name": file_info.get("class_name", "unknown"),
                "original_class_name": file_info.get("original_class_name", "unknown"),
                "bbox": file_info.get("bbox", []),
                "video_id": video_id,
                "file_path": str(cropped_dir / file_info["filename"]),
            }

            point = PointStruct(id=crop_id, vector=embedding.tolist(), payload=metadata)
            points_to_store.append(point)
            unique_files.append(file_info)

        print(
            f"Deduplication complete: {len(unique_files)} unique images, {len(removed_files)} duplicates removed"
        )

        # Store embeddings in Qdrant after deduplication
        try:
            self.client.upsert(collection_name=collection_name, points=points_to_store)
            print(
                f"Stored {len(points_to_store)} embeddings in Qdrant collection: {collection_name}"
            )
        except Exception as e:
            print(f"Error storing embeddings in Qdrant: {e}")

        # Cleanup
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        return unique_files, removed_files, points_to_store

    async def _store_embeddings_async(
        self, points_to_store: List, collection_name: str
    ) -> None:
        """Asynchronously store embeddings in Qdrant"""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.client.upsert(
                    collection_name=collection_name, points=points_to_store
                ),
            )
            print(
                f"Stored {len(points_to_store)} embeddings in Qdrant collection: {collection_name}"
            )
        except Exception as e:
            print(f"Error storing embeddings in Qdrant: {e}")
