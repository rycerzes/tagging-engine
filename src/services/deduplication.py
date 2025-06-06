import numpy as np
import faiss
from typing import List, Dict, Any, Tuple
from PIL import Image
from pathlib import Path
import torch
from transformers import CLIPProcessor, CLIPModel
import os

from ..config import DEVICE


class FaissDeduplicationService:
    def __init__(self, similarity_threshold: float = 0.85):  # Much lower threshold
        self.similarity_threshold = similarity_threshold
        self._processor = None
        self._model = None
        self.model_name = "patrickjohncyh/fashion-clip"

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
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

        return image_features.cpu().numpy().flatten()

    def deduplicate_crops(
        self, cropped_dir: Path, cropped_files: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Deduplicate cropped images using FAISS similarity search

        Returns:
            - List of unique cropped files (deduplicated)
            - List of filenames that were removed as duplicates
        """
        if not cropped_files:
            return cropped_files, []

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
            return [], []

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
                
            class_indices = [item[0] for item in class_items if item[0] < len(valid_files)]
            
            for i in class_indices:
                if i in to_remove:
                    continue
                    
                for j in range(1, min(10, len(similarities[i]))):  # Check top 10 similar
                    similar_idx = indices[i][j]
                    if (similar_idx in class_indices and 
                        similar_idx not in to_remove and 
                        similarities[i][j] >= class_threshold and
                        similar_idx > i):
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

        # Return deduplicated list
        unique_files = [
            valid_files[i] for i in range(len(valid_files)) 
            if i not in to_remove
        ]

        print(f"Deduplication complete: {len(unique_files)} unique images, {len(removed_files)} duplicates removed")

        # Cleanup
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        return unique_files, removed_files
