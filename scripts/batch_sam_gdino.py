import argparse
import os
import json
import torch
import numpy as np
import pycocotools.mask as mask_util
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from sam2.sam2_image_predictor import SAM2ImagePredictor
from tqdm import tqdm
import glob
import time
from datetime import datetime, timedelta

def get_image_files(directory):
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(directory, ext)))
    return sorted(image_files)

def find_product_directories(input_dir):
    product_dirs = []
    for root, dirs, files in os.walk(input_dir):
        if "product_info.json" in files:
            image_files = get_image_files(root)
            if image_files:
                product_dirs.append({
                    "path": root,
                    "images": image_files,
                    "product_info": os.path.join(root, "product_info.json")
                })
    return product_dirs

def crop_image_with_mask(image, mask, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    cropped_img = image.crop((x1, y1, x2, y2))
    cropped_mask = mask[y1:y2, x1:x2]
    cropped_img = cropped_img.convert("RGBA")
    img_array = np.array(cropped_img)
    img_array[:, :, 3] = cropped_mask * 255
    return Image.fromarray(img_array)

def crop_image_bbox_only(image, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    return image.crop((x1, y1, x2, y2))

def single_mask_to_rle(mask):
    rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle

def validate_and_convert_image(image_path):
    try:
        image = Image.open(image_path)
        image.verify()
        image = Image.open(image_path)
        if image.mode in ('RGBA', 'LA', 'P'):
            rgb_image = Image.new('RGB', image.size, (255, 255, 255))
            if image.mode == 'P':
                image = image.convert('RGBA')
            rgb_image.paste(image, mask=image.split()[-1] if image.mode in ('RGBA', 'LA') else None)
            image = rgb_image
        elif image.mode == 'L':
            image = image.convert('RGB')
        elif image.mode != 'RGB':
            image = image.convert('RGB')
        if image.size[0] < 32 or image.size[1] < 32:
            print(f"Image too small: {image_path} ({image.size})")
            return None
        return image
    except Exception as e:
        print(f"Invalid image format in {image_path}: {str(e)}")
        return None

def process_single_image(image_path, sam2_predictor, grounding_processor, grounding_model, text_prompt, device, output_dir):
    try:
        image = validate_and_convert_image(image_path)
        if image is None:
            return False
        image_name = Path(image_path).stem
        grounding_inputs = grounding_processor(images=image, text=text_prompt, return_tensors="pt").to(device)
        with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.float16):
            grounding_outputs = grounding_model(**grounding_inputs)
        grounding_results = grounding_processor.post_process_grounded_object_detection(
            grounding_outputs,
            grounding_inputs.input_ids,
            box_threshold=0.4,
            text_threshold=0.3,
            target_sizes=[image.size[::-1]]
        )
        del grounding_inputs, grounding_outputs
        torch.cuda.empty_cache() if device == "cuda" else None
        if not grounding_results or not grounding_results[0]["boxes"].numel():
            print(f"No objects detected in {image_path}")
            return False
        image_array = np.array(image)
        if len(image_array.shape) != 3 or image_array.shape[2] != 3:
            print(f"Invalid image array shape: {image_array.shape} for {image_path}")
            return False
        sam2_predictor.set_image(image_array)
        input_boxes = grounding_results[0]["boxes"].cpu().numpy()
        with torch.no_grad():
            masks, scores, _ = sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_boxes,
                multimask_output=False
            )
        torch.cuda.empty_cache() if device == "cuda" else None
        if masks.ndim == 4:
            masks = masks.squeeze(1)
        confidences = grounding_results[0]["scores"].cpu().numpy().tolist()
        class_names = grounding_results[0]["text_labels"]
        bbox_dir = output_dir / "bbox_crops"
        masked_dir = output_dir / "masked_crops"
        bbox_dir.mkdir(parents=True, exist_ok=True)
        masked_dir.mkdir(parents=True, exist_ok=True)
        results = []
        for i, (class_name, bbox, mask, confidence) in enumerate(zip(class_names, input_boxes, masks, confidences)):
            bbox_crop = crop_image_bbox_only(image, bbox)
            bbox_filename = f"{image_name}_crop_{i}_{class_name.replace(' ', '_')}_bbox.png"
            bbox_path = bbox_dir / bbox_filename
            bbox_crop.save(bbox_path)
            masked_crop = crop_image_with_mask(image, mask, bbox)
            masked_filename = f"{image_name}_crop_{i}_{class_name.replace(' ', '_')}_masked.png"
            masked_path = masked_dir / masked_filename
            masked_crop.save(masked_path)
            results.append({
                "class_name": class_name,
                "bbox": bbox.tolist(),
                "segmentation": single_mask_to_rle(mask),
                "score": float(confidence),
                "bbox_crop_path": str(bbox_path.relative_to(output_dir)),
                "masked_crop_path": str(masked_path.relative_to(output_dir))
            })
        results_data = {
            "original_image_path": image_path,
            "image_name": image_name,
            "annotations": results,
            "box_format": "xyxy",
            "img_width": image.width,
            "img_height": image.height
        }
        results_path = output_dir / f"{image_name}_results.json"
        with open(results_path, "w") as f:
            json.dump(results_data, f, indent=2)
        return True
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return False

def is_image_already_processed(image_path, output_dir):
    image_name = Path(image_path).stem
    results_path = output_dir / f"{image_name}_results.json"
    if not results_path.exists():
        return False
    try:
        with open(results_path, 'r') as f:
            results_data = json.load(f)
        if 'annotations' not in results_data or not results_data['annotations']:
            return False
        bbox_dir = output_dir / "bbox_crops"  # noqa: F841
        masked_dir = output_dir / "masked_crops"  # noqa: F841
        for annotation in results_data['annotations']:
            bbox_crop_path = output_dir / annotation['bbox_crop_path']
            masked_crop_path = output_dir / annotation['masked_crop_path']
            if not bbox_crop_path.exists() or not masked_crop_path.exists():
                return False
        return True
    except (json.JSONDecodeError, KeyError, FileNotFoundError):
        return False

def filter_images_to_process(product_dirs, output_base, force_reprocess=False):
    if force_reprocess:
        return product_dirs
    filtered_dirs = []
    total_skipped = 0
    for product_dir in product_dirs:
        product_name = Path(product_dir["path"]).name
        product_output_dir = output_base / product_name
        unprocessed_images = []
        for image_path in product_dir["images"]:
            if not is_image_already_processed(image_path, product_output_dir):
                unprocessed_images.append(image_path)
            else:
                total_skipped += 1
        if unprocessed_images:
            filtered_dirs.append({
                "path": product_dir["path"],
                "images": unprocessed_images,
                "product_info": product_dir["product_info"]
            })
    print(f"Skipping {total_skipped} already processed images")
    return filtered_dirs

def main():
    parser = argparse.ArgumentParser(description="Batch process product images with SAM2 + Grounding DINO")
    parser.add_argument("--grounding-model", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--sam2-model", default="facebook/sam2.1-hiera-base-plus")
    parser.add_argument("--text-prompt", default="""wristwear. topwear. bottomwear. footwear. cap. hat. bow. headband. accessories. bag. outerwear.""")
    parser.add_argument("--input-dir", default="/root/flickd-ai/tagging-engine/data/raw/downloaded_images")
    parser.add_argument("--output-dir", default="/root/flickd-ai/tagging-engine/data/processed/product_images")
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--force-reprocess", action="store_true", help="Force reprocessing of all images, even if already processed")
    args = parser.parse_args()
    start_time = time.time()
    start_datetime = datetime.now()
    device = "cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu"
    print(f"Starting batch processing at {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Using device: {device}")
    if args.force_reprocess:
        print("Force reprocess mode: Will process all images")
    else:
        print("Resume mode: Will skip already processed images")
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.enable_flash_sdp(True)
    print(f"Loading SAM2 model: {args.sam2_model}")
    sam2_predictor = SAM2ImagePredictor.from_pretrained(args.sam2_model, device=device)
    print(f"Loading Grounding DINO model: {args.grounding_model}")
    grounding_processor = AutoProcessor.from_pretrained(args.grounding_model)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        args.grounding_model,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    ).to(device)
    print(f"Scanning for product directories in {args.input_dir}...")
    all_product_dirs = find_product_directories(args.input_dir)
    print(f"Found {len(all_product_dirs)} product directories")
    if not all_product_dirs:
        print("No valid product directories found!")
        return
    output_base = Path(args.output_dir)
    total_images_found = sum(len(pd["images"]) for pd in all_product_dirs)
    print(f"Total images found: {total_images_found}")
    product_dirs = filter_images_to_process(all_product_dirs, output_base, args.force_reprocess)
    if not product_dirs:
        print("All images have already been processed!")
        return
    total_images = sum(len(pd["images"]) for pd in product_dirs)
    processed_images = 0
    successful_images = 0
    print(f"Images to process: {total_images}")
    print(f"Text prompt: {args.text_prompt}")
    print(f"Output directory: {args.output_dir}")
    print("Starting batch processing...")
    overall_pbar = tqdm(total=total_images, desc="Overall Progress", position=0)
    for product_idx, product_dir in enumerate(product_dirs):
        product_name = Path(product_dir["path"]).name
        product_output_dir = output_base / product_name
        product_output_dir.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(product_dir["product_info"], product_output_dir / "product_info.json")
        product_image_count = len(product_dir["images"])
        remaining_images = total_images - processed_images
        current_time = time.time()
        elapsed_seconds = current_time - start_time
        elapsed_str = str(timedelta(seconds=int(elapsed_seconds)))
        product_desc = f"Product {product_idx + 1}/{len(product_dirs)}: {product_name} ({product_image_count} imgs)"
        print(f"Processing Product {product_idx + 1}/{len(product_dirs)}: {product_name}")
        print(f"Images in this product: {product_image_count}")
        print(f"Total processed so far: {processed_images}/{total_images}")
        print(f"Time elapsed: {elapsed_str}")
        print(f"Images remaining: {remaining_images}")
        product_pbar = tqdm(product_dir["images"], desc=product_desc, position=1, leave=False)
        for image_path in product_pbar:
            success = process_single_image(
                image_path,
                sam2_predictor,
                grounding_processor,
                grounding_model,
                args.text_prompt,
                device,
                product_output_dir
            )
            processed_images += 1
            if success:
                successful_images += 1
            overall_pbar.update(1)
            remaining_images = total_images - processed_images
            current_elapsed = time.time() - start_time
            elapsed_str = str(timedelta(seconds=int(current_elapsed)))
            overall_pbar.set_description(
                f"Overall Progress | Processed: {processed_images}/{total_images} | Success: {successful_images} | Elapsed: {elapsed_str}"
            )
        product_pbar.close()
    overall_pbar.close()
    end_time = time.time()
    end_datetime = datetime.now()
    total_elapsed = end_time - start_time
    total_elapsed_str = str(timedelta(seconds=int(total_elapsed)))
    avg_time_per_image = total_elapsed / processed_images if processed_images > 0 else 0
    print("Batch processing complete!")
    print(f"Total images found: {total_images}")
    print(f"Total images processed: {processed_images}")
    print(f"Successfully processed: {successful_images}")
    print(f"Failed: {processed_images - successful_images}")
    print(f"Success rate: {(successful_images/processed_images*100):.1f}%")
    print(f"Total time elapsed: {total_elapsed_str}")
    print(f"Average time per image: {avg_time_per_image:.2f} seconds")
    print(f"Started at: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Finished at: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
