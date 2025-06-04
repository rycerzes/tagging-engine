import asyncio
import csv
import json
import os
import urllib.parse
from pathlib import Path
from typing import Dict, List, Tuple

import httpx


async def download_image(client: httpx.AsyncClient, url: str, filepath: Path) -> bool:
    """Download a single image from URL to filepath."""
    try:
        response = await client.get(url, timeout=30.0)
        response.raise_for_status()

        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "wb") as f:
            f.write(response.content)

        print(f"Downloaded: {filepath.name}")
        return True

    except Exception as e:
        print(f"Failed to download {url}: {str(e)}")
        return False


def parse_images_csv(csv_path: str) -> List[Tuple[str, str]]:
    """Parse images CSV file and extract (id, image_url) pairs."""
    image_data = []

    with open(csv_path, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if "id" in row and "image_url" in row:
                image_data.append((row["id"], row["image_url"]))

    return image_data


def get_image_filename(url: str, product_id: str, index: int) -> str:
    """Generate filename from URL, product ID, and index."""
    parsed_url = urllib.parse.urlparse(url)
    original_filename = os.path.basename(parsed_url.path)

    _, ext = os.path.splitext(original_filename)
    if not ext:
        ext = ".jpg"

    return f"{product_id}_{index:03d}{ext}"


async def download_images_for_product(
    client: httpx.AsyncClient,
    product_id: str,
    urls: List[str],
    download_dir: Path,
) -> None:
    """Download all images for a specific product."""
    product_dir = download_dir / f"product_{product_id}"

    tasks = []
    for i, url in enumerate(urls):
        filename = get_image_filename(url, product_id, i)
        filepath = product_dir / filename

        if filepath.exists():
            print(f"Skipping existing file: {filepath.name}")
            continue

        tasks.append(download_image(client, url, filepath))

    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


def parse_product_details_csv(csv_path: str) -> Dict[str, Dict[str, str]]:
    """Parse product details CSV file and extract product information."""
    product_details = {}
    
    try:
        with open(csv_path, "r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            
            print(f"CSV columns found: {reader.fieldnames}")
            
            for row_num, row in enumerate(reader, start=2):  
                if row_num <= 3:
                    print(f"Row {row_num}: {dict(row)}")
                
                if "id" in row and row["id"]:  
                    product_id = str(row["id"]).strip()
                    product_details[product_id] = {
                        "title": row.get("title", "").strip(),
                        "description": row.get("description", "").strip(),
                        "product_type": row.get("product_type", "").strip(),
                        "alias": row.get("alias", "").strip(),
                        "mrp": row.get("mrp", "").strip(),
                        "price_display_amount": row.get("price_display_amount", "").strip(),
                        "discount_percentage": row.get("discount_percentage", "").strip(),
                        "product_tags": row.get("product_tags", "").strip(),
                        "product_collections": row.get("product_collections", "").strip()
                    }
                else:
                    print(f"Warning: Row {row_num} missing 'id' field or empty id")
                    
    except Exception as e:
        print(f"Error parsing CSV file: {str(e)}")
        return {}
    
    print(f"Successfully parsed {len(product_details)} products")
    return product_details


def write_product_info(product_id: str, product_info: Dict[str, str], download_dir: Path) -> None:
    """Write product information to a text file in the product folder."""
    product_dir = download_dir / f"product_{product_id}"
    product_dir.mkdir(parents=True, exist_ok=True)
    
    info_file = product_dir / "product_info.txt"
    
    with open(info_file, "w", encoding="utf-8") as f:
        f.write(f"Product ID: {product_id}\n")
        f.write(f"Title: {product_info['title']}\n")
        f.write(f"Description: {product_info['description']}\n")
        f.write(f"Product Type: {product_info['product_type']}\n")
        f.write(f"Alias: {product_info['alias']}\n")
        f.write(f"MRP: {product_info['mrp']}\n")
        f.write(f"Price Display Amount: {product_info['price_display_amount']}\n")
        f.write(f"Discount Percentage: {product_info['discount_percentage']}\n")
        f.write(f"Product Tags: {product_info['product_tags']}\n")
        f.write(f"Product Collections: {product_info['product_collections']}\n")
    
    print(f"Created product info file: {info_file.name}")


def write_product_info_json(product_id: str, product_info: Dict[str, str], download_dir: Path) -> None:
    """Write product information to a JSON file in the product folder."""
    product_dir = download_dir / f"product_{product_id}"
    product_dir.mkdir(parents=True, exist_ok=True)
    
    info_file = product_dir / "product_info.json"
    
    json_data = {
        "product_id": product_id,
        "title": product_info['title'],
        "description": product_info['description'],
        "product_type": product_info['product_type'],
        "alias": product_info['alias'],
        "mrp": product_info['mrp'],
        "price_display_amount": product_info['price_display_amount'],
        "discount_percentage": product_info['discount_percentage'],
        "product_tags": product_info['product_tags'],
        "product_collections": product_info['product_collections']
    }
    
    with open(info_file, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    print(f"Created product JSON file: {info_file.name}")


def show_menu() -> int:
    """Display menu and get user choice."""
    print("\n" + "="*50)
    print("IMAGE DOWNLOADER")
    print("="*50)
    print("1. Download images")
    print("2. Add textual descriptions to image folders")
    print("3. Exit")
    print("-"*50)
    
    while True:
        try:
            choice = int(input("Enter your choice (1-3): ").strip())
            if choice in [1, 2, 3]:
                return choice
            else:
                print("Please enter 1, 2, or 3.")
        except ValueError:
            print("Please enter a valid number.")


async def download_images_process():
    """Handle the image downloading process."""
    images_csv = input("Enter path to images CSV file (id, image_url): ").strip()
    download_dir = Path("downloaded_images")

    if not os.path.exists(images_csv):
        print(f"Images CSV file not found: {images_csv}")
        return

    print("Parsing images CSV file...")
    image_data = parse_images_csv(images_csv)

    if not image_data:
        print("No image data found in CSV file.")
        return

    product_images = {}
    for product_id, url in image_data:
        if product_id not in product_images:
            product_images[product_id] = []
        product_images[product_id].append(url)

    print(f"Found {len(image_data)} images for {len(product_images)} products")

    async with httpx.AsyncClient() as client:
        for product_id, urls in product_images.items():
            print(f"\nDownloading {len(urls)} images for product {product_id}...")
            await download_images_for_product(client, product_id, urls, download_dir)

    print(f"\nDownload complete! Images saved to: {download_dir.absolute()}")


def add_text_descriptions_process():
    """Handle adding textual descriptions to product folders."""
    product_details_csv = input("Enter path to product details CSV file: ").strip()
    download_dir = Path("data/raw/downloaded_images")

    if not os.path.exists(product_details_csv):
        print(f"Product details CSV file not found: {product_details_csv}")
        return

    print("Parsing product details CSV file...")
    product_details = parse_product_details_csv(product_details_csv)

    if not product_details:
        print("No product details found in CSV file.")
        return

    print(f"Found details for {len(product_details)} products")

    created_count = 0
    for product_id, product_info in product_details.items():
        product_dir = download_dir / f"product_{product_id}"
        if product_dir.exists():
            print(f"Adding info for product {product_id}...")
            write_product_info(product_id, product_info, download_dir)
            write_product_info_json(product_id, product_info, download_dir)
            created_count += 1
        else:
            print(f"Warning: Product folder for ID {product_id} not found, creating anyway...")
            write_product_info(product_id, product_info, download_dir)
            write_product_info_json(product_id, product_info, download_dir)
            created_count += 1

    print(f"\nText descriptions added! Created {created_count} info files (both .txt and .json) in: {download_dir.absolute()}")


async def main():
    """Main function with menu system."""
    while True:
        choice = show_menu()
        
        if choice == 1:
            await download_images_process()
        elif choice == 2:
            add_text_descriptions_process()
        elif choice == 3:
            print("Goodbye!")
            break
        
        input("\nPress Enter to continue...")


if __name__ == "__main__":
    asyncio.run(main())
