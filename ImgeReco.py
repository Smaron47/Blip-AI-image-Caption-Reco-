from google.colab import drive
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import os
import csv
from concurrent.futures import ThreadPoolExecutor
import torch

# Mount Google Drive
drive.mount('/content/gdrive')

# Paths
base_drive_folder = "/content/gdrive/My Drive/downloads"  # Path to the "downloads" folder in Google Drive
output_csv = "/content/gdrive/My Drive/captions1.csv"  # Path to save the CSV file

# Check for GPU availability and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load BLIP model and processor on GPU
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)

def generate_caption(image_path):
    """Generate caption for an image using GPU."""
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(image, return_tensors="pt").to(device)
        caption = model.generate(**inputs)
        return processor.decode(caption[0], skip_special_tokens=True)
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def process_image_batch(image_paths):
    """Process a batch of images and return captions."""
    results = []
    for image_path in image_paths:
        caption = generate_caption(image_path)
        if caption:
            results.append((os.path.basename(image_path), caption))
    return results

def process_images_for_captions(folder_path, output_csv, batch_size=10):
    """Process images in batches using GPU and save captions to a CSV file."""
    fieldnames = ["Image Name", "Caption"]
    all_images = []

    # Collect all image paths
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                all_images.append(os.path.join(root, file))

    print(f"Found {len(all_images)} images to process.")

    # Process images in batches
    with open(output_csv, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        with ThreadPoolExecutor() as executor:
            # Divide images into batches
            for i in range(0, len(all_images), batch_size):
                batch = all_images[i:i + batch_size]
                future = executor.submit(process_image_batch, batch)
                results = future.result()

                # Write results to the CSV
                for image_name, caption in results:
                    writer.writerow({
                        "Image Name": image_name,
                        "Caption": caption,
                    })
                    print(f"Processed {image_name}: {caption}")

if __name__ == "__main__":
    process_images_for_captions(base_drive_folder, output_csv)
    print(f"Captions saved to {output_csv}.")
