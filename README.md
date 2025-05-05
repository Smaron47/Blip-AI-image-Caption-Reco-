# Blip-AI-image-Caption-Reco-
This Python script automates the generation of natural-language captions for large collections of images stored in Google Drive. It leverages the Salesforce BLIP (Bootstrapped Language–Image Pre-training) model via the Hugging Face transformers library, 
**BLIP Image Captioning Batch Processor**

---

## 1. Overview

This Python script automates the generation of natural-language captions for large collections of images stored in Google Drive. It leverages the [Salesforce BLIP (Bootstrapped Language–Image Pre-training)](https://huggingface.co/Salesforce/blip-image-captioning-large) model via the Hugging Face `transformers` library, processes images in parallel batches with `ThreadPoolExecutor`, and writes the results to a CSV file.

**Key capabilities:**

* Mounts Google Drive in Colab to access and save files
* Detects and uses GPU (if available) for faster BLIP inference
* Walks a Drive directory tree to discover all JPEG/PNG images
* Generates captions in batches using multithreading
* Writes per-image captions to a CSV file with columns `Image Name` and `Caption`

---

## 2. Prerequisites & Installation

1. **Google Colab environment** (or any Python runtime with Drive integration).
2. Install required packages:

   ```bash
   pip install transformers torch pillow
   ```
3. A Google Drive account with a `downloads` folder containing your images.

---

## 3. Directory & File Paths

* `base_drive_folder`: Root folder to scan for images (e.g. `/content/gdrive/My Drive/downloads`).
* `output_csv`: CSV file path for writing captions (e.g. `/content/gdrive/My Drive/captions1.csv`).

Customize these at the top of the script.

---

\## 4. Code Walkthrough

### 4.1 Drive Mounting

```python
from google.colab import drive

drive.mount('/content/gdrive')
```

Mounts your Google Drive under `/content/gdrive` to read images and write CSV output.

### 4.2 Model & Device Setup

```python
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# choose GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-large"
).to(device)
```

Loads the BLIP processor and model on the chosen device (GPU/CPU).

### 4.3 Caption Generation

```python
def generate_caption(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt").to(device)
    caption_ids = model.generate(**inputs)
    return processor.decode(caption_ids[0], skip_special_tokens=True)
```

* Reads and converts the image to RGB
* Tokenizes via `BlipProcessor`
* Runs `model.generate()` under GPU
* Decodes token IDs back to a human-readable string

### 4.4 Batch Processing

```python
def process_image_batch(image_paths):
    results = []
    for path in image_paths:
        caption = generate_caption(path)
        if caption:
            results.append((os.path.basename(path), caption))
    return results
```

Handles a small list of images, returns tuples `(filename, caption)`.

### 4.5 Parallelized Scanning & CSV Writing

```python
from concurrent.futures import ThreadPoolExecutor

def process_images_for_captions(folder_path, output_csv, batch_size=10):
    all_images = [...]  # walk folder and collect .jpg/.png
    with open(output_csv, "w", newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["Image Name","Caption"])
        writer.writeheader()
        with ThreadPoolExecutor() as executor:
            for i in range(0, len(all_images), batch_size):
                batch = all_images[i:i+batch_size]
                future = executor.submit(process_image_batch, batch)
                for name, cap in future.result():
                    writer.writerow({"Image Name": name, "Caption": cap})
```

* Recursively finds all supported image files under `folder_path`
* Splits them into batches of `batch_size`
* Uses a thread pool to caption each batch in parallel
* Writes each result row immediately to `output_csv`

---

## 5. Usage

1. Update the `base_drive_folder` and `output_csv` paths.
2. Run the notebook/cell in Colab.
3. Monitor the printout for progress (`Processed filename: caption`).
4. After completion, download or view `captions1.csv` from your Drive.

---

## 6. Error Handling

* Any exceptions in `generate_caption` print an error and skip that image.
* Invalid or missing files are ignored during the folder walk.

---

## 7. Customization & Extensions

* **Batch size**: Increase/decrease `batch_size` for speed/memory trade-offs.
* **Model variant**: Swap `"large"` with `"base"` or fine-tuned BLIP models.
* **Output**: Write JSON or plain text instead of CSV.
* **GPU fallback**: Extend to detect multi‑GPU setups or fall back to CPU.

---

## 8. Keywords

```
BLIP, image captioning, Hugging Face, Transformers, PyTorch, Google Colab, GPU inference, ThreadPoolExecutor, batch processing, PIL, CSV, concurrent processing
BLIP, image captioning, Hugging Face, Transformers, PyTorch, Google Colab, GPU inference, ThreadPoolExecutor, batch processing, PIL, CSV, concurrent processing

```
