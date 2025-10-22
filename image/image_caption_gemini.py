import os
from google import genai
from google.genai import types
import base64
from datasets import load_dataset
from tqdm import tqdm
import json 
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from itertools import islice
import numpy as np

tqdm.pandas()


client = genai.Client(
  vertexai=True, project="", location="us-east1",
)

model = "gemini-2.0-flash-001"

# Checkpoint file
CHECKPOINT_FILE = "fire_prompt_checkpoint.json"
FINAL_OUTPUT_FILE = "fire_prompt_gemini.json"
SAVE_INTERVAL = 100  # Save every 1000 processed samples
MAX_WORKERS = 1  # Number of parallel processes (adjust based on your hardware)
IMAGE_FOLDER = "images"  # Path to your image directory

# Function to encode image files to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()

        # Create an inline data part
        image_part = types.Part.from_bytes(
            data=image_bytes,
            mime_type='image/jpeg',
        )
        return image_part

def fire_caption(entry):
    image_path = entry # Path to the image file
    image_path = IMAGE_FOLDER+"/"+image_path

    # Encode image
    base64_image = encode_image(image_path)

    # Step 1: Ask GPT-4o to reflect and refine while preserving structure
    system_prompt = """
    You are a fire scene expert. Describe only fire-related content. 
    Ignore irrelevant objects, background, and weather.
    """

    # User prompt
    user_prompt = f"""
    TASK: Look at the provided image and produce a single-sentence caption (~75 tokens) describing ONLY fire-related content:
        - What is burning
        - Where it is happening (location/scene context)
        - Fire Severity. 
    In particular, a Fire Severity Level chosen EXACTLY from this list:
        – Controlled Fire (No Risk): A small fire used for cooking, lighting, or heating, posing no immediate danger.
        – Minor Fire (Low Risk): A small, contained fire (e.g., small trash fire) that could be extinguished easily.
        – Moderate Fire (Medium Risk): A fire spreading but still manageable, posing some risk to nearby objects or people.
        – Severe Fire (High Risk): A large, uncontrolled fire that is rapidly spreading and endangering structures or lives.

    OUTPUT RULES:
    - Return ONLY one senctence, ~75 tokens.

    EXAMPLE (format only, not content):
    A large fire engulfing a residential building with thick black smoke, which poses a severe fire risk.
    """

    retries = 5
    wait_time = 60
    while retries > 0:
        try:
            response = client.models.generate_content(
                model=model,
                config=types.GenerateContentConfig(system_instruction=system_prompt,temperature=0.7,maxOutputTokens=2048),
                contents=[
                    base64_image,
                    user_prompt,
                ],
            )
            caption = response.text
            print(caption)
            return {
                "caption": caption,
                "image_path": image_path
            }
        
        except Exception as e:
            print(f"API Error: {e}. Retrying ({6 - retries}/5)...")
            retries -= 1
            time.sleep(wait_time)  # Wait before retrying

    print(f"Skipping due to repeated failures for image {image_path}")
    return {
        "caption": "",  # Mark failed attempts
        "image_path": image_path
    }



# Load checkpoint if exists
if os.path.exists(CHECKPOINT_FILE):
    with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
        caption_dataset = json.load(f)
    print(f"Resuming from checkpoint: {len(caption_dataset)} samples processed.")
else:
    caption_dataset = []

imgs = sorted([p for p in os.listdir(IMAGE_FOLDER)])

# Resume processing from last checkpoint
start_index = len(caption_dataset)

# Function to process dataset in parallel
def process_batch(entry):
    return [fire_caption(entry)]

start = time.time()

# Parallel processing
with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = []
    
    for index in range(start_index, len(imgs)):
        future = executor.submit(process_batch, imgs[index])
        futures.append(future)
    
    for future in as_completed(futures):
        try:
            results = future.result()
            caption_dataset.extend(results)
            # Save checkpoint every SAVE_INTERVAL samples
            if len(caption_dataset) % SAVE_INTERVAL == 0:
                with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
                    json.dump(caption_dataset, f, indent=4)
                print(f"Checkpoint saved at {len(caption_dataset)} samples.")
                end = time.time()
                print(f"Time for processing 100 data is: {end-start}")
                start = time.time()

        except Exception as e:
            print(f"Error processing batch: {e}")

# Final save
with open(FINAL_OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(caption_dataset, f, indent=4)

print("Processing complete. Final results saved.")