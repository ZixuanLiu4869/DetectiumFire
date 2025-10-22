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
  vertexai=True, project="detectium-ml-sandbox", location="us-east1",
)

model = "gemini-2.0-flash-001"

# Checkpoint file
CHECKPOINT_FILE = "detectium_video_checkpoint.json"
FINAL_OUTPUT_FILE = "detectium_video.json"
video_folder = "fire/"
SAVE_INTERVAL = 100  # Save every 1000 processed samples
MAX_WORKERS = 1  # Number of parallel processes (adjust based on your hardware)


def encode_video(video_path):
    return types.Video.from_file(video_path)

def video_caption(video_path):
    video_path = os.path.join(video_folder, video_path)

    # Encode image
    video = encode_video(video_path)

    # Step 1: Ask GPT-4o to reflect and refine while preserving structure
    system_prompt = """
    You are a fire scene expert. Describe only fire-related content in several sentences for the provided video. 
    Include what is burning, where it's happening. 
    Ignore irrelevant objects, background, and weather.
    """


    retries = 5
    wait_time = 60
    while retries > 0:
        try:
            response = client.models.generate_content(
                model=model,
                config=types.GenerateContentConfig(system_instruction=system_prompt,temperature=0.7,maxOutputTokens=128),
                contents=[
                    video,
                ],
            )
            
            print(response.text)
            return {
                "answer": response.text,
                "video_path": video_path
            }
        
        except Exception as e:
            print(f"API Error: {e}. Retrying ({6 - retries}/5)...")
            retries -= 1
            time.sleep(wait_time)  # Wait before retrying

    print(f"Skipping due to repeated failures: {video_path}")
    return {
        "answer": "",  # Mark failed attempts
        "video_path": video_path
    }

video_list = os.listdir(video_folder)


# Load checkpoint if exists
if os.path.exists(CHECKPOINT_FILE):
    with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
        refined_dataset = json.load(f)
    print(f"Resuming from checkpoint: {len(refined_dataset)} samples processed.")
else:
    refined_dataset = []


# Resume processing from last checkpoint
start_index = len(refined_dataset)
remaining_data = video_list[start_index:]  # Get only unprocessed data

# Function to process dataset in parallel
def process_batch(entry):
    return [video_caption(entry)]

start = time.time()

# Parallel processing
with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = []
    
    for index in range(start_index, len(video_list)):
        future = executor.submit(process_batch, video_list[index])
        futures.append(future)
    
    for future in as_completed(futures):
        try:
            results = future.result()
            refined_dataset.extend(results)
            print(len(refined_dataset))
            # Save checkpoint every SAVE_INTERVAL samples
            if len(refined_dataset) % SAVE_INTERVAL == 0:
                with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
                    json.dump(refined_dataset, f, indent=4)
                print(f"Checkpoint saved at {len(refined_dataset)} samples.")
                end = time.time()
                print(f"Time for processing 100 data is: {end-start}")
                start = time.time()

        except Exception as e:
            print(f"Error processing batch: {e}")

# Final save
with open(FINAL_OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(refined_dataset, f, indent=4)

print("Processing complete. Final results saved.")