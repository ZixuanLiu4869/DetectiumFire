import os
import random
import pandas as pd

# Define directories
fire_dir = "../fire"  # Path to fire videos directory
non_fire_dir = "../non_fire"  # Path to non-fire videos directory

# Output CSV files
train_csv = "train.csv"
val_csv = "val.csv"
test_csv = "test.csv"

# Split ratios
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# Collect video paths and labels
fire_videos = [(os.path.join(fire_dir, vid), 1) for vid in os.listdir(fire_dir) if vid.endswith(('.mp4', '.avi', '.mov'))]
non_fire_videos = [(os.path.join(non_fire_dir, vid), 2) for vid in os.listdir(non_fire_dir) if vid.endswith(('.mp4', '.avi', '.mov'))]

all_videos = fire_videos + non_fire_videos

# Shuffle videos
random.shuffle(all_videos)

# Split videos into train, val, test
num_videos = len(all_videos)
train_end = int(train_ratio * num_videos)
val_end = train_end + int(val_ratio * num_videos)

train_videos = all_videos[:train_end]
val_videos = all_videos[train_end:val_end]
test_videos = all_videos[val_end:]

# Save to CSV files
def save_to_csv(data, filename):
    df = pd.DataFrame(data, columns=["path", "label"])
    df.to_csv(filename, index=False)

save_to_csv(train_videos, train_csv)
save_to_csv(val_videos, val_csv)
save_to_csv(test_videos, test_csv)

print(f"Train, validation, and test splits saved to {train_csv}, {val_csv}, and {test_csv}, respectively.")
