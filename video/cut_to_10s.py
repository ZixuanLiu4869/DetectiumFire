import cv2
import os
from tqdm import tqdm

tqdm.pandas()


def process_videos(source_folder, output_folder, clip_length=10):
    # Ensure output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Iterate over all files in the source directory
    for filename in tqdm(os.listdir(source_folder)):
        if filename.endswith((".mp4", ".mov", ".avi")):  # Check for video files
            filepath = os.path.join(source_folder, filename)
            cap = cv2.VideoCapture(filepath)
            
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps
            clip_frames = clip_length * fps

            current_clip = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Start a new video file every `clip_frames` frames
                if current_clip * clip_frames >= total_frames:
                    break

                output_filename = f"{filename[:-4]}_clip_{current_clip + 1:03d}.mp4"
                output_filepath = os.path.join(output_folder, output_filename)
                
                # Define the codec and create VideoWriter object
                fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
                out = cv2.VideoWriter(output_filepath, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

                frame_count = 0
                while frame_count < clip_frames:
                    if frame is not None:
                        out.write(frame)
                    ret, frame = cap.read()
                    frame_count += 1
                    if not ret:
                        break
                
                out.release()  # Release the file
                current_clip += 1

            cap.release()  # Release the capture


# Example usage
source_folder = 'non_fire'
output_folder = 'non_fire_clip'
process_videos(source_folder, output_folder)
