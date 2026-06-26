import os
import scipy.io
import numpy as np
from dataset import VideoDatasetLoader
from base_model import ATBMultiModel

def generate_submission_files(input_dir="Dev_data", output_dir="output"):
    while True:
        try:
            n_val = int(input("\nEnter the value of n (1 to 5): "))
            if 1 <= n_val <= 5:
                break
            else:
                print("Please enter a valid number between 1 and 5.")
        except ValueError:
            print("Invalid input. Please enter an integer.")

    print(f"\n--- Generating Predictions for n = {n_val} ---")
    
    os.makedirs(output_dir, exist_ok=True)
    loader = VideoDatasetLoader(input_dir)
    videos_dict = loader.load_all_videos()
    
    if not videos_dict:
        print(f"No videos found in '{input_dir}'. Please ensure your Dev Data is there.")
        return

    base_model = ATBMultiModel(n=n_val, num_points=100)

    #  Process each video
    for video_name, frames in videos_dict.items():
        total_frames = len(frames)
        valid_frames = range(n_val, total_frames - n_val)
        
        video_predictions = []
        
        print(f"Processing '{video_name}' (Target Frames: {len(valid_frames)})...")
        
        for t in range(total_frames):
            frame_output = {'frame_index': t + 1} 
            
            if t in valid_frames:
                prev_f = frames[t - n_val]
                next_f = frames[t + n_val]
                
                # Generate the ATB contours
                preds = base_model.predict(prev_f, next_f)
                
                for c_name in base_model.contour_names:
                    contour_data = preds.get(c_name)
                    if contour_data is None:
                        frame_output[c_name] = np.full((100, 2), np.nan)
                    else:
                        frame_output[c_name] = contour_data
            else:
                for c_name in base_model.contour_names:
                    frame_output[c_name] = np.full((100, 2), np.nan)
                    
            video_predictions.append(frame_output)
            
        # Save to a new .mat file
        output_filename = video_name.replace('.mat', f'_predicted_n{n_val}.mat')
        output_path = os.path.join(output_dir, output_filename)
        
        scipy.io.savemat(output_path, {'predictions': video_predictions})
        
    print(f"\nPredicted contour files saved to the '{os.path.abspath(output_dir)}' folder.")

if __name__ == "__main__":
    DEV_DATA_DIRECTORY = "Dev_data" 
    
    generate_submission_files(input_dir=DEV_DATA_DIRECTORY)