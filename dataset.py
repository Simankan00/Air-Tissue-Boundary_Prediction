import os
import scipy.io
import numpy as np

class VideoDatasetLoader:
    def __init__(self, directory):
        self.directory = directory
        self.dataset = {}

    def load_all_videos(self):
        """
        Loads all .mat files in the directory.
        Maintains the MATLAB structure to support multi-contour extraction.
        """
        if not os.path.exists(self.directory):
            raise FileNotFoundError(f"Directory not found: {self.directory}")

        for filename in os.listdir(self.directory):
            if filename.endswith(".mat"):
                file_path = os.path.join(self.directory, filename)
                try:
                    # Squeeze internal dimensions for standard format
                    mat_data = scipy.io.loadmat(file_path, squeeze_me=True)
                    
                    # Target the 'annotation' struct
                    annotation_struct = mat_data['annotation']
                    
                    # Ensure it's a iterable array of frames
                    if annotation_struct.dtype != 'object' and annotation_struct.ndim == 1:
                        frames_list = annotation_struct
                    elif annotation_struct.dtype == 'object':
                        # Handle potential cell array formats
                        frames_list = [annotation_struct[i, 0] for i in range(annotation_struct.shape[0])]
                    else:
                        raise ValueError(f"Unrecognized annotation structure in {filename}")
                        
                    self.dataset[filename] = frames_list
                    
                except Exception as e:
                    print(f"Error loading {filename}: {e}")

        return self.dataset