import os
import numpy as np
import joblib
from sklearn.linear_model import Ridge
from dataset import VideoDatasetLoader
from base_model import ATBMultiModel

def train_split_regression(data_dir="videos", max_n=5):
    print("=== Training Independent ATB Ridge Models ===")
    
    loader = VideoDatasetLoader(data_dir)
    videos_dict = loader.load_all_videos()
    base_model = ATBMultiModel(num_points=100)
    
    X_train = {'contour1': [], 'contour2': [], 'contour3': []}
    Y_train = {'contour1': [], 'contour2': [], 'contour3': []}

    print("Pre-processing training data...")
    
    for video_name, frames in videos_dict.items():
        total_frames = len(frames)
        resampled_frames = []
        
        for t in range(total_frames):
            frame_data = {}
            is_valid = True
            for c_name in base_model.contour_names:
                coords = base_model.extract_coords(frames[t], c_name)
                try:
                    frame_data[c_name] = base_model._resample(coords)
                except Exception:
                    is_valid = False
            resampled_frames.append(frame_data if is_valid else None)
        
        for n in range(1, max_n + 1):
            for t in range(n, total_frames - n):
                past_f = resampled_frames[t - n]
                target_f = resampled_frames[t]
                future_f = resampled_frames[t + n]
                
                if past_f and target_f and future_f:
                    for c_name in base_model.contour_names:
                        past_flat = past_f[c_name].flatten()
                        future_flat = future_f[c_name].flatten()
                        target_flat = target_f[c_name].flatten()
                        
                        if np.isnan(past_flat).any() or np.isnan(future_flat).any() or np.isnan(target_flat).any():
                            continue
                        
                        X_train[c_name].append(np.concatenate([past_flat, future_flat]))
                        Y_train[c_name].append(target_flat)

    models = {}
    print("\nTraining 3 Independent Models...")
    
    for c_name in base_model.contour_names:
        X = np.array(X_train[c_name])
        Y = np.array(Y_train[c_name])
        
        alpha_val = 0.5 if c_name == 'contour2' else 1.0
        
        model = Ridge(alpha=alpha_val, solver='auto')
        model.fit(X, Y)
        models[c_name] = model
        
        distances = np.linalg.norm(model.predict(X).reshape(-1, 2) - Y.reshape(-1, 2), axis=1)
        acc = (np.sum(distances < 2.0) / len(distances)) * 100.0
        print(f"{c_name.capitalize()} Model -> R^2: {model.score(X, Y):.4f} | Accuracy: {acc:.2f}%")

    os.makedirs("saved_models", exist_ok=True)
    joblib.dump(models, "saved_models/split_ridge_models.pkl")
    print("\n All models saved to 'saved_models/split_ridge_models.pkl'")

if __name__ == "__main__":
    train_split_regression()
