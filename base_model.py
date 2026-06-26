import numpy as np
from scipy.spatial.distance import cdist, euclidean
from fastdtw import fastdtw
import scipy.interpolate

class ATBMultiModel:
    def __init__(self, n=2, num_points=100, threshold=2.0):
        self.n = n
        self.num_points = num_points
        self.threshold = threshold
        self.contour_names = ['contour1', 'contour2', 'contour3']

    def extract_coords(self, frame_struct, contour_name):
        try:
            data = frame_struct[contour_name]
            if isinstance(data, np.ndarray) and data.size >= 4: 
                return data.reshape(-1, 2)
        except Exception:
            pass
        return None

    def _resample(self, contour):
        if contour is None or len(contour) < 2:
            raise ValueError("Invalid contour")
            
        dist = np.cumsum(np.sqrt(np.sum(np.diff(contour, axis=0)**2, axis=1)))
        dist = np.insert(dist, 0, 0)
        
        if dist[-1] == 0: 
            raise ValueError("Zero length contour")
            
        interp_x = scipy.interpolate.interp1d(dist/dist[-1], contour[:, 0], fill_value="extrapolate")
        interp_y = scipy.interpolate.interp1d(dist/dist[-1], contour[:, 1], fill_value="extrapolate")
        
        alpha = np.linspace(0, 1, self.num_points)
        return np.column_stack((interp_x(alpha), interp_y(alpha)))

    def predict(self, prev_frame, next_frame):
        predictions = {}
        for c_name in self.contour_names:
            coords_prev = self.extract_coords(prev_frame, c_name)
            coords_next = self.extract_coords(next_frame, c_name)

            try:
                res_prev = self._resample(coords_prev)
                res_next = self._resample(coords_next)
                
                predicted_contour = (res_prev + res_next) / 2.0
                predictions[c_name] = predicted_contour
            except ValueError:
                predictions[c_name] = None
                
        return predictions

    def evaluate(self, predictions, gt_struct):
        metrics = {}
        for c_name in self.contour_names:
            gt_coords = self.extract_coords(gt_struct, c_name)
            try:
                gt_res = self._resample(gt_coords)
                pred = predictions[c_name]
                if pred is None:
                    raise ValueError("No prediction generated")
                    
                dtw_err, _ = fastdtw(pred, gt_res, dist=euclidean)
                dtw_err = dtw_err / self.num_points
                
                dists = np.linalg.norm(pred - gt_res, axis=1)
                accuracy = (np.sum(dists < self.threshold) / self.num_points) * 100
                
                metrics[c_name] = {'dtw': dtw_err, 'acc': accuracy, 'gt_resampled': gt_res}
            except Exception:
                metrics[c_name] = None
                
        return metrics