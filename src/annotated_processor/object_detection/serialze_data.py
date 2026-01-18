import os
import numpy as np
import json

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        return super().default(obj)

def save_json(obj, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(obj, f, cls=NumpyEncoder, indent=2)

def load_json(in_path):
    with open(in_path, "r") as f:
        return json.load(f)

def save_npz(np_arr, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, arr=np_arr)

def load_npz(in_path):
    arr = np.load(in_path)
    return arr['arr']