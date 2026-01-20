import json
import numpy as np

def read_camera_intrinsics_new(intrinsic_file: str):
    with open(intrinsic_file, "r") as f:
        intrinsic_data = json.load(f)
        params = intrinsic_data['Calibration']['cameras'][0]['model']['ptr_wrapper']['data']['parameters']

        f = params['f']['val']
        cx = params['cx']['val']
        cy = params['cy']['val']
        
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
        ks = [params[f'k{i}']['val'] for i in range(1, 5)]
        dist_coeffs = np.array(ks)

    return K, dist_coeffs