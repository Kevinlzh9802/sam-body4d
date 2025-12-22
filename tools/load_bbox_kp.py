import os
import pickle

def load_bbox_kp(bbox_kp_folder: str, folder_name: str):
    """
    Load bboxes and kps from a folder.
    """
    bboxes_kps_data = None
    if len(bbox_kp_folder):
        keypoint_path = os.path.join(bbox_kp_folder, f"{folder_name}.pkl")
        with open(keypoint_path, "rb") as kp_f:
            bboxes_kps_data = pickle.load(kp_f)
    return bboxes_kps_data

if __name__ == "__main__":
    bbox_kp_folder = "./experiments/bboxes_kps_refined"
    folder_name = "428"
    bboxes_kps_data = load_bbox_kp(bbox_kp_folder, folder_name)
    print(bboxes_kps_data)