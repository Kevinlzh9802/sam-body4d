import json
from math import sqrt
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pickle
import os
import cv2
import matplotlib.pyplot as plt
from undistortion import adjust_K
from data_utils import read_camera_intrinsics_new

RAW_JSON_PATH = Path("/home/zonghuan/tudelft/projects/datasets/conflab/annotations/pose/coco")
INTRINSIC_PATH = Path("./experiments/intrinsics/parameters-camera-04.json")
IMG_WIDTH = 960
IMG_HEIGHT = 540
# NOTE:
# - Videos / images are assumed to be 960x540.
# - Intrinsics in `INTRINSIC_PATH` are for ~1920x1080, so we always scale K by 0.5.
IMG_SCALE = 0.5
REMOVE_BBOX = {
    '428': [20, 25, 26, 27]
}

# The 10 joints you mentioned, just for reference / debugging
JOINT_NAMES = [
    "head",
    "nose",
    "leftShoulder",
    "rightShoulder",
    "leftHip",
    "rightHip",
    "leftAnkle",
    "rightAnkle",
    "leftToe",
    "rightToe",
]

MHR70_MAP = {
    0: 69,   # head → neck
    1: 0,    # nose
    2: 5,    # left shoulder
    3: 6,    # right shoulder
    4: 9,    # left hip
    5: 10,   # right hip
    6: 13,   # left ankle
    7: 14,   # right ankle
    8: 15,   # left big toe tip
    9: 18,   # right big toe tip
}

def extract_raw_keypoints(skeleton: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    Extract 10 keypoints per person from a single frame annotation.

    Returns:
        dict: {person_id: (10, 2) array} in RELATIVE coords [0,1] (NaN for missing)
    """
    keypoint_names = [
        ("head", 0), ("nose", 2),
        ("leftShoulder", 12), ("rightShoulder", 6),
        ("leftHip", 24), ("rightHip", 18),
        ("leftAnkle", 28), ("rightAnkle", 22),
        ("leftFoot", 32), ("rightFoot", 30),
    ]
    coords = {}
    for person_id, kps in skeleton.items():
        kp = kps.get("keypoints")
        if kp is None:
            coords[person_id] = np.full((10, 2), np.nan)
            continue
        xy = []
        for _, idx in keypoint_names:
            xy.append(kp[idx])  # add x coord
            xy.append(kp[idx + 1])  # add y coord
        coords[person_id] = np.asarray(xy, dtype=float).reshape(10, 2)
    return coords


def undistort_keypoints_fisheye(
    kps_rel: np.ndarray,
    K_scaled: np.ndarray,
    dist_coeffs: np.ndarray,
    *,
    balance: float = 1.0,
    fov_scale: float = 0.85,
) -> np.ndarray:
    """
    Fisheye undistort keypoints from RELATIVE coords to PIXEL coords.

    Input:
        kps_rel: (10,2) relative coords in [0,1] (NaN allowed)
    Output:
        (10,2) pixel coords in the undistorted image coordinate system.
        Out-of-bounds points are set to NaN.
    """
    dim = (IMG_WIDTH, IMG_HEIGHT)
    R = np.eye(3)
    Knew = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K_scaled, dist_coeffs, dim, R, balance=balance, new_size=dim, fov_scale=fov_scale
    )

    kps_px = kps_rel.astype(np.float32) * np.array([IMG_WIDTH, IMG_HEIGHT], dtype=np.float32)
    pts = kps_px.reshape(-1, 1, 2)
    pts_u = cv2.fisheye.undistortPoints(pts, K_scaled, dist_coeffs, R=R, P=Knew).reshape(-1, 2)

    oob = (
        (pts_u[:, 0] < 0) | (pts_u[:, 0] >= IMG_WIDTH) |
        (pts_u[:, 1] < 0) | (pts_u[:, 1] >= IMG_HEIGHT)
    )
    pts_u[oob] = np.nan
    return pts_u


def undistort_keypoints_fisheye_pixels(
    kps_px: np.ndarray,
    K_scaled: np.ndarray,
    dist_coeffs: np.ndarray,
    *,
    balance: float = 1.0,
    fov_scale: float = 0.85,
) -> np.ndarray:
    """
    Fisheye undistort keypoints from PIXEL coords to PIXEL coords.

    Input:
        kps_px: (10,2) pixel coords in the original (distorted) image (NaN allowed)
    Output:
        (10,2) pixel coords in the undistorted image coordinate system.
        Out-of-bounds points are set to NaN.
    """
    dim = (IMG_WIDTH, IMG_HEIGHT)
    R = np.eye(3)
    Knew = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K_scaled, dist_coeffs, dim, R, balance=balance, new_size=dim, fov_scale=fov_scale
    )

    pts = kps_px.astype(np.float32).reshape(-1, 1, 2)
    pts_u = cv2.fisheye.undistortPoints(pts, K_scaled, dist_coeffs, R=R, P=Knew).reshape(-1, 2)

    oob = (
        (pts_u[:, 0] < 0) | (pts_u[:, 0] >= IMG_WIDTH) |
        (pts_u[:, 1] < 0) | (pts_u[:, 1] >= IMG_HEIGHT)
    )
    pts_u[oob] = np.nan
    return pts_u

def build_save_bbox_kps_all(output_path: Path, undistort: bool = False):
    """
    Main pipeline:
      json (relative kps) -> optional undistort -> build bbox + kps prompts -> save pkl

    Saved outputs are ALWAYS in ABSOLUTE PIXEL coordinates (for 960x540).
    """
    json_files = sorted(RAW_JSON_PATH.glob("*.json"))
    
    for json_path in json_files:

        stem_parts = json_path.stem.split("_")
        cam = vid = seg = None
        for part in stem_parts:
            if part.startswith("cam"):
                cam = int(part.replace("cam", ""))
            elif part.startswith("vid"):
                vid = int(part.replace("vid", ""))
            elif part.startswith("seg"):
                seg = int(part.replace("seg", ""))
        seg_name = f"{cam}{vid}{seg}"

        with open(json_path, "r") as file:
            raw_annotation = json.load(file)
        skeletons = raw_annotation.get("annotations", {}).get("skeletons", [])

        bboxes_kps = []

        K, dist_coeffs = read_camera_intrinsics_new(INTRINSIC_PATH)
        K_scaled = adjust_K(K, IMG_SCALE)

        for idx, skeleton in enumerate(skeletons):
            frame_coords = extract_raw_keypoints(skeleton)

            # Convert to pixel coords (undistorted or original), then build bbox/kps using identical logic.
            frame_coords_px: Dict[str, np.ndarray] = {}
            for pid, kps_rel in frame_coords.items():
                if undistort:
                    frame_coords_px[pid] = undistort_keypoints_fisheye(
                        kps_rel, K_scaled, dist_coeffs, balance=1.0, fov_scale=1
                    )
                else:
                    frame_coords_px[pid] = kps_rel.astype(np.float32) * np.array(
                        [IMG_WIDTH, IMG_HEIGHT], dtype=np.float32
                    )

            bboxes, kps, pids = build_bbox_kps_single(frame_coords_px, IMG_WIDTH, IMG_HEIGHT)
            bboxes_kps.append({
                "bboxes": bboxes,
                "kps": kps,
                "pids": pids,
            })
        # save to json
        if not output_path.exists():
            output_path.mkdir(parents=True)
        with open(output_path / f"{seg_name}.pkl", "wb") as file:
            pickle.dump(bboxes_kps, file)


# Backward-compatible alias (old name)
build_save_bboxex_kps_all = build_save_bbox_kps_all

def build_bbox_kps_single(frame_coords, img_width, img_height, pad_ratio=0.20, min_valid_kps=3):
    """
    Build per-person bounding boxes and keypoint prompts for SAM-3D-Body
    from a dict of PIXEL 2D keypoints.

    Parameters
    ----------
    frame_coords : dict
        {person_id: (10, 2) ndarray} with x,y in pixel coordinates.
        NaNs are allowed and treated as missing keypoints.
    img_width : int
        Image width in pixels.
    img_height : int
        Image height in pixels.
    pad_ratio : float
        How much to expand the bbox, relative to its width/height.
    min_valid_kps : int
        Minimum number of valid keypoints required to keep a person.

    Returns
    -------
    bboxes : np.ndarray or None
        (num_person, 4) array of [x_min, y_min, x_max, y_max] in pixels,
        or None if no valid persons.
    keypoint_prompt : np.ndarray or None
        (num_person, 10, 3) array: [x_px, y_px, label].
        - x_px, y_px are pixel coordinates (still valid even if label is -2).
        - label in {0..9} for valid joints, -2 for invalid / missing joints.
        Returns None if no valid persons.

    Notes
    -----
    - This is designed to be passed into your modified `SAM3DBodyEstimator`
      where you convert from full-image pixels to crop space.
    - Labels -2 follow the SAM-3D-Body convention:
        label == -2 → invalid point (ignored by the prompt encoder)
    """

    person_ids = sorted(frame_coords.keys())
    all_bboxes = []
    all_keypoints = []
    pids = []
    for pid in person_ids:
        coords = np.asarray(frame_coords[pid], dtype=float)  # (10, 2)
        if coords.shape != (10, 2):
            raise ValueError(
                f"Expected (10,2) coords per person, got {coords.shape} for {pid}"
            )

        # Valid keypoints: both x and y finite (NaNs count as invalid)
        valid_mask = np.isfinite(coords[:, 0]) & np.isfinite(coords[:, 1])
        num_valid = int(valid_mask.sum())
        if num_valid < min_valid_kps:
            # Too few keypoints to build a stable bbox → skip this person
            continue

        xs = coords[:, 0]
        ys = coords[:, 1]
        xy_pix = np.stack([xs, ys], axis=-1).astype(np.float32)  # (10, 2)

        # Compute bbox from valid keypoints only
        xs_valid = xs[valid_mask]
        ys_valid = ys[valid_mask]

        x_min = xs_valid.min()
        x_max = xs_valid.max()
        y_min = ys_valid.min()
        y_max = ys_valid.max()

        # Add padding around bbox
        w = x_max - x_min
        h = y_max - y_min
        # If w or h is zero (all points at same place), give a minimum size
        if w <= 0:
            w = img_width * 0.02
        if h <= 0:
            h = img_height * 0.02

        cx = 0.5 * (x_min + x_max)
        cy = 0.5 * (y_min + y_max)

        pad_w = sqrt(w) * pad_ratio
        pad_h = sqrt(h) * pad_ratio

        x_min_p = max(0.0, cx - 0.5 * w - pad_w - 20)
        x_max_p = min(float(img_width - 1), cx + 0.5 * w + pad_w + 20)
        y_min_p = max(0.0, cy - 0.5 * h - pad_h - 20)
        y_max_p = min(float(img_height - 1), cy + 0.5 * h + pad_h + 20)

        absolute_pad = 15
        if x_max_p - x_min_p < 80:
            x_min_p = max(0.0, x_min_p - absolute_pad)
            x_max_p = min(float(img_width - 1), x_max_p + absolute_pad)
        if y_max_p - y_min_p < 80:
            y_min_p = max(0.0, y_min_p - absolute_pad)
            y_max_p = min(float(img_height - 1), y_max_p + absolute_pad)

        bbox = np.array([x_min_p, y_min_p, x_max_p, y_max_p], dtype=np.float32)

        # Build keypoint [x_px, y_px, label] array
        labels = np.full((10,), -2.0, dtype=np.float32)  # -2 = invalid
        # Assign labels only to valid joints (indices 0..9)
        for j in range(10):
            if valid_mask[j]:
                labels[j] = float(MHR70_MAP[j])
        # Assign (0, 0) to invalid joints
        invalid_mask = (labels == -2)
        xy_pix[invalid_mask] = (0, 0)
        kps_with_labels = np.concatenate(
            [xy_pix.astype(np.float32), labels[:, None]], axis=-1
        )  # (10, 3)
        assert not np.any(np.isnan(kps_with_labels))

        all_bboxes.append(bbox)
        all_keypoints.append(kps_with_labels)
        pids.append(pid)

    if not all_bboxes:
        return None, None, None

    bboxes = np.stack(all_bboxes, axis=0)           # (num_person, 4)
    keypoint_prompt = np.stack(all_keypoints, axis=0)  # (num_person, 10, 3)
    pids = np.array(pids)
    return bboxes, keypoint_prompt, pids


def refine_bboxes_kps_single(
    data,
    seg_name,
    img_width=IMG_WIDTH,
    img_height=IMG_HEIGHT,
    min_valid_kps=6,          # 1) minimum number of valid kps
    min_area_ratio=0.002,      # 2) bbox area limits (as ratio of full image)
    max_area_ratio=0.15,
    min_hw_ratio=1/6,         # 3) height/width ratio limits
    max_hw_ratio=6,
    edge_sum_thresh=0.3,      # 4) keypoints too close to image edge (in normalized coords)
    kps_are_normalized=False,  # whether kps[..., :2] are in [0,1]
):
    """
    data: list of dicts, each with:
        'bboxes': (N, 4) or None
        'kps':    (N, 10, 3) or None
        'pids':   (N,) or None
    Modifies data in-place:
        - removes persons that fail any condition
        - sets bboxes/kps to empty arrays if no one remains
    """

    img_area = img_width * img_height

    for idx, item in enumerate(data):
        bboxes = item.get('bboxes', None)
        kps = item.get('kps', None)
        pids = item.get('pids', None)
        if bboxes is None or kps is None:
            continue

        bboxes = np.asarray(bboxes, dtype=float)
        kps = np.asarray(kps, dtype=float)

        if bboxes.ndim != 2 or bboxes.shape[1] != 4:
            raise ValueError(f"Expected bboxes shape (N,4), got {bboxes.shape}")
        if kps.ndim != 3 or kps.shape[1:] != (10, 3):
            raise ValueError(f"Expected kps shape (N,10,3), got {kps.shape}")

        N = bboxes.shape[0]
        assert kps.shape[0] == N, "bboxes and kps must have same N"

        keep_mask = np.ones(N, dtype=bool)

        for i in range(N):
            bbox = bboxes[i]       # [x1, y1, x2, y2]
            kp_person = kps[i]     # (10, 3) [x, y, label]
            pid = pids[i]
            # Skip if bbox has NaNs
            if not np.all(np.isfinite(bbox)):
                keep_mask[i] = False
                continue

            x1, y1, x2, y2 = bbox
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            area = w * h

            # ------------------------------------------------------------------
            # 1) #valid keypoints >= min_valid_kps
            # ------------------------------------------------------------------
            # valid if x,y finite (you could also include label >= 0 if you use -2 flag)
            kp_xy = kp_person[:, :2]                   # (10,2)
            kp_valid = np.isfinite(kp_xy).all(axis=-1) # (10,)
            num_valid = int(kp_valid.sum())

            if num_valid < min_valid_kps:
                keep_mask[i] = False
                continue

            # ------------------------------------------------------------------
            # 2) bbox too large or too small -> check area ratio
            # ------------------------------------------------------------------
            if area <= 0:
                keep_mask[i] = False
                continue

            area_ratio = area / img_area
            if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
                keep_mask[i] = False
                continue

            # ------------------------------------------------------------------
            # 3) bbox height-width ratio too high/low
            # ------------------------------------------------------------------
            if w <= 0 or h <= 0:
                keep_mask[i] = False
                continue

            hw_ratio = h / w  # height / width
            if hw_ratio < min_hw_ratio or hw_ratio > max_hw_ratio:
                keep_mask[i] = False
                continue

            # ------------------------------------------------------------------
            # 4) keypoints too close to image edge
            # ------------------------------------------------------------------
            # Here we assume kp_xy is normalized to [0,1].
            # If they are in pixels instead, replace this with kp_xy[...,0]/img_width etc.
            if kps_are_normalized:
                x_norm = kp_xy[kp_valid, 0]
                y_norm = kp_xy[kp_valid, 1]
            else:
                x_norm = kp_xy[kp_valid, 0] / img_width
                y_norm = kp_xy[kp_valid, 1] / img_height

            if x_norm.size > 0:
                # Distances in normalized units [0,1]
                # left edge:   dist = x
                # right edge:  dist = 1 - x
                # top edge:    dist = y
                # bottom edge: dist = 1 - y
                left_sum   = float(np.sum(x_norm))
                right_sum  = float(np.sum(1.0 - x_norm))
                top_sum    = float(np.sum(y_norm))
                bottom_sum = float(np.sum(1.0 - y_norm))

                # If for any edge the total distance of all valid kps is
                # very small, that means many kps are hugging that edge.
                if (
                    left_sum   < edge_sum_thresh or
                    right_sum  < edge_sum_thresh or
                    top_sum    < edge_sum_thresh or
                    bottom_sum < edge_sum_thresh
                ):
                    keep_mask[i] = False
                    continue
            # ------------------------------------------------------------------
            # 5) manually remove based on REMOVE_BBOX
            # ------------------------------------------------------------------
            if seg_name in REMOVE_BBOX.keys():
                if int(pid) in REMOVE_BBOX[seg_name]:
                    keep_mask[i] = False
                    continue

        # Apply mask
        bboxes_refined = bboxes[keep_mask]
        kps_refined = kps[keep_mask]
        pids_refined = pids[keep_mask]
        # You can choose between [] or None if nothing remains; I'll use empty arrays
        if bboxes_refined.size == 0:
            bboxes_refined = np.empty((0, 4), dtype=float)
            kps_refined = np.empty((0, 10, 3), dtype=float)
            pids_refined = np.empty((0,), dtype=int)
        data[idx]['bboxes'] = bboxes_refined
        data[idx]['kps'] = kps_refined
        data[idx]['pids'] = pids_refined
    return data

def refine_bboxes_kps(bbox_kp_folder: Path, output_folder: Path):
    """
    Refine bboxes and kps by removing invalid bboxes and kps.
    """
    if not output_folder.exists():
        output_folder.mkdir(parents=True)
    for pkl_file in bbox_kp_folder.glob("*.pkl"):
        with open(pkl_file, "rb") as f:
            data = pickle.load(f)
        seg_name = pkl_file.stem
        data = refine_bboxes_kps_single(data, seg_name)
        with open(output_folder / pkl_file.name, "wb") as f:
            pickle.dump(data, f)


def plot_bboxes_kps(frame_num, seg_num, bbox_path, K, dist_coeffs, undistort=False):
    bbox_filename = f"{seg_num}.pkl"
    # video_filename = "cam04_cut_10s.mp4"
    video_filename = "cam04_cut_10s_undistorted_scaled_s1.mp4"
    # img_filename = f"{seg_num}_{frame_num:06d}.jpg"
    # img = cv2.imread(os.path.join(image_path, img_filename))
    bbox_path = os.path.join(bbox_path, bbox_filename)
    with open(bbox_path, "rb") as f:
        data = pickle.load(f)
    bboxes = data[frame_num]["bboxes"]
    kps = data[frame_num]["kps"]
    pids = data[frame_num]["pids"]

    # read first frame
    cap = cv2.VideoCapture("./experiments/" + video_filename)
    ret, img = cap.read()
    cap.release()
    if not ret:
        raise ValueError(f"Could not read first frame from ./experiments/{video_filename}")
    # Visualization helper: if your PKL was generated with undistort=True,
    # you likely want to show an undistorted image here as well.
    K_scaled = adjust_K(K, IMG_SCALE)
    if undistort:
        dim = (IMG_WIDTH, IMG_HEIGHT)
        R = np.eye(3)
        Knew = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K_scaled, dist_coeffs, dim, R, balance=1.0, new_size=dim, fov_scale=1
        )
        img_undistorted = cv2.fisheye.undistortImage(img, K_scaled, dist_coeffs, Knew=Knew, new_size=dim)
    else:
        img_undistorted = img

    for pid in range(kps.shape[0]):
        if undistort:
            # PKL stores pixel coords in the *distorted* image; undistort them to match `img_undistorted`.
            kp_undistorted = undistort_keypoints_fisheye_pixels(
                kps[pid][:, :2], K_scaled, dist_coeffs, balance=1.0, fov_scale=1
            )
        else:
            kp_undistorted = kps[pid][:, :2]
        for pt in kp_undistorted:
            if not np.all(np.isfinite(pt)):
                continue
            cv2.circle(img_undistorted, (int(pt[0]), int(pt[1])), 3, (0, 0, 255), -1)
        # Put the person id label near the first finite keypoint (skip if all are NaN)
        finite_idx = np.where(np.isfinite(kp_undistorted).all(axis=1))[0]
        if finite_idx.size > 0:
            x0, y0 = kp_undistorted[finite_idx[0]]
            cv2.putText(
                img_undistorted,
                str(pid),
                (int(x0), int(y0) + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )

    for bbox, kp, pid in zip(bboxes, kps, pids):
        color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
        # for kp in kp:
        #     cv2.circle(img, (int(kp[0]), int(kp[1])), 3, color, -1)
        cv2.putText(img, str(pid), (int(bbox[0]), int(bbox[1])+30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    plt.imshow(img_undistorted)
    plt.show()
    c = 9
    # cv2.imwrite(os.path.join(output_folder, img_filename.replace(".jpg", f"_bbox_kps_{frame_num}.jpg")), img)
    return img_undistorted



if __name__ == "__main__":
    undistort = True
    suffix = "_undistort" if undistort else ""
    # 1) json -> (optional undistort) -> bbox/kps -> pkl
    # build_save_bbox_kps_all(Path(f"./experiments/inputs/bboxes_kps{suffix}/"), undistort=undistort)
    
    # 2) optional refinement stage
    # refine_bboxes_kps(Path(f"./experiments/inputs/bboxes_kps{suffix}/"), Path(f"./experiments/inputs/bboxes_kps_refined{suffix}/"))
    #
    # 3) visualize one frame from the saved pkl
    K, dist_coeffs = read_camera_intrinsics_new(INTRINSIC_PATH)
    plot_bboxes_kps(0, 428, Path(f"./experiments/inputs/bboxes_kps_refined{suffix}/"), K, dist_coeffs, undistort=False)