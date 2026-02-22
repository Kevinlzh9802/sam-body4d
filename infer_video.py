#!/usr/bin/env python3
"""
Simplified inference script for SAM-Body4D on a single video.
Runs the full pipeline: mask generation → 4D generation.
"""

import os
import sys
import argparse
import time
import cv2
import glob
import random
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from omegaconf import OmegaConf
import pickle
import json
# from dataset.undistortion import read_camera_intrinsics_new
# from tools.load_bbox_kp import load_bbox_kp

# Add model paths
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'models', 'sam_3d_body'))
sys.path.append(os.path.join(current_dir, 'models', 'diffusion_vas'))

from utils import (
    mask_painter, images_to_mp4, DAVIS_PALETTE, jpg_folder_to_mp4,
    is_super_long_or_wide, keep_largest_component, is_skinny_mask,
    bbox_from_mask, resize_mask_with_unique_label
)

from models.sam_3d_body.sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
from models.sam_3d_body.notebook.utils import process_image_with_mask, save_mesh_results
from models.sam_3d_body.tools.vis_utils import visualize_sample_together, visualize_sample
from models.diffusion_vas.demo import (
    init_amodal_segmentation_model, init_rgb_model, init_depth_model,
    load_and_transform_masks, load_and_transform_rgbs, rgb_to_depth
)
from utils.proj_utils import build_pred_cam_t_debug_record
from utils.image_utils import save_sanity_first_frame_overlay
from utils.camera_utils import adjust_K, read_camera_intrinsics, read_camera_intrinsics_new
from utils.model_factory import build_sam3_from_config


def build_sam3_3d_body_config(cfg, device):
    """Construct SAM-3D-Body model from config."""
    mhr_path = cfg.sam_3d_body['mhr_path']
    fov_path = cfg.sam_3d_body['fov_path']
    
    model, model_cfg = load_sam_3d_body(
        cfg.sam_3d_body['ckpt_path'], device=device, mhr_path=mhr_path
    )
    
    from models.sam_3d_body.tools.build_fov_estimator import FOVEstimator
    fov_estimator = FOVEstimator(name='moge2', device=device, path=fov_path)
    
    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=None,
        human_segmentor=None,
        fov_estimator=fov_estimator,
    )
    return estimator


def build_diffusion_vas_config(cfg):
    """Construct Diffusion-VAS models for completion."""
    model_path_mask = cfg.completion['model_path_mask']
    model_path_rgb = cfg.completion['model_path_rgb']
    depth_encoder = cfg.completion['depth_encoder']
    model_path_depth = cfg.completion['model_path_depth']
    
    generator = torch.manual_seed(23)
    pipeline_mask = init_amodal_segmentation_model(model_path_mask)
    pipeline_rgb = init_rgb_model(model_path_rgb)
    depth_model = init_depth_model(model_path_depth, depth_encoder)
    
    return pipeline_mask, pipeline_rgb, depth_model, generator


def read_video_metadata(path: str):
    """Return FPS and total frame count."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, total


def mask_generation(video_path: str, predictor, inference_state, output_dir, fps, out_obj_ids):
    """
    Run SAM-3 propagation across the video and save masks.
    Returns: video_segments dict
    """
    print("[INFO] Running mask generation...")
    
    # Run propagation throughout the video
    video_segments = {}
    for frame_idx, obj_ids, low_res_masks, video_res_masks, obj_scores, iou_scores in predictor.propagate_in_video(
        inference_state,
        start_frame_idx=0,
        max_frame_num_to_track=1800,
        reverse=False,
        propagate_preflight=True,
    ):
        video_segments[frame_idx] = {
            out_obj_id: (video_res_masks[i] > 0.0).cpu().float().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    
    # Render segmentation results
    vis_frame_stride = 1
    out_h = inference_state['video_height']
    out_w = inference_state['video_width']
    img_to_video = []
    
    IMAGE_PATH = os.path.join(output_dir, 'images')
    MASKS_PATH = os.path.join(output_dir, 'masks')
    os.makedirs(IMAGE_PATH, exist_ok=True)
    os.makedirs(MASKS_PATH, exist_ok=True)
    
    for out_frame_idx in tqdm(range(0, len(video_segments), vis_frame_stride), desc="Saving masks"):
        img = inference_state['images'][out_frame_idx].detach().float().cpu()
        img = (img + 1) / 2
        img = img.clamp(0, 1)
        img = F.interpolate(
            img.unsqueeze(0),
            size=(out_h, out_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        img = img.permute(1, 2, 0)
        img = (img.float().numpy() * 255).astype("uint8")
        img_pil = Image.fromarray(img).convert('RGB')
        msk = np.zeros_like(img[:, :, 0])
        
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            mask = (out_mask[0] > 0).astype(np.uint8) * 255
            img = mask_painter(img, mask, mask_color=4 + out_obj_id)
            msk[mask == 255] = out_obj_id
        
        img_to_video.append(img)
        
        msk_pil = Image.fromarray(msk).convert('P')
        msk_pil.putpalette(DAVIS_PALETTE)
        img_pil.save(os.path.join(IMAGE_PATH, f"{out_frame_idx:08d}.jpg"))
        msk_pil.save(os.path.join(MASKS_PATH, f"{out_frame_idx:08d}.png"))
    
    out_video_path = os.path.join(output_dir, "video_mask.mp4")
    images_to_mp4(img_to_video, out_video_path, fps=fps)
    print(f"[INFO] Mask video saved to: {out_video_path}")
    
    return video_segments


def generate_4d(output_dir, estimator, out_obj_ids, batch_size, fps, 
                pipeline_mask=None, pipeline_rgb=None, depth_model=None,
                detection_resolution=[256, 512], completion_resolution=[512, 1024], camera_intrinsics=None,
                bboxes_kps_data=None, obj_id_to_bbox_idx=None, consecutive_to_actual=None):
    """
    Run 4D generation with optional completion.
    
    Args:
        consecutive_to_actual: Optional mapping {consecutive_id: actual_pid} to convert
            SAM3's consecutive IDs back to actual person IDs when saving outputs.
    """
    print("[INFO] Running 4D generation...")
    
    IMAGE_PATH = os.path.join(output_dir, 'images')
    MASKS_PATH = os.path.join(output_dir, 'masks')
    
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.gif", "*.bmp", "*.tiff", "*.webp"]
    images_list = sorted([
        image for ext in image_extensions
        for image in glob.glob(os.path.join(IMAGE_PATH, ext))
    ])
    masks_list = sorted([
        image for ext in image_extensions
        for image in glob.glob(os.path.join(MASKS_PATH, ext))
    ])
    
    os.makedirs(f"{output_dir}/rendered_frames", exist_ok=True)
    # Debug: store per-person camera translation (pred_cam_t) to inspect units/axes.
    debug_cam_t_dir = os.path.join(output_dir, "debug_pred_cam_t")
    os.makedirs(debug_cam_t_dir, exist_ok=True)
    
    # Helper to convert consecutive ID to actual PID (if mapping provided)
    def to_actual_pid(consecutive_id):
        if consecutive_to_actual is not None and int(consecutive_id) in consecutive_to_actual:
            return consecutive_to_actual[int(consecutive_id)]
        return consecutive_id  # fallback to original ID
    
    # Create directories using actual PIDs
    for obj_id in out_obj_ids:
        actual_pid = to_actual_pid(obj_id)
        os.makedirs(f"{output_dir}/mesh_4d_individual/{actual_pid}", exist_ok=True)
        os.makedirs(f"{output_dir}/rendered_frames_individual/{actual_pid}", exist_ok=True)
        os.makedirs(os.path.join(debug_cam_t_dir, str(actual_pid)), exist_ok=True)

    # Cache debug file handles for speed (one jsonl per person).
    debug_fhs = {}
    for obj_id in out_obj_ids:
        actual_pid = to_actual_pid(obj_id)
        debug_path = os.path.join(debug_cam_t_dir, str(actual_pid), "pred_cam_t.jsonl")
        debug_fhs[actual_pid] = open(debug_path, "w", encoding="utf-8")
    
    n = len(images_list)
    pred_res = detection_resolution
    pred_res_hi = completion_resolution
    generator = torch.manual_seed(23)
    
    # Prepare completion if enabled
    modal_pixels_list = []
    if pipeline_mask is not None:
        print("[INFO] Loading masks and images for completion...")
        for obj_id in out_obj_ids:
            modal_pixels, ori_shape = load_and_transform_masks(MASKS_PATH, resolution=pred_res, obj_id=obj_id)
            modal_pixels_list.append(modal_pixels)
        rgb_pixels, _, raw_rgb_pixels = load_and_transform_rgbs(IMAGE_PATH, resolution=pred_res)
        depth_pixels = rgb_to_depth(rgb_pixels, depth_model)
    
    mhr_shape_scale_dict = {}
    obj_ratio_dict = {}
    try:
        for i in tqdm(range(0, n, batch_size), desc="Processing batches"):
            batch_images = images_list[i:i + batch_size]
            batch_masks = masks_list[i:i + batch_size]
            if len(batch_masks) == 0:
                continue

            W, H = Image.open(batch_masks[0]).size

            # Detect occlusions
            idx_dict = {}
            idx_path = {}
            occ_dict = {}

            if len(modal_pixels_list) > 0:
                pred_amodal_masks_dict = {}
                for (modal_pixels, obj_id) in zip(modal_pixels_list, out_obj_ids):
                    # Predict amodal masks
                    pred_amodal_masks = pipeline_mask(
                        modal_pixels[:, i:i + batch_size, :, :, :],
                        depth_pixels[:, i:i + batch_size, :, :, :],
                        height=pred_res[0],
                        width=pred_res[1],
                        num_frames=modal_pixels[:, i:i + batch_size, :, :, :].shape[1],
                        decode_chunk_size=8,
                        motion_bucket_id=127,
                        fps=8,
                        noise_aug_strength=0.02,
                        min_guidance_scale=1.5,
                        max_guidance_scale=1.5,
                        generator=generator,
                    ).frames[0]

                    # Process amodal masks
                    pred_amodal_masks_com = [np.array(img.resize((pred_res_hi[1], pred_res_hi[0]))) for img in pred_amodal_masks]
                    pred_amodal_masks_com = np.array(pred_amodal_masks_com).astype('uint8')
                    pred_amodal_masks_com = (pred_amodal_masks_com.sum(axis=-1) > 600).astype('uint8')
                    pred_amodal_masks_com = [keep_largest_component(pamc) for pamc in pred_amodal_masks_com]

                    pred_amodal_masks = [np.array(img.resize((W, H))) for img in pred_amodal_masks]
                    pred_amodal_masks = np.array(pred_amodal_masks).astype('uint8')
                    pred_amodal_masks = (pred_amodal_masks.sum(axis=-1) > 600).astype('uint8')
                    pred_amodal_masks = [keep_largest_component(pamc) for pamc in pred_amodal_masks]

                    # Compute IoU
                    masks = [(np.array(Image.open(bm).convert('P')) == obj_id).astype('uint8') for bm in batch_masks]
                    ious = []
                    masks_margin_shrink = [bm.copy() for bm in masks]
                    mask_H, mask_W = masks_margin_shrink[0].shape

                    for bi, (a, b) in enumerate(zip(masks, pred_amodal_masks)):
                        zero_mask_cp = np.zeros_like(masks_margin_shrink[bi])
                        zero_mask_cp[masks_margin_shrink[bi] == 1] = 255
                        mask_binary_cp = zero_mask_cp.astype(np.uint8)
                        mask_binary_cp[:int(mask_H*0.05), :] = mask_binary_cp[-int(mask_H*0.05):, :] = \
                            mask_binary_cp[:, :int(mask_W*0.05)] = mask_binary_cp[:, -int(mask_W*0.05):] = 0

                        if mask_binary_cp.max() == 0:
                            ious.append(1.0)
                            continue

                        area_a = (a > 0).sum()
                        area_b = (b > 0).sum()
                        if area_a == 0 and area_b == 0:
                            ious.append(1.0)
                        elif area_a > area_b:
                            ious.append(1.0)
                        else:
                            inter = np.logical_and(a > 0, b > 0).sum()
                            uni = np.logical_or(a > 0, b > 0).sum()
                            obj_iou = inter / (uni + 1e-6)
                            ious.append(obj_iou)

                        if i == 0 and bi == 0:
                            if ious[0] < 0.7:
                                obj_ratio_dict[obj_id] = bbox_from_mask(b)
                            else:
                                obj_ratio_dict[obj_id] = bbox_from_mask(a)

                    # Remove fake completions
                    for pi, pamc in enumerate(pred_amodal_masks_com):
                        if masks[pi].sum() > pred_amodal_masks[pi].sum():
                            ious[pi] = 1.0
                            pred_amodal_masks_com[pi] = resize_mask_with_unique_label(masks[pi], pred_res_hi[0], pred_res_hi[1], obj_id)
                        elif is_super_long_or_wide(pred_amodal_masks[pi], obj_id):
                            ious[pi] = 1.0
                            pred_amodal_masks_com[pi] = resize_mask_with_unique_label(masks[pi], pred_res_hi[0], pred_res_hi[1], obj_id)
                        elif is_skinny_mask(pred_amodal_masks[pi]):
                            ious[pi] = 1.0
                            pred_amodal_masks_com[pi] = resize_mask_with_unique_label(masks[pi], pred_res_hi[0], pred_res_hi[1], obj_id)

                    pred_amodal_masks_dict[obj_id] = pred_amodal_masks_com

                    # Confirm occlusions
                    start, end = (idxs := [ix for ix, x in enumerate(ious) if x < 0.7]) and (idxs[0], idxs[-1]) or (None, None)
                    occ_dict[obj_id] = [1 if ix > 0.7 else 0 for ix in ious]

                    if start is not None and end is not None:
                        start = max(0, start - 2)
                        end = min(modal_pixels[:, i:i + batch_size, :, :, :].shape[1] - 1, end + 2)
                        idx_dict[obj_id] = (start, end)
                        completion_path = ''.join(random.choices('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=4))
                        completion_image_path = f'{output_dir}/completion/{completion_path}/images'
                        completion_masks_path = f'{output_dir}/completion/{completion_path}/masks'
                        os.makedirs(completion_image_path, exist_ok=True)
                        os.makedirs(completion_masks_path, exist_ok=True)
                        idx_path[obj_id] = {'images': completion_image_path, 'masks': completion_masks_path}

                        for idx_ in range(start, end):
                            mask_idx_ = pred_amodal_masks[idx_].copy()
                            mask_idx_[mask_idx_ > 0] = obj_id
                            mask_idx_ = Image.fromarray(mask_idx_).convert('P')
                            mask_idx_.putpalette(DAVIS_PALETTE)
                            mask_idx_.save(os.path.join(completion_masks_path, f"{idx_:08d}.png"))

                # Content completion
                for obj_id, (start, end) in idx_dict.items():
                    completion_image_path = idx_path[obj_id]['images']
                    modal_pixels_current, ori_shape = load_and_transform_masks(MASKS_PATH, resolution=pred_res_hi, obj_id=obj_id)
                    rgb_pixels_current, _, raw_rgb_pixels_current = load_and_transform_rgbs(IMAGE_PATH, resolution=pred_res_hi)
                    modal_pixels_current = modal_pixels_current[:, i:i + batch_size, :, :, :]
                    modal_pixels_current = modal_pixels_current[:, start:end]
                    pred_amodal_masks_current = pred_amodal_masks_dict[obj_id][start:end]
                    modal_mask_union = (modal_pixels_current[0, :, 0, :, :].cpu().float().numpy() > 0).astype('uint8')
                    pred_amodal_masks_current = np.logical_or(pred_amodal_masks_current, modal_mask_union).astype('uint8')
                    pred_amodal_masks_tensor = torch.from_numpy(np.where(pred_amodal_masks_current == 0, -1, 1)).float().unsqueeze(0).unsqueeze(2).repeat(1, 1, 3, 1, 1)

                    rgb_pixels_current = rgb_pixels_current[:, i:i + batch_size, :, :, :][:, start:end]
                    modal_obj_mask = (modal_pixels_current > 0).float()
                    modal_background = 1 - modal_obj_mask
                    rgb_pixels_current = (rgb_pixels_current + 1) / 2
                    modal_rgb_pixels = rgb_pixels_current * modal_obj_mask + modal_background
                    modal_rgb_pixels = modal_rgb_pixels * 2 - 1

                    # Predict amodal RGB
                    pred_amodal_rgb = pipeline_rgb(
                        modal_rgb_pixels,
                        pred_amodal_masks_tensor,
                        height=pred_res_hi[0],
                        width=pred_res_hi[1],
                        num_frames=end - start,
                        decode_chunk_size=8,
                        motion_bucket_id=127,
                        fps=8,
                        noise_aug_strength=0.02,
                        min_guidance_scale=1.5,
                        max_guidance_scale=1.5,
                        generator=generator,
                    ).frames[0]

                    pred_amodal_rgb = [np.array(img) for img in pred_amodal_rgb]
                    pred_amodal_rgb = np.array(pred_amodal_rgb).astype('uint8')
                    pred_amodal_rgb_save = np.array([cv2.resize(frame, (ori_shape[1], ori_shape[0]), interpolation=cv2.INTER_LINEAR)
                                                    for frame in pred_amodal_rgb])
                    idx_ = start
                    for img in pred_amodal_rgb_save:
                        cv2.imwrite(os.path.join(completion_image_path, f"{idx_:08d}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                        idx_ += 1
            else:
                for obj_id in out_obj_ids:
                    occ_dict[obj_id] = [1] * len(batch_masks)

            # Process with SAM-3D-Body
            mask_outputs, id_batch, empty_frame_list = process_image_with_mask(
                estimator, batch_images, batch_masks, idx_path, idx_dict, mhr_shape_scale_dict, occ_dict, camera_intrinsics
            )

            num_empth_ids = 0
            for frame_id in range(len(batch_images)):
                image_path = batch_images[frame_id]
                if frame_id in empty_frame_list:
                    mask_output = None
                    id_current = None
                    num_empth_ids += 1
                else:
                    mask_output = mask_outputs[frame_id - num_empth_ids]
                    id_current = id_batch[frame_id - num_empth_ids]

                img = cv2.imread(image_path)
                rend_img = visualize_sample_together(img, mask_output, estimator.faces, id_current)
                cv2.imwrite(
                    f"{output_dir}/rendered_frames/{os.path.basename(image_path)[:-4]}.jpg",
                    rend_img.astype(np.uint8),
                )

                # Save rendered frames for individual person (by tracked obj_id if available)
                # Convert consecutive IDs to actual PIDs for saving
                rend_img_list = visualize_sample(img, mask_output, estimator.faces, id_current)
                for ri, rend_img in enumerate(rend_img_list):
                    if id_current is not None and ri < len(id_current):
                        consecutive_id = int(id_current[ri])
                    else:
                        consecutive_id = int(ri + 1)
                    actual_pid = to_actual_pid(consecutive_id)
                    cv2.imwrite(
                        f"{output_dir}/rendered_frames_individual/{actual_pid}/{os.path.basename(image_path)[:-4]}_{actual_pid}.jpg",
                        rend_img.astype(np.uint8),
                    )

                # Debug dump pred_cam_t (and pelvis proxy) per person for this frame.
                # One json record per line in debug_pred_cam_t/<actual_pid>/pred_cam_t.jsonl
                frame_name = os.path.basename(image_path)[:-4]
                if mask_output is not None:
                    for pid, person_output in enumerate(mask_output):
                        if id_current is not None and pid < len(id_current):
                            consecutive_id = int(id_current[pid])
                        else:
                            consecutive_id = int(pid + 1)
                        actual_pid = to_actual_pid(consecutive_id)
                        if actual_pid not in debug_fhs:
                            continue

                        pelvis_local = None
                        pelvis_cam = None
                        try:
                            rec = build_pred_cam_t_debug_record(
                                frame_name=frame_name,
                                obj_id=actual_pid,  # Use actual PID in debug record
                                person_output=person_output,
                                bboxes_kps_data=bboxes_kps_data,
                                obj_id_to_bbox_idx=obj_id_to_bbox_idx,
                            )
                        except Exception:
                            rec = {"frame": frame_name, "obj_id": int(actual_pid)}

                        debug_fhs[actual_pid].write(json.dumps(rec) + "\n")

                # Save mesh for individual person - convert id_current to actual PIDs
                id_current_actual = None
                if id_current is not None:
                    id_current_actual = [to_actual_pid(int(cid)) for cid in id_current]
                save_mesh_results(
                    outputs=mask_output,
                    faces=estimator.faces,
                    save_dir=f"{output_dir}/mesh_4d_individual",
                    image_path=image_path,
                    id_current=id_current_actual,
                )
    finally:
        for _k, fh in debug_fhs.items():
            try:
                fh.close()
            except Exception:
                pass
    
    out_4d_path = os.path.join(output_dir, "4d_result.mp4")
    jpg_folder_to_mp4(f"{output_dir}/rendered_frames", out_4d_path, fps=fps)
    print(f"[INFO] 4D video saved to: {out_4d_path}")
    
    return out_4d_path

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

def main():
    parser = argparse.ArgumentParser(
        description="SAM-Body4D Video Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using bounding boxes (format: obj_id,frame_idx,x_min,y_min,x_max,y_max)
  python infer_video.py --video input.mp4 \\
      --boxes "1,0,100,50,300,400" "2,0,400,100,600,500"
  
  # Using points (format: obj_id,frame_idx,x,y,label where label: 1=positive, 0=negative)
  python infer_video.py --video input.mp4 \\
      --points "1,0,200,300,1" "1,0,250,350,1"
  
  # Note: You MUST provide either --boxes or --points for the model to track objects.
        """
    )
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--config", type=str, default="configs/body4d.yaml", help="Path to config file")
    parser.add_argument("--output", type=str, default=None, help="Output directory (default: auto-generated)")
    # parser.add_argument("--boxes", type=str, nargs="+", default=None, 
                        # help="Bounding boxes for tracking (format: 'obj_id,frame_idx,x_min,y_min,x_max,y_max')")
    parser.add_argument("--points", type=str, nargs="+", default=None,
                        help="Points for tracking (format: 'obj_id,frame_idx,x,y,label' where label: 1=positive, 0=negative)")
    parser.add_argument("--no-completion", action="store_true", help="Disable completion module")
    args = parser.parse_args()
    
    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[INFO] Using device: {device}")
    
    if device.type == "cuda" and torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Load config
    config_path = os.path.join(os.path.dirname(__file__), args.config)
    if not os.path.exists(config_path):
        config_path = args.config
    
    print(f"[INFO] Loading config from: {config_path}")
    cfg = OmegaConf.load(config_path)
    
    # Setup output directory
    if args.output is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(cfg.runtime['output_dir'], f"inference_{timestamp}")
    else:
        output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] Output directory: {output_dir}")
    
    # Initialize models
    print("[INFO] Initializing SAM-3 model...")
    sam3_model, predictor = build_sam3_from_config(cfg)
    
    print("[INFO] Initializing SAM-3D-Body model...")
    estimator = build_sam3_3d_body_config(cfg, device)
    
    # Initialize completion models if enabled
    pipeline_mask, pipeline_rgb, depth_model = None, None, None
    if cfg.completion.get('enable', False) and not args.no_completion:
        print("[INFO] Initializing completion models...")
        pipeline_mask, pipeline_rgb, depth_model, _ = build_diffusion_vas_config(cfg)
    
    # Read video metadata
    video_path = args.video
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    fps, total_frames = read_video_metadata(video_path)
    print(f"[INFO] Video: {video_path}")
    print(f"[INFO] FPS: {fps}, Total frames: {total_frames}")
    
    # Initialize inference state
    print("[INFO] Initializing inference state...")
    inference_state = predictor.init_state(video_path=video_path)
    predictor.clear_all_points_in_video(inference_state)
    
    # Get video dimensions for relative coordinates
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    # Parse and add prompts
    out_obj_ids = []
    
    # if args.boxes is not None:
    print("[INFO] Adding bounding box prompts...")
    
    bboxes_kps_data = load_bbox_kp("/mnt/neon/zonghuan/data/sam4d_body/inputs/bboxes_kps_refined", "428")
    # Sanity check: visualize bboxes + kps on the first SAM3 input frame.
    save_sanity_first_frame_overlay(
        inference_state=inference_state,
        bboxes_kps_data=bboxes_kps_data,
        output_dir=output_dir,
        frame_idx=0,
    )
    # Don't use 0 for object id as it is reserved for background in mask PNGs.
    # SAM3D-Body expects consecutive IDs (1, 2, 3, ...) internally.
    # Build a mapping from consecutive IDs to actual PIDs for later conversion.
    selected_boxes = list(range(len(bboxes_kps_data[0]['bboxes'])))
    pid_list = bboxes_kps_data[0]['pids']
    
    # Build bidirectional mappings:
    # - consecutive_to_actual: {1: actual_pid1, 2: actual_pid2, ...}
    # - actual_to_consecutive: {actual_pid1: 1, actual_pid2: 2, ...}
    consecutive_to_actual = {}
    actual_to_consecutive = {}
    for i, pid in enumerate(pid_list):
        consecutive_id = i + 1  # 1-based consecutive IDs
        actual_pid = int(pid)
        consecutive_to_actual[consecutive_id] = actual_pid
        actual_to_consecutive[actual_pid] = consecutive_id
    
    # Explicit mapping for debug/projection utilities (supports discontinuous obj_id values).
    # Note: obj_id_to_bbox_idx uses ACTUAL PIDs as keys (for backward compatibility with kps lookup)
    obj_id_to_bbox_idx = {int(pid): int(i) for i, pid in enumerate(pid_list)}
    
    # Save the ID mapping to the output directory
    id_mapping_path = os.path.join(output_dir, "id_mapping.json")
    with open(id_mapping_path, "w", encoding="utf-8") as f:
        json.dump({
            "consecutive_to_actual": {str(k): v for k, v in consecutive_to_actual.items()},
            "actual_to_consecutive": {str(k): v for k, v in actual_to_consecutive.items()},
        }, f, indent=2)
    print(f"[INFO] Saved ID mapping to: {id_mapping_path}")
    
    for bbox_idx in selected_boxes:
        consecutive_id = bbox_idx + 1  # Use consecutive IDs (1, 2, 3, ...) for SAM3
        actual_pid = int(pid_list[bbox_idx])
        bbox = bboxes_kps_data[0]['bboxes'][bbox_idx]
        rel_box = bbox / [width, height, width, height]
        
        print(f"  Consecutive ID {consecutive_id} (actual PID {actual_pid}) at frame {0}: box relative coordinates {rel_box}")
        
        _, out_obj_ids, low_res_masks, video_res_masks = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=consecutive_id,  # Use consecutive ID for SAM3
            box=rel_box,
        )
        
    if args.points is not None:
        print("[INFO] Adding point prompts...")
        # Group points by (obj_id, frame_idx)
        points_by_obj_frame = {}
        for point_str in args.points:
            parts = point_str.split(',')
            if len(parts) != 5:
                raise ValueError(f"Invalid point format: {point_str}. Expected: obj_id,frame_idx,x,y,label")
            
            obj_id = int(parts[0])
            frame_idx = int(parts[1])
            x, y = float(parts[2]), float(parts[3])
            label = int(parts[4])
            
            key = (obj_id, frame_idx)
            if key not in points_by_obj_frame:
                points_by_obj_frame[key] = {'points': [], 'labels': []}
            
            points_by_obj_frame[key]['points'].append([x / width, y / height])
            points_by_obj_frame[key]['labels'].append(label)
        
        # Add points for each object/frame
        for (obj_id, frame_idx), data in points_by_obj_frame.items():
            points_tensor = torch.tensor(data['points'], dtype=torch.float32)
            labels_tensor = torch.tensor(data['labels'], dtype=torch.int32)
            
            print(f"  Object {obj_id} at frame {frame_idx}: {len(data['points'])} point(s)")
            
            _, out_obj_ids, low_res_masks, video_res_masks = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                points=points_tensor,
                labels=labels_tensor,
            )
    
    out_obj_ids = sorted(list(set(out_obj_ids)))
    print(f"[INFO] Tracking {len(out_obj_ids)} object(s) with IDs: {out_obj_ids}")
    
    # Run mask generation
    video_segments = mask_generation(video_path, predictor, inference_state, output_dir, fps, out_obj_ids)
    
    # Run 4D generation
    batch_size = cfg.sam_3d_body.get('batch_size', 1)
    detection_resolution = cfg.completion.get('detection_resolution', [256, 512])
    completion_resolution = cfg.completion.get('completion_resolution', [512, 1024])

    # Read video info
    cam_num = video_path.split("/")[-1].split(".")[0].split("_")[0][-1]
    # cam_int = read_camera_intrinsics(f"/mnt/data/sam4d_body/inputs/camera_params/intrinsic_{cam_num}.json", scale=0.5)
    cam_int, dist = read_camera_intrinsics_new(f"/mnt/data/sam4d_body/inputs/camera_params_new/parameters-camera-0{cam_num}.json")
    cam_int = torch.from_numpy(cam_int).float().unsqueeze(0) 
    generate_4d(
        output_dir, estimator, out_obj_ids, batch_size, fps,
        pipeline_mask, pipeline_rgb, depth_model,
        detection_resolution, completion_resolution, cam_int,
        bboxes_kps_data=bboxes_kps_data,
        obj_id_to_bbox_idx=obj_id_to_bbox_idx,
        consecutive_to_actual=consecutive_to_actual,
    )
    
    print(f"[INFO] Inference complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
