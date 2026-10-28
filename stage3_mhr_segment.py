#!/usr/bin/env python3
"""
Stage 3 — per-segment variant.

Splits raw_mhr.pt by segment boundaries, runs the full Stage 3 pipeline
(Option 1 smoothing + mask reproj + ground constraint) on each segment
independently, then reassembles the segments into one sequence and exports
per-person NPZ meshes.

This avoids cross-segment discontinuity issues that arise when the
smoothers treat the concatenated multi-segment sequence as one
continuous trajectory.

Usage:
  python stage3_mhr_segment.py \\
      --raw /path/to/raw_mhr.pt \\
      --config configs/body4d.yaml \\
      --camera-intrinsics-json /path/to/intrinsics.json \\
      --extrinsics-json /path/to/extrinsics.json \\
      --enable-ground \\
      --out /path/to/output_dir
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

import sys

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
sam3d_body_pkg_dir = os.path.join(REPO_DIR, "models", "sam_3d_body")
if sam3d_body_pkg_dir not in sys.path:
    sys.path.insert(0, sam3d_body_pkg_dir)

from models.sam_3d_body.sam_3d_body.models.meta_arch.mhr_io import load_raw_mhr
from smoothing.stage3_core import (
    Stage3Config,
    stack_frames_to_tensors,
    freeze_shape_scale_first_frame,
    run_stage3_post_optimizations,
)
from smoothing.ground_plane_opt import load_extrinsics_json, GroundOptConfig
from smoothing.feet_z_plot import plot_feet_z_from_stage3, plot_body_landmarks_z
from smoothing.reproj_opt import ReprojOptConfig
from smoothing.mask_reproj_opt import MaskReprojOptConfig
from utils import kalman_smooth_mhr_params_per_obj_id_adaptive, ema_smooth_global_rot_per_obj_id_adaptive
from utils.camera_utils import adjust_K, read_camera_intrinsics_new
from utils.extract_mesh_ground_info import run_extract_ground_info
from utils.id_mapping import load_segment_id_mappings_from_meta
from utils.model_factory import build_sam3d_body_from_config
from utils.plot_ground_info import run_plot_ground_info
from utils.plot_mesh_heights import plot_mesh_heights_from_stage3
from utils.zip_utils import cleanup_extracted_dir, make_extract_root, resolve_archive_dir
from scripts.diagnose_pred_cam_t import run_diagnostics as run_pred_cam_t_diagnostics

from stage3_mhr_full import (
    run_batched_mhr_forward,
    export_per_person_meshes,
    generate_reproj_overlays,
)


# ---------------------------------------------------------------------------
# Segment splitting
# ---------------------------------------------------------------------------

def split_frames_by_segment(
    frames: List[Dict[str, Any]],
    segment_id_mappings: List[Dict[str, Any]],
) -> List[Tuple[Dict[str, Any], List[Dict[str, Any]]]]:
    """
    Split raw frames into per-segment lists.

    Returns list of (segment_info, segment_frames) tuples, where
    segment_info has frame_start, frame_end, consecutive_to_actual, segment_key.
    """
    frame_name_to_idx = {}
    for i, fr in enumerate(frames):
        fname = str(fr.get("frame", f"{i:08d}"))
        frame_name_to_idx[fname] = i

    segments_out = []
    for seg in segment_id_mappings:
        fs = int(seg["frame_start"])
        fe = int(seg["frame_end"])
        seg_frames = []
        for fr in frames:
            fname = str(fr.get("frame", ""))
            try:
                fidx = int(fname)
            except ValueError:
                continue
            if fs <= fidx <= fe:
                seg_frames.append(fr)
        if seg_frames:
            segments_out.append((seg, seg_frames))
    return segments_out


# ---------------------------------------------------------------------------
# Per-segment Stage 3
# ---------------------------------------------------------------------------

def process_one_segment(
    *,
    seg_info: Dict[str, Any],
    seg_frames: List[Dict[str, Any]],
    estimator,
    cfg_path: str,
    device: torch.device,
    args,
    stage1_dir: str,
    extract_root: Optional[str],
    out_dir: str,
    seg_idx: int,
) -> Optional[Dict[str, Any]]:
    """
    Run full Stage 3 on one segment's frames.

    Returns a dict with the post-optimization tensors on CPU (j3d, pred_cam_t,
    verts, faces_np, frame_names, obj_ids_all, frame_obj_ids_slots, T, N), used
    for mesh export and combined plotting, or None for an empty segment.
    """
    seg_key = seg_info.get("segment_key", f"seg{seg_idx}")
    fs = int(seg_info["frame_start"])
    fe = int(seg_info["frame_end"])
    print(f"\n{'='*60}")
    print(f"[SEGMENT {seg_idx}] {seg_key}  frames [{fs}, {fe}]  ({len(seg_frames)} frames)")
    print(f"{'='*60}")

    mhr, frame_obj_ids_slots, vis_flags, frame_names, obj_ids_all = stack_frames_to_tensors(
        seg_frames, device=device,
    )
    T = len(frame_names)
    N = len(obj_ids_all)
    print(f"[INFO] T={T}, N={N}, obj_ids={obj_ids_all}")

    if T == 0 or N == 0:
        print(f"[WARN] Empty segment {seg_key}, skipping")
        return None

    # Option 1 smoothing
    if not args.no_option1:
        mhr = kalman_smooth_mhr_params_per_obj_id_adaptive(
            mhr_dict=mhr, num_frames=T, frame_obj_ids=frame_obj_ids_slots,
            keys_to_smooth=[k for k in ["body_pose", "hand", "pred_cam_t"] if k in mhr],
            kalman_cfg=None, vis_flags=vis_flags,
        )
        freeze_shape_scale_first_frame(mhr, T=T, N=N)
        if "global_rot" in mhr:
            mhr = ema_smooth_global_rot_per_obj_id_adaptive(
                mhr_dict=mhr, num_frames=T, frame_obj_ids=frame_obj_ids_slots,
                vis_flags=vis_flags, key_name="global_rot",
            )

    for k in ("global_rot", "body_pose", "scale", "shape", "pred_cam_t"):
        if k not in mhr:
            raise ValueError(f"Missing required key: {k}")

    # MHR forward
    verts, j3d, faces_np = run_batched_mhr_forward(
        estimator, mhr, device, int(args.mhr_batch_size), T, N,
    )
    pred_cam_t = mhr.get("pred_cam_t", None)

    # Post-optimizations (mask reproj, kps reproj, ground)
    enable_mask_reproj = not args.no_mask_reproj
    enable_ground = args.enable_ground

    mask_dir = resolve_archive_dir(os.path.join(stage1_dir, "masks"), extract_root)
    if enable_mask_reproj:
        if not os.path.isdir(mask_dir):
            print(f"[WARN] Mask directory not found: {mask_dir} — disabling mask reproj.")
            enable_mask_reproj = False
        elif not args.camera_intrinsics_json:
            print("[WARN] --camera-intrinsics-json required for mask reproj — disabling.")
            enable_mask_reproj = False

    # Build the segment_id_mappings for *this* segment only, with frame offsets
    # relative to the segment's own indexing (0-based within segment).
    seg_mapping_for_opt = [{
        "segment_key": seg_key,
        "frame_start": 0,
        "frame_end": T - 1,
        "consecutive_to_actual": seg_info.get("consecutive_to_actual", {}),
    }]

    if enable_mask_reproj or enable_ground:
        cfg3 = Stage3Config(
            enable_option1=(not args.no_option1),
            enable_mask_reproj=enable_mask_reproj,
            enable_kps_reproj=False,
            enable_ground=enable_ground,
            mask_dir=mask_dir if enable_mask_reproj else None,
            mask_reproj_cfg=MaskReprojOptConfig(
                iters=int(args.mask_reproj_iters), lr=float(args.mask_reproj_lr),
                lambda_vertex_in_mask=float(args.mask_reproj_lambda_vertex),
                lambda_mask_coverage=float(args.mask_reproj_lambda_coverage),
                lambda_prior=float(args.mask_reproj_lambda_prior),
                lambda_vel=float(args.mask_reproj_lambda_vel),
                lambda_accel=float(args.mask_reproj_lambda_accel),
                num_sample_vertices=int(args.mask_reproj_num_verts),
                num_sample_mask_points=int(args.mask_reproj_num_mask_pts),
            ),
            camera_intrinsics_json=args.camera_intrinsics_json,
            camera_scale=float(args.camera_scale),
            reproj_cfg=ReprojOptConfig(),
            extrinsics_json=args.extrinsics_json,
            ground_cfg=GroundOptConfig(
                iters=int(args.ground_iters), lr=float(args.ground_lr),
                lambda_plane=float(args.ground_lambda_plane),
                lambda_slide=float(args.ground_lambda_slide),
                lambda_prior=float(args.ground_lambda_prior),
                lambda_vel=float(args.ground_lambda_vel),
                lambda_accel=float(args.ground_lambda_accel),
                contact_z_thresh=float(args.contact_z_thresh),
                contact_v_xy_thresh=float(args.contact_vxy_thresh),
                world_scale=float(args.world_scale),
            ),
        )
        mhr, opt_summary = run_stage3_post_optimizations(
            cfg=cfg3, device=device, mhr=mhr, frame_names=frame_names,
            obj_ids_all=obj_ids_all, frame_obj_ids_slots=frame_obj_ids_slots,
            vis_flags=vis_flags, keypoints3d_local=j3d,
            vertices_local=verts if enable_mask_reproj else None,
            segment_id_mappings=seg_mapping_for_opt,
        )
        pred_cam_t = mhr["pred_cam_t"]

        seg_summary_path = os.path.join(out_dir, f"stage3_opt_summary_{seg_key}.json")
        with open(seg_summary_path, "w", encoding="utf-8") as f:
            json.dump(opt_summary, f, indent=2)

    # Move tensors to CPU for mesh export and combined plotting; free GPU copies
    seg_tensor_data = {
        "j3d": j3d.detach().cpu(),
        "pred_cam_t": pred_cam_t.detach().cpu() if isinstance(pred_cam_t, torch.Tensor) else pred_cam_t,
        "verts": verts.detach().cpu(),
        "faces_np": faces_np,
        "frame_names": list(frame_names),
        "obj_ids_all": list(obj_ids_all),
        "frame_obj_ids_slots": [list(s) for s in frame_obj_ids_slots],
        "T": T,
        "N": N,
    }
    del mhr, verts, j3d, pred_cam_t
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return seg_tensor_data


# ---------------------------------------------------------------------------
# Reassemble per-segment tensors for combined plotting
# ---------------------------------------------------------------------------

def reassemble_full_sequence(
    all_segment_data: List[Optional[Dict[str, Any]]],
) -> Dict[str, Any]:
    """
    Merge per-segment post-optimization tensors into a single (T_total, N_all)
    tensor layout for mesh export and the plotting functions (feet_z,
    ground_plane, reproj_overlay) that expect full-sequence inputs.

    Each segment's vertices are released as soon as they are copied (popped
    from its dict), so the full vertex tensor never coexists with all segment
    copies; absent (frame, person) slots stay zero.
    """
    non_empty = [sd for sd in all_segment_data if sd is not None]
    if not non_empty:
        return {}

    all_oids: set = set()
    for sd in non_empty:
        all_oids.update(sd["obj_ids_all"])
    obj_ids_all = sorted(all_oids)
    N_all = len(obj_ids_all)
    oid_to_slot = {oid: i for i, oid in enumerate(obj_ids_all)}

    first = non_empty[0]
    J = first["j3d"].shape[1]
    V = first["verts"].shape[1]
    faces_np = first["faces_np"]
    T_total = sum(sd["T"] for sd in non_empty)

    full_j3d = torch.zeros(T_total, N_all, J, 3)
    full_pct = torch.zeros(T_total, N_all, 3)
    # torch.empty: memory is committed block by block as segments are written below
    full_verts = torch.empty(T_total, N_all, V, 3)
    all_frame_names: List[str] = []
    all_frame_obj_ids: List[List[int]] = []

    t_off = 0
    for sd in non_empty:
        T_seg, N_seg = sd["T"], sd["N"]
        seg_oids = sd["obj_ids_all"]
        seg_slots = sd["frame_obj_ids_slots"]

        seg_j3d = sd["j3d"].view(T_seg, N_seg, J, 3)
        seg_pct = sd["pred_cam_t"]
        if seg_pct.dim() == 2:
            seg_pct = seg_pct.view(T_seg, N_seg, 3)
        seg_verts = sd.pop("verts").view(T_seg, N_seg, V, 3)
        full_verts[t_off:t_off + T_seg].zero_()

        for ti in range(T_seg):
            global_ids = [0] * N_all
            for si_local, oid in enumerate(seg_oids):
                gi = oid_to_slot[oid]
                if seg_slots[ti][si_local] == oid:
                    global_ids[gi] = oid
                    full_j3d[t_off + ti, gi] = seg_j3d[ti, si_local]
                    full_pct[t_off + ti, gi] = seg_pct[ti, si_local]
                    full_verts[t_off + ti, gi] = seg_verts[ti, si_local]
            all_frame_obj_ids.append(global_ids)

        all_frame_names.extend(sd["frame_names"])
        t_off += T_seg
        del seg_verts

    return {
        "j3d": full_j3d.view(T_total * N_all, J, 3),
        "pred_cam_t": full_pct.view(T_total * N_all, 3),
        "verts": full_verts.view(T_total * N_all, V, 3),
        "faces_np": faces_np,
        "frame_names": all_frame_names,
        "obj_ids_all": obj_ids_all,
        "frame_obj_ids_slots": all_frame_obj_ids,
        "T": T_total,
        "N": N_all,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 3 (per-segment): smooth and export meshes per segment, then concatenate",
    )
    parser.add_argument("--raw", required=True, help="raw_mhr.pt from Stage 2")
    parser.add_argument("--config", default=None, help="Config YAML")
    parser.add_argument("--out", default=None, help="Output directory")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-option1", action="store_true")
    parser.add_argument("--enable-ground", action="store_true")
    parser.add_argument("--no-mask-reproj", action="store_true")
    parser.add_argument("--mask-reproj-iters", type=int, default=200)
    parser.add_argument("--mask-reproj-lr", type=float, default=0.01)
    parser.add_argument("--mask-reproj-lambda-vertex", type=float, default=1.0)
    parser.add_argument("--mask-reproj-lambda-coverage", type=float, default=0.5)
    parser.add_argument("--mask-reproj-lambda-prior", type=float, default=0.05)
    parser.add_argument("--mask-reproj-lambda-vel", type=float, default=0.5)
    parser.add_argument("--mask-reproj-lambda-accel", type=float, default=2.0)
    parser.add_argument("--mask-reproj-num-verts", type=int, default=500)
    parser.add_argument("--mask-reproj-num-mask-pts", type=int, default=200)
    parser.add_argument("--camera-intrinsics-json", default=None)
    parser.add_argument("--camera-scale", type=float, default=0.5)
    parser.add_argument("--extrinsics-json", default=None)
    parser.add_argument("--ground-iters", type=int, default=200)
    parser.add_argument("--ground-lr", type=float, default=0.05)
    parser.add_argument("--ground-lambda-plane", type=float, default=5.0)
    parser.add_argument("--ground-lambda-slide", type=float, default=1.0)
    parser.add_argument("--ground-lambda-prior", type=float, default=0.2)
    parser.add_argument("--ground-lambda-vel", type=float, default=1.0)
    parser.add_argument("--ground-lambda-accel", type=float, default=0.5)
    parser.add_argument("--contact-z-thresh", type=float, default=3.0)
    parser.add_argument("--contact-vxy-thresh", type=float, default=5.0)
    parser.add_argument("--mhr-batch-size", type=int, default=256)
    parser.add_argument("--export-camera-space", action="store_true")
    parser.add_argument("--world-scale", type=float, default=100.0)
    parser.add_argument("--input-dir", default=None,
                        help="Stage1 folder with images(.zip)/masks(.zip). Only read: archives are "
                             "extracted to a temp dir. Default: the --raw folder (extracted in place).")
    args = parser.parse_args()

    out_dir = args.out or os.path.dirname(args.raw)
    os.makedirs(out_dir, exist_ok=True)

    # Stage1 images/masks: a separate --input-dir is never written to
    stage1_dir = args.input_dir or os.path.dirname(args.raw)
    extract_root = make_extract_root("stage3_") if args.input_dir else None
    with open(os.path.join(out_dir, "stage3_per_segment_options.txt"), "w") as f:
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")

    # Load raw payload
    payload = load_raw_mhr(args.raw, map_location="cpu")
    frames = payload["frames"]
    meta = payload.get("meta", {})
    segment_id_mappings = load_segment_id_mappings_from_meta(meta, fallback_dir=stage1_dir)

    if not segment_id_mappings:
        print("[ERROR] No segment_id_mappings found. This script requires per-segment "
              "metadata from stage1_masklet.py. Use stage3_mhr_full.py "
              "for single-segment data.")
        sys.exit(1)

    print(f"[INFO] {len(segment_id_mappings)} segments found:")
    for seg in segment_id_mappings:
        print(f"  [{seg['frame_start']}, {seg['frame_end']}] {seg.get('segment_key', '')}")

    # Diagnostics: visualise raw pred_cam_t before any processing
    diag_dir = os.path.join(out_dir, "diagnostics")
    _mhr_diag, _slots_diag, _vis_diag, _names_diag, _oids_diag = stack_frames_to_tensors(frames, device=torch.device("cpu"))
    run_pred_cam_t_diagnostics(
        mhr=_mhr_diag, frame_obj_ids_slots=_slots_diag,
        frame_names=_names_diag, obj_ids_all=_oids_diag,
        segment_id_mappings=segment_id_mappings, out_dir=diag_dir, label="raw",
    )
    del _mhr_diag, _slots_diag, _vis_diag, _names_diag, _oids_diag

    # Split frames by segment
    segment_splits = split_frames_by_segment(frames, segment_id_mappings)
    print(f"[INFO] Split into {len(segment_splits)} non-empty segments")

    # Build model once
    cfg_path = args.config or meta.get("config_path") or os.path.join(REPO_DIR, "configs", "body4d.yaml")
    if not os.path.exists(cfg_path):
        cfg_path = os.path.join(REPO_DIR, cfg_path)
    cfg = OmegaConf.load(cfg_path)
    device = torch.device(args.device)
    print(f"[INFO] Building SAM-3D-Body model on {device}")
    estimator = build_sam3d_body_from_config(cfg, device=device)

    # Process each segment
    all_segment_tensors: List[Optional[Dict[str, Any]]] = []
    for seg_idx, (seg_info, seg_frames) in enumerate(segment_splits):
        seg_tensors = process_one_segment(
            seg_info=seg_info,
            seg_frames=seg_frames,
            estimator=estimator,
            cfg_path=cfg_path,
            device=device,
            args=args,
            stage1_dir=stage1_dir,
            extract_root=extract_root,
            out_dir=out_dir,
            seg_idx=seg_idx,
        )
        all_segment_tensors.append(seg_tensors)

    # Release model
    del estimator
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print("[INFO] Released MHR model from GPU")

    # Extrinsics for world-space export
    extr = None
    export_world = not args.export_camera_space
    if export_world:
        if not args.extrinsics_json:
            print("[WARN] World-space export requires --extrinsics-json. Falling back to camera space.")
            export_world = False
        else:
            extr = load_extrinsics_json(args.extrinsics_json, device=torch.device("cpu"))
            print("[INFO] Exporting meshes in WORLD coordinates")
    if not export_world:
        print("[INFO] Exporting meshes in CAMERA coordinates")

    # ------------------------------------------------------------------
    # Reassemble segments, export meshes, combined plots
    # ------------------------------------------------------------------
    combined = reassemble_full_sequence(all_segment_tensors)
    del all_segment_tensors
    gc.collect()

    if not combined:
        print("[WARN] No segment data to export or plot")
    else:
        T_all = combined["T"]
        N_all = combined["N"]
        j3d_all = combined["j3d"]
        pred_cam_t_all = combined["pred_cam_t"]
        verts_all = combined["verts"]
        faces_np = combined["faces_np"]
        frame_names_all = combined["frame_names"]
        obj_ids_all_combined = combined["obj_ids_all"]
        frame_obj_ids_all = combined["frame_obj_ids_slots"]

        world_scale = float(args.world_scale)

        # Per-person meshes (camera-space verts + post-opt pred_cam_t, then extrinsics)
        export_per_person_meshes(
            verts=verts_all, pred_cam_t=pred_cam_t_all,
            obj_ids_all=obj_ids_all_combined, frame_obj_ids_slots=frame_obj_ids_all,
            frame_names=frame_names_all, faces_np=faces_np, extr=extr,
            world_scale=world_scale,
            mesh_dir=os.path.join(out_dir, "meshes_4d_individual"),
            T=T_all, N=N_all,
        )

        # Mesh height plot
        if extr is not None:
            try:
                plot_mesh_heights_from_stage3(
                    vertices=verts_all, pred_cam_t=pred_cam_t_all, extr=extr,
                    world_scale=world_scale, T=T_all, N=N_all,
                    obj_ids_all=obj_ids_all_combined,
                    frame_obj_ids_slots=frame_obj_ids_all,
                    output_path=os.path.join(out_dir, "mesh_heights_world.png"),
                    title="Mesh Height (World Space) — Per-Segment Stage 3",
                )
            except Exception as e:
                print(f"[WARN] Failed to generate mesh height plot: {e}")

        # Feet z-coordinate plot
        if extr is not None:
            try:
                plot_feet_z_from_stage3(
                    keypoints3d_local=j3d_all, pred_cam_t=pred_cam_t_all, extr=extr,
                    T=T_all, N=N_all, obj_ids_all=obj_ids_all_combined,
                    frame_obj_ids_slots=frame_obj_ids_all,
                    output_path=os.path.join(out_dir, "feet_z_world.png"),
                    title="Feet Z (World) — Per-Segment Stage 3",
                    world_scale=world_scale,
                )
            except Exception as e:
                print(f"[WARN] Failed to generate feet z-coordinate plot: {e}")

        # Head / pelvis / feet z-coordinate combined plot
        if extr is not None:
            try:
                plot_body_landmarks_z(
                    keypoints3d_local=j3d_all, pred_cam_t=pred_cam_t_all, extr=extr,
                    T=T_all, N=N_all, obj_ids_all=obj_ids_all_combined,
                    frame_obj_ids_slots=frame_obj_ids_all,
                    world_scale=world_scale,
                    output_path=os.path.join(out_dir, "body_landmarks_z_world.png"),
                    title="Body Landmark Z (World) — Per-Segment Stage 3",
                )
            except Exception as e:
                print(f"[WARN] Failed to generate body landmark z-plot: {e}")

        # Ground-plane info + plots
        ground_rows: List[Dict[str, Any]] = []
        if extr is not None:
            try:
                _pkl, ground_rows = run_extract_ground_info(
                    keypoints3d_local=j3d_all.detach().cpu().numpy(),
                    pred_cam_t=pred_cam_t_all.detach().cpu().numpy(),
                    extr=extr, world_scale=world_scale,
                    frame_names=frame_names_all, obj_ids_all=obj_ids_all_combined,
                    frame_obj_ids_slots=frame_obj_ids_all, T=T_all, N=N_all,
                    output_dir=out_dir, basename="ground_plane_info",
                    device=torch.device("cpu"),
                )
                if _pkl:
                    print(f"[INFO] Saved ground-plane info: {_pkl}")
            except Exception as e:
                print(f"[WARN] Ground-plane info extraction failed: {e}")
        if ground_rows:
            try:
                plot_dir = run_plot_ground_info(
                    rows=ground_rows, frame_names=frame_names_all,
                    output_dir=out_dir, frame_interval=200,
                    plot_subdir="ground_plane_plots",
                )
                print(f"[INFO] Saved ground-plane plots: {plot_dir}")
            except Exception as e:
                print(f"[WARN] Ground-plane plotting failed: {e}")

        # Reprojection overlays
        overlay_result = generate_reproj_overlays(
            verts=verts_all, j3d=j3d_all, pred_cam_t=pred_cam_t_all,
            faces_np=faces_np,
            camera_intrinsics_json=args.camera_intrinsics_json,
            camera_scale=float(args.camera_scale),
            bbox_kps_pkl=None,
            payload_meta=payload.get("meta", {}),
            frame_names=frame_names_all, obj_ids_all=obj_ids_all_combined,
            frame_obj_ids_slots=frame_obj_ids_all,
            segment_id_mappings=segment_id_mappings,
            out_dir=out_dir, T=T_all, N=N_all,
            input_dir=args.input_dir, extract_root=extract_root,
        )

        # Clean up dirs extracted in place (a temp extract_root is removed at exit)
        if overlay_result and extract_root is None:
            for d in dict.fromkeys(overlay_result):
                if d:
                    cleanup_extracted_dir(d)

        del combined

    print(f"\n[INFO] Done. Results in: {out_dir}")


if __name__ == "__main__":
    main()
