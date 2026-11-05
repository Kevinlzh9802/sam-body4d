#!/usr/bin/env python3
"""
Stage 3: raw MHR params (Stage 2) -> smoothed, image-aligned, grounded meshes.

Steps (each person separately, so memory is one person's frames, not the
whole frames x people grid):
  1. smooth     zero-phase temporal smoothing of the params (smoothing/temporal.py);
                shape/scale fixed to the person's median
  2. MHR forward on the person's frames
  3. mask       optimize pred_cam_t so the mesh projects onto the person's Stage 1
                mask, per run of consecutive frames (smoothing/mask_reproj_opt.py)
  4. ground     with --extrinsics-json: one scale per person about the camera
                centre + a small per-frame shift along the viewing ray, so the
                lowest foot is on the floor z=0 while the projection stays put
                (smoothing/ground_plane_opt.py)
  5. export     meshes_4d_individual/<pid>.npz (``vertices`` (T_p, V, 3) and mhr70
                ``keypoints`` (T_p, 70, 3) float32, ``faces``, ``frame_names``),
                in world coordinates (world units, cm by default) with extrinsics,
                else camera space (m); ``cam_t`` (T_p, 3) is the camera-space
                translation (m) of the person's local mesh

Steps 1, 3 and 4 can be switched off (--no-smooth, --no-mask-reproj, --no-ground)
to compare. Plots: diagnostics/ (raw pred_cam_t), reproj_overlay/ (final meshes
and body skeletons on the Stage 1 frames, every --overlay-interval frames, 100 by
default), and with extrinsics the world-space height / feet / ground plots.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in (REPO_DIR, os.path.join(REPO_DIR, "models", "sam_3d_body")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models.sam_3d_body.sam_3d_body.models.meta_arch.mhr_io import load_raw_mhr
from smoothing.feet_z_plot import plot_body_landmarks_z, plot_feet_z_from_stage3
from smoothing.ground_plane_opt import GroundOptConfig, load_extrinsics_json, optimize_ground_ray_scale
from smoothing.mask_reproj_opt import MaskReprojOptConfig, load_masks_for_sequence, optimize_mask_reprojection
from smoothing.reproj_overlay_plot import plot_reproj_overlays_from_stage3
from smoothing.stage3_core import stack_frames_to_tensors
from smoothing.temporal import SmoothConfig, person_runs, smooth_mhr_params
from utils.camera_utils import adjust_K, read_camera_intrinsics_new
from utils.extract_mesh_ground_info import run_extract_ground_info
from utils.id_mapping import consecutive_to_actual_per_frame, load_segment_id_mappings_from_meta
from utils.model_factory import build_sam3d_body_from_config
from utils.plot_ground_info import run_plot_ground_info
from utils.plot_mesh_heights import plot_mesh_heights
from utils.zip_utils import cleanup_extracted_dir, make_extract_root, resolve_archive_dir
from scripts.diagnose_pred_cam_t import run_diagnostics as run_pred_cam_t_diagnostics

MHR_KEYS = ("global_rot", "body_pose", "hand", "scale", "shape", "face")
OVERLAY_INTERVAL = 100


# ---------------------------------------------------------------------------
# MHR forward
# ---------------------------------------------------------------------------

def mhr_forward(head_pose, params: Dict[str, torch.Tensor], device: torch.device,
                batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """MHR forward of (P, D) params in batches. Returns camera-axis (y, z flipped)
    local vertices (P, V, 3) and body keypoints (P, 70, 3), both on the CPU."""
    P = params["global_rot"].shape[0]
    zeros = {"hand": 108, "face": 72}
    p = {k: (params[k] if k in params else torch.zeros((P, zeros[k])))
         .to(device=device, dtype=torch.float32) for k in MHR_KEYS}
    verts: Optional[torch.Tensor] = None
    j3d = None
    with torch.no_grad():
        for bs in range(0, P, batch_size):
            be = min(bs + batch_size, P)
            out = head_pose.mhr_forward(
                global_trans=p["global_rot"][bs:be] * 0, global_rot=p["global_rot"][bs:be],
                body_pose_params=p["body_pose"][bs:be], hand_pose_params=p["hand"][bs:be],
                scale_params=p["scale"][bs:be], shape_params=p["shape"][bs:be],
                expr_params=p["face"][bs:be], return_keypoints=True,
                return_joint_coords=False, return_model_params=False, return_joint_rotations=False,
            )
            v, k = out[0].detach(), out[1].detach()[:, :70]
            if verts is None:
                verts = torch.empty((P,) + tuple(v.shape[1:]), dtype=torch.float32)
                j3d = torch.empty((P,) + tuple(k.shape[1:]), dtype=torch.float32)
            verts[bs:be].copy_(v)
            j3d[bs:be].copy_(k)
    # Camera axes: flip y and z
    verts[..., 1:3] *= -1
    j3d[..., 1:3] *= -1
    return verts, j3d


class FrameSubset:
    """verts[bi] for the (frame, person) slots kept for the overlay plots."""

    def __init__(self) -> None:
        self.rows: Dict[int, torch.Tensor] = {}

    def __getitem__(self, bi: int) -> torch.Tensor:
        return self.rows[int(bi)]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True, help="raw_mhr.pt from stage2_sam3body.py")
    ap.add_argument("--out", default=None, help="Output dir (default: alongside raw file)")
    ap.add_argument("--config", default=None, help="Config YAML (default: from raw meta, else configs/body4d.yaml)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", choices=["cuda", "cpu"])
    ap.add_argument("--input-dir", default=None,
                    help="Stage 1 folder with images(.zip)/masks(.zip). Only read: archives are "
                         "extracted to a temp dir. Default: the --raw folder (extracted in place).")
    ap.add_argument("--camera-intrinsics-json", required=True)
    ap.add_argument("--camera-scale", type=float, default=0.5, help="Stage 1 image size / calibration size")
    ap.add_argument("--extrinsics-json", default=None,
                    help="world_to_camera extrinsics, floor z=0: enables the ground step and world export")
    ap.add_argument("--export-camera-space", action="store_true", help="Export camera space even with extrinsics")
    ap.add_argument("--world-scale", type=float, default=100.0, help="World units per metre (100 = cm)")
    ap.add_argument("--mhr-batch-size", type=int, default=256)
    ap.add_argument("--overlay-interval", type=int, default=OVERLAY_INTERVAL,
                    help="Frames between reprojection overlays (reproj_overlay/); 0 = none")
    # 1. smoothing
    ap.add_argument("--no-smooth", action="store_true")
    ap.add_argument("--smooth-sigma-pose", type=float, default=SmoothConfig.sigma_pose,
                    help="Gaussian width (frames) for body_pose, hand, face")
    ap.add_argument("--smooth-sigma-rot", type=float, default=SmoothConfig.sigma_rot, help="... for global_rot")
    ap.add_argument("--smooth-sigma-cam", type=float, default=SmoothConfig.sigma_cam, help="... for pred_cam_t")
    ap.add_argument("--smooth-sigma", type=float, default=None, help="Set all three widths")
    # 3. mask reprojection
    ap.add_argument("--no-mask-reproj", action="store_true")
    ap.add_argument("--mask-reproj-iters", type=int, default=200)
    ap.add_argument("--mask-reproj-lr", type=float, default=0.01)
    ap.add_argument("--mask-reproj-lambda-vertex", type=float, default=1.0)
    ap.add_argument("--mask-reproj-lambda-coverage", type=float, default=0.5)
    ap.add_argument("--mask-reproj-lambda-prior", type=float, default=0.05)
    ap.add_argument("--mask-reproj-lambda-vel", type=float, default=0.5)
    ap.add_argument("--mask-reproj-lambda-accel", type=float, default=2.0)
    ap.add_argument("--mask-reproj-num-verts", type=int, default=500)
    ap.add_argument("--mask-reproj-num-mask-pts", type=int, default=200)
    # 4. ground
    ap.add_argument("--no-ground", action="store_true")
    ap.add_argument("--ground-iters", type=int, default=200)
    ap.add_argument("--ground-lambda-plane", type=float, default=5.0)
    ap.add_argument("--ground-lambda-prior", type=float, default=0.2)
    ap.add_argument("--ground-lambda-vel", type=float, default=1.0)
    ap.add_argument("--ground-lambda-accel", type=float, default=0.5)
    args = ap.parse_args()

    out_dir = args.out or os.path.dirname(args.raw)
    os.makedirs(out_dir, exist_ok=True)
    stage1_dir = args.input_dir or os.path.dirname(args.raw)
    extract_root = make_extract_root("stage3_") if args.input_dir else None
    with open(os.path.join(out_dir, "stage3_run_options.txt"), "w", encoding="utf-8") as f:
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")

    payload = load_raw_mhr(args.raw, map_location="cpu")
    frames = payload.get("frames")
    if frames is None:
        raise ValueError("Invalid raw payload: missing 'frames'.")
    meta = payload.get("meta", {})
    segment_id_mappings = load_segment_id_mappings_from_meta(meta, fallback_dir=stage1_dir)

    device = torch.device(args.device)
    mhr, slots, _vis, frame_names, obj_ids = stack_frames_to_tensors(frames, device=torch.device("cpu"))
    T, N = len(frame_names), len(obj_ids)
    frame_numbers = np.array([int(n) for n in frame_names])
    print(f"[INFO] T={T}, N={N}, device={device}")

    run_pred_cam_t_diagnostics(
        mhr=mhr, frame_obj_ids_slots=slots, frame_names=frame_names, obj_ids_all=obj_ids,
        segment_id_mappings=segment_id_mappings, out_dir=os.path.join(out_dir, "diagnostics"), label="raw",
    )

    # 1. smoothing
    if args.smooth_sigma is not None:
        args.smooth_sigma_pose = args.smooth_sigma_rot = args.smooth_sigma_cam = args.smooth_sigma
    if not args.no_smooth:
        mhr = smooth_mhr_params(mhr, frame_names, slots, obj_ids, SmoothConfig(
            sigma_pose=args.smooth_sigma_pose, sigma_rot=args.smooth_sigma_rot, sigma_cam=args.smooth_sigma_cam))

    K_np, _dist = read_camera_intrinsics_new(args.camera_intrinsics_json)
    K_np = adjust_K(K_np, scale=float(args.camera_scale))
    K = torch.from_numpy(K_np).to(device=device, dtype=torch.float32)

    extr = load_extrinsics_json(args.extrinsics_json, device=device) if args.extrinsics_json else None
    do_ground = extr is not None and not args.no_ground
    export_world = extr is not None and not args.export_camera_space
    ws = float(args.world_scale)
    ground_cfg = GroundOptConfig(
        iters=int(args.ground_iters), lambda_plane=float(args.ground_lambda_plane),
        lambda_prior=float(args.ground_lambda_prior), lambda_vel=float(args.ground_lambda_vel),
        lambda_accel=float(args.ground_lambda_accel), world_scale=ws,
    )
    print(f"[INFO] Steps: smooth={not args.no_smooth}, mask={not args.no_mask_reproj}, ground={do_ground}; "
          f"export in {'WORLD' if export_world else 'CAMERA'} coordinates")

    # 3. masks (Stage 1 pixel values are per-segment tracking IDs)
    mask_dir = resolve_archive_dir(os.path.join(stage1_dir, "masks"), extract_root)
    masks: Dict[int, torch.Tensor] = {}
    a2c_per_frame: Optional[List[Dict[int, int]]] = None
    if not args.no_mask_reproj:
        if not os.path.isdir(mask_dir):
            raise FileNotFoundError(f"Mask directory not found: {mask_dir} (use --no-mask-reproj to skip)")
        masks = load_masks_for_sequence(mask_dir, frame_names, device=device)
        print(f"[INFO] Loaded {len(masks)} mask frames")
        if segment_id_mappings:
            a2c_per_frame = [{a: c for c, a in m.items()}
                             for m in consecutive_to_actual_per_frame(frame_names, segment_id_mappings)]
    mask_cfg = MaskReprojOptConfig(
        iters=int(args.mask_reproj_iters), lr=float(args.mask_reproj_lr),
        lambda_vertex_in_mask=float(args.mask_reproj_lambda_vertex),
        lambda_mask_coverage=float(args.mask_reproj_lambda_coverage),
        lambda_prior=float(args.mask_reproj_lambda_prior),
        lambda_vel=float(args.mask_reproj_lambda_vel),
        lambda_accel=float(args.mask_reproj_lambda_accel),
        num_sample_vertices=int(args.mask_reproj_num_verts),
        num_sample_mask_points=int(args.mask_reproj_num_mask_pts),
    )

    cfg_path = args.config or meta.get("config_path") or os.path.join(REPO_DIR, "configs", "body4d.yaml")
    if not os.path.exists(cfg_path):
        cfg_path = os.path.join(REPO_DIR, cfg_path)
    estimator = build_sam3d_body_from_config(OmegaConf.load(cfg_path), device=device)
    head_pose = estimator.model.head_pose
    faces_np = np.asarray(estimator.faces, dtype=np.int32)

    mesh_dir = os.path.join(out_dir, "meshes_4d_individual")
    os.makedirs(mesh_dir, exist_ok=True)
    cam_t = mhr["pred_cam_t"].view(T, N, 3).clone()
    j3d_all = torch.zeros(T, N, 70, 3)
    overlay_verts = FrameSubset()
    heights: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    summary: Dict[str, Any] = {"mask_reproj": {}, "ground": {}}
    params = {k: mhr[k].view(T, N, -1) for k in MHR_KEYS if k in mhr}
    if extr is not None:
        R_np = extr.R.cpu().numpy().astype(np.float32)
        t_np = extr.t.cpu().numpy().reshape(1, 3).astype(np.float32)

    for si, oid in enumerate(obj_ids):
        idx = np.array([t for t in range(T) if slots[t][si] == oid])
        if idx.size == 0:
            continue
        idx_t = torch.from_numpy(idx)
        print(f"[INFO] Person {oid}: {idx.size} frames")
        # 2. MHR forward on this person's frames
        verts, j3d = mhr_forward(head_pose, {k: v[idx_t, si] for k, v in params.items()}, device,
                                 int(args.mhr_batch_size))
        t_p = cam_t[idx_t, si].to(device)

        # 3. mask reprojection per run of consecutive frames
        if not args.no_mask_reproj:
            runs_metrics = []
            for run in person_runs(np.ones(idx.size, bool), frame_numbers[idx]):
                fr = idx[run]
                run = torch.from_numpy(run)
                run_masks = {i: masks[int(t)] for i, t in enumerate(fr) if int(t) in masks}
                mask_ids = None
                if a2c_per_frame is not None:
                    mask_ids = {i: a2c_per_frame[int(t)][oid] for i, t in enumerate(fr) if oid in a2c_per_frame[int(t)]}
                t_run, m = optimize_mask_reprojection(
                    K=K, vertices_local=verts[run].to(device), t0=t_p[run], masks=run_masks,
                    obj_id=oid, mask_ids=mask_ids, cfg=mask_cfg,
                )
                t_p[run] = t_run
                runs_metrics.append({"frames": [frame_names[int(fr[0])], frame_names[int(fr[-1])]], **m})
            summary["mask_reproj"][str(oid)] = runs_metrics

        # 4. ground: scale about the camera centre + shift along the ray
        if do_ground:
            t_p, scale, m = optimize_ground_ray_scale(
                extr=extr, X3d=j3d.to(device), t0=t_p,
                frame_numbers=torch.from_numpy(frame_numbers[idx]), cfg=ground_cfg,
            )
            verts.mul_(scale)
            j3d.mul_(scale)
            summary["ground"][str(oid)] = m
            print(f"[INFO]   ground: scale {scale:.3f}, lowest foot {m['min_foot_z_mean']:.1f} "
                  f"+- {m['min_foot_z_std']:.1f} {'cm' if ws == 100 else 'units'}")

        # 5. export
        t_cpu = t_p.detach().cpu().float()
        v_cam = verts + t_cpu[:, None, :]
        if extr is not None:
            v_world = (v_cam.numpy() * ws - t_np) @ R_np
            z = v_world[..., 2]
            heights[oid] = (idx, z.max(axis=1) - z.min(axis=1))
        k_cam = (j3d + t_cpu[:, None, :]).numpy()
        v_out = v_world if export_world else v_cam.numpy()
        k_out = (k_cam * ws - t_np) @ R_np if export_world else k_cam
        np.savez_compressed(os.path.join(mesh_dir, f"{oid}.npz"), vertices=v_out.astype(np.float32, copy=False),
                            keypoints=k_out.astype(np.float32, copy=False), cam_t=t_cpu.numpy(),
                            faces=faces_np, frame_names=np.array([frame_names[t] for t in idx]))

        cam_t[idx_t, si] = t_cpu
        j3d_all[idx_t, si] = j3d
        for i, t in enumerate(idx):
            if args.overlay_interval > 0 and t % args.overlay_interval == 0:
                overlay_verts.rows[int(t) * N + si] = verts[i].clone()
        del verts, v_cam, j3d
    del estimator, head_pose, masks
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"[INFO] Exported meshes to {mesh_dir}")
    with open(os.path.join(out_dir, "stage3_opt_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Plots
    cam_t_flat = cam_t.view(T * N, 3)
    j3d_flat = j3d_all.view(T * N, 70, 3)
    if extr is not None:
        extr_cpu = load_extrinsics_json(args.extrinsics_json, device=torch.device("cpu"))
        wu = "cm" if ws == 100 else "world units"
        plot_mesh_heights(heights, os.path.join(out_dir, "mesh_heights_world.png"),
                          title="Mesh Height (World Space) - Stage 3", world_units=wu)
        for name, fn, kw in (
            ("feet_z_world.png", plot_feet_z_from_stage3, {"title": "Average Feet Z (World) - Stage 3"}),
            ("body_landmarks_z_world.png", plot_body_landmarks_z, {"title": "Body Landmark Z (World) - Stage 3"}),
        ):
            try:
                fn(keypoints3d_local=j3d_flat, pred_cam_t=cam_t_flat, extr=extr_cpu, T=T, N=N,
                   obj_ids_all=obj_ids, frame_obj_ids_slots=slots, world_scale=ws,
                   output_path=os.path.join(out_dir, name), **kw)
            except Exception as e:
                print(f"[WARN] {name} failed: {e}")
        try:
            _pkl, rows = run_extract_ground_info(
                keypoints3d_local=j3d_flat.numpy(), pred_cam_t=cam_t_flat.numpy(), extr=extr_cpu,
                world_scale=ws, frame_names=frame_names, obj_ids_all=obj_ids, frame_obj_ids_slots=slots,
                T=T, N=N, output_dir=out_dir, basename="ground_plane_info", device=torch.device("cpu"),
            )
            if rows:
                run_plot_ground_info(rows=rows, frame_names=frame_names, output_dir=out_dir,
                                     frame_interval=200, plot_subdir="ground_plane_plots")
        except Exception as e:
            print(f"[WARN] Ground-plane info failed: {e}")

    input_dir = args.input_dir or meta.get("input_dir", "")
    images_dir = resolve_archive_dir(os.path.join(input_dir, "images"), extract_root) if input_dir else ""
    if args.overlay_interval > 0 and images_dir and os.path.isdir(images_dir):
        try:
            plot_reproj_overlays_from_stage3(
                verts=overlay_verts, keypoints3d_local=j3d_flat, pred_cam_t=cam_t_flat, faces=faces_np,
                K=K_np, images_dir=images_dir, masks_dir=mask_dir if os.path.isdir(mask_dir) else None,
                T=T, N=N, obj_ids_all=obj_ids, frame_obj_ids_slots=slots, frame_names=frame_names,
                output_dir=os.path.join(out_dir, "reproj_overlay"), frame_interval=args.overlay_interval,
                segment_id_mappings=segment_id_mappings,
            )
        except Exception as e:
            print(f"[WARN] Reproj overlays failed: {e}")
    elif args.overlay_interval > 0:
        print(f"[INFO] Skipping reproj overlay: images dir not found ({images_dir!r})")

    # Archives extracted in place next to the raw file (a temp extract_root is removed at exit)
    if extract_root is None:
        for d in dict.fromkeys([mask_dir, images_dir]):
            if d:
                cleanup_extracted_dir(d)
    print(f"[INFO] Done. Results in: {out_dir}")


if __name__ == "__main__":
    main()
