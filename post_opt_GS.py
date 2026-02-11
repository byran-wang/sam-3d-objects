# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Standalone Gaussian Splatting post-optimization script.

Loads saved inference output from demo.py and runs layout_post_optimization_method_GS
without re-running the full inference pipeline.

Usage:
    python post_opt_GS.py --out-dir <out_dir>

Expects the following files in <out_dir> (produced by demo.py):
    - gaussian.ply: Saved Gaussian splatting model
    - post_opt_data.pt: Saved pose, intrinsics, mask, rgb, and pointmap data
"""
import os
import sys
import json
import argparse

import numpy as np
import torch

sys.path.append("notebook")

from sam3d_objects.model.backbone.tdfy_dit.representations.gaussian.gaussian_model import (
    Gaussian,
)
from sam3d_objects.pipeline.inference_utils import layout_post_optimization_method_GS
from sam3d_objects.data.dataset.tdfy.transforms_3d import compose_transform
from pytorch3d.transforms import quaternion_to_matrix

from demo_scene import _GL_TO_CV, _R_ZUP_TO_YUP
from demo import render_novel_view, gaussian_zup_to_yup


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gaussian_path = os.path.join(args.out_dir, "gaussian.ply")
    data_path = os.path.join(args.out_dir, "post_opt_data.pt")

    if not os.path.exists(gaussian_path):
        print(f"Gaussian PLY not found: {gaussian_path}")
        return
    if not os.path.exists(data_path):
        print(f"Post-optimization data not found: {data_path}")
        return

    # Load saved data
    data = torch.load(data_path, map_location=device, weights_only=False)

    # Reconstruct Gaussian from PLY
    gs = Gaussian(**data["gaussian_init_params"], device=device)
    gs.load_ply(gaussian_path)

    # Extract inputs
    rotation = data["rotation"].to(device)
    translation = data["translation"].to(device)
    scale = data["scale"].to(device)
    intrinsics = data["intrinsics"].to(device)
    mask = data["mask"][0, 0]  # (1, 1, H, W) -> (H, W)
    rgb_gt = data["rgb"][0]  # (1, 3, H, W) -> (3, H, W)

    pointmap_unnorm = data.get("pointmap_unnorm")
    if pointmap_unnorm is not None:
        point_map = pointmap_unnorm[0].permute(1, 2, 0).to(device)  # (1, 3, H, W) -> (H, W, 3)
    else:
        H, W = mask.shape
        point_map = torch.zeros(H, W, 3, device=device)

    # Normalize intrinsics (make isotropic, same as pipeline)
    intrinsics = intrinsics.clone()
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    re_focal = min(fx, fy)
    intrinsics[0, 0], intrinsics[1, 1] = re_focal, re_focal

    print(f"Running GS post-optimization on {args.out_dir}")
    print(f"  Gaussian: {gaussian_path}")
    print(f"  Mask: {mask.shape}, RGB: {rgb_gt.shape}, Pointmap: {point_map.shape}")
    print(f"  Intrinsics (isotropic focal): {re_focal:.2f}")

    # Run post-optimization
    revised_quat, revised_t, revised_scale, final_iou, initial_iou, _, flag_optim = (
        layout_post_optimization_method_GS(
            gs,
            rotation,
            translation,
            scale,
            mask,
            rgb_gt,
            point_map,
            intrinsics,
            Enable_occlusion_check=False,
            Enable_manual_alignment=False,
            Enable_shape_ICP=False,
            Enable_rendering_optimization=True,
            min_size=518,
            device=device,
            backend="gsplat",
        )
    )

    # Refine scale (make isotropic — same as pipeline.refine_scale)
    mean_scale = revised_scale.mean(dim=1, keepdim=True)
    revised_scale = mean_scale.expand_as(revised_scale).clone()

    print(f"  Initial IoU: {initial_iou:.4f}")
    print(f"  Final IoU:   {final_iou:.4f}")
    print(f"  Optimization accepted: {flag_optim}")

    # Recompute o2c transform with optimized pose
    R_l2c = quaternion_to_matrix(revised_quat)
    l2c_transform = compose_transform(
        scale=revised_scale,
        rotation=R_l2c,
        translation=revised_t,
    )
    transform_matrix = l2c_transform.get_matrix()[0].cpu().numpy().T
    o2c = _GL_TO_CV.T @ _R_ZUP_TO_YUP.T @ transform_matrix @ _R_ZUP_TO_YUP

    # De-normalize intrinsics to pixel values for camera.json
    # The intrinsics in data are normalized [0,1]; read existing camera.json for pixel-space K
    camera_json_path = os.path.join(args.out_dir, "camera.json")
    if os.path.exists(camera_json_path):
        with open(camera_json_path, "r") as f:
            existing = json.load(f)
        K_pixel = np.array(existing["K"], dtype=np.float32)
    else:
        K_pixel = intrinsics.cpu().numpy()

    # Save updated camera.json with optimized o2c
    camera_data = {
        "K": K_pixel.tolist(),
        "blw2cvc": o2c.tolist(),
    }
    with open(camera_json_path, "w") as f:
        json.dump(camera_data, f)
    print(f"  Updated camera.json in {args.out_dir}")

    # Save optimized pose
    optimized = {
        "rotation": revised_quat.cpu(),
        "translation": revised_t.cpu(),
        "scale": revised_scale.cpu(),
        "iou": final_iou,
        "iou_before_optim": initial_iou,
        "optim_accepted": flag_optim,
        "intrinsics": intrinsics.cpu(),
    }
    result_path = os.path.join(args.out_dir, "post_opt_result.pt")
    torch.save(optimized, result_path)
    print(f"  Saved optimized pose to {result_path}")

    # Render novel view with optimized pose
    gs_fresh = Gaussian(**data["gaussian_init_params"], device=device)
    gs_fresh.load_ply(gaussian_path)
    gaussian_zup_to_yup(gs_fresh)
    output_for_render = {
        "gaussian": [gs_fresh],
        "rotation": revised_quat,
        "translation": revised_t,
        "scale": revised_scale,
    }
    render_novel_view(
        output_for_render,
        args.out_dir,
        filename="rendered_novel_view_post_opt.png",
        distance=1.5,
        hfov=50.0,
        elevation=45.0,
        azimuth=135.0,
        resolution=512,       
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run GS post-optimization from saved inference output."
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Directory containing gaussian.ply and post_opt_data.pt from demo.py.",
    )
    args = parser.parse_args()
    main(args)
