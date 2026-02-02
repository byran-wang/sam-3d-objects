# Copyright (c) Meta Platforms, Inc. and affiliates.
import sys
import json
import os
import numpy as np
import torch
import argparse

sys.path.append("notebook")
from inference import load_image, load_mask
from demo import (
    _load_camera_data,
    load_pointmap_from_depth,
    load_intrinsics,
    normalize_intrinsics,
    convert_pointmap_to_pytorch3d,
    visualize_in_rerun,
)
from demo_scene import _GL_TO_CV, _R_ZUP_TO_YUP
from sam3d_objects.model.backbone.tdfy_dit.representations.gaussian.gaussian_model import Gaussian
from sam3d_objects.pipeline.inference_utils import layout_post_optimization_method_GS
from pytorch3d.transforms import matrix_to_quaternion, Transform3d
from sam3d_objects.data.dataset.tdfy.transforms_3d import decompose_transform


def load_gaussian_from_ply(ply_path: str, device: str = "cuda") -> Gaussian:
    """Load Gaussian splatting model from PLY file."""
    gaussian = Gaussian(
        aabb=[-1, -1, -1, 1, 1, 1],
        sh_degree=0,
        device=device,
    )
    gaussian.load_ply(ply_path)
    return gaussian


def decompose_o2c_to_pose(o2c: np.ndarray, device: str = "cuda"):
    """Decompose o2c matrix back to quaternion, translation, scale.

    The o2c in camera.json was computed as:
        o2c = _GL_TO_CV.T @ _R_ZUP_TO_YUP.T @ transform_matrix @ _R_ZUP_TO_YUP

    We need to reverse this to get back the original transform.

    Returns:
        quaternion: (1, 1, 4) tensor
        translation: (1, 3) tensor
        scale: (1, 3) tensor
    """
    # Reverse the coordinate transforms
    # o2c = _GL_TO_CV.T @ _R_ZUP_TO_YUP.T @ M @ _R_ZUP_TO_YUP
    # M = _R_ZUP_TO_YUP @ _GL_TO_CV @ o2c @ _R_ZUP_TO_YUP.T
    transform_matrix = _R_ZUP_TO_YUP @ _GL_TO_CV @ o2c @ _R_ZUP_TO_YUP.T

    # Convert to torch and create Transform3d
    # Note: transform_matrix is in row-major form, need to transpose for Transform3d
    M = torch.from_numpy(transform_matrix.T.astype(np.float32)).to(device)
    tfm = Transform3d(matrix=M.unsqueeze(0), device=device)

    # Decompose into scale, rotation, translation
    decomposed = decompose_transform(tfm)
    scale = decomposed.scale  # (1, 3)
    rotation = decomposed.rotation  # (1, 3, 3)
    translation = decomposed.translation  # (1, 3)

    # Convert rotation matrix to quaternion
    quaternion = matrix_to_quaternion(rotation)  # (1, 4)
    quaternion = quaternion.unsqueeze(1)  # (1, 1, 4)

    return quaternion, translation, scale


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Check required input files
    if not os.path.exists(args.image_path):
        print(f"Image {args.image_path} not found.")
        return
    if not os.path.exists(args.mask_path):
        print(f"Mask {args.mask_path} not found.")
        return
    if not os.path.exists(args.depth_file):
        print(f"Depth file {args.depth_file} not found.")
        return
    if not os.path.exists(args.meta_file):
        print(f"Meta file {args.meta_file} not found.")
        return

    # Check demo.py outputs
    camera_json_path = os.path.join(args.demo_out_dir, "camera.json")
    scene_ply_path = os.path.join(args.demo_out_dir, "scene.ply")

    if not os.path.exists(camera_json_path):
        print(f"camera.json not found in {args.demo_out_dir}. Run demo.py first.")
        return
    if not os.path.exists(scene_ply_path):
        print(f"scene.ply not found in {args.demo_out_dir}. Run demo.py with gaussian output first.")
        return

    # --- Load image, mask ---
    print(f"Loading image: {args.image_path}")
    image = load_image(args.image_path)
    print(f"Loading mask: {args.mask_path}")
    mask = load_mask(args.mask_path)
    height, width = image.shape[:2]

    # --- Load depth and intrinsics to create pointmap ---
    print(f"Loading depth: {args.depth_file}")
    print(f"Loading intrinsics from: {args.meta_file}")
    K = load_intrinsics(args.meta_file)
    pointmap = load_pointmap_from_depth(args.depth_file, K)
    K_normalized = normalize_intrinsics(K, height, width)
    print(f"Pointmap shape: {pointmap.shape}")

    # --- Load camera.json for initial pose ---
    print(f"Loading camera data from: {camera_json_path}")
    K_camera, o2c = _load_camera_data(camera_json_path)

    # --- Load Gaussian from PLY ---
    print(f"Loading Gaussian from: {scene_ply_path}")
    gaussian = load_gaussian_from_ply(scene_ply_path, device=device)

    # --- Decompose o2c to get initial pose parameters ---
    print("Decomposing initial pose...")
    quaternion, translation, scale = decompose_o2c_to_pose(o2c, device=device)
    print(f"Initial quaternion: {quaternion.squeeze()}")
    print(f"Initial translation: {translation.squeeze()}")
    print(f"Initial scale: {scale.squeeze()}")

    # --- Prepare inputs for post-optimization ---
    # Convert mask to tensor (H, W) -> for layout_post_optimization_method_GS
    mask_tensor = torch.from_numpy(mask.astype(np.float32)).to(device)

    # Convert image to tensor (3, H, W) for RGB supervision
    rgb_gt = torch.from_numpy(image[..., :3].transpose(2, 0, 1).astype(np.float32) / 255.0).to(device)

    # Pointmap is already a tensor (H, W, 3)
    pointmap_tensor = pointmap.to(device)

    # Intrinsics normalized
    intrinsics_tensor = torch.from_numpy(K_normalized).to(device)

    # --- Run post-optimization ---
    if args.vis:
        # Visualization mode: show current state in rerun
        print("Visualizing in rerun...")
        # Flip pointmap for visualization (reverse pytorch3d coords)
        pointmap_vis = convert_pointmap_to_pytorch3d(pointmap.numpy().copy())
        visualize_in_rerun(
            image, mask, camera_json_path,
            os.path.join(args.demo_out_dir, "scene.glb"),
            pointmap=pointmap_vis
        )
        return

    print("Running post-optimization...")
    result = layout_post_optimization_method_GS(
        gaussian=gaussian,
        quaternion=quaternion,
        translation=translation,
        scale=scale,
        mask=mask_tensor,
        rgb_gt=rgb_gt,
        point_map=pointmap_tensor,
        intrinsics=intrinsics_tensor,
        Enable_occlusion_check=args.enable_occlusion_check,
        Enable_manual_alignment=args.enable_manual_alignment,
        Enable_shape_ICP=args.enable_shape_icp,
        Enable_rendering_optimization=args.enable_rendering_optimization,
        min_size=512,
        device=device,
    )

    # Unpack results
    (
        optimized_quaternion,
        optimized_translation,
        optimized_scale,
        initial_iou,
        final_iou,
        flag_manual,
        flag_icp,
    ) = result

    print(f"\n=== Post-optimization Results ===")
    print(f"Initial IoU: {initial_iou:.4f}")
    print(f"Final IoU: {final_iou:.4f}")
    print(f"Manual alignment applied: {flag_manual}")
    print(f"ICP applied: {flag_icp}")
    print(f"Optimized quaternion: {optimized_quaternion.squeeze()}")
    print(f"Optimized translation: {optimized_translation.squeeze()}")
    print(f"Optimized scale: {optimized_scale.squeeze()}")

    # --- Save optimized results ---
    os.makedirs(args.out_dir, exist_ok=True)

    # Recompute o2c from optimized parameters
    from pytorch3d.transforms import quaternion_to_matrix
    from sam3d_objects.data.dataset.tdfy.transforms_3d import compose_transform

    R_optimized = quaternion_to_matrix(optimized_quaternion.squeeze(1))
    l2c_transform = compose_transform(
        scale=optimized_scale,
        rotation=R_optimized,
        translation=optimized_translation,
    )
    transform_matrix = l2c_transform.get_matrix()[0].cpu().numpy().T
    o2c_optimized = _GL_TO_CV.T @ _R_ZUP_TO_YUP.T @ transform_matrix @ _R_ZUP_TO_YUP

    # Save optimized camera.json
    camera_data_optimized = {
        "K": K.tolist(),
        "blw2cvc": o2c_optimized.tolist(),
        "initial_iou": initial_iou,
        "final_iou": final_iou,
    }
    optimized_camera_path = os.path.join(args.out_dir, "camera_optimized.json")
    with open(optimized_camera_path, "w") as f:
        json.dump(camera_data_optimized, f, indent=2)
    print(f"Saved optimized camera to: {optimized_camera_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-optimization for SAM-3D layout refinement")
    parser.add_argument(
        "--image-path",
        type=str,
        required=True,
        help="Path to the input RGB image.",
    )
    parser.add_argument(
        "--mask-path",
        type=str,
        required=True,
        help="Path to the input mask image.",
    )
    parser.add_argument(
        "--depth-file",
        type=str,
        required=True,
        help="Path to the depth file (PNG encoded).",
    )
    parser.add_argument(
        "--meta-file",
        type=str,
        required=True,
        help="Path to the meta pickle file containing intrinsics (camMat key).",
    )
    parser.add_argument(
        "--demo-out-dir",
        type=str,
        required=True,
        help="Directory containing demo.py outputs (camera.json, scene.ply).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Directory to save optimized outputs.",
    )
    parser.add_argument(
        "--vis",
        action="store_true",
        help="Visualize in rerun instead of running optimization.",
    )
    parser.add_argument(
        "--enable-occlusion-check",
        action="store_true",
        help="Enable occlusion checking.",
    )
    parser.add_argument(
        "--enable-manual-alignment",
        action="store_true",
        help="Enable manual alignment step.",
    )
    parser.add_argument(
        "--enable-shape-icp",
        action="store_true",
        help="Enable shape ICP step.",
    )
    parser.add_argument(
        "--enable-rendering-optimization",
        action="store_true",
        default=True,
        help="Enable rendering optimization step (default: True).",
    )

    args = parser.parse_args()
    main(args)
