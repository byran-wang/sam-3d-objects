# Copyright (c) Meta Platforms, Inc. and affiliates.
import sys
import json
from demo_scene import make_scene_untextured_mesh_without_transform, _GL_TO_CV, _R_ZUP_TO_YUP
# import inference code
sys.path.append("notebook")
from inference import Inference, load_image, load_single_mask, load_mask
import os
import numpy as np
from PIL import Image
import torch
from pytorch3d.transforms import quaternion_to_matrix
from sam3d_objects.data.dataset.tdfy.transforms_3d import compose_transform
from third_party.utils_simba.utils_simba.depth import save_depth, get_depth, depth2xyzmap
from pytorch3d.renderer import MeshRasterizer, RasterizationSettings, PerspectiveCameras
from pytorch3d.structures import Meshes


def visualize_in_rerun(image, mask, camera_json_path, scene_glb_path):
    """Visualize input image, masked image, camera intrinsics, and mesh in rerun."""
    import rerun as rr
    import rerun.blueprint as rrb
    import trimesh

    # Load camera data
    with open(camera_json_path, "r") as f:
        camera_data = json.load(f)
    K = np.array(camera_data["K"], dtype=np.float32)
    o2c = np.array(camera_data["blw2cvc"], dtype=np.float32)

    # Set up blueprint with 3D view and two 2D views side by side
    blueprint = rrb.Horizontal(
        rrb.Vertical(
            rrb.Spatial2DView(name="Camera Image", origin="/image"),
            rrb.Spatial2DView(name="Masked Image", origin="world/camera/masked_image"),
        ),
        rrb.Spatial3DView(name="3D Scene", origin="world"),
        column_shares=[1, 2],
    )

    rr.init("sam3d_demo", spawn=True)
    rr.send_blueprint(blueprint)

    # Get image dimensions
    height, width = image.shape[:2]

    # Extract intrinsics from K matrix
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Log the pinhole camera
    rr.log(
        "world/camera",
        rr.Pinhole(
            focal_length=[fx, fy],
            principal_point=[cx, cy],
            resolution=[width, height],
            camera_xyz=rr.ViewCoordinates.RDF,  # Right-Down-Forward (OpenCV convention)
            image_plane_distance=3.0,
        ),
    )

    # Log the original and masked images under the camera
    rr.log("/image", rr.Image(image[..., :3]))
    mask_bool = mask.astype(bool)
    masked_rgb = image[..., :3].copy()
    masked_rgb[~mask_bool] = 0
    rr.log("world/camera/masked_image", rr.Image(masked_rgb))
    print(f"Camera intrinsics: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")

    # Load and transform mesh
    if os.path.exists(scene_glb_path):
        loaded = trimesh.load(scene_glb_path)
        if isinstance(loaded, trimesh.Scene):
            meshes = list(loaded.geometry.values())
        else:
            meshes = [loaded]

        all_vertices = []
        all_faces = []
        all_colors = []
        vertex_offset = 0

        for mesh in meshes:
            # Apply o2c transform to vertices
            vertices_h = np.hstack([mesh.vertices, np.ones((len(mesh.vertices), 1))])
            transformed_vertices = (o2c @ vertices_h.T).T[:, :3]

            all_vertices.append(transformed_vertices)
            all_faces.append(mesh.faces + vertex_offset)
            vertex_offset += len(mesh.vertices)

            if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
                all_colors.append(mesh.visual.vertex_colors[:, :3])
            else:
                all_colors.append(np.full((len(mesh.vertices), 3), 128, dtype=np.uint8))

        if all_vertices:
            combined_vertices = np.vstack(all_vertices)
            combined_faces = np.vstack(all_faces)
            combined_colors = np.vstack(all_colors)

            rr.log(
                "world/scene_mesh",
                rr.Mesh3D(
                    vertex_positions=combined_vertices,
                    triangle_indices=combined_faces,
                    vertex_colors=combined_colors,
                ),
            )

            # Log point cloud with blue color
            num_points = min(50000, len(combined_vertices))
            indices = np.random.choice(len(combined_vertices), num_points, replace=False)
            sampled_vertices = combined_vertices[indices]
            rr.log(
                "world/scene_pc",
                rr.Points3D(
                    positions=sampled_vertices,
                    colors=np.full((num_points, 3), [0, 100, 255], dtype=np.uint8),
                    radii=0.001,
                ),
            )

    print("Rerun visualization started. Check your browser or rerun viewer.")

def save_rgba_with_mask(image, mask, output_path):
    alpha = mask.astype(np.uint8) * 255
    if image.ndim == 3 and image.shape[2] == 4:
        rgba = image.copy()
        rgba[..., 3] = alpha
    else:
        rgba = np.dstack([image, alpha])
    Image.fromarray(rgba).save(output_path)

import pickle


def load_intrinsics(meta_file):
    """Load camera intrinsics from meta pickle file."""
    with open(meta_file, "rb") as f:
        meta_data = pickle.load(f)
    return np.array(meta_data["camMat"], dtype=np.float32)


def normalize_intrinsics(K, height, width):
    """Normalize intrinsics to [0, 1] range expected by pipeline."""
    K_norm = np.array([
        [K[0, 0] / width, 0, K[0, 2] / width],
        [0, K[1, 1] / height, K[1, 2] / height],
        [0, 0, 1]
    ], dtype=np.float32)
    return K_norm


def load_pointmap_and_intrinsics(depth_file, meta_file):
    """Load pointmap from depth file and intrinsics from meta file.

    Returns:
        pointmap: torch.Tensor of shape (H, W, 3)
        K: original intrinsics matrix (pixel values)
        K_normalized: normalized intrinsics for pipeline
    """
    K = load_intrinsics(meta_file)
    pointmap = load_pointmap_from_depth(depth_file, K)
    height, width = pointmap.shape[:2]
    K_normalized = normalize_intrinsics(K, height, width)
    return pointmap, K, K_normalized


def load_pointmap_from_depth(depth_file, K):
    """Load depth and convert to pointmap using intrinsics K."""
    # Load depth
    depth = get_depth(depth_file)

    # Convert depth to pointmap (H, W, 3)
    pointmap = depth2xyzmap(depth, K)

    # Flip x and y axis to comply with pytorch3d camera coordinate system
    pointmap[..., 0] = -pointmap[..., 0]
    pointmap[..., 1] = -pointmap[..., 1]

    # Convert to torch tensor
    pointmap = torch.from_numpy(pointmap).float()

    return pointmap


def main(args):
    image_path = args.image_path
    mask_path = args.mask_path
    out_dir = args.out_dir
    if not os.path.exists(image_path) or not os.path.exists(mask_path):
        print(f"Image {image_path} or mask {mask_path}  not found.")
        return
    
    image = load_image(image_path)
    # mask = load_single_mask("notebook/images/shutterstock_stylish_kidsroom_1640806567", index=14)
    mask = load_mask(mask_path)
    if args.vis:
        camera_json_path = os.path.join(out_dir, "camera.json")
        scene_glb_path = os.path.join(out_dir, "scene.glb")
        if not os.path.exists(camera_json_path) or not os.path.exists(scene_glb_path):
            print(f"Visualization requires camera.json and scene.glb in {out_dir}.")
            print("Run without --vis first to generate outputs.")
            return
        visualize_in_rerun(image, mask, camera_json_path, scene_glb_path)
        return

    # Load pointmap from depth if provided
    pointmap = None
    K_input = None
    K_input_normalized = None
    if args.depth_file and args.meta_file:
        if not os.path.exists(args.depth_file):
            print(f"Depth file {args.depth_file} not found.")
            return
        if not os.path.exists(args.meta_file):
            print(f"Meta file {args.meta_file} not found.")
            return
        print(f"Loading pointmap from depth: {args.depth_file}")
        print(f"Using intrinsics from: {args.meta_file}")
        pointmap, K_input, K_input_normalized = load_pointmap_and_intrinsics(
            args.depth_file, args.meta_file
        )
        print(f"Pointmap shape: {pointmap.shape}")

    os.makedirs(out_dir, exist_ok=True)
    save_rgba_with_mask(image, mask, os.path.join(out_dir, "input.png"))
    
    # display_image(image, mask, output_path=os.path.join(out_dir, f"{scene}_{index:04d}_inputs.png"))
    # load model
    tag = "hf"
    config_path = f"checkpoints/{tag}/pipeline.yaml"
    inference = Inference(config_path, compile=False)
    outputs = [inference(image, mask, seed=42, pointmap=pointmap, intrinsics=K_input_normalized)]

    # Save o2c transforms to out_dir
    assert len(outputs) == 1, "Only single object inference is supported in demo.py"
    output = outputs[0]
    R_l2c = quaternion_to_matrix(output["rotation"])
    l2c_transform = compose_transform(
        scale=output["scale"],
        rotation=R_l2c,
        translation=output["translation"],
    )
    transform_matrix = l2c_transform.get_matrix()[0].cpu().numpy().T
    o2c = _GL_TO_CV.T @ _R_ZUP_TO_YUP.T @ transform_matrix @ _R_ZUP_TO_YUP

    # Get intrinsics: use input K if provided, otherwise denormalize from pipeline output
    height, width = image.shape[:2]
    if K_input is not None:
        K = K_input
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
    else:
        intrinsics = output["intrinsics"]
        if isinstance(intrinsics, torch.Tensor):
            intrinsics = intrinsics.cpu().numpy()
        intrinsics = intrinsics.squeeze()
        fx = intrinsics[0, 0] * width
        fy = intrinsics[1, 1] * height
        cx = intrinsics[0, 2] * width
        cy = intrinsics[1, 2] * height
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    # Save K and o2c to camera.json
    camera_data = {
        "K": K.tolist(),
        "blw2cvc": o2c.tolist()
    }
    with open(os.path.join(out_dir, "camera.json"), "w") as f:
        json.dump(camera_data, f)
    print(f"Saved camera.json to {out_dir}")

    if 1:
        scene_mesh = make_scene_untextured_mesh_without_transform(*outputs)
        scene_mesh.export(f"{out_dir}/scene.glb")
        print(f"Your reconstruction has been saved to {out_dir}/scene.glb")

        # Render depth with MeshRasterizer
        device = output["rotation"].device

        # Convert trimesh to pytorch3d Meshes
        verts = torch.from_numpy(scene_mesh.vertices.astype(np.float32)).to(device)
        faces = torch.from_numpy(scene_mesh.faces.astype(np.int64)).to(device)

        # Apply o2c transform to vertices
        verts_h = torch.cat([verts, torch.ones(verts.shape[0], 1, device=device)], dim=1)
        o2c_tensor = torch.from_numpy(o2c.astype(np.float32)).to(device)
        verts_cam = (o2c_tensor @ verts_h.T).T[:, :3]

        mesh = Meshes(verts=[verts_cam], faces=[faces])

        # Set up camera (identity since vertices are already in camera space)
        cameras = PerspectiveCameras(
            focal_length=torch.tensor([[fx, fy]], dtype=torch.float32, device=device),
            principal_point=torch.tensor([[cx, cy]], dtype=torch.float32, device=device),
            image_size=torch.tensor([[height, width]], dtype=torch.float32, device=device),
            in_ndc=False,
            device=device
        )

        raster_settings = RasterizationSettings(
            image_size=(height, width),
            blur_radius=0.0,
            faces_per_pixel=1,
        )

        rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
        fragments = rasterizer(mesh)
        depth = fragments.zbuf[0, ..., 0].cpu().numpy()

        # Save depth
        save_depth(depth, os.path.join(out_dir, "depth.png"))
        print(f"Saved depth to {out_dir}/depth.png")

    else:
        # export gaussian splat
        outputs[0]["gs"].save_ply(f"{out_dir}/scene.ply")
        print(f"Your reconstruction has been saved to {out_dir}/scene.ply")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image-path",
        type=str,
        required=True,
        help="Path to the input RGBA image.",
    )
    parser.add_argument(
        "--mask-path",
        type=str,
        required=True,
        help="Path to the input mask image.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Directory to save outputs.",
    )
    parser.add_argument(
        "--vis",
        action="store_true",
        help="Visualize outputs in rerun (requires existing camera.json and scene.glb in out-dir).",
    )
    parser.add_argument(
        "--depth-file",
        type=str,
        default=None,
        help="Path to the depth file (PNG encoded). If provided with --meta-file, uses this instead of depth model.",
    )
    parser.add_argument(
        "--meta-file",
        type=str,
        default=None,
        help="Path to the meta pickle file containing intrinsics (camMat key). Required with --depth-file.",
    )

    args = parser.parse_args()
    main(args)