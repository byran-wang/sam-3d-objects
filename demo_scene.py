# Copyright (c) Meta Platforms, Inc. and affiliates.
import os
import argparse
import sys
import numpy as np
import torch
from copy import deepcopy
from pytorch3d.transforms import quaternion_to_matrix
from sam3d_objects.data.dataset.tdfy.transforms_3d import compose_transform
# import inference code
sys.path.append("notebook")

_R_ZUP_TO_YUP = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.float32)
_R_YUP_TO_ZUP = _R_ZUP_TO_YUP.T
_GL_TO_CV = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.float32)


def visualize_in_rerun(image, meta_data, output_glb_path=None):
    """Visualize image, camera, poses and meshes in rerun."""
    import rerun as rr
    import rerun.blueprint as rrb
    import trimesh

    # Set up blueprint with 3D view and 2D image view side by side
    blueprint = rrb.Horizontal(
        rrb.Spatial3DView(name="3D Scene", origin="world"),
        rrb.Spatial2DView(name="Camera Image", origin="world/camera"),
        column_shares=[2, 1],
    )

    rr.init("sam3d_scene", spawn=True)
    rr.send_blueprint(blueprint)

    # Get image dimensions for denormalizing intrinsics
    height, width = image.shape[:2]

    # Log camera with intrinsics (use first output's intrinsics as they should be the same)
    if meta_data and "intrinsics" in meta_data[0]:
        intrinsics = meta_data[0]["intrinsics"]
        if isinstance(intrinsics, torch.Tensor):
            intrinsics = intrinsics.cpu().numpy()
        intrinsics = intrinsics.squeeze()

        # Denormalize intrinsics (normalized focal length -> pixel focal length)
        fx = intrinsics[0, 0] * width
        fy = intrinsics[1, 1] * height
        cx = intrinsics[0, 2] * width
        cy = intrinsics[1, 2] * height

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

        # Log the image under the camera
        rr.log("world/camera/image", rr.Image(image[..., :3]))

        print(f"Camera intrinsics: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
    else:
        # Fallback: just log the image without camera model
        rr.log("world/image", rr.Image(image[..., :3]))

    # Get meshes - either from metadata or from GLB file
    meshes = []
    for obj_meta in meta_data:
        mesh = obj_meta.get("glb")
        if mesh is not None:
            meshes.append(mesh)

    # If no meshes in metadata, load from GLB file
    if not meshes and output_glb_path and os.path.exists(output_glb_path):
        loaded = trimesh.load(output_glb_path)
        if isinstance(loaded, trimesh.Scene):
            meshes = list(loaded.geometry.values())
        else:
            meshes = [loaded]

    # Transform each mesh with its corresponding metadata
    all_vertices = []
    all_faces = []
    all_colors = []
    vertex_offset = 0

    for i, mesh in enumerate(meshes):
        if i >= len(meta_data):
            break

        obj_meta = meta_data[i]
        R_l2c = quaternion_to_matrix(obj_meta["rotation"])
        l2c_transform = compose_transform(
            scale=obj_meta["scale"],
            rotation=R_l2c,
            translation=obj_meta["translation"],
        )

        # Get the 4x4 transformation matrix and apply from left
        transform_matrix = l2c_transform.get_matrix()[0].cpu().numpy().T
        # Add homogeneous coordinate
        vertices_h = np.hstack([mesh.vertices, np.ones((len(mesh.vertices), 1))])
        o2c = _GL_TO_CV.T @ _R_ZUP_TO_YUP.T @ transform_matrix @ _R_ZUP_TO_YUP
        transformed_vertices = (o2c @ vertices_h.T).T[:, :3]

        # Collect vertices, faces (with offset), and colors
        all_vertices.append(transformed_vertices)
        all_faces.append(mesh.faces + vertex_offset)
        vertex_offset += len(mesh.vertices)

        if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
            all_colors.append(mesh.visual.vertex_colors[:, :3])
        else:
            all_colors.append(np.full((len(mesh.vertices), 3), 128, dtype=np.uint8))

    if all_vertices:
        # Combine all meshes
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

        # Log point cloud with blue color (random sample for performance)
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

def make_scene_untextured_mesh(*outputs, in_place=False):
    import trimesh

    if not in_place:
        outputs = [deepcopy(output) for output in outputs]

    all_meshes = []
    for output in outputs:
        mesh = output["glb"]
        if mesh is None:
            continue
        # GLB is Y-up, transforms are Z-up; convert, apply, convert back
        R_l2c = quaternion_to_matrix(output["rotation"])
        l2c_transform = compose_transform(
            scale=output["scale"],
            rotation=R_l2c,
            translation=output["translation"],
        )
        # Get the 4x4 transformation matrix and apply from left
        transform_matrix = l2c_transform.get_matrix()[0].cpu().numpy().T  # (4, 4), transpose for left multiply
        # Add homogeneous coordinate
        vertices_h = np.hstack([mesh.vertices, np.ones((len(mesh.vertices), 1))])  # (N, 4)
        o2c = _GL_TO_CV.T @ _R_ZUP_TO_YUP.T @ transform_matrix @ _R_ZUP_TO_YUP
        mesh.vertices = (o2c @ vertices_h.T).T[:, :3]
        all_meshes.append(mesh)

    if not all_meshes:
        return None

    if len(all_meshes) == 1:
        return all_meshes[0]

    return trimesh.util.concatenate(all_meshes)

def make_scene_untextured_mesh_without_transform(*outputs, in_place=False):
    import trimesh

    if not in_place:
        outputs = [deepcopy(output) for output in outputs]

    all_meshes = []
    for output in outputs:
        mesh = output["glb"]
        if mesh is None:
            continue

        # # GLB is Y-up, transforms are Z-up; convert, apply, convert back
        # vertices = mesh.vertices.astype(np.float32) @ _R_YUP_TO_ZUP
        # vertices_tensor = torch.from_numpy(vertices).float().to(output["rotation"].device)
        # R_l2c = quaternion_to_matrix(output["rotation"])
        # l2c_transform = compose_transform(
        #     scale=output["scale"],
        #     rotation=R_l2c,
        #     translation=output["translation"],
        # )
        # vertices = l2c_transform.transform_points(vertices_tensor.unsqueeze(0))
        # mesh.vertices = (vertices.squeeze(0).cpu().numpy() @ _R_ZUP_TO_YUP) @ _GL_TO_CV
        all_meshes.append(mesh)

    if not all_meshes:
        return None

    if len(all_meshes) == 1:
        return all_meshes[0]

    return trimesh.util.concatenate(all_meshes)

def load_outputs_from_npz(npz_path):
    """Load outputs from saved npz file, reconstructing the list of dicts."""
    data = np.load(npz_path)

    # Find number of outputs by looking at keys
    indices = set()
    for key in data.keys():
        idx = int(key.split("_")[0])
        indices.add(idx)

    outputs = []
    for i in sorted(indices):
        output = {}
        for key in data.keys():
            if key.startswith(f"{i}_"):
                field_name = key[len(f"{i}_"):]
                output[field_name] = torch.from_numpy(data[key])
        outputs.append(output)

    return outputs


def parse_args():
    parser = argparse.ArgumentParser(description="Generate 3D scene mesh from image and masks")
    parser.add_argument("--image", type=str, required=True, help="Path to input image (RGBA)")
    parser.add_argument("--masks-dir", type=str, required=True, help="Directory containing masks")
    parser.add_argument("--output_scene", type=str, default="scene.glb", help="Output GLB file path")
    parser.add_argument("--config", type=str, default="checkpoints/hf/pipeline.yaml", help="Path to pipeline config")
    parser.add_argument("--num-masks", type=int, default=None, help="Number of masks to process (default: all)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--compile", action="store_true", help="Enable model compilation")
    parser.add_argument("--meta_data", type=str, default=None, help="Path to save outputs as .npz file")
    parser.add_argument("--vis", action="store_true", help="Visualize outputs in rerun (requires --save-outputs)")
    return parser.parse_args()


def main():
    args = parse_args()
    from inference import load_image
    # load image
    image = load_image(args.image)

    # visualize mode: load from saved outputs if available
    if args.vis and args.meta_data and os.path.exists(args.meta_data):
        meta_data = load_outputs_from_npz(args.meta_data)
        visualize_in_rerun(image, meta_data, args.output_scene.replace(".glb", "_without_transform.glb"))
        return


    from inference import Inference, load_masks, display_image

    # load masks
    masks = load_masks(args.masks_dir)
    if args.num_masks is not None:
        masks = masks[: args.num_masks]
    
    os.makedirs(os.path.dirname(args.output_scene) or ".", exist_ok=True)
    display_image(image, masks, output_path=os.path.join(os.path.dirname(args.output_scene), "inputs.png"))
    # load model
    inference = Inference(args.config, compile=args.compile)

    # run model
    outputs = [inference(image, mask, seed=args.seed) for mask in masks]

    # save outputs to numpy file
    if args.meta_data:
        save_data = {}
        for i, output in enumerate(outputs):
            for key, value in output.items():
                if isinstance(value, torch.Tensor):
                    save_data[f"{i}_{key}"] = value.cpu().numpy()
                elif isinstance(value, np.ndarray):
                    save_data[f"{i}_{key}"] = value
        os.makedirs(os.path.dirname(args.meta_data) or ".", exist_ok=True)
        np.savez(args.meta_data, **save_data)
        print(f"Output metadata to {args.meta_data}")

    scene_mesh = make_scene_untextured_mesh(*outputs)
    scene_mesh.export(args.output_scene)
    scene_mesh_without_transform = make_scene_untextured_mesh_without_transform(*outputs)
    scene_mesh_without_transform.export(args.output_scene.replace(".glb", "_without_transform.glb"))    

    print(f"Your scene reconstruction has been saved to {args.output_scene}")

    # visualize after inference if requested
    if args.vis:
        visualize_in_rerun(image, outputs, args.output_scene.replace(".glb", "_without_transform.glb"))


if __name__ == "__main__":
    main()
