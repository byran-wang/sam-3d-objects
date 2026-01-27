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
from third_party.utils_simba.utils_simba.depth import save_depth
from pytorch3d.renderer import MeshRasterizer, RasterizationSettings, PerspectiveCameras
from pytorch3d.structures import Meshes

def save_rgba_with_mask(image, mask, output_path):
    alpha = mask.astype(np.uint8) * 255
    if image.ndim == 3 and image.shape[2] == 4:
        rgba = image.copy()
        rgba[..., 3] = alpha
    else:
        rgba = np.dstack([image, alpha])
    Image.fromarray(rgba).save(output_path)

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

    os.makedirs(out_dir, exist_ok=True)
    save_rgba_with_mask(image, mask, os.path.join(out_dir, "input.png"))
    
    # display_image(image, mask, output_path=os.path.join(out_dir, f"{scene}_{index:04d}_inputs.png"))
    # load model
    tag = "hf"
    config_path = f"checkpoints/{tag}/pipeline.yaml"
    inference = Inference(config_path, compile=False)
    outputs = [inference(image, mask, seed=42)]

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

    # Get intrinsics and denormalize
    intrinsics = output["intrinsics"]
    if isinstance(intrinsics, torch.Tensor):
        intrinsics = intrinsics.cpu().numpy()
    intrinsics = intrinsics.squeeze()
    height, width = image.shape[:2]
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
    args = parser.parse_args()
    main(args)