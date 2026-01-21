# Copyright (c) Meta Platforms, Inc. and affiliates.
import sys
import numpy as np
import torch
from copy import deepcopy
from pytorch3d.transforms import quaternion_to_matrix
from sam3d_objects.data.dataset.tdfy.transforms_3d import compose_transform

_R_ZUP_TO_YUP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
_R_YUP_TO_ZUP = _R_ZUP_TO_YUP.T


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
        vertices = mesh.vertices.astype(np.float32) @ _R_YUP_TO_ZUP
        vertices_tensor = torch.from_numpy(vertices).float().to(output["rotation"].device)
        R_l2c = quaternion_to_matrix(output["rotation"])
        l2c_transform = compose_transform(
            scale=output["scale"],
            rotation=R_l2c,
            translation=output["translation"],
        )
        vertices = l2c_transform.transform_points(vertices_tensor.unsqueeze(0))
        mesh.vertices = vertices.squeeze(0).cpu().numpy() @ _R_ZUP_TO_YUP
        all_meshes.append(mesh)

    if not all_meshes:
        return None

    if len(all_meshes) == 1:
        return all_meshes[0]

    return trimesh.util.concatenate(all_meshes)

# import inference code
sys.path.append("notebook")
from inference import Inference, load_image, load_single_mask, load_masks

# load model
tag = "hf"
config_path = f"checkpoints/{tag}/pipeline.yaml"
inference = Inference(config_path, compile=False)

# load image (RGBA only, mask is embedded in the alpha channel)
image = load_image("notebook/images/shutterstock_stylish_kidsroom_1640806567/image.png")
# mask = load_single_mask("notebook/images/shutterstock_stylish_kidsroom_1640806567", index=14)
masks = load_masks("notebook/images/shutterstock_stylish_kidsroom_1640806567")

# run model
outputs = [inference(image, mask, seed=42) for mask in masks]
scene_mesh = make_scene_untextured_mesh(*outputs)
scene_mesh.export("scene.glb")

print("Your scene reconstruction has been saved to scene.glb")
