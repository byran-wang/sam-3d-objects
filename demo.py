# Copyright (c) Meta Platforms, Inc. and affiliates.
import sys
from demo_scene import make_scene_untextured_mesh_without_transform
# import inference code
sys.path.append("notebook")
from inference import Inference, load_image, load_single_mask, load_mask
import os
import numpy as np
from PIL import Image

scenes = {
    "ABF14": 0,
    "BB10": 1266,
    "GPMF10": 0,
    "GSF11": 0,
    "MC1": 0,
    "MC2": 730,
    "MDF11": 2588,
    "SB10": 0,
    "SB14": 0,
    "ShSu10": 611,
    "SM2": 18,
    "SMu1": 1738,
}

out_dir = "output_HO3D"

def save_rgba_with_mask(image, mask, output_path):
    alpha = mask.astype(np.uint8) * 255
    if image.ndim == 3 and image.shape[2] == 4:
        rgba = image.copy()
        rgba[..., 3] = alpha
    else:
        rgba = np.dstack([image, alpha])
    Image.fromarray(rgba).save(output_path)

# load model
tag = "hf"
config_path = f"checkpoints/{tag}/pipeline.yaml"
inference = Inference(config_path, compile=False)

for scene, index in scenes.items():
    # load image (RGBA only, mask is embedded in the alpha channel)
    image_path = f"/home/simba/Documents/dataset/BundleSDF/HO3D_v3/train/{scene}/rgb/{index:04d}.jpg"
    mask_path = f"/home/simba/Documents/dataset/BundleSDF/HO3D_v3/masks_XMem/{scene}/{index:05d}.png"
    if not os.path.exists(image_path) or not os.path.exists(mask_path):
        print(f"Skipping missing scene {scene} index {index}")
        continue
    image = load_image(image_path)
    # mask = load_single_mask("notebook/images/shutterstock_stylish_kidsroom_1640806567", index=14)
    mask = load_mask(mask_path)

    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, f"{scene}_{index:04d}_inputs.png")
    save_rgba_with_mask(image, mask, output_path)
    
    # display_image(image, mask, output_path=os.path.join(out_dir, f"{scene}_{index:04d}_inputs.png"))
    # run model
    outputs = [inference(image, mask, seed=42)]
    if 1:
        scene_mesh = make_scene_untextured_mesh_without_transform(*outputs)
        scene_mesh.export(f"{out_dir}/{scene}_{index:04d}.glb")
        print(f"Your reconstruction has been saved to {out_dir}/{scene}_{index:04d}.glb")

    else:
        # export gaussian splat
        outputs[0]["gs"].save_ply(f"{out_dir}/{scene}_{index:04d}.ply")
        print(f"Your reconstruction has been saved to {out_dir}/{scene}_{index:04d}.ply")