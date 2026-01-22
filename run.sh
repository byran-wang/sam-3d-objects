eval "$(conda shell.bash hook)"
conda activate sam3d-objects

# python demo.py
# LIDRA_SKIP_INIT=1 python demo_scene.py

out_dir=output
LIDRA_SKIP_INIT=1 python demo_scene.py \
    --image notebook/images/shutterstock_stylish_kidsroom_1640806567/image.png \
    --masks-dir notebook/images/shutterstock_stylish_kidsroom_1640806567 \
    --num-masks -1 \
    --output $out_dir/scene.glb \
    --vis \
    --meta_data $out_dir/data.npz
# notebook/demo_multi_object.ipynb