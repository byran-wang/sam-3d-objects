eval "$(conda shell.bash hook)"
conda activate sam-3d-objects

python demo.py
LIDRA_SKIP_INIT=1 python demo_scene.py