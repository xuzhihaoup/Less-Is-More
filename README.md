# Less Is More

The same interfaces are used across PH2, BraTS2020, and COVID-19-CT with U-Net, CMUNeXt, MK-UNet, DAG-UNet, and
EGE-UNet backbones.

Prepare each dataset according to [docs/DATASETS.md](docs/DATASETS.md) before running an experiment.

## Repository layout

```text
Less-Is-More/
├── docs/DATASETS.md
├── scripts/train_all_models.sh
├── src/less_is_more/
│   ├── models/
│   ├── cli.py
│   ├── config.py
│   ├── data.py
│   ├── engine.py
│   └── search.py
├── genetic_search.py
├── pyproject.toml
├── requirements.txt
└── train.py
```

## Installation

Python 3.9 or newer is required. Create an isolated environment and install a PyTorch build compatible with your
hardware before installing this project. For example:

```bash
conda create -n less-is-more python=3.9 -y
conda activate less-is-more

# Select the appropriate PyTorch command for your platform. The reference
# GPU environment used CUDA 12.8:
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install -e . --no-deps
```

The reference environment used Python 3.9.19, PyTorch 2.8.0+cu128, NumPy 1.26.4, Pillow 10.4.0, nibabel 5.3.3,
TensorBoard 2.19.0, timm 1.0.27, einops 0.8.1, and scikit-image 0.24.0. Exact Python package versions are recorded in
`requirements.txt`. The CUDA build is platform-specific; CPU and other CUDA users should install the matching PyTorch
wheel first.

## Demo

Download the PH2 U-Net checkpoint from [Google Drive](https://drive.google.com/file/d/1sIOiEnBOqpa87-f-8VEqHuxnBkAnfAVj),
then pass its local path to the bundled demo.

Run with automatic device selection:

```bash
python demo_ph2.py --checkpoint /path/to/ph2_unet.pth
```

Run on CPU:

```bash
python demo_ph2.py --checkpoint /path/to/ph2_unet.pth --device cpu
```

Run on CUDA with automatic mixed precision:

```bash
python demo_ph2.py --checkpoint /path/to/ph2_unet.pth --device cuda --amp
```

## Training

Train U-Net on PH2:

```bash
less-is-more-train \
  --model unet \
  --dataset ph2 \
  --dataset-root datasets/PH2 \
  --run-name ph2_unet \
  --epochs 300 \
  --batch-size 8 \
  --amp
```

The same entry point supports all backbones:

```text
unet, cmunext, mk_unet, dag_unet, ege_unet
```

Examples for volumetric datasets:

```bash
# BraTS2020: middle FLAIR slice from each subject
less-is-more-train --model cmunext --dataset brats2020 \
  --dataset-root datasets/BraTS2020 --brats-modality flair --amp

# COVID-19-CT: all axial slices
less-is-more-train --model mk_unet --dataset covid19ct \
  --dataset-root datasets/COVID-19-CT --covid-slice-index -1 --amp
```

Run all five backbones sequentially:

```bash
bash scripts/train_all_models.sh ph2 datasets/PH2
```

Override batch settings through environment variables when using the batch script:

```bash
EPOCHS=300 BATCH_SIZE=16 NUM_WORKERS=8 AMP=true \
  bash scripts/train_all_models.sh brats2020 datasets/BraTS2020
```

Use `less-is-more-train --help` for the complete option list.

The default training schedule uses `300` epochs, including when launched through `scripts/train_all_models.sh`.

## Genetic search

Run a single-image PH2 search:

```bash
less-is-more-search \
  --model unet \
  --checkpoint outputs/ph2_unet/checkpoints/best.pt \
  --dataset ph2 \
  --dataset-root datasets/PH2 \
  --image-name IMD006 \
  --output-dir outputs/search_ph2_unet \
  --population-size 300 \
  --generations 300 \
  --n-segments 128 \
  --amp
```

Search an entire split:

```bash
less-is-more-search \
  --model cmunext \
  --checkpoint outputs/brats2020_cmunext/checkpoints/best.pt \
  --dataset brats2020 \
  --dataset-root datasets/BraTS2020 \
  --all-images \
  --output-dir outputs/search_brats2020_cmunext \
  --skip-existing \
  --amp
```

## Reproducibility notes

- Set the random seed with `--seed`.
- Add `--deterministic` to request deterministic PyTorch behavior. Some GPU operations may still vary by platform.
- SLIC results depend on the scikit-image version; the reference version is pinned to 0.24.0.

## Model attribution

The backbone implementations correspond to U-Net, CMUNeXt, MK-UNet, DAG-UNet, and EGE-UNet. When using a backbone,
please cite both the Less-Is-More paper and the original architecture paper.
