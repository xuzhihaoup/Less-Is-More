# Less Is More

Official implementation of the Less-Is-More framework for medical image segmentation and decision-aware genetic
search over SLIC superpixels. The same interfaces are used across PH2, BraTS2020, and COVID-19-CT with U-Net,
CMUNeXt, MK-UNet, DAG-UNet, and EGE-UNet backbones.

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

## Training

Train U-Net on PH2:

```bash
less-is-more-train \
  --model unet \
  --dataset ph2 \
  --dataset-root datasets/PH2 \
  --run-name ph2_unet \
  --epochs 320 \
  --batch-size 8 \
  --amp
```

The same entry point supports all released backbones:

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
EPOCHS=320 BATCH_SIZE=16 NUM_WORKERS=8 AMP=true \
  bash scripts/train_all_models.sh brats2020 datasets/BraTS2020
```

Use `less-is-more-train --help` for the complete option list.

## Genetic search

The search represents an individual as a binary vector over SLIC regions: `1` retains a superpixel and `0` masks it.
Each individual is evaluated by running the fixed segmentation model on the masked image. The all-ones individual
provides the unmasked baseline.

Four decisions are evaluated in every generation:

- **Best individual:** highest-IoU gene in the population
- **Dominance:** strict per-position majority vote
- **Consensus:** vote thresholds selected from configurable consensus ratios
- **Frequency:** high-frequency regions selected from configurable retained fractions

Consensus and Frequency select the sparsest candidate satisfying `baseline IoU - decision_iou_delta` when feasible.
The decision-aware population score combines the four decision scores, their minimum IoU, an explanation sparsity
term, and an IoU-shortfall penalty. It selects the population saved for downstream decisions; individual selection,
elitism, and crossover continue to use individual fitness. Selected decision genes can be injected into the next
generation as elites.

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
  --generations 180 \
  --n-segments 128 \
  --decision-iou-delta 0.15 \
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

Use `--image-list-file PATH` for a custom subset. Candidate sets can be changed with `--consensus-ratios` and
`--frequency-fractions`, using comma-separated values. Use `less-is-more-search --help` for all options.

## Outputs

Each run is self-contained under `outputs/<run-name>/`:

```text
outputs/<run-name>/
├── checkpoints/
│   ├── best.pt
│   └── last.pt
├── tensorboard/
├── config.json
└── history.csv
```

`best.pt` is selected by validation mIoU. Resume an interrupted run with
`--resume outputs/<run-name>/checkpoints/last.pt`, or initialize model weights only with `--init-checkpoint PATH`.

Inspect curves with:

```bash
tensorboard --logdir outputs
```

Each searched sample receives its own directory containing the selected population, Best Individual, Dominance,
Consensus, and Frequency genes as pickle files. The IoU is encoded in each filename at four-decimal precision. The
metadata JSON records the baseline, decision IoUs, keep ratios, selected Consensus/Frequency thresholds, generation,
and file paths. Run-level parameters and metrics are written to `search_config.json`, `summary.jsonl`, and TensorBoard.

## Reproducibility notes

- The default random seed is `42`; change it with `--seed`.
- Add `--deterministic` to request deterministic PyTorch behavior. Some GPU operations may still vary by platform.
- Automatic BCE positive-class weighting is estimated from training masks. Pass `--pos-weight VALUE` to override it.
- Dataset versions, preprocessing, hardware, and PyTorch/CUDA builds should be reported alongside experimental results.
- SLIC results depend on the scikit-image version; the reference version is pinned to 0.24.0.

## Model attribution

The backbone implementations correspond to U-Net, CMUNeXt, MK-UNet, DAG-UNet, and EGE-UNet. When using a backbone,
please cite both the Less-Is-More paper and the original architecture paper. The implementations are kept structurally
compatible with those used for the reported experiments.
