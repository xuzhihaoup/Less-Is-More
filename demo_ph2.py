"""Run a compact genetic-search demo on the bundled PH2 sample."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPOSITORY_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from less_is_more.engine import outputs_to_probabilities, resolve_device
from less_is_more.search import (
    load_checkpoint_model,
    load_ph2_sample,
    main as search_main,
    parse_args as parse_search_args,
)


DEMO_IMAGE_NAME = "demo"

DECISION_OUTPUTS = (
    ("dominance", "dominance"),
    ("consensus", "consensus"),
    ("frequency", "frequency"),
    ("best_individual", "best_gene"),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"Search SLIC explanations for the bundled PH2 {DEMO_IMAGE_NAME} sample.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to ph2_unet.pth.")
    parser.add_argument("--output-dir", type=Path, default=None)
    # Small defaults keep the demo quick; users can adjust them for a larger search.
    parser.add_argument("--population-size", type=int, default=32, help="Genetic population size.")
    parser.add_argument("--generations", type=int, default=32, help="Maximum search generations.")
    parser.add_argument("--eval-batch-size", type=int, default=16, help="Inference batch size.")
    parser.add_argument("--n-segments", type=int, default=128, help="Number of SLIC segments.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--amp", action="store_true", help="Enable CUDA automatic mixed precision.")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def load_gene(path: Path, expected_length: int) -> np.ndarray:
    with path.open("rb") as file:
        gene = np.asarray(pickle.load(file), dtype=np.uint8).squeeze()
    if gene.ndim != 1 or len(gene) < expected_length:
        raise ValueError(f"Invalid gene shape {gene.shape} in {path}")
    return gene[:expected_length]


def binary_iou(target: np.ndarray, prediction: np.ndarray) -> float:
    target_bool = target.astype(bool)
    prediction_bool = prediction.astype(bool)
    union = np.logical_or(target_bool, prediction_bool).sum(dtype=np.float64)
    if union == 0:
        return 1.0
    intersection = np.logical_and(target_bool, prediction_bool).sum(dtype=np.float64)
    return float(intersection / union)


def comparison_image(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    comparison = np.zeros((*target.shape, 3), dtype=np.uint8)
    comparison[(target == 0) & (prediction == 1)] = (255, 0, 0)
    comparison[(target == 1) & (prediction == 0)] = (0, 0, 255)
    comparison[(target == 1) & (prediction == 1)] = (0, 255, 0)
    return comparison


def save_rgb(image: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image_uint8 = np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(image_uint8).save(path)


def render_visualizations(output_dir: Path, search_arguments: list[str]) -> None:
    args = parse_search_args(search_arguments)
    device = resolve_device(args.device)
    args.amp = args.amp and device.type == "cuda"
    model = load_checkpoint_model(args, device)
    sample = load_ph2_sample(args, DEMO_IMAGE_NAME)

    sample_dir = output_dir / sample.name
    metadata_path = sample_dir / f"{sample.name}_best_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    visualization_dir = output_dir / "visualizations"
    original_dir = visualization_dir / "original"
    explanation_dir = visualization_dir / "explanations"
    segmentation_dir = visualization_dir / "segmentations"
    original_dir.mkdir(parents=True, exist_ok=True)
    explanation_dir.mkdir(parents=True, exist_ok=True)
    segmentation_dir.mkdir(parents=True, exist_ok=True)

    Image.open(sample.image_path).convert("RGB").save(original_dir / f"{sample.name}_original.png")
    Image.fromarray((sample.mask * 255).astype(np.uint8)).save(
        original_dir / f"{sample.name}_ground_truth.png"
    )

    labels = ["original"]
    model_inputs = [sample.image]
    for decision_name, output_name in DECISION_OUTPUTS:
        gene_path = Path(metadata["paths"][decision_name])
        gene = load_gene(gene_path, sample.gene_length)
        keep_mask = gene[sample.segments].astype(np.float32)
        masked_image = sample.image * keep_mask[:, :, None]
        save_rgb(masked_image, explanation_dir / f"{output_name}.png")
        labels.append(output_name)
        model_inputs.append(masked_image)

    images = torch.from_numpy(np.stack(model_inputs)).permute(0, 3, 1, 2).to(
        device=device,
        dtype=torch.float32,
    )
    autocast = torch.autocast(device_type="cuda", dtype=torch.float16) if args.amp else nullcontext()
    with torch.inference_mode(), autocast:
        probabilities = outputs_to_probabilities(
            args.model,
            model(images),
            target_size=sample.mask.shape,
        )
    predictions = (probabilities >= args.threshold).squeeze(1).cpu().numpy().astype(np.uint8)

    metrics = {}
    for label, prediction in zip(labels, predictions):
        iou = binary_iou(sample.mask, prediction)
        suffix = f"{int(round(iou * 10000)):04d}"
        base_name = f"{label}_{suffix}"
        Image.fromarray(prediction * 255).save(segmentation_dir / f"{base_name}_pre.png")
        Image.fromarray(comparison_image(sample.mask, prediction)).save(
            segmentation_dir / f"{base_name}_con.png"
        )
        metrics[label] = {"iou": iou, "iou_suffix": suffix}

    metrics_path = visualization_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Visualizations: {visualization_dir}")


def main() -> None:
    args = build_parser().parse_args()
    dataset_root = REPOSITORY_ROOT / "examples" / "ph2_demo"
    output_dir = args.output_dir or REPOSITORY_ROOT / "outputs" / (
        f"ph2_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    search_arguments = [
        "--model",
        "unet",
        "--checkpoint",
        str(args.checkpoint.expanduser().resolve()),
        "--dataset",
        "ph2",
        "--dataset-root",
        str(dataset_root),
        "--image-name",
        DEMO_IMAGE_NAME,
        "--output-dir",
        str(output_dir),
        "--population-size",
        str(args.population_size),
        "--generations",
        str(args.generations),
        "--eval-batch-size",
        str(args.eval_batch_size),
        "--n-segments",
        str(args.n_segments),
        "--device",
        args.device,
        "--seed",
        str(args.seed),
        "--no-tensorboard",
    ]
    if args.amp:
        search_arguments.append("--amp")

    search_main(search_arguments)
    render_visualizations(output_dir, search_arguments)
    print(f"Demo results: {output_dir}")


if __name__ == "__main__":
    main()
