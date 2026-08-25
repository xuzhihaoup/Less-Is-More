"""Decision-aware genetic search over SLIC superpixels."""

from __future__ import annotations

import argparse
import json
import pickle
import random
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from skimage.segmentation import slic
from skimage.util import img_as_float
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from .config import DATASET_NAMES, MODEL_NAMES
from .data import BraTSSliceDataset, COVID19CTSliceDataset
from .engine import outputs_to_probabilities, resolve_device
from .models import build_model

EPS = 1e-7
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
DEFAULT_CONSENSUS_RATIOS = (0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.97)
DEFAULT_FREQUENCY_FRACTIONS = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)


@dataclass(frozen=True)
class SearchSample:
    name: str
    image_path: Path
    mask_path: Path
    image: np.ndarray
    mask: np.ndarray
    segments: np.ndarray
    gene_length: int


def parse_float_list(value: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected comma-separated floating-point values.") from error
    if not values:
        raise argparse.ArgumentTypeError("At least one value is required.")
    return values


def read_names(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Image list not found: {path}")
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not names:
        raise RuntimeError(f"Image list is empty: {path}")
    return names


def resolve_image(directory: Path, name: str) -> Path:
    direct = directory / name
    if direct.suffix and direct.exists():
        return direct
    for extension in IMAGE_EXTENSIONS:
        candidate = directory / f"{name}{extension}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Image `{name}` not found in {directory}")


def resolve_brats_root(dataset_root: Path) -> Path:
    candidates = (
        dataset_root,
        dataset_root / "BraTS2020_TrainingData" / "MICCAI_BraTS2020_TrainingData",
        dataset_root / "MICCAI_BraTS2020_TrainingData",
    )
    for candidate in candidates:
        if candidate.exists() and any(candidate.glob("BraTS20_Training_*")):
            return candidate
    raise FileNotFoundError(f"Could not find BraTS2020 subject directories under {dataset_root}")


def resolve_image_names(args: argparse.Namespace) -> list[str]:
    if args.image_name is not None:
        return [args.image_name]
    if args.image_list_file is not None:
        return read_names(args.image_list_file)

    if args.dataset == "ph2":
        return read_names(args.dataset_root / "ImageSets" / "test.txt")
    if args.dataset == "covid19ct":
        return read_names(args.dataset_root / "ImageSets" / f"{args.covid_split}.txt")

    root = resolve_brats_root(args.dataset_root)
    names = []
    for subject_dir in sorted(root.glob("BraTS20_Training_*")):
        if not subject_dir.is_dir():
            continue
        try:
            BraTSSliceDataset._subject_file(subject_dir, args.brats_modality)
            BraTSSliceDataset._subject_file(subject_dir, "seg")
        except FileNotFoundError:
            continue
        names.append(subject_dir.name)
    if not names:
        raise RuntimeError(f"No labeled BraTS2020 subjects found under {root}")
    return names


def _prepare_segments(image: np.ndarray, n_segments: int, compactness: float) -> np.ndarray:
    slic_input = image[:, :, 0] if image.shape[2] == 1 else image
    channel_axis = None if slic_input.ndim == 2 else -1
    segments = slic(
        img_as_float(slic_input),
        n_segments=n_segments,
        compactness=compactness,
        start_label=0,
        channel_axis=channel_axis,
    )
    return np.asarray(segments, dtype=np.int64)


def _make_sample(
    name: str,
    image_path: Path,
    mask_path: Path,
    image_tensor: torch.Tensor,
    mask_tensor: torch.Tensor,
    n_segments: int,
    compactness: float,
) -> SearchSample:
    image = image_tensor.permute(1, 2, 0).numpy().astype(np.float32)
    mask = (mask_tensor.numpy() > 0).astype(np.uint8)
    if image.shape[:2] != mask.shape:
        raise ValueError(f"Image/mask shape mismatch for {name}: {image.shape[:2]} vs {mask.shape}")
    segments = _prepare_segments(image, n_segments, compactness)
    return SearchSample(
        name=name,
        image_path=image_path,
        mask_path=mask_path,
        image=image,
        mask=mask,
        segments=segments,
        gene_length=int(segments.max()) + 1,
    )


def load_ph2_sample(args: argparse.Namespace, name: str) -> SearchSample:
    image_path = resolve_image(args.dataset_root / "Images" / "test", name)
    mask_path = resolve_image(args.dataset_root / "ProcessedSegmentationClass" / "test", name)
    image_mode = "L" if args.input_channels == 1 else "RGB"
    image = np.asarray(Image.open(image_path).convert(image_mode), dtype=np.float32) / 255.0
    if image.ndim == 2:
        image = image[:, :, None]
    mask = (np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8) > 0).astype(np.int64)
    image_tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))
    return _make_sample(
        name,
        image_path,
        mask_path,
        image_tensor,
        torch.from_numpy(mask),
        args.n_segments,
        args.slic_compactness,
    )


def load_brats_sample(args: argparse.Namespace, name: str) -> SearchSample:
    subject_dir = resolve_brats_root(args.dataset_root) / name
    if not subject_dir.is_dir():
        raise FileNotFoundError(f"BraTS2020 subject not found: {subject_dir}")
    dataset = BraTSSliceDataset(
        [subject_dir],
        modality=args.brats_modality,
        slice_index=args.brats_slice_index,
        image_size=args.brats_image_size,
        input_channels=args.input_channels,
        cache=True,
    )
    image_tensor, mask_tensor = dataset[0]
    return _make_sample(
        name,
        dataset._subject_file(subject_dir, args.brats_modality),
        dataset._subject_file(subject_dir, "seg"),
        image_tensor,
        mask_tensor,
        args.n_segments,
        args.slic_compactness,
    )


def build_covid_dataset(args: argparse.Namespace) -> COVID19CTSliceDataset:
    root = args.dataset_root
    split_file = args.image_list_file or root / "ImageSets" / f"{args.covid_split}.txt"
    return COVID19CTSliceDataset(
        image_dir=root / args.covid_split / "images",
        mask_dir=root / args.covid_split / "masks",
        split_file=split_file,
        slice_index=args.covid_slice_index,
        time_index=args.covid_time_index,
        image_size=args.covid_image_size,
        input_channels=args.input_channels,
    )


def load_covid_sample(args: argparse.Namespace, dataset: COVID19CTSliceDataset, name: str) -> SearchSample:
    if name not in dataset.image_paths:
        raise FileNotFoundError(f"COVID-19-CT sample `{name}` is not present in the selected split.")
    image_tensor, mask_tensor = dataset._load_sample(name, args.covid_slice_index)
    return _make_sample(
        name,
        dataset.image_paths[name],
        dataset.mask_paths[name],
        image_tensor,
        mask_tensor,
        args.n_segments,
        args.slic_compactness,
    )


def load_checkpoint_model(args: argparse.Namespace, device: torch.device) -> nn.Module:
    model = build_model(
        model_name=args.model,
        input_channels=args.input_channels,
        dag_threshold=args.dag_threshold,
        dag_fraction=args.dag_fraction,
    ).to(device)
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(args.checkpoint, map_location=device)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint format: {args.checkpoint}")
    checkpoint_model = checkpoint.get("model_name")
    if checkpoint_model is not None and checkpoint_model != args.model:
        raise ValueError(f"Checkpoint model is {checkpoint_model}, but --model is {args.model}.")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if any(key.startswith("module.") for key in state_dict):
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
    target = model.state_dict()
    matched = {key: value for key, value in state_dict.items() if key in target and target[key].shape == value.shape}
    if len(matched) < max(1, len(target) // 2):
        raise RuntimeError(f"Only {len(matched)}/{len(target)} checkpoint tensors matched {args.model}.")
    target.update(matched)
    model.load_state_dict(target)
    model.eval()
    print(f"Loaded {args.model}: {len(matched)}/{len(target)} tensors from {args.checkpoint}")
    return model


def tournament_selection(population: np.ndarray, fitness: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    population_size = len(population)
    size = min(max(size, 1), population_size)
    selected_indices = []
    for _ in range(population_size):
        candidates = np.random.choice(population_size, size, replace=False)
        selected_indices.append(int(candidates[np.argmin(fitness[candidates])]))
    indices = np.asarray(selected_indices)
    return population[indices].copy(), fitness[indices].copy()


def crossover_population(
    population: np.ndarray,
    fitness: np.ndarray,
    probability: float,
    elite_count: int,
) -> np.ndarray:
    population_size = len(population)
    elite_count = min(max(elite_count, 0), population_size)
    order = np.argsort(fitness)
    elites = population[order[:elite_count]].copy()
    parents = population[order[elite_count:]].copy()
    np.random.shuffle(parents)
    children = []
    for index in range(0, len(parents), 2):
        first = parents[index]
        second = parents[(index + 1) % len(parents)]
        if len(first) > 1 and np.random.random() < probability:
            point = np.random.randint(1, len(first))
            children.extend(
                (
                    np.concatenate((first[:point], second[point:])),
                    np.concatenate((second[:point], first[point:])),
                )
            )
        else:
            children.extend((first.copy(), second.copy()))
    parts = [part for part in (elites, np.asarray(children, dtype=np.uint8)) if len(part)]
    offspring = np.vstack(parts) if parts else population.copy()
    if len(offspring) < population_size:
        fill = np.random.choice(population_size, population_size - len(offspring), replace=True)
        offspring = np.vstack((offspring, population[fill]))
    return offspring[:population_size].astype(np.uint8)


def mutate_population(population: np.ndarray, probability: float) -> np.ndarray:
    mutated = population.copy()
    mutation_mask = np.random.random(mutated.shape) < probability
    mutated[mutation_mask] = 1 - mutated[mutation_mask]
    return mutated.astype(np.uint8)


def initial_population(gene_length: int, population_size: int, include_all_ones: bool) -> np.ndarray:
    population = np.random.randint(2, size=(population_size, gene_length), dtype=np.uint8)
    if include_all_ones:
        population[0] = 1
    return population


def majority_vote(population: np.ndarray) -> np.ndarray:
    return (population.sum(axis=0) > len(population) // 2).astype(np.uint8)


def consensus_vote(population: np.ndarray, ratio: float) -> np.ndarray:
    return (population.sum(axis=0) >= int(np.ceil(ratio * len(population)))).astype(np.uint8)


def frequency_vote(population: np.ndarray, fraction: float) -> np.ndarray:
    votes = population.sum(axis=0)
    threshold = np.percentile(votes, 100.0 * (1.0 - fraction))
    return (votes >= threshold).astype(np.uint8)


def keep_ratio(gene: np.ndarray) -> float:
    return float(np.mean(gene, dtype=np.float64)) if gene.size else 0.0


def mask_population(sample: SearchSample, population: np.ndarray) -> np.ndarray:
    keep_masks = population[:, sample.segments].astype(np.float32)
    return sample.image[None, :, :, :] * keep_masks[:, :, :, None]


def iou_scores(target: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    target_mask = target.astype(bool)
    predicted_masks = predictions.astype(bool)
    intersection = np.logical_and(predicted_masks, target_mask[None]).sum(axis=(1, 2), dtype=np.float64)
    predicted_area = predicted_masks.sum(axis=(1, 2), dtype=np.float64)
    target_area = float(target_mask.sum())
    union = predicted_area + target_area - intersection
    scores = np.ones(len(predictions), dtype=np.float64)
    nonempty = union > 0
    scores[nonempty] = intersection[nonempty] / union[nonempty]
    return scores.astype(np.float32)


def evaluate_genes_iou(
    genes: np.ndarray,
    sample: SearchSample,
    model: nn.Module,
    args: argparse.Namespace,
    device: torch.device,
) -> np.ndarray:
    genes = np.asarray(genes, dtype=np.uint8)
    if genes.ndim == 1:
        genes = genes[None]
    scores = np.empty(len(genes), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(genes), args.eval_batch_size):
            end = min(start + args.eval_batch_size, len(genes))
            masked = mask_population(sample, genes[start:end])
            images = torch.from_numpy(masked).permute(0, 3, 1, 2).to(device, dtype=torch.float32)
            autocast = torch.autocast(device_type="cuda", dtype=torch.float16) if args.amp else nullcontext()
            with autocast:
                probabilities = outputs_to_probabilities(
                    args.model,
                    model(images),
                    target_size=sample.mask.shape,
                )
            predictions = (probabilities >= args.threshold).squeeze(1).cpu().numpy()
            scores[start:end] = iou_scores(sample.mask, predictions)
    return scores


def evaluate_population_fitness(
    population: np.ndarray,
    sample: SearchSample,
    model: nn.Module,
    args: argparse.Namespace,
    device: torch.device,
) -> np.ndarray:
    """Compute Eq. (7) with segmentation quality F defined as IoU in Eq. (8)."""
    population_ious = evaluate_genes_iou(population, sample, model, args, device)
    return (1.0 - population_ious).astype(np.float32)


def sparse_score(iou: float, ratio: float, minimum_iou: float, args: argparse.Namespace) -> float:
    shortfall = max(0.0, minimum_iou - iou)
    return float(iou - args.sparsity_weight * ratio - args.penalty_weight * shortfall)


def quality_score(iou: float, minimum_iou: float, args: argparse.Namespace) -> float:
    return float(iou - args.penalty_weight * max(0.0, minimum_iou - iou))


def select_sparse_candidate(candidates: list[dict[str, Any]], minimum_iou: float) -> dict[str, Any]:
    feasible = [candidate for candidate in candidates if candidate["iou"] >= minimum_iou]
    if feasible:
        return min(feasible, key=lambda item: (item["keep_ratio"], -item["iou"]))
    return max(candidates, key=lambda item: (item["iou"], -item["keep_ratio"]))


def evaluate_decisions(
    population: np.ndarray,
    population_ious: np.ndarray,
    baseline_iou: float,
    sample: SearchSample,
    model: nn.Module,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    best_index = int(np.argmax(population_ious))
    best_gene = population[best_index].copy()
    best_iou = float(population_ious[best_index])
    majority_gene = majority_vote(population)
    consensus_candidates = [
        {"value": ratio, "gene": consensus_vote(population, ratio)} for ratio in args.consensus_ratios
    ]
    frequency_candidates = [
        {"value": fraction, "gene": frequency_vote(population, fraction)}
        for fraction in args.frequency_fractions
    ]
    genes = [majority_gene]
    genes.extend(candidate["gene"] for candidate in consensus_candidates)
    genes.extend(candidate["gene"] for candidate in frequency_candidates)
    scores = evaluate_genes_iou(np.asarray(genes), sample, model, args, device)

    majority_iou = float(scores[0])
    cursor = 1
    for candidate in consensus_candidates + frequency_candidates:
        candidate["iou"] = float(scores[cursor])
        candidate["keep_ratio"] = keep_ratio(candidate["gene"])
        cursor += 1

    minimum_iou = max(0.0, baseline_iou - args.decision_iou_delta)
    consensus = select_sparse_candidate(consensus_candidates, minimum_iou)
    frequency = select_sparse_candidate(frequency_candidates, minimum_iou)
    decisions = {
        "best_individual": {
            "gene": best_gene,
            "iou": best_iou,
            "keep_ratio": keep_ratio(best_gene),
            "score": quality_score(best_iou, baseline_iou, args),
        },
        "majority": {
            "gene": majority_gene,
            "iou": majority_iou,
            "keep_ratio": keep_ratio(majority_gene),
            "score": quality_score(majority_iou, minimum_iou, args),
        },
        "consensus": {
            "gene": consensus["gene"],
            "iou": consensus["iou"],
            "keep_ratio": consensus["keep_ratio"],
            "score": sparse_score(consensus["iou"], consensus["keep_ratio"], minimum_iou, args),
            "ratio": consensus["value"],
            "candidates": [
                {"ratio": item["value"], "iou": item["iou"], "keep_ratio": item["keep_ratio"]}
                for item in consensus_candidates
            ],
        },
        "frequency": {
            "gene": frequency["gene"],
            "iou": frequency["iou"],
            "keep_ratio": frequency["keep_ratio"],
            "score": sparse_score(frequency["iou"], frequency["keep_ratio"], minimum_iou, args),
            "fraction": frequency["value"],
            "candidates": [
                {"fraction": item["value"], "iou": item["iou"], "keep_ratio": item["keep_ratio"]}
                for item in frequency_candidates
            ],
        },
    }
    ious = [decision["iou"] for decision in decisions.values()]
    overall_score = (
        args.majority_weight * decisions["majority"]["score"]
        + args.best_weight * decisions["best_individual"]["score"]
        + args.consensus_weight * decisions["consensus"]["score"]
        + args.frequency_weight * decisions["frequency"]["score"]
        + args.minimum_iou_weight * min(ious)
    )
    return {
        "baseline_iou": baseline_iou,
        "minimum_iou": minimum_iou,
        "mean_iou": float(np.mean(ious)),
        "minimum_decision_iou": float(min(ious)),
        "overall_score": float(overall_score),
        "decisions": decisions,
    }


def serializable_decisions(metrics: dict[str, Any]) -> dict[str, Any]:
    result = {
        "baseline_iou": metrics["baseline_iou"],
        "minimum_iou": metrics["minimum_iou"],
        "mean_iou": metrics["mean_iou"],
        "minimum_decision_iou": metrics["minimum_decision_iou"],
        "overall_score": metrics["overall_score"],
        "decisions": {},
    }
    for name, decision in metrics["decisions"].items():
        result["decisions"][name] = {key: value for key, value in decision.items() if key != "gene"}
    return result


def save_pickle(value: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(value, file)


def iou_suffix(iou: float) -> str:
    return f"{int(round(iou * 10000)):04d}"


def replace_pickle(sample_dir: Path, pattern: str, value: np.ndarray, filename: str) -> Path:
    for old_path in sample_dir.glob(pattern):
        old_path.unlink(missing_ok=True)
    path = sample_dir / filename
    save_pickle(value, path)
    return path


def update_best_decision(
    records: dict[str, dict[str, Any]],
    sample_dir: Path,
    sample_name: str,
    name: str,
    decision: dict[str, Any],
    generation: int,
) -> None:
    current = records[name]
    better_iou = decision["iou"] > current["iou"] + EPS
    sparser_tie = abs(decision["iou"] - current["iou"]) <= EPS and decision["keep_ratio"] < current["keep_ratio"]
    if not (better_iou or sparser_tie):
        return
    suffix = iou_suffix(decision["iou"])
    path = replace_pickle(
        sample_dir,
        f"{sample_name}_best_decision_{name}_*.pkl",
        decision["gene"],
        f"{sample_name}_best_decision_{name}_{suffix}.pkl",
    )
    current.clear()
    current.update(
        iou=float(decision["iou"]),
        keep_ratio=float(decision["keep_ratio"]),
        score=float(decision["score"]),
        generation=generation,
        path=str(path),
    )
    if "ratio" in decision:
        current["ratio"] = float(decision["ratio"])
    if "fraction" in decision:
        current["fraction"] = float(decision["fraction"])


def write_selected_population(
    sample_dir: Path,
    sample: SearchSample,
    population: np.ndarray,
    metrics: dict[str, Any],
) -> dict[str, str]:
    best_iou = metrics["decisions"]["best_individual"]["iou"]
    paths = {
        "population": replace_pickle(
            sample_dir,
            f"{sample.name}_best_population_*.pkl",
            population,
            f"{sample.name}_best_population_{iou_suffix(best_iou)}.pkl",
        )
    }
    for name, decision in metrics["decisions"].items():
        file_label = "best_individual" if name == "best_individual" else f"decision_{name}"
        paths[name] = replace_pickle(
            sample_dir,
            f"{sample.name}_{file_label}_*.pkl",
            decision["gene"],
            f"{sample.name}_{file_label}_{iou_suffix(decision['iou'])}.pkl",
        )
    return {name: str(path) for name, path in paths.items()}


def log_generation(
    writer: SummaryWriter,
    sample_name: str,
    generation: int,
    fitness: np.ndarray,
    global_best: float,
    metrics: dict[str, Any],
    mutation_probability: float,
) -> None:
    values = {
        "fitness/best": float(fitness.min()),
        "fitness/mean": float(fitness.mean()),
        "fitness/worst": float(fitness.max()),
        "fitness/global_best": global_best,
        "decision/score": metrics["overall_score"],
        "decision/mean_iou": metrics["mean_iou"],
        "decision/minimum_iou": metrics["minimum_decision_iou"],
        "search/mutation_probability": mutation_probability,
    }
    for name, decision in metrics["decisions"].items():
        values[f"decision/{name}_iou"] = decision["iou"]
        values[f"decision/{name}_keep_ratio"] = decision["keep_ratio"]
        values[f"decision/{name}_score"] = decision["score"]
    for key, value in values.items():
        writer.add_scalar(f"{sample_name}/{key}", value, generation)


def run_search(
    sample: SearchSample,
    model: nn.Module,
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
    writer: SummaryWriter,
) -> dict[str, Any]:
    sample_dir = output_dir / sample.name
    sample_dir.mkdir(parents=True, exist_ok=True)
    population = initial_population(sample.gene_length, args.population_size, args.include_all_ones)
    baseline_iou = float(
        evaluate_genes_iou(np.ones(sample.gene_length, dtype=np.uint8), sample, model, args, device)[0]
    )
    best_score = float("-inf")
    global_best_fitness = float("inf")
    starting_best_fitness: float | None = None
    mutation_probability = args.mutation_probability
    selected_metadata_path: Path | None = None
    selected_metrics: dict[str, Any] | None = None
    selected_generation = -1
    selected_paths: dict[str, str] = {}
    best_decisions = {
        name: {"iou": float("-inf"), "keep_ratio": float("inf")}
        for name in ("best_individual", "majority", "consensus", "frequency")
    }

    print(
        f"{sample.name}: segments={sample.gene_length}, baseline_iou={baseline_iou:.4f}, "
        f"population={args.population_size}, generations={args.generations}"
    )
    for generation in range(args.generations):
        fitness = evaluate_population_fitness(population, sample, model, args, device)
        population_ious = 1.0 - fitness
        generation_best = float(fitness.min())
        global_best_fitness = min(global_best_fitness, generation_best)
        if starting_best_fitness is None:
            starting_best_fitness = generation_best

        metrics = evaluate_decisions(population, population_ious, baseline_iou, sample, model, args, device)
        selection_score = metrics["overall_score"] if args.decision_aware else float(population_ious.max())
        for name, decision in metrics["decisions"].items():
            update_best_decision(best_decisions, sample_dir, sample.name, name, decision, generation)
        log_generation(writer, sample.name, generation, fitness, global_best_fitness, metrics, mutation_probability)

        if selection_score > best_score:
            best_score = selection_score
            selected_generation = generation
            selected_metrics = metrics
            selected_paths = write_selected_population(sample_dir, sample, population, metrics)
            selected_metadata_path = sample_dir / f"{sample.name}_best_metadata.json"
            metadata = {
                "image_name": sample.name,
                "image_path": str(sample.image_path),
                "mask_path": str(sample.mask_path),
                "model": args.model,
                "checkpoint": str(args.checkpoint),
                "generation": generation,
                "requested_n_segments": args.n_segments,
                "actual_n_segments": sample.gene_length,
                "population_size": args.population_size,
                "decision_aware": args.decision_aware,
                "decision_iou_delta": args.decision_iou_delta,
                "metrics": serializable_decisions(metrics),
                "paths": selected_paths,
            }
            selected_metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            print(
                f"  generation={generation:03d} saved score={selection_score:.4f} "
                f"best={metrics['decisions']['best_individual']['iou']:.4f} "
                f"majority={metrics['decisions']['majority']['iou']:.4f} "
                f"consensus={metrics['decisions']['consensus']['iou']:.4f} "
                f"frequency={metrics['decisions']['frequency']['iou']:.4f}"
            )

        print(
            f"  generation={generation:03d} fitness best={fitness.min():.4f} mean={fitness.mean():.4f} "
            f"worst={fitness.max():.4f} decision_score={selection_score:.4f}"
        )
        if global_best_fitness <= args.target_fitness:
            break
        if (
            starting_best_fitness > 0
            and (starting_best_fitness - global_best_fitness) / starting_best_fitness > 0.65
        ):
            mutation_probability = min(mutation_probability, args.late_mutation_probability)

        selected, selected_fitness = tournament_selection(population, fitness, args.tournament_size)
        population = crossover_population(selected, selected_fitness, args.crossover_probability, args.elite_count)
        population = mutate_population(population, mutation_probability)
        if args.decision_aware and args.inject_decision_genes:
            decision_genes = [decision["gene"] for decision in metrics["decisions"].values()]
            for index, gene in enumerate(decision_genes[: len(population)]):
                population[index] = gene

    if selected_metrics is None or selected_metadata_path is None:
        raise RuntimeError(f"No population was selected for {sample.name}.")
    metadata = json.loads(selected_metadata_path.read_text(encoding="utf-8"))
    metadata["best_decisions_across_generations"] = best_decisions
    selected_metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {
        "image_name": sample.name,
        "model": args.model,
        "checkpoint": str(args.checkpoint),
        "baseline_iou": baseline_iou,
        "selection_score": best_score,
        "generation": selected_generation,
        "metrics": serializable_decisions(selected_metrics),
        "paths": selected_paths,
        "best_decisions_across_generations": best_decisions,
        "metadata_path": str(selected_metadata_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run decision-aware genetic search over SLIC superpixels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", choices=DATASET_NAMES, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image-name", default=None)
    source.add_argument("--image-list-file", type=Path, default=None)
    source.add_argument("--all-images", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--skip-existing", action="store_true")

    parser.add_argument("--n-segments", type=int, default=128)
    parser.add_argument("--slic-compactness", type=float, default=10.0)
    parser.add_argument("--population-size", type=int, default=300)
    parser.add_argument("--generations", type=int, default=180)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--crossover-probability", type=float, default=0.7)
    parser.add_argument("--mutation-probability", type=float, default=0.02)
    parser.add_argument("--late-mutation-probability", type=float, default=0.01)
    parser.add_argument("--elite-count", type=int, default=2)
    parser.add_argument("--tournament-size", type=int, default=5)
    parser.add_argument("--target-fitness", type=float, default=0.0)

    parser.add_argument("--decision-aware", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--inject-decision-genes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-all-ones", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--decision-iou-delta", type=float, default=0.15)
    parser.add_argument("--sparsity-weight", type=float, default=0.15)
    parser.add_argument("--penalty-weight", type=float, default=2.0)
    parser.add_argument("--majority-weight", type=float, default=0.30)
    parser.add_argument("--best-weight", type=float, default=0.25)
    parser.add_argument("--consensus-weight", type=float, default=0.25)
    parser.add_argument("--frequency-weight", type=float, default=0.20)
    parser.add_argument("--minimum-iou-weight", type=float, default=0.30)
    parser.add_argument(
        "--consensus-ratios",
        type=parse_float_list,
        default=DEFAULT_CONSENSUS_RATIOS,
    )
    parser.add_argument(
        "--frequency-fractions",
        type=parse_float_list,
        default=DEFAULT_FREQUENCY_FRACTIONS,
    )

    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--input-channels", type=int, choices=(1, 3), default=3)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dag-threshold", type=float, default=0.5)
    parser.add_argument("--dag-fraction", type=float, default=0.75)

    parser.add_argument("--brats-modality", choices=("flair", "t1", "t1ce", "t2"), default="flair")
    parser.add_argument("--brats-slice-index", type=int, default=-1)
    parser.add_argument("--brats-image-size", type=int, default=256)
    parser.add_argument("--covid-split", choices=("train", "test"), default="test")
    parser.add_argument("--covid-slice-index", type=int, default=-1)
    parser.add_argument("--covid-time-index", type=int, default=-1)
    parser.add_argument("--covid-image-size", type=int, default=256)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    args.brats_slice_index = None if args.brats_slice_index < 0 else args.brats_slice_index
    args.covid_slice_index = None if args.covid_slice_index < 0 else args.covid_slice_index
    args.covid_time_index = None if args.covid_time_index < 0 else args.covid_time_index
    if args.population_size <= 0 or args.generations <= 0 or args.eval_batch_size <= 0:
        raise ValueError("Population size, generations, and evaluation batch size must be positive.")
    if args.n_segments <= 0 or args.slic_compactness <= 0:
        raise ValueError("SLIC segment count and compactness must be positive.")
    if not 0 <= args.threshold <= 1:
        raise ValueError("--threshold must be between 0 and 1.")
    if not all(0 < value <= 1 for value in args.consensus_ratios):
        raise ValueError("Consensus ratios must be in (0, 1].")
    if not all(0 < value < 1 for value in args.frequency_fractions):
        raise ValueError("Frequency fractions must be in (0, 1).")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    return args


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = resolve_device(args.device)
    args.amp = args.amp and device.type == "cuda"
    model = load_checkpoint_model(args, device)
    names = resolve_image_names(args)
    output_dir = args.output_dir or Path("outputs") / (
        f"search_{args.dataset}_{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    if output_dir.exists() and any(output_dir.iterdir()) and not args.skip_existing:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Choose another directory or add --skip-existing."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            config[key] = str(value)
        elif isinstance(value, tuple):
            config[key] = list(value)
        else:
            config[key] = value
    (output_dir / "search_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    covid_dataset = build_covid_dataset(args) if args.dataset == "covid19ct" else None

    summary_path = output_dir / "summary.jsonl"
    writer = SummaryWriter(str(output_dir / "tensorboard"))
    try:
        for name in names:
            metadata_path = output_dir / name / f"{name}_best_metadata.json"
            if args.skip_existing and metadata_path.exists():
                print(f"Skipping completed sample: {name}")
                continue
            if args.dataset == "ph2":
                sample = load_ph2_sample(args, name)
            elif args.dataset == "brats2020":
                sample = load_brats_sample(args, name)
            else:
                if covid_dataset is None:
                    raise RuntimeError("COVID-19-CT dataset was not initialized.")
                sample = load_covid_sample(args, covid_dataset, name)
            result = run_search(sample, model, args, device, output_dir, writer)
            with summary_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(result) + "\n")
    finally:
        writer.close()
    print(f"Search complete: {output_dir}")


if __name__ == "__main__":
    main()
