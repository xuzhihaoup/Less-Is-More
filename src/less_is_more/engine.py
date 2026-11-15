"""Training loop and checkpoint management."""

from __future__ import annotations

import csv
import json
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .config import TrainConfig
from .data import create_dataloaders
from .models import create_model

EPS = 1e-7
METRIC_NAMES = (
    "loss",
    "miou",
    "dice",
    "accuracy",
    "predicted_foreground_ratio",
    "target_foreground_ratio",
    "mean_probability",
)


def seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but no CUDA device is available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def initialize_model(model: nn.Module, model_name: str) -> None:
    """Apply the initialization used by the original training implementation."""
    if model_name not in {"unet", "cmunext", "dag_unet"}:
        return

    def initialize(module: nn.Module) -> None:
        class_name = module.__class__.__name__
        if hasattr(module, "weight") and "Conv" in class_name:
            nn.init.normal_(module.weight.data, 0.0, 0.02)
        elif "BatchNorm2d" in class_name:
            nn.init.normal_(module.weight.data, 1.0, 0.02)
            nn.init.constant_(module.bias.data, 0.0)

    model.apply(initialize)


def _torch_load(path: Path, device: torch.device) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint format: {path}")
    return checkpoint


def _state_dict_from_checkpoint(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if any(key.startswith("module.") for key in state_dict):
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
    return state_dict


def load_initial_weights(model: nn.Module, path: Path, device: torch.device) -> None:
    checkpoint = _torch_load(path, device)
    source = _state_dict_from_checkpoint(checkpoint)
    target = model.state_dict()
    matched = {key: value for key, value in source.items() if key in target and target[key].shape == value.shape}
    if not matched:
        raise RuntimeError(f"No model parameters matched checkpoint: {path}")
    target.update(matched)
    model.load_state_dict(target)
    print(f"Initialized {len(matched)}/{len(target)} tensors from {path}")


def resume_training(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    path: Path,
    device: torch.device,
    config: TrainConfig,
) -> tuple[int, float]:
    checkpoint = _torch_load(path, device)
    checkpoint_model = checkpoint.get("model_name")
    if checkpoint_model is not None and checkpoint_model != config.model:
        raise ValueError(f"Checkpoint model is {checkpoint_model}, but --model is {config.model}.")
    checkpoint_config = checkpoint.get("config", {})
    checkpoint_dataset = checkpoint_config.get("dataset") if isinstance(checkpoint_config, dict) else None
    if checkpoint_dataset is not None and checkpoint_dataset != config.dataset:
        raise ValueError(f"Checkpoint dataset is {checkpoint_dataset}, but --dataset is {config.dataset}.")
    model.load_state_dict(_state_dict_from_checkpoint(checkpoint), strict=True)
    if "optimizer_state_dict" not in checkpoint or "scheduler_state_dict" not in checkpoint:
        raise KeyError("A resume checkpoint must contain optimizer and scheduler state.")
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    start_epoch = int(checkpoint.get("epoch", 0)) + 1
    best_miou = float(checkpoint.get("best_miou", -1.0))
    print(f"Resumed from {path} at epoch {start_epoch}")
    return start_epoch, best_miou


def _split_outputs(model_name: str, outputs: Any) -> tuple[torch.Tensor, list[torch.Tensor]]:
    if model_name == "ege_unet" and isinstance(outputs, tuple) and len(outputs) == 2:
        auxiliary, main = outputs
        return main, list(auxiliary)
    if model_name == "mk_unet" and isinstance(outputs, (list, tuple)):
        predictions = list(outputs)
        return predictions[0], predictions[1:]
    if isinstance(outputs, (list, tuple)):
        return outputs[0], []
    return outputs, []


def _ensure_channels(prediction: torch.Tensor) -> torch.Tensor:
    return prediction.unsqueeze(1) if prediction.ndim == 3 else prediction


def _resize(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if prediction.shape[-2:] == target.shape[-2:]:
        return prediction
    return F.interpolate(prediction, size=target.shape[-2:], mode="bilinear", align_corners=False)


def _to_logits(model_name: str, prediction: torch.Tensor) -> torch.Tensor:
    prediction = _ensure_channels(prediction)
    if model_name == "ege_unet":
        return torch.logit(prediction.clamp(EPS, 1.0 - EPS))
    return prediction


def _to_probabilities(model_name: str, prediction: torch.Tensor) -> torch.Tensor:
    prediction = _ensure_channels(prediction)
    return prediction if model_name == "ege_unet" else torch.sigmoid(prediction)


def outputs_to_probabilities(
    model_name: str,
    outputs: Any,
    target_size: tuple[int, int],
) -> torch.Tensor:
    """Convert a backbone-specific output to a resized foreground probability map."""
    main, _ = _split_outputs(model_name, outputs)
    probabilities = _to_probabilities(model_name, main)
    if probabilities.shape[-2:] != target_size:
        probabilities = F.interpolate(probabilities, size=target_size, mode="bilinear", align_corners=False)
    return probabilities


def compute_loss(
    model_name: str,
    outputs: Any,
    targets: torch.Tensor,
    criterion: nn.Module,
    auxiliary_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    main, auxiliary = _split_outputs(model_name, outputs)
    main_logits = _resize(_to_logits(model_name, main), targets)
    probabilities = _resize(_to_probabilities(model_name, main), targets)
    loss = criterion(main_logits, targets)
    if auxiliary:
        auxiliary_losses = [criterion(_resize(_to_logits(model_name, item), targets), targets) for item in auxiliary]
        loss = loss + auxiliary_weight * torch.stack(auxiliary_losses).mean()
    return loss, probabilities


def compute_metrics(probabilities: torch.Tensor, targets: torch.Tensor, threshold: float) -> dict[str, float]:
    predictions = probabilities >= threshold
    target_mask = targets >= 0.5
    intersection = (predictions & target_mask).sum(dim=(1, 2, 3)).float()
    predicted_area = predictions.sum(dim=(1, 2, 3)).float()
    target_area = target_mask.sum(dim=(1, 2, 3)).float()
    union = predicted_area + target_area - intersection

    return {
        "miou": ((intersection + EPS) / (union + EPS)).mean().item(),
        "dice": ((2.0 * intersection + EPS) / (predicted_area + target_area + EPS)).mean().item(),
        "accuracy": (predictions == target_mask).float().mean(dim=(1, 2, 3)).mean().item(),
        "predicted_foreground_ratio": predictions.float().mean().item(),
        "target_foreground_ratio": target_mask.float().mean().item(),
        "mean_probability": probabilities.mean().item(),
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    model_name: str,
    threshold: float,
    auxiliary_weight: float,
    amp: bool,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {name: 0.0 for name in METRIC_NAMES}
    sample_count = 0
    grad_context = torch.enable_grad() if training else torch.inference_mode()

    with grad_context:
        for images, targets in loader:
            images = images.to(device, dtype=torch.float32, non_blocking=True)
            targets = targets.to(device, dtype=torch.float32, non_blocking=True).unsqueeze(1)
            batch_size = images.shape[0]

            if training:
                optimizer.zero_grad(set_to_none=True)
            autocast = torch.autocast(device_type="cuda", dtype=torch.float16) if amp else nullcontext()
            with autocast:
                outputs = model(images)
                loss, probabilities = compute_loss(model_name, outputs, targets, criterion, auxiliary_weight)

            if training:
                if scaler is None:
                    raise RuntimeError("A gradient scaler is required during training.")
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            metrics = compute_metrics(probabilities.detach(), targets, threshold)
            totals["loss"] += loss.item() * batch_size
            for name, value in metrics.items():
                totals[name] += value * batch_size
            sample_count += batch_size

    if sample_count == 0:
        raise RuntimeError("The data loader produced no samples.")
    return {name: value / sample_count for name, value in totals.items()}


def resolve_positive_weight(dataset: Any, configured_value: float, device: torch.device) -> tuple[torch.Tensor, float]:
    foreground, total = dataset.foreground_pixel_counts()
    if total <= 0:
        raise RuntimeError("The training masks contain no pixels.")
    foreground_ratio = foreground / total
    if configured_value > 0:
        value = configured_value
    else:
        clipped_ratio = min(max(foreground_ratio, EPS), 1.0 - EPS)
        value = (1.0 - clipped_ratio) / clipped_ratio
    return torch.tensor([value], dtype=torch.float32, device=device), foreground_ratio


def _write_history(
    path: Path,
    epoch: int,
    learning_rate: float,
    train: dict[str, float],
    val: dict[str, float],
) -> None:
    columns = ["epoch", "learning_rate"]
    columns.extend(f"train_{name}" for name in METRIC_NAMES)
    columns.extend(f"val_{name}" for name in METRIC_NAMES)
    row = {"epoch": epoch, "learning_rate": learning_rate}
    row.update({f"train_{name}": value for name, value in train.items()})
    row.update({f"val_{name}": value for name, value in val.items()})
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _save_checkpoint(
    path: Path,
    epoch: int,
    best_miou: float,
    config: TrainConfig,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    metrics: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "best_miou": best_miou,
            "model_name": config.model,
            "config": config.as_serializable_dict(),
            "metrics": metrics,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
        },
        path,
    )


def _log_metrics(writer: SummaryWriter, stage: str, metrics: dict[str, float], epoch: int) -> None:
    for name, value in metrics.items():
        writer.add_scalar(f"{stage}/{name}", value, epoch)


def _format_epoch(epoch: int, epochs: int, learning_rate: float, train: dict[str, float], val: dict[str, float]) -> str:
    return (
        f"Epoch {epoch:03d}/{epochs:03d} | lr={learning_rate:.6g} | "
        f"train loss={train['loss']:.4f} mIoU={train['miou']:.4f} Dice={train['dice']:.4f} | "
        f"val loss={val['loss']:.4f} mIoU={val['miou']:.4f} Dice={val['dice']:.4f}"
    )


def train(config: TrainConfig) -> Path:
    seed_everything(config.seed, config.deterministic)
    device = resolve_device(config.device)
    amp_enabled = config.amp and device.type == "cuda"
    if config.amp and not amp_enabled:
        print("AMP requested without CUDA; continuing with full precision.")

    run_dir = config.output_dir / config.run_name
    if config.resume is None and run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {run_dir}. Use a new --run-name or resume from a checkpoint."
        )
    checkpoint_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.json"
    config_path.write_text(json.dumps(config.as_serializable_dict(), indent=2), encoding="utf-8")

    train_loader, val_loader = create_dataloaders(config, pin_memory=device.type == "cuda")
    model = create_model(config).to(device)
    initialize_model(model, config.model)

    positive_weight, foreground_ratio = resolve_positive_weight(train_loader.dataset, config.pos_weight, device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.scheduler_step_size,
        gamma=config.scheduler_gamma,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    start_epoch = 1
    best_miou = -1.0
    if config.resume is not None:
        start_epoch, best_miou = resume_training(
            model,
            optimizer,
            scheduler,
            scaler,
            config.resume,
            device,
            config,
        )
    elif config.init_checkpoint is not None:
        load_initial_weights(model, config.init_checkpoint, device)

    print(f"Run: {config.run_name}")
    print(f"Model: {config.model} | Dataset: {config.dataset} | Device: {device}")
    print(f"Training samples: {len(train_loader.dataset)} | Validation samples: {len(val_loader.dataset)}")
    print(f"Foreground ratio: {foreground_ratio:.6f} | BCE pos_weight: {positive_weight.item():.4f}")
    print(f"Output: {run_dir}")

    history_path = run_dir / "history.csv"
    writer = SummaryWriter(str(run_dir / "tensorboard"))
    try:
        for epoch in range(start_epoch, config.epochs + 1):
            train_metrics = run_epoch(
                model,
                train_loader,
                criterion,
                device,
                config.model,
                config.threshold,
                config.aux_loss_weight,
                amp_enabled,
                optimizer=optimizer,
                scaler=scaler,
            )
            val_metrics = run_epoch(
                model,
                val_loader,
                criterion,
                device,
                config.model,
                config.threshold,
                config.aux_loss_weight,
                amp_enabled,
            )
            learning_rate = optimizer.param_groups[0]["lr"]
            _log_metrics(writer, "train", train_metrics, epoch)
            _log_metrics(writer, "val", val_metrics, epoch)
            writer.add_scalar("train/learning_rate", learning_rate, epoch)
            _write_history(history_path, epoch, learning_rate, train_metrics, val_metrics)
            print(_format_epoch(epoch, config.epochs, learning_rate, train_metrics, val_metrics))

            improved = val_metrics["miou"] > best_miou
            if improved:
                best_miou = val_metrics["miou"]
            scheduler.step()
            _save_checkpoint(
                checkpoint_dir / "last.pt",
                epoch,
                best_miou,
                config,
                model,
                optimizer,
                scheduler,
                scaler,
                val_metrics,
            )
            if improved:
                _save_checkpoint(
                    checkpoint_dir / "best.pt",
                    epoch,
                    best_miou,
                    config,
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    val_metrics,
                )
                print(f"Best checkpoint updated: validation mIoU={best_miou:.4f}")
    finally:
        writer.close()

    return run_dir
