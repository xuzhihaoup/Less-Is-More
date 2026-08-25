from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


MODEL_NAMES = ("unet", "cmunext", "mk_unet", "dag_unet", "ege_unet")
DATASET_NAMES = ("ph2", "brats2020", "covid19ct")


@dataclass(frozen=True)
class TrainConfig:
    model: str
    dataset: str
    dataset_root: Path
    run_name: str
    output_dir: Path
    epochs: int
    batch_size: int
    learning_rate: float
    num_workers: int
    device: str
    amp: bool
    deterministic: bool
    seed: int
    input_channels: int
    threshold: float
    pos_weight: float
    aux_loss_weight: float
    scheduler_step_size: int
    scheduler_gamma: float
    init_checkpoint: Path | None
    resume: Path | None
    brats_modality: str
    brats_slice_index: int | None
    brats_image_size: int
    brats_val_ratio: float
    covid_slice_index: int | None
    covid_time_index: int | None
    covid_image_size: int
    dag_threshold: float
    dag_fraction: float

    def as_serializable_dict(self) -> dict[str, Any]:
        values = asdict(self)
        for key, value in values.items():
            if isinstance(value, Path):
                values[key] = str(value)
        return values


def _optional_index(value: int) -> int | None:
    return None if value < 0 else value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a binary medical image segmentation model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", choices=MODEL_NAMES, default="unet")
    parser.add_argument("--dataset", choices=DATASET_NAMES, default="ph2")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--run-name", default="", help="Run identifier. A timestamped name is generated when omitted.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))

    parser.add_argument("--epochs", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--amp", action="store_true", help="Use automatic mixed precision on CUDA.")
    parser.add_argument("--deterministic", action="store_true", help="Prefer deterministic CUDA operations.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input-channels", type=int, choices=(1, 3), default=3)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--pos-weight",
        type=float,
        default=-1.0,
        help="BCE positive-class weight. Values <= 0 enable automatic estimation.",
    )
    parser.add_argument("--aux-loss-weight", type=float, default=0.4)
    parser.add_argument("--scheduler-step-size", type=int, default=150)
    parser.add_argument("--scheduler-gamma", type=float, default=0.65)

    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument("--init-checkpoint", type=Path, default=None, help="Initialize model weights only.")
    checkpoint_group.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume model, optimizer, and scheduler state.",
    )

    parser.add_argument("--brats-modality", choices=("flair", "t1", "t1ce", "t2"), default="flair")
    parser.add_argument("--brats-slice-index", type=int, default=-1, help="-1 selects the middle axial slice.")
    parser.add_argument("--brats-image-size", type=int, default=256)
    parser.add_argument("--brats-val-ratio", type=float, default=0.2)

    parser.add_argument("--covid-slice-index", type=int, default=-1, help="-1 exposes all axial slices.")
    parser.add_argument("--covid-time-index", type=int, default=-1, help="-1 selects the middle frame for 4D data.")
    parser.add_argument("--covid-image-size", type=int, default=256)

    parser.add_argument("--dag-threshold", type=float, default=0.5)
    parser.add_argument("--dag-fraction", type=float, default=0.75)
    return parser


def parse_config(argv: list[str] | None = None) -> TrainConfig:
    args = build_parser().parse_args(argv)

    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive.")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative.")
    if args.scheduler_step_size <= 0:
        raise ValueError("--scheduler-step-size must be positive.")
    if args.aux_loss_weight < 0:
        raise ValueError("--aux-loss-weight cannot be negative.")
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1.")
    if not 0.0 < args.scheduler_gamma <= 1.0:
        raise ValueError("--scheduler-gamma must be in (0, 1].")
    if not 0.0 < args.brats_val_ratio < 1.0:
        raise ValueError("--brats-val-ratio must be in (0, 1).")
    if not 0.0 < args.dag_fraction <= 1.0:
        raise ValueError("--dag-fraction must be in (0, 1].")
    if args.brats_image_size <= 0 or args.covid_image_size <= 0:
        raise ValueError("Dataset image sizes must be positive.")
    if args.brats_image_size % 32 or args.covid_image_size % 32:
        raise ValueError("BraTS and COVID-19-CT image sizes must be divisible by 32.")

    run_name = args.run_name or f"{args.dataset}_{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return TrainConfig(
        model=args.model,
        dataset=args.dataset,
        dataset_root=args.dataset_root,
        run_name=run_name,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_workers=args.num_workers,
        device=args.device,
        amp=args.amp,
        deterministic=args.deterministic,
        seed=args.seed,
        input_channels=args.input_channels,
        threshold=args.threshold,
        pos_weight=args.pos_weight,
        aux_loss_weight=args.aux_loss_weight,
        scheduler_step_size=args.scheduler_step_size,
        scheduler_gamma=args.scheduler_gamma,
        init_checkpoint=args.init_checkpoint,
        resume=args.resume,
        brats_modality=args.brats_modality,
        brats_slice_index=_optional_index(args.brats_slice_index),
        brats_image_size=args.brats_image_size,
        brats_val_ratio=args.brats_val_ratio,
        covid_slice_index=_optional_index(args.covid_slice_index),
        covid_time_index=_optional_index(args.covid_time_index),
        covid_image_size=args.covid_image_size,
        dag_threshold=args.dag_threshold,
        dag_fraction=args.dag_fraction,
    )
