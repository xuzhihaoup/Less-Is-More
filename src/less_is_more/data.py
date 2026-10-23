from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .config import TrainConfig


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
NIFTI_EXTENSIONS = (".nii", ".nii.gz")


def _read_names(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not names:
        raise RuntimeError(f"Split file is empty: {path}")
    return names


def _resolve_image(directory: Path, name: str) -> Path:
    direct = directory / name
    if direct.suffix and direct.exists():
        return direct
    for extension in IMAGE_EXTENSIONS:
        candidate = directory / f"{name}{extension}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Image `{name}` not found in {directory}")


def _resolve_nifti(directory: Path, name: str) -> Path:
    for extension in NIFTI_EXTENSIONS:
        candidate = directory / f"{name}{extension}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"NIfTI file `{name}` not found in {directory}")


def _resize_float_image(image: np.ndarray, size: int | None) -> np.ndarray:
    if size is None:
        return image.astype(np.float32, copy=False)
    resized = Image.fromarray(image.astype(np.float32)).resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32)


def _resize_mask(mask: np.ndarray, size: int | None) -> np.ndarray:
    if size is None:
        return mask.astype(np.uint8, copy=False)
    resized = Image.fromarray(mask.astype(np.uint8)).resize((size, size), Image.Resampling.NEAREST)
    return np.asarray(resized, dtype=np.uint8)


def _to_image_tensor(image: np.ndarray, channels: int) -> torch.Tensor:
    if image.ndim == 2:
        image = image[None, :, :]
    else:
        image = np.transpose(image, (2, 0, 1))
    if channels == 3 and image.shape[0] == 1:
        image = np.repeat(image, 3, axis=0)
    if channels == 1 and image.shape[0] == 3:
        image = image.mean(axis=0, keepdims=True)
    return torch.from_numpy(np.ascontiguousarray(image)).float()


class PNGSegmentationDataset(Dataset):
    def __init__(self, image_dir: Path, mask_dir: Path, split_file: Path, input_channels: int = 3):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.names = _read_names(split_file)
        self.input_channels = input_channels

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        name = self.names[index]
        image = np.asarray(Image.open(_resolve_image(self.image_dir, name)).convert("RGB"), dtype=np.float32) / 255.0
        mask = np.asarray(Image.open(_resolve_image(self.mask_dir, name)).convert("L"), dtype=np.uint8)
        if image.shape[:2] != mask.shape:
            raise ValueError(f"Image/mask shape mismatch for {name}: {image.shape[:2]} vs {mask.shape}")
        if image.shape[0] % 32 or image.shape[1] % 32:
            raise ValueError(f"Image dimensions must be divisible by 32 for {name}, got {image.shape[:2]}")
        mask = (mask > 0).astype(np.int64)
        return _to_image_tensor(image, self.input_channels), torch.from_numpy(mask)

    def foreground_pixel_counts(self) -> tuple[int, int]:
        foreground = 0
        total = 0
        for name in self.names:
            mask = np.asarray(Image.open(_resolve_image(self.mask_dir, name)).convert("L"), dtype=np.uint8)
            foreground += int(np.count_nonzero(mask))
            total += int(mask.size)
        return foreground, total


class BraTSSliceDataset(Dataset):
    def __init__(
        self,
        subject_dirs: list[Path],
        modality: str,
        slice_index: int | None,
        image_size: int,
        input_channels: int,
        cache: bool = True,
    ):
        self.subject_dirs = subject_dirs
        self.modality = modality
        self.slice_index = slice_index
        self.image_size = image_size
        self.input_channels = input_channels
        self._cache = [self._load_sample(path) for path in subject_dirs] if cache else None

    def __len__(self) -> int:
        return len(self.subject_dirs)

    @staticmethod
    def _subject_file(subject_dir: Path, suffix: str) -> Path:
        return _resolve_nifti(subject_dir, f"{subject_dir.name}_{suffix}")

    @staticmethod
    def _load_volume(path: Path) -> np.ndarray:
        import nibabel as nib

        return np.asarray(nib.load(str(path)).get_fdata(dtype=np.float32), dtype=np.float32)

    def _select_slice(self, volume: np.ndarray) -> np.ndarray:
        index = volume.shape[2] // 2 if self.slice_index is None else self.slice_index
        if not 0 <= index < volume.shape[2]:
            raise ValueError(f"Slice index {index} is out of range for volume shape {volume.shape}")
        return volume[:, :, index]

    @staticmethod
    def _normalize(image: np.ndarray) -> np.ndarray:
        foreground = image[image > 0]
        if foreground.size == 0:
            return np.zeros_like(image, dtype=np.float32)
        lower, upper = np.percentile(foreground, (1, 99))
        if upper <= lower:
            return np.zeros_like(image, dtype=np.float32)
        normalized = (np.clip(image, lower, upper) - lower) / (upper - lower)
        return normalized.astype(np.float32)

    def _load_sample(self, subject_dir: Path) -> tuple[torch.Tensor, torch.Tensor]:
        image = self._select_slice(self._load_volume(self._subject_file(subject_dir, self.modality)))
        mask = self._select_slice(self._load_volume(self._subject_file(subject_dir, "seg")))
        image = _resize_float_image(self._normalize(image), self.image_size)
        mask = _resize_mask((mask > 0).astype(np.uint8), self.image_size).astype(np.int64)
        return _to_image_tensor(image, self.input_channels), torch.from_numpy(mask)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self._cache is not None:
            return self._cache[index]
        return self._load_sample(self.subject_dirs[index])

    def foreground_pixel_counts(self) -> tuple[int, int]:
        samples = self._cache if self._cache is not None else [self._load_sample(path) for path in self.subject_dirs]
        return sum(int(mask.sum()) for _, mask in samples), sum(int(mask.numel()) for _, mask in samples)


class COVID19CTSliceDataset(Dataset):
    def __init__(
        self,
        image_dir: Path,
        mask_dir: Path,
        split_file: Path,
        slice_index: int | None,
        time_index: int | None,
        image_size: int,
        input_channels: int,
    ):
        self.names = _read_names(split_file)
        self.image_paths = {name: _resolve_nifti(image_dir, name) for name in self.names}
        self.mask_paths = {name: _resolve_nifti(mask_dir, name) for name in self.names}
        self.slice_index = slice_index
        self.time_index = time_index
        self.image_size = image_size
        self.input_channels = input_channels
        self.records = self._build_records()

    @staticmethod
    def _load_volume(path: Path) -> np.ndarray:
        import nibabel as nib

        return np.asarray(nib.load(str(path)).get_fdata(dtype=np.float32), dtype=np.float32)

    def _select_time(self, volume: np.ndarray) -> np.ndarray:
        if volume.ndim != 4:
            return volume
        index = volume.shape[3] // 2 if self.time_index is None else self.time_index
        if not 0 <= index < volume.shape[3]:
            raise ValueError(f"Time index {index} is out of range for volume shape {volume.shape}")
        return volume[:, :, :, index]

    @staticmethod
    def _depth(path: Path) -> int:
        import nibabel as nib

        shape = tuple(dim for dim in nib.load(str(path)).shape if dim != 1)
        return 1 if len(shape) == 2 else shape[2]

    def _build_records(self) -> list[tuple[str, int | None]]:
        records = []
        for name in self.names:
            depth = self._depth(self.image_paths[name])
            mask_depth = self._depth(self.mask_paths[name])
            if depth != mask_depth:
                raise ValueError(f"Image/mask depth mismatch for {name}: {depth} vs {mask_depth}")
            if depth == 1:
                records.append((name, None))
            elif self.slice_index is None:
                records.extend((name, index) for index in range(depth))
            elif 0 <= self.slice_index < depth:
                records.append((name, self.slice_index))
            else:
                raise ValueError(f"Slice index {self.slice_index} is out of range for {name} (depth={depth})")
        return records

    @staticmethod
    def _normalize(image: np.ndarray) -> np.ndarray:
        finite = image[np.isfinite(image)]
        if finite.size == 0:
            return np.zeros_like(image, dtype=np.float32)
        lower, upper = np.percentile(finite, (1, 99))
        if upper <= lower:
            return np.zeros_like(image, dtype=np.float32)
        normalized = (np.clip(image, lower, upper) - lower) / (upper - lower)
        return np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)

    @staticmethod
    def _select_slice(volume: np.ndarray, index: int | None) -> np.ndarray:
        volume = np.squeeze(volume)
        if volume.ndim == 2:
            return volume
        if volume.ndim != 3:
            raise ValueError(f"Expected 2D or 3D data after squeezing, got shape {volume.shape}")
        if index is None:
            index = volume.shape[2] // 2
        return volume[:, :, index]

    def _load_sample(self, name: str, slice_index: int | None) -> tuple[torch.Tensor, torch.Tensor]:
        image_volume = self._select_time(self._load_volume(self.image_paths[name]))
        mask_volume = self._select_time(self._load_volume(self.mask_paths[name]))
        image = self._select_slice(image_volume, slice_index)
        mask = self._select_slice(mask_volume, slice_index)
        image = _resize_float_image(self._normalize(image), self.image_size)
        mask = _resize_mask((mask > 0).astype(np.uint8), self.image_size).astype(np.int64)
        return _to_image_tensor(image, self.input_channels), torch.from_numpy(mask)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._load_sample(*self.records[index])

    def foreground_pixel_counts(self) -> tuple[int, int]:
        foreground = 0
        total = 0
        for name in self.names:
            mask = np.squeeze(self._select_time(self._load_volume(self.mask_paths[name])))
            if mask.ndim == 2:
                selected = mask
            elif mask.ndim == 3 and self.slice_index is None:
                selected = mask
            elif mask.ndim == 3 and 0 <= self.slice_index < mask.shape[2]:
                selected = mask[:, :, self.slice_index]
            else:
                raise ValueError(f"Unsupported mask shape or slice index for {name}: {mask.shape}")
            foreground += int(np.count_nonzero(selected))
            total += int(selected.size)
        return foreground, total


def _resolve_brats_root(dataset_root: Path) -> Path:
    candidates = (
        dataset_root,
        dataset_root / "BraTS2020_TrainingData" / "MICCAI_BraTS2020_TrainingData",
        dataset_root / "MICCAI_BraTS2020_TrainingData",
    )
    for candidate in candidates:
        if candidate.exists() and any(candidate.glob("BraTS20_Training_*")):
            return candidate
    raise FileNotFoundError(f"Could not find BraTS2020 training subjects under {dataset_root}")


def _build_ph2(config: TrainConfig) -> tuple[Dataset, Dataset]:
    root = config.dataset_root
    train = PNGSegmentationDataset(
        root / "Images" / "train",
        root / "ProcessedSegmentationClass" / "train",
        root / "ImageSets" / "train.txt",
        config.input_channels,
    )
    val = PNGSegmentationDataset(
        root / "Images" / "test",
        root / "ProcessedSegmentationClass" / "test",
        root / "ImageSets" / "test.txt",
        config.input_channels,
    )
    return train, val


def _build_brats(config: TrainConfig) -> tuple[Dataset, Dataset]:
    root = _resolve_brats_root(config.dataset_root)
    subjects = []
    for path in sorted(root.glob("BraTS20_Training_*")):
        if not path.is_dir():
            continue
        try:
            BraTSSliceDataset._subject_file(path, config.brats_modality)
            BraTSSliceDataset._subject_file(path, "seg")
        except FileNotFoundError:
            continue
        subjects.append(path)
    if len(subjects) < 2:
        raise RuntimeError(f"Fewer than two labeled BraTS subjects were found under {root}")

    generator = np.random.default_rng(config.seed)
    order = generator.permutation(len(subjects))
    val_count = min(max(1, round(len(subjects) * config.brats_val_ratio)), len(subjects) - 1)
    val_indices = set(order[:val_count].tolist())
    train_subjects = [path for index, path in enumerate(subjects) if index not in val_indices]
    val_subjects = [path for index, path in enumerate(subjects) if index in val_indices]
    kwargs = {
        "modality": config.brats_modality,
        "slice_index": config.brats_slice_index,
        "image_size": config.brats_image_size,
        "input_channels": config.input_channels,
    }
    return BraTSSliceDataset(train_subjects, **kwargs), BraTSSliceDataset(val_subjects, **kwargs)


def _build_covid(config: TrainConfig) -> tuple[Dataset, Dataset]:
    root = config.dataset_root
    kwargs = {
        "slice_index": config.covid_slice_index,
        "time_index": config.covid_time_index,
        "image_size": config.covid_image_size,
        "input_channels": config.input_channels,
    }
    train = COVID19CTSliceDataset(
        root / "train" / "images",
        root / "train" / "masks",
        root / "ImageSets" / "train.txt",
        **kwargs,
    )
    val = COVID19CTSliceDataset(
        root / "test" / "images",
        root / "test" / "masks",
        root / "ImageSets" / "test.txt",
        **kwargs,
    )
    return train, val


def create_datasets(config: TrainConfig) -> tuple[Dataset, Dataset]:
    builders = {"ph2": _build_ph2, "brats2020": _build_brats, "covid19ct": _build_covid}
    return builders[config.dataset](config)


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_dataloaders(config: TrainConfig, pin_memory: bool) -> tuple[DataLoader, DataLoader]:
    train_dataset, val_dataset = create_datasets(config)
    generator = torch.Generator().manual_seed(config.seed)
    common = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": config.num_workers > 0,
        "worker_init_fn": _seed_worker,
        "generator": generator,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **common)
    val_loader = DataLoader(val_dataset, shuffle=False, **common)
    return train_loader, val_loader
