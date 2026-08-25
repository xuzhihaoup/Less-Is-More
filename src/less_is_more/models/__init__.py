"""Model registry for the five segmentation backbones used in the paper."""

from __future__ import annotations

from torch import nn

from ..config import TrainConfig

__all__ = ["build_model", "create_model"]


def build_model(
    model_name: str,
    input_channels: int = 3,
    dag_threshold: float = 0.5,
    dag_fraction: float = 0.75,
) -> nn.Module:
    """Build one of the released binary segmentation backbones."""
    if model_name == "unet":
        from .unet import UNet

        return UNet(
            in_channels=input_channels,
            out_channels=1,
            normalize_input=False,
            return_features=False,
        )
    if model_name == "cmunext":
        from .cmunext import CMUNeXt

        return CMUNeXt(input_channel=input_channels, num_classes=1)
    if model_name == "mk_unet":
        from .mk_unet import MK_UNet

        # MK-UNet repeats one-channel input to RGB in its forward pass.
        model_channels = 3 if input_channels == 1 else input_channels
        return MK_UNet(in_channels=model_channels, num_classes=1)
    if model_name == "dag_unet":
        from .dag_unet import DAG_UNet

        return DAG_UNet(
            in_c=input_channels,
            out_c=1,
            threshold=dag_threshold,
            frac=dag_fraction,
        )
    if model_name == "ege_unet":
        from .ege_unet import EGEUNet

        return EGEUNet(num_classes=1, input_channels=input_channels, gt_ds=True)
    raise ValueError(f"Unsupported model: {model_name}")


def create_model(config: TrainConfig) -> nn.Module:
    """Build the model described by a training configuration."""
    return build_model(
        model_name=config.model,
        input_channels=config.input_channels,
        dag_threshold=config.dag_threshold,
        dag_fraction=config.dag_fraction,
    )
