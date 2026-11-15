# Less Is More

[English](README.md) | 简体中文

本项目提供 Less Is More 医学图像分割实验的训练与遗传搜索代码。统一接口支持 PH2、BraTS2020 和
COVID-19-CT 数据集，以及 U-Net、CMUNeXt、MK-UNet、DAG-UNet 和 EGE-UNet 五种分割网络。

运行完整实验前，请先按照 [数据集准备说明](docs/DATASETS.md) 整理数据目录。

## 仓库结构

```text
Less-Is-More/
├── docs/DATASETS.md              # 数据集目录与预处理说明
├── examples/ph2_demo/            # PH2 Demo 图像与标注
├── scripts/train_all_models.sh   # 五种网络的批量训练脚本
├── src/less_is_more/
│   ├── models/                   # 分割网络
│   ├── cli.py                    # 训练命令行入口
│   ├── config.py                 # 训练配置
│   ├── data.py                   # 数据加载
│   ├── engine.py                 # 训练与推理逻辑
│   └── search.py                 # SLIC 遗传搜索
├── demo_ph2.py                   # 单图快速演示
├── genetic_search.py             # 遗传搜索入口
├── pyproject.toml
├── requirements.txt
└── train.py                      # 训练入口
```

## 环境安装

项目要求 Python 3.9 或更高版本。建议先创建独立环境，并根据本机硬件安装对应版本的 PyTorch：

```bash
conda create -n less-is-more python=3.9 -y
conda activate less-is-more

# 以下命令适用于 CUDA 12.8；其他平台请替换为对应的 PyTorch 安装命令。
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install -e . --no-deps
```

参考实验环境为 Python 3.9.19、PyTorch 2.8.0+cu128、NumPy 1.26.4、Pillow 10.4.0、
nibabel 5.3.3、TensorBoard 2.19.0、timm 1.0.27、einops 0.8.1 和 scikit-image 0.24.0。
Python 依赖版本记录在 `requirements.txt` 中。

## 快速 Demo

从 [Google Drive](https://drive.google.com/file/d/1sIOiEnBOqpa87-f-8VEqHuxnBkAnfAVj) 下载 PH2 U-Net
权重，并将本地权重路径传给 Demo：

```bash
python demo_ph2.py --checkpoint /path/to/ph2_unet.pth
```

指定 CPU：

```bash
python demo_ph2.py \
  --checkpoint /path/to/ph2_unet.pth \
  --device cpu
```

使用 CUDA 和自动混合精度：

```bash
python demo_ph2.py \
  --checkpoint /path/to/ph2_unet.pth \
  --device cuda \
  --amp
```

Demo 默认使用仓库中的 `examples/ph2_demo` 样例，执行 `128` 个 SLIC 超像素上的单图遗传搜索。
为了缩短演示时间，默认种群规模和最大迭代次数均为 `32`；可通过参数扩大搜索规模：

```bash
python demo_ph2.py \
  --checkpoint /path/to/ph2_unet.pth \
  --population-size 300 \
  --generations 300 \
  --device cuda \
  --amp
```

运行结果默认保存在带时间戳的 `outputs/ph2_demo_*` 目录中：

```text
outputs/ph2_demo_<timestamp>/
├── demo/                         # 基因、种群与搜索元数据
├── visualizations/
│   ├── original/                # 原图与真实标注
│   ├── explanations/            # 四种遮挡解释图
│   ├── segmentations/           # 分割掩码与误差对比图
│   └── metrics.json             # 各输入对应的 IoU
├── search_config.json
└── summary.jsonl
```

分割误差对比图中，绿色表示真正例，红色表示假正例，蓝色表示假负例。可使用 `--output-dir PATH`
指定输出目录。

## 模型训练

在 PH2 上训练 U-Net：

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

`--model` 支持以下网络：

```text
unet, cmunext, mk_unet, dag_unet, ege_unet
```

BraTS2020 示例：

```bash
less-is-more-train \
  --model cmunext \
  --dataset brats2020 \
  --dataset-root datasets/BraTS2020 \
  --brats-modality flair \
  --amp
```

COVID-19-CT 示例：

```bash
less-is-more-train \
  --model mk_unet \
  --dataset covid19ct \
  --dataset-root datasets/COVID-19-CT \
  --covid-slice-index -1 \
  --amp
```

依次训练同一数据集上的五种网络：

```bash
bash scripts/train_all_models.sh ph2 datasets/PH2
```

批量训练参数可通过环境变量调整：

```bash
EPOCHS=300 BATCH_SIZE=16 NUM_WORKERS=8 AMP=true \
  bash scripts/train_all_models.sh brats2020 datasets/BraTS2020
```

完整参数可通过以下命令查看：

```bash
less-is-more-train --help
```

## 遗传搜索

对 PH2 单张图像执行搜索：

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

搜索整个数据集划分：

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

## 复现实验

- 可通过 `--seed` 设置随机种子。
- 训练时增加 `--deterministic` 可请求 PyTorch 使用确定性算法；部分 GPU 运算仍可能因平台而产生差异。
- SLIC 结果与 scikit-image 版本有关，参考实验使用 `0.24.0`。

## 模型引用

网络实现对应 U-Net、CMUNeXt、MK-UNet、DAG-UNet 和 EGE-UNet。使用相关网络时，请同时引用
Less Is More 论文和对应模型的原始论文。
