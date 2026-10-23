# Dataset preparation

The repository does not redistribute medical images or annotations. Download each dataset from its official source and
confirm that your use complies with its license and data-use agreement.

All tasks are treated as binary segmentation. Label values greater than zero are mapped to foreground.

## PH2

Arrange preprocessed RGB images and binary masks as follows:

```text
PH2/
├── Images/
│   ├── train/
│   └── test/
├── ProcessedSegmentationClass/
│   ├── train/
│   └── test/
└── ImageSets/
    ├── train.txt
    └── test.txt
```

Each split file contains one image identifier per line. Identifiers may include an extension. Images and masks must use
the same identifier and spatial dimensions. For all five backbones, use image dimensions divisible by 32 (the released
PH2 preprocessing uses 512 x 512 inputs).

## BraTS2020

The loader accepts either a directory containing the subject folders directly or the standard nested training-data
layout. Each usable subject must contain the selected modality and segmentation files:

```text
BraTS2020/
└── MICCAI_BraTS2020_TrainingData/
    ├── BraTS20_Training_001/
    │   ├── BraTS20_Training_001_flair.nii.gz
    │   └── BraTS20_Training_001_seg.nii.gz
    └── ...
```

Subjects are split reproducibly into training and validation subsets using `--seed` and `--brats-val-ratio`. By
default, the middle axial slice is used. Select another slice with `--brats-slice-index`.

## COVID-19-CT

Arrange NIfTI volumes and split files as follows:

```text
COVID-19-CT/
├── train/
│   ├── images/
│   └── masks/
├── test/
│   ├── images/
│   └── masks/
└── ImageSets/
    ├── train.txt
    └── test.txt
```

Image and mask files must have matching identifiers. With the default `--covid-slice-index -1`, every axial slice is
used. For 4D input, `--covid-time-index -1` selects the middle frame.
