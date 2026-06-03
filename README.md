# EEMSAGAN: Edge-Enhanced Multi-Scale Attention Generative Adversarial Network for Remote Sensing Image Super-Resolution

## Environment Installation

```bash
pip install -r requirements.txt
```

## Dataset Preparation

The training set should be placed in the 'datasets/train_HR' folder, the high-resolution validation set in 'datasets/val_HR', and the degraded low-resolution validation set in 'datasets/val_LR'.

You need to use the [generate_meta_info.py](generate_meta_info.py) script to generate meta information for the training set.

```bash
 python generate_meta_info.py --input datasets/train_HR --root datasets --meta_info datasets/meta_info/meta_info_RemoteSensing.txt
```

## Train EEMSAGAN

Our experiment is implemented using the Basicsr framework. First, we need to import the edge loss function into the 'basicsr/losses' directory to integrate it successfully into the model.

### Train Generator

1. Modify the contents of the configuration file 'options/train_eemsanet_x4plus.yml' accordingly based on the experiment.

2. Train on a GPU:

    ```bash
    python eemsagan/train.py -opt options/train_eemsanet_x4plus.yml
    ```

### Train EEMSAGAN

1. Modify the contents of the configuration file 'options/train_eemsagan_x4plus.yml' accordingly based on the experiment.

2. Train on a GPU:

    ```bash
    python eemsagan/train.py -opt options/train_eemsagan_x4plus.yml
    ```
