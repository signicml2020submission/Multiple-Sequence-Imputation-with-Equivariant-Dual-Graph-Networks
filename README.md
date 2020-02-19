# Multiple Sequence Imputation with Equivariant Dual Graph Networks
This repository is the official PyTorch implementation of "Multiple Sequence Imputation with Equivariant Dual Graph Networks".

## Requirements
- `pytorch` > 1.2
- `dgl`: [Deep Graph Library](https://www.dgl.ai)
- `tensorboard`
- `scikit-learn`
- `tqdm`

## Data

Download `dataset.tar.gz` from [Google Drive](https://drive.google.com/file/d/1FRimRINjSz4FiGIqo8KVLn_AQiJDTI37/view?usp=sharing). Then,
```bash
tar -xzvf dataset.tar.gz -C /path/to/the/root/of/project
```

## Run

### Sequence Imputation

Configuration files are under `yaml/exp`.

#### SIGN, GRU
```bash
# SIGN
## NBA dataset
python main.py -f yaml/exp/SNATA-nba-20-8-0_1-noself.yaml --gpu 0 --mode train
## weather station dataset
python main.py -f yaml/exp/SNATA-noaa-8-0_1-noself.yaml --gpu 0 --mode train

# GRU
## NBA dataset
python main.py -f yaml/exp/GRU-NBA-20-8-0_1.yaml --gpu 0 --mode train
## weather station dataset
python main.py -f yaml/exp/GRU-noaa-8-0_1.yaml --gpu 0 --mode train
```

#### h-hop, kNN, Linear, MissForest(ExtraTree)

```bash
# models run on CPU and do not require training data
python main.py -f yaml/exp/{config_filename} --gpu -1 --mode test
```

### Robustness to the Missing Rate

Configuration files are under `yaml/ablation_missing_rate`.

### Imputation to Newly Added Sequences

Configuration files are under `yaml/ablation_add-sequences`.

### Spatial Interpolation and Extrapolation

Configuration files are under `yaml/ablation_space_inter_extra-polation`.