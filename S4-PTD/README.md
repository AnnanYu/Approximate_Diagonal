# S4-PTD

This is the code repository accompanying the S4-PTD part of the manuscript ''Robustifying State-space Models for Long Sequences via Approximate Diagonalization." The repository is heavily adapted from the ''state-spaces" GitHub repository (https://github.com/HazyResearch/state-spaces.git). While it contains references to existing papers and code repositories, it includes no information that reveals the identities of the manuscript authors.

## Setup

### Requirements
This repository requires Python 3.9+ and Pytorch 1.10+.
It has been tested up to Pytorch 1.13.1.
Other packages are listed in [requirements.txt](./requirements.txt).
Some care may be needed to make some of the library versions compatible, particularly torch/torchvision/torchaudio/torchtext.

Example installation:
```
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
pip install -r requirements.txt
```

### Data

Basic datasets are auto-downloaded, including MNIST, CIFAR, and Speech Commands.
All logic for creating and loading datasets is in [src/dataloaders](./src/dataloaders/) directory.
The README inside this subdirectory documents how to download and organize other datasets.

### Configs and Hyperparameters

Configurations can be changed in [configs/experiment/lra](./configs/experiment/lra/).

### WandB

Logging with [WandB](https://wandb.ai/site) is built into this repository.
In order to use this, simply set your `WANDB_API_KEY` environment variable, and change the `wandb.project` attribute of [configs/config.yaml](configs/config.yaml) (or pass it on the command line e.g. `python -m train .... wandb.project=s4`).

Set `wandb=null` to turn off WandB logging.

## Execution

### Ablation Study

The ablation study of the S4-PTD model can be reproduced using the command
```
python ablation.py --epochs 200 --delta 1 --batch_size 64 --n_layers 6 --d_model 512 --weight_decay 0.01 --dropout 0.1
```
One can change the value of `delta` to obtain perturbation matrices of different sizes.

### Robustness Test

The robustness test can be reproduced using the command
```
python robustness.py
```
One can change the perturbation matrix of the S4-PTD model by replacing `ptbHiPPO32.mat` with a different perturbation matrix. In particular, one can recover the S4D model by setting `ptbHiPPO32.mat` to be the rank-1 part of the HiPPO-LegS matrix.

### LRA Benchmarks

The Long-Range Arena benchmarks can be tested by running
```
python -m train experiment=lra/s4-PTD-foo
```
where `foo` is the name of the problem, choosing from `listops`, `imdb`, `aan`, `cifar`, `pathfinder`, and `pathx`.



