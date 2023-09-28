# S5-PTD

This is the code repository accompanying the S5-PTD part of the manuscript ''Robustifying State-space Models for Long Sequences via Approximate Diagonalization." The repository is heavily adapted from the ''S5" GitHub repository (https://github.com/lindermanlab/S5.git). While it contains references to existing papers and code repositories, it includes no information that reveals the identities of the manuscript authors.


## Setup

### Requirements
To run the code on your own machine, run either `pip install -r requirements_cpu.txt` or `pip install -r requirements_gpu.txt`. Run from within the root directory `pip install -e .` to install the package. 


### Data
Downloading the raw data is done differently for each dataset. The following datasets require no action:
- Text (IMDb)
- Image (Cifar black & white)
- sMNIST
- psMNIST
- Cifar (Color)

The remaining datasets need to be manually downloaded.  To download _everything_, run `./bin/download_all.sh`.  This will download quite a lot of data and will take some time.  

Below is a summary of the steps for each dataset:
- ListOps: run `./bin/download_lra.sh` to download the full LRA dataset.  
- Retrieval (AAN): run `./bin/download_aan.sh`
- Pathfinder: run `./bin/download_lra.sh` to download the full LRA dataset.
- Path-X: run `./bin/download_lra.sh` to download the full LRA dataset.


## Execution
The Long-Range Arena benchmarks can be tested by running the script
```
./bin/run_experiments/run_lra_foo.sh
```
where `foo` is the name of the problem, choosing from `listops`, `imdb`, `aan`, `cifar`, `pathfinder`, and `pathx`. In this repository, we included pre-calculated perturbation matrices of different sizes. Since the LTI systems in each task have a different size, one needs to change the file name in the `make_DPLR` function in [./s5/ssm_init.py](./s5/ssm_init.py). One can also create their own perturbation matrices and feed them into the `make_DPLR` function. Note that to reproduce the `pathx` result, one needs to decrease the batch size. So far, we have not implemented a procedure for automatically changing the batch size. This can be done, however, by storing and restoring checkpoints, as implemented in [./s5/train.py](./s5/train.py). For the `cifar` task, it is better to initialize the matrix B to be exactly the HiPPO-LegS matrix. Hence, one needs to use the `else` statement in the function `init_VinvB` in [./s5/ssm_init.py](./s5/ssm_init.py). This is hard-coded for now and will be revised later.