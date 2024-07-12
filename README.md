# Fast and high-fidelity whole-event simulation in high-energy heavy-ion experiments

<p align="center">
  <img src="https://github.com/LS4GAN/gallery/blob/main/calo-ddpm/event_display_cent0.jpg" width="35%" title="Centrality 0-10% Event">
  <img src="https://github.com/LS4GAN/gallery/blob/main/calo-ddpm/event_display_cent4.jpg" width="35%" title="Centrality 40-50% Event">
</p>

This package provides implementations of GAN and DDPM/DDIM models used in the
["Effectiveness of denoising diffusion probabilistic models for fast and
high-fidelity whole-event simulation in high-energy heavy-ion
experiments"](https://arxiv.org/abs/2406.01602) paper. The instructions below
describe how to setup this package, train generative models, and synthesize
new data.


# Installation & Requirements

## Environment

This code was developed and tested in the official `pytorch` container
`pytorch_1.12.1-cuda11.3-cudnn8-runtime`. An environment similar to that
container can be set up with conda, using the provided configuration:

```bash
conda env create -f contrib/conda_env.yaml
```

NOTE: this environment was tested only on Linux machines.

## Requirements

`calo-ddpm` relies on the reference implementation of the iDDPM architecture
by [OpenAI](https://github.com/openai/improved-diffusion).
[improved-diffusion](https://github.com/openai/improved-diffusion) package
needs to be manually installed inside of the created environment.
We used commit 783b6740edb79fdb7d063250db2c51cc9545dcd1 in our work.

## Installation

Finally, to install the `calo-ddpm` package, run the following command

```bash
python3 setup.py develop --user
```

## Environment Variables

By default, `calo-ddpm` will search for data in the `./data` directory and
store trained models in the `./outdir` directory. If one wants to change
this behavior, modify the following environment variables:

```
export JETGEN_DATA=PATH_TO_DATA
export JETGEN_OUTDIR=PATH_TO_OUTDIR
```

# Training

<p align="center">
  <img src="https://github.com/LS4GAN/gallery/blob/main/calo-ddpm/diffusion_process.jpg" width="75%" title="Diffusion Process">
</p>

NOTE: Due to the sPhenix collaboration policies, we are unable to share the
training dataset outside the sPhenix collaboration.

In this section, we describe the following:
1. How to obtain pre-trained models
2. How to prepare your own dataset for training.
3. How to train DDPM/GAN models using the official sPhenix dataset, or custom
   data.

## 1. Obtaining Pre-Trained Models

The pre-trained GAN and DDPM models have been uploaded to
[Zenodo](https://zenodo.org/records/12535659).
One can download them with the help of the provided convenience script
`./scripts/download_model.sh` .

## 2. Using your own dataset

To train the DDPM/GAN models on your own dataset, you can take one of the
available training scripts as a starting point (e.g.
`scripts/train/sphenix/train_cent0_dcgan.py`
or
`scripts/train/sphenix/train_cent0_ddpm.py`
).
These scripts describe the training configuration, which should be
straightforward to navigate.

Next, you would need to prepare your dataset to match the format that
`calo-ddpm` expects or write an alternative pytorch dataset implementation.
By default, `calo-ddpm` expects the dataset to be packed into `hdf5` files
and arranged in the following directory structure:

```
DATASET/
    train/
        DOMAIN.h5
    val/            # optional
        DOMAIN.h5
    test/           # optional
        DOMAIN.h5
```

where `DATASET` and `DOMAIN` are arbitrary names (make sure to change the
`path` and `domain` fields of the training configuration to match).

`DOMAIN.h5` is an HDF5 file containing the dataset. The dataset should be saved
in an `hdf5` dataset called `data`. The `data` should have a shape of
`(N, H, W, C)` or `(N, H, W)`, where *N* is the number of samples in the
dataset, *(H, W)* spatial dimensions of the data samples, and *C* is the number
of channels.


## 3. Trainining Models

To train the GAN/DDPM models, one can run one of the following scripts:

```
scripts/train/sphenix/train_cent0_dcgan.py
scripts/train/sphenix/train_cent0_ddpm.py
scripts/train/sphenix/train_cent4_dcgan.py
scripts/train/sphenix/train_cent4_ddpm.py
```

These scripts contain the default training configurations used in the paper.
Once the models are trained, they will be saved in the `./outdir` directory
(or `JETGEN_OUTDIR`).

Note: by default, the training will attempt to use all the available GPUs.
To bind the training to a single GPU -- set `CUDA_VISIBLE_DEVICES` environment
variable to the index of the desired GPU.


# Data Generation

`calo-ddpm` provides two scripts `scripts/eval_dp.py` and `scripts/eval_gan.py`
to generate new data with Diffusion Models and with GANs respectively.
For example, if one has a trained DDPM model, one can generate new data by
running:

```
python3 scripts/eval_dp.py -n N_SAMPLES_TO_GENERATE --domain 0 PATH_TO_TRAINED_MODEL
```

Run `eval_dp.py --help` to see the additional generation options.


# LICENSE

This package is distributed under `BSD-2` license.

`calo-ddpm` repository contains some code (primarily in jetgen/base
subdirectory) from
[pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).
This code is also licensed under `BSD-2`
(please refer to `jetgen/base/LICENSE` for details).

Each code snippet that was taken from `pytorch-CycleGAN-and-pix2pix` has a note
about proper copyright attribution.

