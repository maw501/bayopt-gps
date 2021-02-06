<h2 align="center">Bayesian optimization of NNs using GPyOpt and PyTorch</h2>

<div align="center">
  <!--Python version -->
  <a href="https://www.python.org/downloads/release/python-380/">
    <img src="https://img.shields.io/badge/python-3.8-blue.svg"
      alt="Python version" />
  </a>
  <!--Commits  -->
  <a href="https://github.com/maw501/bayopt-gps/commits/master">
    <img src="https://img.shields.io/github/last-commit/maw501/bayopt-gps.svg"
      alt="Status version" />
  </a>
</div>
<br />

## Overview

A simple but practical example of how to use Bayesian optimization with Gaussian processes to tune a neural network using PyTorch and GPyOpt.

## Getting started

Clone the repository then create the conda environment:

```
git clone git@github.com:maw501/bayopt-gps.git
cd bayopt-gps
conda env create -f environment.yml
```

In order to use the conda environment in a notebook run:

```
python -m ipykernel install --user --name=bayopt
```

The version of [torch](https://pytorch.org/) installed is CPU only and training takes a few minutes per epoch on 4 cores depending on the parameters chosen by the Bayesian optimization.

## Data

The data will be automatically downloaded when training is run for the first time and stored in a `data` directory under the root of the repository.

## Example notebook

There is currently a notebook which walks through the training process and shows how to set-up the objective function: [Using GPyOpt to tune a CNN on MNIST](https://nbviewer.jupyter.org/github/maw501/bayopt-gps/blob/master/notebooks/Using_GPyOpt_to_tune_NN.ipynb).
