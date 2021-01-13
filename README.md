<h2 align="center">Bayesian optimization of NNs using GPyOpt and PyTorch</h2>

<div align="center">
  <!--Python version -->
  <a href="https://www.python.org/downloads/release/python-360/">
    <img src="https://img.shields.io/pypi/pyversions/fastai.svg"
      alt="Python version" />
  </a>
  <!--Project status -->
  <a href="https://github.com/maw501/bayopt-gps">
    <img src="https://img.shields.io/badge/Status-Under%20development-green.svg"
      alt="Status version" />
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
### Requirements

The main dependencies are:
* Gpy==1.9.6
* GPyOpt==1.2.5
* PyTorch==1.1

The Gpy and GPyOpt packages can be directly installed via conda:

```
conda install -c conda-forge gpy
conda install -c kgullikson gpyopt
```

PyTorch can also be installed via conda though if you don't have it installed already it's worth referring to their [getting started](https://pytorch.org/get-started/locally/) instructions.

### Running the example

Then clone the repo and run the example which will automatically download the MNIST dataset.

```
git clone https://github.com/maw501/bayopt-gps.git
cd bayopt-gps
python3 mnist_example.py
```

## Notebooks
1. [Introduction: using GPyOpt to tune a CNN on MNIST](https://nbviewer.jupyter.org/github/maw501/bayopt-gps/blob/master/notebooks/Using_GPyOpt_to_tune_NN.ipynb) - an introductory notebook walking through `mnist_example.py` in a little more detail.
