<h1 align="center">bayopt-gps</h1>

<!--NPM version
<div align="center">
  :steam_locomotive::train::train::train::train::train:
</div>
-->

<div align="left">

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
* Gpy=1.9.6
* GPyOpt=1.2.5
* PyTorch>=1.0.1

all can be installed via conda.

### Running the example

Then clone the repo and run the example which takes 2 arguments:

1. number of epochs
2. number of iterations to run the Bayesian optimization for

```
git clone https://github.com/maw501/bayopt-gps.git
cd bayopt-gps
python3 mnist_simple 5 5
```

Note: 5 initial iterations of the Bayesian optimization are run for exploratory purposes on top of those specified.

## TODO

* Saving and loading models based on best results
* Loading previous results and continuing
* Adding extra parameters after some initial training
