
# PyTorch Implementation of the LCA Sparse Coding Algorithm

[![tests](https://github.com/MichaelTeti/lca-pytorch/actions/workflows/build.yml/badge.svg)](https://github.com/MichaelTeti/lca-pytorch/actions/workflows/build.yml)
[![codecov](https://codecov.io/gh/MichaelTeti/lca-pytorch/branch/main/graph/badge.svg?token=4EPI05G5CY)](https://codecov.io/gh/MichaelTeti/lca-pytorch)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/MichaelTeti/lca-pytorch.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/MichaelTeti/lca-pytorch/context:python)
[![Total alerts](https://img.shields.io/lgtm/alerts/g/MichaelTeti/lca-pytorch.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/MichaelTeti/lca-pytorch/alerts/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

LCA-PyTorch (lcapt) provides the ability to flexibly build single- or multi-layer convolutional sparse coding networks in PyTorch with the [Locally Competitive Algorithm (LCA)](https://www.ece.rice.edu/~eld1/papers/Rozell08.pdf). LCA-Pytorch currently supports 1D, 2D, and 3D convolutional LCA layers, which maintain all the functionality and behavior of PyTorch convolutional layers. We currently do not support Linear (a.k.a. fully-connected) layers, but it is possible to implement the equivalent of a Linear layer with convolutions.  

LCA is a neuroscientific model that performs sparse coding by modeling the feature specific lateral competition observed throughout many different sensory areas in the brain, including the [visual cortex](https://www.nature.com/articles/s41586-019-0997-6). Under lateral competition, neurons with overlapping receptive fields compete to represent a shared portion of the input. This is a discrete implementation, but LCA can also be implemented in [analog circuits](https://patentimages.storage.googleapis.com/30/8f/6e/5d9da903f0d635/US7783459.pdf) and neuromorphic chips, such as [IBM's TrueNorth](https://www.frontiersin.org/articles/10.3389/fnins.2019.00754/full) and [Intel's Loihi](https://ieeexplore.ieee.org/abstract/document/9325356?casa_token=0kxjP50T3IIAAAAA:EOCnIf4-fMYowF7HgTLo0UQyKLWbrWW7VnOT1TZ2DI0U_cUCBYBQv1GN8r49LtISezWQ--A).

## Installation  

### Pip Installation

```
pip install git+https://github.com/MichaelTeti/lca-pytorch.git
```

### Manual Installation

```
git clone git@github.com:MichaelTeti/lca-pytorch.git
cd lca-pytorch
pip install .
```

## LCA Parameters

Below is a mapping between the variable names used in this implementation and the symbols used in [Rozell et al.'s](https://www.ece.rice.edu/~eld1/papers/Rozell08.pdf) formulation of LCA.

| **LCA-PyTorch Variable** | **Rozell Variable** | **Description** |
| --- | --- | --- |
| input_drive | *b(t)* | Drive from the inputs/stimulus |
| states | *u(t)* | Internal state/membrane potential |
| acts | *a(t)* | Code/Representation or External Communication |
| <img src="https://latex.codecogs.com/svg.image?\large&space;\bg{black}\lambda" title="https://latex.codecogs.com/svg.image?\large \bg{white}\lambda" /> | <img src="https://latex.codecogs.com/svg.image?\large&space;\bg{white}\lambda" title="https://latex.codecogs.com/svg.image?\large \bg{white}\lambda" /> | Transfer function threshold value |
| weights | <img src="https://latex.codecogs.com/svg.image?\large&space;\bg{black}\Phi" title="https://latex.codecogs.com/svg.image?\large \bg{black}\Phi" /> | Dictionary/Features |
| inputs | *s(t)* | Input data |
| recons | <img src="https://latex.codecogs.com/svg.image?\hat{s}(t)" title="https://latex.codecogs.com/svg.image?\hat{s}(t)" /> | Reconstruction of the input *s(t)* |

## Examples

  * Dictionary Learning Using Built-In Update Method
    * [Dictionary Learning on Cifar-10 Images](https://github.com/MichaelTeti/lca-pytorch/blob/main/examples/builtin_dictionary_learning_cifar.ipynb)
    * [Dictionary Learning on Google Speech Commands Audio](https://github.com/MichaelTeti/lca-pytorch/blob/main/examples/builtin_dictionary_learning_speech_commands.ipynb)  
  
  * Dictionary Learning Using PyTorch Optimizer  
    * [Dictionary Learning on Cifar-10 Images](https://github.com/MichaelTeti/lca-pytorch/blob/main/examples/pytorch_optim_dictionary_learning_cifar.ipynb)
