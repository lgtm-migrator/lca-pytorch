
# PyTorch Implementation of the LCA Sparse Coding Algorithm

LCA-PyTorch (lcapt) provides the ability to flexibly build single- or multi-layer sparse coding networks in PyTorch. 
This package implements the [Locally Competitive Algorithm (LCA)](https://www.ece.rice.edu/~eld1/papers/Rozell08.pdf), which performs sparse coding by modeling the feature specific lateral competition observed throughout many
different sensory areas in the brain, including the [visual cortex](https://www.nature.com/articles/s41586-019-0997-6).
Feature specific lateral competition is where neurons compete with their neighbors to represent a shared input based on their receptive field similarity.
This is a discrete implementation, but LCA can be implemented in [analog circuits](https://patentimages.storage.googleapis.com/30/8f/6e/5d9da903f0d635/US7783459.pdf) and
multiple neuromorphic chips such as [IBM's TrueNorth](https://www.frontiersin.org/articles/10.3389/fnins.2019.00754/full) and 
[Intel's Loihi](https://ieeexplore.ieee.org/abstract/document/9325356?casa_token=0kxjP50T3IIAAAAA:EOCnIf4-fMYowF7HgTLo0UQyKLWbrWW7VnOT1TZ2DI0U_cUCBYBQv1GN8r49LtISezWQ--A). This package allows for the creation of convolutional LCA layers which
maintain all of the functionality present in typical PyTorch layers.

## Installation  

To install LCA-Pytorch via pip, run the following command:

```
pip install git+https://github.com/MichaelTeti/lca-pytorch.git
```

Alternatively, clone the repository and install the package manually:

```
git clone git@github.com:MichaelTeti/lca-pytorch.git
cd lca-pytorch
pip install .
```

## LCA Parameters

Below is a mapping between the variable names used in this implementation and the symbols used in [Rozell et al.'s](https://www.ece.rice.edu/~eld1/papers/Rozell08.pdf) formulation of LCA.

| **LCA-PyTorch Variable Name** | **Rozell Symbol** | **Description** |
| --- | --- | --- |
| input_drive | *b(t)* | Drive from the inputs/stimulus |
| states | *u(t)* | Internal state/membrane potential |
| acts | *a(t)* | Code/Representation or External Communication |
| <img src="https://latex.codecogs.com/svg.image?\large&space;\bg{black}\lambda" title="https://latex.codecogs.com/svg.image?\large \bg{white}\lambda" /> | <img src="https://latex.codecogs.com/svg.image?\large&space;\bg{white}\lambda" title="https://latex.codecogs.com/svg.image?\large \bg{white}\lambda" /> | Transfer function threshold value |
| weights | <img src="https://latex.codecogs.com/svg.image?\large&space;\bg{black}\Phi" title="https://latex.codecogs.com/svg.image?\large \bg{black}\Phi" /> | Dictionary/Features |
| inputs | *s(t)* | Input data |
| recons | <img src="https://latex.codecogs.com/svg.image?\hat{s}(t)" title="https://latex.codecogs.com/svg.image?\hat{s}(t)" /> | Reconstruction of the input *s(t)* |

## Examples

Dictionary Learning
  * [Dictionary Learning on Cifar-10 Images](https://github.com/MichaelTeti/lca-pytorch/blob/main/examples/builtin_dictionary_learning_cifar.ipynb)
