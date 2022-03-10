
# PyTorch Implementation of the LCA Sparse Coding Algorithm

LCA-PyTorch (lcapt) provides the ability to build single- or multi-layer sparse coding networks in PyTorch. 
Sparse coding, which was inspired by the sparse activity of neurons in the visual cortex, is a method
which aims to represent a given input with only a few features. This package implements sparse coding
via the Locally Competitive Algorithm (LCA) ([Rozell et al. 2008](https://www.ece.rice.edu/~eld1/papers/Rozell08.pdf)),
which solves the sparse coding objective by modeling the feature specific lateral competition observed throughout many
different sensory areas in the brain, including the visual cortex ([Chettih et al. 2019](https://www.nature.com/articles/s41586-019-0997-6)).
LCA has been implemented in analog circuits ([Rozell et al. 2010](https://patentimages.storage.googleapis.com/30/8f/6e/5d9da903f0d635/US7783459.pdf)) and
multiple neuromorphic chips such as IBM's TrueNorth ([Fair et al. 2019](https://www.frontiersin.org/articles/10.3389/fnins.2019.00754/full)) and 
Intel's Loihi ([Henke et al. 2020](https://ieeexplore.ieee.org/abstract/document/9325356?casa_token=0kxjP50T3IIAAAAA:EOCnIf4-fMYowF7HgTLo0UQyKLWbrWW7VnOT1TZ2DI0U_cUCBYBQv1GN8r49LtISezWQ--A)).
It was originally formulated as vectorized implementation for single image patches, and as a result, many implementations
of LCA are designed for this scenario (i.e. non-convolutional implementation). This package allows for the creation of convolutional LCA layers which
maintain all of the functionality present in typical PyTorch layers.
  
## Installation  

To install lcapt via pip, run the following command:

```
pip install git+https://github.com/MichaelTeti/lca-pytorch.git
```

Alternatively, clone the repository and install the package manually:

```
git clone git@github.com:MichaelTeti/lca-pytorch.git
cd lca-pytorch
pip install .
```
