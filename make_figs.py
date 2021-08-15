from argparse import ArgumentParser 
from math import sqrt
import os

import h5py
import imageio
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import torch
from torchvision.utils import make_grid 

parser = ArgumentParser()
parser.add_argument('result_dir', type=str)
args = parser.parse_args()

with h5py.File(os.path.join(args.result_dir, 'tensors.h5'), 'r') as h5f:
    keys = list(h5f.keys())
    latest_ckpt = str(max([int(k.split('_')[-1]) for k in keys]))
    acts = h5f['a_' + latest_ckpt][()]
    feats = torch.from_numpy(h5f['D_' + latest_ckpt][()])
    recons = torch.from_numpy(h5f['recon_' + latest_ckpt][()])
    inputs = torch.from_numpy(h5f['input_' + latest_ckpt][()])

sparsity = (acts != 0).mean(axis=(0, 2, 3, 4))
plt.plot(np.sort(sparsity)[::-1])
plt.xlabel('Feature Index')
plt.ylabel('Fraction Active')
plt.savefig(os.path.join(args.result_dir, 'sparsity.png'))
plt.close()

sorted_inds = [ind for _, ind in sorted(zip(sparsity, list(range(sparsity.size))), reverse=True)]
feats = feats[sorted_inds]

fgrids = [make_grid(feats[:, :, t], nrow=int(sqrt(feats.shape[0]))) for t in range(feats.shape[2])]
rgrids = [make_grid(recons[:, :, t], nrow=int(sqrt(recons.shape[0])), normalize=True, pad_value=0, scale_each=True) for t in range(recons.shape[2])]
igrids = [make_grid(inputs[:, :, t], nrow=int(sqrt(inputs.shape[0])), normalize=True, pad_value=0, scale_each=True) for t in range(inputs.shape[2])]

imageio.mimwrite(os.path.join(args.result_dir, 'feats.gif'), [g[0] for g in fgrids], fps=5)
imageio.mimwrite(os.path.join(args.result_dir, 'recons.gif'), [g[0] for g in rgrids], fps=5)
imageio.mimwrite(os.path.join(args.result_dir, 'inputs.gif'), [g[0] for g in igrids], fps=5)