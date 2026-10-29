import os
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, TensorDataset
from xautodl.datasets import get_datasets

from config import DATA_ROOTS


@dataclass
class Loaders:
    calib: DataLoader
    search: DataLoader
    test: DataLoader
    input_shape: list          # e.g. [1, 3, 16, 16]


def _materialize(ds, indices):
    """Apply transforms once and keep the result in memory."""
    xs, ys = zip(*(ds[int(i)] for i in indices))
    return TensorDataset(torch.stack(xs), torch.as_tensor(ys))


def build_loaders(args) -> Loaders:
    root = os.path.join(args.datapath, DATA_ROOTS[args.dataset])
    train, test, xshape, _ = get_datasets(args.dataset, root, cutout=0)
    train.transform = test.transform                    # no augmentation

    perm = torch.randperm(len(train), generator=torch.Generator().manual_seed(0))
    calib = _materialize(train, perm[:args.n_calib])                               # disjoint
    search = _materialize(train, perm[args.n_calib:args.n_calib + args.n_search])  # subsets
    test = _materialize(test, range(len(test)))

    return Loaders(
        calib=DataLoader(calib, batch_size=args.calib_bs, shuffle=False),
        search=DataLoader(search, batch_size=args.eval_bs, shuffle=False),
        test=DataLoader(test, batch_size=args.eval_bs, shuffle=False),
        input_shape=list(xshape),
    )


def collate_x_only(batch):
    """Loaders yield (images, labels); PPQ only needs the images."""
    return batch[0] if isinstance(batch, (list, tuple)) else batch