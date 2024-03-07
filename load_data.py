import numpy as np

import jax.numpy as jnp
from jax.tree_util import tree_map

from torch.utils.data import Dataset, DataLoader, default_collate

def numpy_collate(batch):
    return tree_map(jnp.asarray, default_collate(batch))

class LorenzDataset(Dataset):
    def __init__(self, path: str, seq_len: int = 100):
        super().__init__()
        self.data = np.load(path)
        window = np.lib.stride_tricks.sliding_window_view(self.data, (seq_len,), axis=0) # ( # , DIM , SEQ_LEN )
        self.window = np.swapaxes(window, 1, 2) # ( # , SEQ_LEN, DIM )
        self.seq_len = seq_len

    def __len__(self):
        return len(self.window)

    def __getitem__(self, index):
        src = self.window[index]
        tgt_index = (index + 1) % len(self)
        tgt = self.window[tgt_index]
        return src, tgt

def get_datasets(seq_len: int = 100):
    l63tr = LorenzDataset("./data/lorenz63_train.npy", seq_len=seq_len)
    l63te = LorenzDataset("./data/lorenz63_test.npy", seq_len=seq_len)
    l96tr = LorenzDataset("./data/lorenz96_train.npy", seq_len=seq_len)
    l96te = LorenzDataset("./data/lorenz96_test.npy", seq_len=seq_len)
    return l63tr, l63te, l96tr, l96te

def get_dataloaders(batch_size: int = 32, shuffle: bool = False):
    l63tr, l63te, l96tr, l96te = get_datasets()
    loader63tr = DataLoader(l63tr, batch_size=batch_size, shuffle=shuffle, collate_fn=numpy_collate)
    loader63te = DataLoader(l63te, batch_size=batch_size, shuffle=shuffle, collate_fn=numpy_collate)
    loader96tr = DataLoader(l96tr, batch_size=batch_size, shuffle=shuffle, collate_fn=numpy_collate)
    loader96te = DataLoader(l96te, batch_size=batch_size, shuffle=shuffle, collate_fn=numpy_collate)
    return loader63tr, loader63te, loader96tr, loader96te
