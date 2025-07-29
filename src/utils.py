import torch
import torch.nn as nn
from torchprofile import profile_macs
import numpy as np
from matplotlib import pyplot as plt

def get_model_macs(model, inputs) -> int:
    return profile_macs(model, inputs)

def get_sparsity(tensor: torch.Tensor) -> float:
    """
    calculate the sparsity of the given tensor
        sparsity = #zeros / #elements = 1 - #nonzeros / #elements
    """
    return 1 - float(tensor.count_nonzero()) / tensor.numel()


def get_model_sparsity(model: nn.Module) -> float:
    """
    calculate the sparsity of the given model
        sparsity = #zeros / #elements = 1 - #nonzeros / #elements
    """
    num_nonzeros, num_elements = 0, 0
    for param in model.parameters():
        num_nonzeros += param.count_nonzero()
        num_elements += param.numel()
    return 1 - float(num_nonzeros) / num_elements

def get_num_parameters(model: nn.Module, count_nonzero_only=False) -> int:
    """
    calculate the total number of parameters of model
    :param count_nonzero_only: only count nonzero weights
    """
    num_counted_elements = 0
    for param in model.parameters():
        if count_nonzero_only:
            num_counted_elements += param.count_nonzero()
        else:
            num_counted_elements += param.numel()
    return num_counted_elements


def get_model_size(model: nn.Module, data_width=32, count_nonzero_only=False) -> int:
    """
    calculate the model size in bits
    :param data_width: #bits per element
    :param count_nonzero_only: only count nonzero weights
    """
    return get_num_parameters(model, count_nonzero_only) * data_width

def norm_value(val, type='min_max'):
    if type == 'min_max':
        norm_val = (val - val.min()) / (val.max() - val.min() + 1e-9)
    elif type == 'zero_mean':
        norm_val = (val - val.mean()) / (val.std() + 1e-9)
    else:
        raise ValueError
    return norm_val

def plot_wanda_ent(wanda, ent, layer_idx, wanda_idx=None, ent_idx=None, mag_ent_idx=None, pr=0, a=0):
    """
    Scatter: x=wanda, y=ent.
    Highlight wanda_idx (red), ent_idx (blue), base points gray.

    Args:
        wanda: 1D torch.Tensor or np.ndarray (already normalized).
        ent:   1D torch.Tensor or np.ndarray (already normalized; same length as wanda).
        wanda_idx: indices to highlight in red.  int | seq[int] | torch.Tensor | np.ndarray | None
        ent_idx:   indices to highlight in blue. int | seq[int] | torch.Tensor | np.ndarray | None
        title: optional str.
        figsize: tuple.

    Returns:
        (fig, ax) matplotlib figure and axes.
    """

    # ---- to numpy ----
    def _to_np(x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        else:
            x = np.asarray(x)
        return x

    wanda = _to_np(wanda).reshape(-1)
    ent   = _to_np(ent).reshape(-1)
    wanda_idx = _to_np(wanda_idx)
    ent_idx = _to_np(ent_idx)
    mag_ent_idx = _to_np(mag_ent_idx)
    assert wanda.shape == ent.shape, "wanda and ent must be same length."
    n = wanda.shape[0]

    wanda_idx = np.asarray(wanda_idx, dtype=int).ravel() if wanda_idx is not None else []
    ent_idx   = np.asarray(ent_idx, dtype=int).ravel() if ent_idx is not None else []
    mag_ent_idx = np.asarray(mag_ent_idx, dtype=int).ravel() if mag_ent_idx is not None else []

    mask_all = np.ones(n, dtype=bool)
    mask_all[:] = True

    mask_wanda = np.zeros(n, dtype=bool)
    mask_wanda[wanda_idx] = True

    mask_ent = np.zeros(n, dtype=bool)
    mask_ent[ent_idx] = True
    
    mask_mag_ent = np.zeros(n, dtype=bool)
    mask_mag_ent[mag_ent_idx] = True

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(4, 3), dpi=300)

    ax.scatter(wanda[~(mask_wanda | mask_ent | mask_mag_ent)], 
               ent[~(mask_wanda | mask_ent | mask_mag_ent)], 
               s=10, c='#999999', alpha=0.5, label='Unpruned Neurons', marker='o')

    # magnitude 
    if mask_wanda.any():
        ax.scatter(wanda[mask_wanda], ent[mask_wanda], marker='s', 
                   s=10, c='#FF7F0E', alpha=0.7, label='Pruned by Magnitude')

    # entropy 
    if mask_ent.any():
        ax.scatter(wanda[mask_ent], ent[mask_ent], marker='^',
                   s=10, c='#1F77B4', alpha=0.7, label='Pruned by Entropy')

    # magent 
    if mask_mag_ent.any():
        ax.scatter(wanda[mask_mag_ent], ent[mask_mag_ent], marker='v',
                   s=10, c='#2CA02C', alpha=0.8, label='Pruned by Magent')

    ax.set_xlabel('Magnitude Score', fontsize=9)
    ax.set_ylabel('Entropy Score', fontsize=9)
    ax.set_title(f'Magnitude vs Entropy Distribution for InternVL3-2B Layer {layer_idx}', fontsize=9)

    ax.legend(loc='best', frameon=False, fontsize=9)
    ax.grid(alpha=0.3)

    fig.savefig(f"img/{pr}/magent_db_l{layer_idx}_a{a}.png", dpi=300, bbox_inches='tight')

    plt.close()