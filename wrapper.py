import torch
import torch.nn as nn
import numpy as np
from scipy.stats import gaussian_kde

def find_layers(module, layers=[nn.Linear], name='', prune='mlp'):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers and prune in name:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def find_sub_layers(module, layers=[nn.Linear], name='', prune_layer='mlp'):
    if type(module) in layers and prune_layer in name:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

class WrappedLayer:
    def __init__(self, layer, layer_id=0, layer_name='none', total_sample=30, num_bins=100):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.cols = layer.weight.data.shape[1]
        self.num_bins = num_bins
        
        self.x_norm_l1 = torch.zeros(self.cols, device=self.dev)
        self.x_norm_l2 = torch.zeros(self.cols, device=self.dev)

        self.min_vals = torch.full((self.cols,), float('inf'), device=self.dev)
        self.max_vals = torch.full((self.cols,), float('-inf'), device=self.dev)
        self.activation_histograms = torch.zeros((self.cols, self.num_bins), device=self.dev)
        self.input_matrix = torch.zeros((self.cols, total_sample), device=self.dev)

        self.nsamples = 0
        self.layer_id = layer_id
        self.layer_name = layer_name

        self.mode = 'get_statistic'

    def add_batch(self, input, output):
        if len(input.shape) == 2:
            input = input.unsqueeze(0)
        if isinstance(self.layer, nn.Linear):
            if len(input.shape) == 3: # (sample_num=1, seq_len, channel)
                input = input.reshape((-1, input.shape[-1])) # (seq_len, channel)
            input = input.t() # (channel, seq_len)

        if input.shape[1] == 0:
            return 
        input = input.type(torch.float32)

        if self.mode == 'get_statistic':
            self.get_data(input, output)
        elif self.mode == 'build_hist':
            self.get_hist(input)
        elif self.mode == 'build_kde':
            self.get_kde(input)

    def get_data(self, input, output):
        self.input = input
        tmp = input.shape[1] # (sample_num,)
 
        if self.layer_name == 'mlp.down_proj':
            self.input_matrix[:, self.nsamples] = input.squeeze(1) # (channel, )

        self.x_norm_l2 *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp

        input = input.type(torch.float32)
        self.x_norm_l1 += torch.norm(input, p=1, dim=1) / self.nsamples
        self.x_norm_l2 += torch.norm(input, p=2, dim=1)**2 / self.nsamples

        self.min_vals = torch.minimum(self.min_vals, torch.min(input, dim=1).values)
        self.max_vals = torch.maximum(self.max_vals, torch.max(input, dim=1).values)

    def get_hist(self, input):
        for i in range(self.cols):
            clamped_inp_row = torch.clamp(input[i], self.min_vals[i], self.max_vals[i]).float()
            hist = torch.histc(clamped_inp_row, bins=self.num_bins, min=self.min_vals[i], max=self.max_vals[i])
            self.activation_histograms[i] += hist

    def prepare_for_hist(self):
        self.mode = 'build_hist'

    def prepare_for_kde(self):
        self.mode = 'build_kde'

    def calculate_entropy(self, epsilon=1e-10):
        if self.nsamples == 0:
            return torch.zeros(self.cols, device=self.dev)
        
        total_counts_per_channel = self.activation_histograms.sum(dim=1)
        # Avoid division by zero for channels that never saw any activation
        total_counts_per_channel[total_counts_per_channel == 0] = 1

        probs = self.activation_histograms / total_counts_per_channel.unsqueeze(1)
        entropy = -torch.sum(probs * torch.log(probs + epsilon), dim=1)
        return entropy

    def calculate_entropy_kde(self):
        c, m = self.input_matrix.shape
        entropies = torch.zeros(c, device=self.dev)

        def kde_entropy(values, num_points=100):
            values_np = values.float().cpu().numpy()
            kde = gaussian_kde(values_np)
            xs = np.linspace(values_np.min(), values_np.max(), num_points)
            probs = kde(xs)
            probs /= probs.sum() + 1e-8  # normalize (not strictly necessary)
            entropy = -np.sum(probs * np.log(probs + 1e-8)) * (xs[1] - xs[0]) * 1000
            return torch.tensor(entropy, dtype=values.dtype, device=values.device)

        for i in range(c):
            values = self.input_matrix[i, :].abs().cpu()
            entropy = kde_entropy(values)  # or hist_entropy(values, bins)
            entropies[i] = entropy.to(self.dev)
        return entropies