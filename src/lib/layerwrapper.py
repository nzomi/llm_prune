import torch
import torch.nn as nn
import torch.nn.functional as F

# Define WrappedGPT class
class WrappedGPT:
    """
    This class wraps a GPT layer for specific operations.
    """

    def __init__(self, layer, layer_id=0, layer_name="none"):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]

        self.scaler_row = torch.zeros((self.columns), device=self.dev)
        self.nsamples = 0

        self.layer_id = layer_id 
        self.layer_name = layer_name

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        self.scaler_row *= self.nsamples / (self.nsamples+tmp)
        self.nsamples += tmp

        inp = inp.type(torch.float32)
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2  / self.nsamples

class WrappedGPTE:
    def __init__(
        self, 
        layer, 
        layer_id=0, 
        layer_name="none",
        num_bins=100
    ):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]
        self.num_bins = num_bins

        self.scaler_row_l2 = torch.zeros(self.columns, device=self.dev)
        self.activation_histograms = torch.zeros((self.columns, self.num_bins), device=self.dev)
        
        self.min_vals = torch.full((self.columns,), float('inf'), device=self.dev)
        self.max_vals = torch.full((self.columns,), float('-inf'), device=self.dev)

        self.mode = 'range_finding'
        self.nsamples = 0
        self.bins = None

    def find_range_and_update_l2(self, inp):
        tmp = inp.shape[1]
        
        self.scaler_row_l2 *= self.nsamples / (self.nsamples + tmp)
        self.scaler_row_l2 += torch.norm(inp, p=2, dim=1) ** 2 / (self.nsamples + tmp)

        self.min_vals = torch.minimum(self.min_vals, torch.min(inp, dim=1).values)
        self.max_vals = torch.maximum(self.max_vals, torch.max(inp, dim=1).values)
        
        self.nsamples += tmp

    def build_histogram(self, inp):
        if self.bins is None:
            raise ValueError("Must call prepare_for_histogram_pass() before building histograms.")
        
        # NOTE: Using a loop here for clarity, but it can be vectorized if needed for extreme performance.
        # Vectorized version is more complex to write correctly for per-channel bins.
        for i in range(self.columns):
            clamped_inp_row = torch.clamp(inp[i], self.min_vals[i], self.max_vals[i])
            hist = torch.histc(clamped_inp_row, bins=self.num_bins, min=self.min_vals[i], max=self.max_vals[i])
            self.activation_histograms[i] += hist

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        
        if inp.shape[1] == 0:
            return
        inp = inp.type(torch.float32)

        if self.mode == 'range_finding':
            self.find_range_and_update_l2(inp)
        elif self.mode == 'histogram_building':
            self.build_histogram(inp)

    def prepare_for_histogram_pass(self):
        self.mode = 'histogram_building'
        self.bins = True

    def calculate_entropy(self, epsilon=1e-10):
        if self.nsamples == 0:
            return torch.zeros(self.columns, device=self.dev)
        
        total_counts_per_channel = self.activation_histograms.sum(dim=1)
        # Avoid division by zero for channels that never saw any activation
        total_counts_per_channel[total_counts_per_channel == 0] = 1

        probs = self.activation_histograms / total_counts_per_channel.unsqueeze(1)
        entropy = -torch.sum(probs * torch.log(probs + epsilon), dim=1)
        return entropy