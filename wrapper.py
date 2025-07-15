import torch
import torch.nn as nn

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
    def __init__(self, layer, layer_id=0, layer_name='none', total_sample=30):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.cols = layer.weight.data.shape[1]
        
        self.x_norm_l1 = torch.zeros(self.cols, device=self.dev)
        self.x_norm_l2 = torch.zeros(self.cols, device=self.dev)
        # self.act_hist = torch.zeros((total_sample, self.rows, self.cols), device=self.dev)

        self.nsamples = 0
        self.layer_id = layer_id
        self.layer_name = layer_name

    def get_data(self, input, output):
        if len(input.shape) == 2:
            input = input.unsqueeze(0)
        tmp = input.shape[0] # (sample_num,)
        if isinstance(self.layer, nn.Linear):
            if len(input.shape) == 3: # (sample_num=1, seq_len, channel)
                input = input.reshape((-1, input.shape[-1])) # (seq_len, channel)
            input = input.t() # (channel, seq_len)
        self.x_norm_l2 *= self.nsamples / (self.nsamples + tmp)
        # self.act_hist[self.nsamples, :] = input.t()
        self.nsamples += tmp

        input = input.type(torch.float32)
        self.x_norm_l1 += torch.norm(input, p=1, dim=1) / self.nsamples
        self.x_norm_l2 += torch.norm(input, p=2, dim=1)**2 / self.nsamples