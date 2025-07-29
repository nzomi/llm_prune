import torch
from ..utils import norm_value

def get_importance(method, sub_layer, weight, x_norm_l2, entropy=None, alpha=0, model_type='internvl'):
    imp_fn = {
        'weight': get_weight_imp,
        'wanda': get_wanda_imp,
        'group_wanda': get_group_wanda_imp,
        'entropy': get_entropy_imp,
        'esparse': get_esparse_imp,
        'magent': get_magent_imp,
        'abnorm_group_wanda': get_norm_group_wanda_imp,
        'test': get_magent_imp,
    }
    return imp_fn.get(method)(sub_layer, weight, x_norm_l2, entropy, alpha, model_type)

def get_weight_imp(sub_layer, weight, x_norm_l2, entropy, alpha, model_type='internvl'):
    if model_type == '9b':
        if any(keyword in sub_layer for keyword in ['w1', 'w3']):
            imp = weight.t()
        elif 'w2' in sub_layer:
            imp = weight
    else:
        if any(keyword in sub_layer for keyword in ['up', 'gate']):
            imp = weight.t()
        elif 'down' in sub_layer:
            imp = weight
    return norm_value(imp.mean(dim=0))

def get_group_wanda_imp(sub_layer, weight, x_norm_l2, entropy, alpha, model_type='internvl'):
    if model_type == '9b':
        if any(keyword in sub_layer for keyword in ['w1', 'w3']):
            imp = x_norm_l2.t() * weight.t()
        elif 'w2' in sub_layer:
            imp = weight * x_norm_l2
    else:
        if any(keyword in sub_layer for keyword in ['up', 'gate']):
            imp = x_norm_l2.t() * weight.t()
        elif 'down' in sub_layer:
            imp = weight * x_norm_l2
    return imp.mean(dim=0)

def get_norm_group_wanda_imp(sub_layer, weight, x_norm_l2, entropy, alpha, model_type='internvl'):
    if model_type == '9b':
        if any(keyword in sub_layer for keyword in ['w1', 'w3']):
            imp = x_norm_l2.t() * weight.t()
        elif 'w2' in sub_layer:
            imp = weight * x_norm_l2
    else:
        if any(keyword in sub_layer for keyword in ['up', 'gate']):
            imp = x_norm_l2.t() * weight.t()
        elif 'down' in sub_layer:
            imp = weight * x_norm_l2
    return norm_value(imp.mean(dim=0))

def get_wanda_imp(sub_layer, weight, x_norm_l2, entropy, alpha, model_type='internvl'):
    if model_type == '9b':
        if any(keyword in sub_layer for keyword in ['w1', 'w3']):
            imp = torch.zeros_like(x_norm_l2.t() * weight.t())
        elif 'w2' in sub_layer:
            imp = weight * x_norm_l2
    else:
        if any(keyword in sub_layer for keyword in ['up', 'gate']):
            imp = torch.zeros_like(x_norm_l2.t() * weight.t())
        elif 'down' in sub_layer:
            imp = weight * x_norm_l2
    return norm_value(imp.mean(dim=0))

def get_entropy_imp(sub_layer, weight, x_norm_l2, entropy, alpha, model_type='internvl'):
    return norm_value(entropy)

def get_esparse_imp(sub_layer, weight, x_norm_l2, entropy, alpha, model_type='internvl'):
    if model_type == '9b':
        if any(keyword in sub_layer for keyword in ['w1', 'w3']):
            imp = x_norm_l2.t() * weight.t()
        elif 'w2' in sub_layer:
            imp = weight * (alpha * entropy + x_norm_l2)
    else:
        if any(keyword in sub_layer for keyword in ['up', 'gate']):
            imp = x_norm_l2.t() * weight.t()
        elif 'down' in sub_layer:
            imp = weight * (alpha * entropy + x_norm_l2)
    return norm_value(imp.mean(dim=0))

def get_magent_imp(sub_layer, weight, x_norm_l2, entropy, alpha, model_type='internvl'):
    return 0