from utils import *

def get_importance(method, sub_layer, weight, x_norm_l2, entropy=None, alpha=1):
    imp_fn = {
        'weight': get_weight_imp,
        'wanda': get_wanda_imp,
        'entropy': get_entropy_imp,
        'esparse': get_esparse_imp,
        'magent': get_magent_imp
    }
    return imp_fn.get(method)(sub_layer, weight, x_norm_l2, entropy, alpha)

def get_weight_imp(sub_layer, weight, x_norm_l2, entropy, alpha):
    if any(keyword in sub_layer for keyword in ['up', 'gate']):
        imp = weight.t()
    elif 'down' in sub_layer:
        imp = weight
    return imp.mean(dim=0)

def get_wanda_imp(sub_layer, weight, x_norm_l2, entropy, alpha):
    if any(keyword in sub_layer for keyword in ['up', 'gate']):
        imp = x_norm_l2.t() * weight.t()
    elif 'down' in sub_layer:
        imp = weight * x_norm_l2
    return imp.mean(dim=0)

def get_entropy_imp(sub_layer, weight, x_norm_l2, entropy, alpha):
    return entropy

def get_esparse_imp(sub_layer, weight, x_norm_l2, entropy, alpha=1):
    if any(keyword in sub_layer for keyword in ['up', 'gate']):
        imp = x_norm_l2.t() * weight.t()
    elif 'down' in sub_layer:
        imp = weight * (alpha * entropy + x_norm_l2)
    return imp.mean(dim=0)

def get_magent_imp(sub_layer, weight, x_norm_l2, entropy, alpha=0.8):
    if any(keyword in sub_layer for keyword in ['up', 'gate']):
        imp = (1-alpha) * x_norm_l2.t() * weight.t()
    elif 'down' in sub_layer:
        imp = (1-alpha) * weight * x_norm_l2 + alpha * entropy
    return imp.mean(dim=0)