def get_importance(method, weight, x_norm_l2, sub_layer):
    imp_fn = {
        'weight': get_weight_imp,
        'wanda': get_wanda_imp,
        'entropy': get_entropy_imp,
        'esparse': get_esparse_imp,
        'magent': get_magent_imp
    }
    return imp_fn.get(method)(weight, x_norm_l2, sub_layer)

def get_weight_imp(weight, x_norm_l2, sub_layer):
    if any(keyword in sub_layer for keyword in ['up', 'gate']):
        imp = weight.t()
    elif 'down' in sub_layer:
        imp = weight
    return imp.mean(dim=0)

def get_wanda_imp(weight, x_norm_l2, sub_layer):
    if any(keyword in sub_layer for keyword in ['up', 'gate']):
        imp = x_norm_l2.t() * weight.t()
    elif 'down' in sub_layer:
        imp = weight * x_norm_l2
    return imp.mean(dim=0)

def get_entropy_imp():
    pass

def get_esparse_imp():
    pass

def get_magent_imp():
    pass