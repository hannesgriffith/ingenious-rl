from torch import nn

def set_model_to_half(model):
    model.half()
    for layer in model.modules():
        if isinstance(layer, nn.BatchNorm2d):
            layer.float()

def set_model_to_float(model):
    model.float()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def set_optimizer_params(optimizer, lr=None, weight_decay=None):
    for param_group in optimizer.param_groups:

        if lr:
            param_group['lr'] = lr

        if weight_decay:
            param_group['weight_decay'] = weight_decay