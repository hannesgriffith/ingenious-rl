from torch import nn

def set_model_to_half(model):
    model.half()
    for layer in model.modules():
        if isinstance(layer, nn.BatchNorm2d):
            layer.float()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def set_optimizer_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr