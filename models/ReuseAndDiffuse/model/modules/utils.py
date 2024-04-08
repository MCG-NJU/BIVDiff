import torch


def checkpoint(func, inputs, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.
    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        return torch.utils.checkpoint.checkpoint(func, *inputs, use_reentrant=False)
    else:
        return func(*inputs)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module
