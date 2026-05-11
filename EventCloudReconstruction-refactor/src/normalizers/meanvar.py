def meanvar_normalizer(x, axis=1):
    """
    Normalize the input tensor `x` along the specified axis using mean and variance normalization.
    Args:
        x (torch.Tensor): Input tensor to be normalized.
        axis (int): Axis along which to normalize. Default is 1.
    """
    x = (x - x.mean(axis, keepdim=True)) / x.std(axis, keepdim=True)
    return x