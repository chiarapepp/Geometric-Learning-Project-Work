def minmax_normalizer(x, axis=1):
    """
    Normalize the input tensor `x` along the specified axis using min-max normalization.
    Args:
        x (torch.Tensor): Input tensor to be normalized.
        axis (int): Axis along which to normalize. Default is 1.
    """
    x = (x - x.min(axis, keepdim=True).values) / (x.max(axis, keepdim=True).values - x.min(axis, keepdim=True).values)
    return x