import torch


_EMD_MODULE_CACHE = {}


def _load_emd_module(device):
    try:
        import emd
    except ImportError as exc:
        raise ImportError(
            "The 'emd_cuda' loss requires the meder411/PyTorch-EMDLoss package. "
            "Install it in the active environment from "
            "https://github.com/meder411/PyTorch-EMDLoss before using this backend."
        ) from exc

    if not hasattr(emd, "emdModule"):
        raise ImportError(
            "Imported package 'emd' does not expose emdModule(). This usually means "
            "a different package named 'emd' is installed. Install "
            "meder411/PyTorch-EMDLoss, whose Python package also imports as 'emd'."
        )

    key = str(device)
    if key not in _EMD_MODULE_CACHE:
        module = emd.emdModule()
        module = module.to(device)
        _EMD_MODULE_CACHE[key] = module
    return _EMD_MODULE_CACHE[key]


def emd_cuda_loss(
    y_hat,
    y,
    eps=0.005,
    iterations=300,
    reduction="mean",
    sqrt=True,
):
    """
    Approximate Earth Mover's Distance using meder411/PyTorch-EMDLoss.

    This backend is optional and CUDA-only. The external extension returns a
    per-point matching cost; by default we take sqrt(cost) so the scale is
    closer to the existing Hungarian EMD implementation based on Euclidean
    distances.
    """
    if y_hat.ndim != 3 or y.ndim != 3:
        raise ValueError(
            f"Expected tensors of shape (B, N, D), got {y_hat.shape} and {y.shape}"
        )

    if y_hat.shape[0] != y.shape[0]:
        raise ValueError(
            f"Batch sizes must match, got {y_hat.shape[0]} and {y.shape[0]}"
        )

    if y_hat.shape[2] != y.shape[2]:
        raise ValueError(
            f"Point dimensions must match, got {y_hat.shape[2]} and {y.shape[2]}"
        )

    if y_hat.device != y.device:
        raise ValueError(f"Input tensors must be on the same device, got {y_hat.device} and {y.device}")

    if y_hat.device.type != "cuda":
        raise RuntimeError("The 'emd_cuda' loss requires CUDA tensors.")

    if not torch.cuda.is_available():
        raise RuntimeError("The 'emd_cuda' loss requires a CUDA-enabled PyTorch build.")

    module = _load_emd_module(y_hat.device)
    pred = y_hat.contiguous().float()
    target = y.contiguous().float()

    distances, _assignment = module(pred, target, float(eps), int(iterations))
    distances = distances.clamp_min(0.0)
    if sqrt:
        distances = torch.sqrt(distances)

    if distances.ndim == 1:
        batch_losses = distances
    else:
        reduce_dims = tuple(range(1, distances.ndim))
        batch_losses = distances.mean(dim=reduce_dims)

    if reduction == "mean":
        return batch_losses.mean()
    elif reduction == "sum":
        return batch_losses.sum()
    elif reduction == "none":
        return batch_losses
    else:
        raise ValueError(f"Unknown reduction '{reduction}'")
