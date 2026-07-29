import torch


def point_cloud_fscore(
    prediction,
    target,
    threshold,
    reduction="mean",
    eps=1e-8,
):
    """Symmetric point-cloud F-score at a Euclidean distance threshold."""
    if prediction.ndim != 3 or target.ndim != 3:
        raise ValueError(
            "Expected tensors of shape (B, N, D), "
            f"got {prediction.shape} and {target.shape}"
        )
    if prediction.shape[0] != target.shape[0]:
        raise ValueError(
            f"Batch sizes must match, got {prediction.shape[0]} and {target.shape[0]}"
        )
    if prediction.shape[2] != target.shape[2]:
        raise ValueError(
            "Point dimensions must match, "
            f"got {prediction.shape[2]} and {target.shape[2]}"
        )
    if threshold <= 0:
        raise ValueError(f"threshold must be positive, got {threshold}")

    distances = torch.cdist(prediction.float(), target.float(), p=2)
    prediction_to_target = distances.min(dim=2).values
    target_to_prediction = distances.min(dim=1).values

    precision = (prediction_to_target <= threshold).float().mean(dim=1)
    recall = (target_to_prediction <= threshold).float().mean(dim=1)
    scores = 2.0 * precision * recall / (precision + recall + eps)

    if reduction == "mean":
        return scores.mean()
    if reduction == "sum":
        return scores.sum()
    if reduction == "none":
        return scores
    raise ValueError(f"Unknown reduction '{reduction}'")

def point_cloud_hd95(prediction, target, reduction="mean"):
    """Symmetric 95th-percentile Hausdorff distance."""
    if prediction.ndim != 3 or target.ndim != 3:
        raise ValueError(
            "Expected tensors of shape (B, N, D), "
            f"got {prediction.shape} and {target.shape}"
        )
    if prediction.shape[0] != target.shape[0]:
        raise ValueError(
            f"Batch sizes must match, got {prediction.shape[0]} and {target.shape[0]}"
        )
    if prediction.shape[2] != target.shape[2]:
        raise ValueError(
            "Point dimensions must match, "
            f"got {prediction.shape[2]} and {target.shape[2]}"
        )

    distances = torch.cdist(prediction.float(), target.float(), p=2)
    prediction_to_target = distances.min(dim=2).values
    target_to_prediction = distances.min(dim=1).values
    directed_prediction = torch.quantile(prediction_to_target, 0.95, dim=1)
    directed_target = torch.quantile(target_to_prediction, 0.95, dim=1)
    values = torch.maximum(directed_prediction, directed_target)

    if reduction == "mean":
        return values.mean()
    if reduction == "sum":
        return values.sum()
    if reduction == "none":
        return values
    raise ValueError(f"Unknown reduction '{reduction}'")