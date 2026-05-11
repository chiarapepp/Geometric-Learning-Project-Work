import torch
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import tonic

def seed_everything(seed=42):
    """
    Fix all random seeds for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    os.environ['PYTHONHASHSEED'] = str(seed)  # Fix hashing seed in Python
    
    # Ensure deterministic operations in cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Optional: Make PyTorch fully deterministic, which may be slower
    # torch.use_deterministic_algorithms(True)

def plot_frames(example_to_plot, dataset, name="frames", event_count=96):
    """
    Function to plot frames starting from tonic dataset format.
    """
    transform = tonic.transforms.ToFrame(
        sensor_size=dataset.sensor_size,
        event_count=event_count
    )
    frames = transform(example_to_plot)
    print(len(frames))
    fig, axes = plt.subplots(1, len(frames))
    i = 0
    for axis, frame in zip(axes, frames):
        axis.imshow(frame[0], cmap="gray")
        axis.axis("off")
    plt.tight_layout()
    os.makedirs("images", exist_ok=True)
    plt.savefig(f"images/{name}.png")
    

def generate_frames_normal(point_cloud, event_count, frame_shape):
    """
    Generates frames from a point cloud by grouping events into frames of size `event_count`.

    Args:
        point_cloud (torch.Tensor): Point cloud data with shape (N, 3), where columns are (x, y, t).
        event_count (int): Number of events per frame.
        frame_shape (tuple): Shape of the output frames (height, width).

    Returns:
        np.ndarray: Array of frames with shape (num_frames, height, width).
    """
    # Initialize variables
    frames = []
    current_frame = np.zeros(frame_shape, dtype=np.float32)

    # Sort point cloud along the temporal axis (t)
    sorted_indices = torch.argsort(point_cloud[:, 2])
    point_cloud = point_cloud[sorted_indices]

    # Process point cloud to create frames
    for i, event in enumerate(point_cloud):
        x, y = int(event[0]), int(event[1])
        
        # Accumulate events into the current frame
        if 0 <= x < frame_shape[1] and 0 <= y < frame_shape[0]:
            current_frame[y, x] += 1
        else:
            print(f"OUT OF BOUND PREDICTION at index {i}: (x={x}, y={y})")

        # If the frame reaches the event count, finalize it and start a new frame
        if (i + 1) % event_count == 0:
            frames.append(current_frame)
            current_frame = np.zeros(frame_shape, dtype=np.float32)

    # Convert to numpy array
    frames = np.array(frames, dtype=np.float32)

    return frames


def generate_temporal_frames(point_cloud, num_frames, frame_shape, max_t):
    """
    Generates frames from a point cloud by grouping events into frames of size `event_count`.

    Args:
        point_cloud (torch.Tensor): Point cloud data with shape (N, 3), where columns are (x, y, t).
        num_frames (int): Number frame (time intervals).
        frame_shape (tuple): Shape of the output frames (height, width).

    Returns:
        np.ndarray: Array of frames with shape (num_frames, height, width).
    """
    # Initialize variables
    frames = []
    current_frame = np.zeros(frame_shape, dtype=np.float32)

    # Sort point cloud along the temporal axis (t)
    sorted_indices = torch.argsort(point_cloud[:, 2])
    point_cloud = point_cloud[sorted_indices]

    time_boundaries = torch.linspace(0, max_t, num_frames + 1)

    for i in range(num_frames):
        start_t = time_boundaries[i]
        end_t = time_boundaries[i + 1]
        cur_events = point_cloud[(point_cloud[:, 2] >= start_t) & (point_cloud[:, 2] < end_t)]
        for event in cur_events:
            x, y = int(event[0]), int(event[1])
            if 0 <= x < frame_shape[1] and 0 <= y < frame_shape[0]:
                current_frame[y, x] += 1
            else:
                print(f"OUT OF BOUND PREDICTION at index {i}: (x={x}, y={y})")

        frames.append(current_frame)
        current_frame = np.zeros(frame_shape, dtype=np.float32)

    # Convert to numpy array
    frames = np.array(frames, dtype=np.float32)

    return frames
    

def generate_frames_normal_batched(point_clouds, event_count, frame_shape):
    """
    Generates frames from a batch of point clouds by grouping events into frames of size `event_count`.

    Args:
        point_clouds (torch.Tensor): Batch of point cloud data with shape (B, N, 3), where columns are (x, y, t).
        event_count (int): Number of events per frame.
        frame_shape (tuple): Shape of the output frames (height, width).

    Returns:
        list of np.ndarray: List of arrays of frames with shape (num_frames, height, width) for each point cloud in the batch.
    """
    batch_frames = []

    for point_cloud in point_clouds:
        # Initialize variables
        frames = []
        current_frame = np.zeros(frame_shape, dtype=np.float32)

        # Sort point cloud along the temporal axis (t)
        sorted_indices = torch.argsort(point_cloud[:, 2])
        point_cloud = point_cloud[sorted_indices]

        # Process point cloud to create frames
        for i, event in enumerate(point_cloud):
            x, y = int(event[0]), int(event[1])

            # Accumulate events into the current frame
            if 0 <= x < frame_shape[1] and 0 <= y < frame_shape[0]:
                current_frame[y, x] += 1
            else:
                print(f"OUT OF BOUND PREDICTION at index {i}: (x={x}, y={y})")

            # If the frame reaches the event count, finalize it and start a new frame
            if (i + 1) % event_count == 0:
                frames.append(current_frame)
                current_frame = np.zeros(frame_shape, dtype=np.float32)

        # Convert to numpy array
        frames = np.array(frames, dtype=np.float32)
        batch_frames.append(frames)

    batch_frames = np.array(batch_frames)
    batch_frames = torch.from_numpy(batch_frames)
    return batch_frames



def point_cloud_to_frames(batch, grid_size=(128, 128), frame_size=500):
    """
    Converts a batch of point clouds into spatial frames (2D grids) for each subset of points.

    Args:
        batch: Tensor of shape (B, P, 3), where P=total points and 3=(x, y, t).
        grid_size: Tuple defining the output frame spatial resolution (H, W).
        frame_size: Number of points per frame.

    Returns:
        frames: Tensor of shape (B, num_frames, H, W), where num_frames = P // frame_size.
    """
    B, P, C = batch.shape
    H, W = grid_size
    num_frames = P // frame_size

    # Normalize x, y coordinates to fit within the grid [0, H), [0, W)
    x = batch[:, :, 0]  # x-coordinates
    y = batch[:, :, 1]  # y-coordinates

    # Rescale x and y to grid indices
    x_grid = x.long().clamp(min=0, max=W - 1)
    y_grid = y.long().clamp(min=0, max=H - 1)

    # Initialize frames
    frames = torch.zeros((B, num_frames, H, W), dtype=torch.float, device=batch.device)

    for b in range(B):
        for f in range(num_frames):
            # Get the subset of points for the current frame
            start = f * frame_size
            end = start + frame_size
            x_f = x_grid[b, start:end]  # x-coordinates for the current frame
            y_f = y_grid[b, start:end]  # y-coordinates for the current frame

            # Create a 2D grid and accumulate points
            grid = torch.zeros((H, W), dtype=torch.float, device=batch.device)

            # Scatter-add the points into the grid
            grid.index_put_((y_f, x_f), torch.ones_like(x_f, dtype=torch.float), accumulate=True)

            # Assign the grid to the current frame
            frames[b, f] = grid

    return frames



def batch_cov(points):
    """
    Computes the unbiased covariance matrix for a batch of point clouds.
    """
    B, N, D = points.size()
    mean = points.mean(dim=1).unsqueeze(1)
    diffs = (points - mean).reshape(B * N, D)
    prods = torch.bmm(diffs.unsqueeze(2), diffs.unsqueeze(1)).reshape(B, N, D, D)
    bcov = prods.sum(dim=1) / (N - 1)  # Unbiased estimate
    return bcov



    
if __name__ == "__main__":
    seed_everything()
    print("Listing seed sample results for check...")
    print("Random seed:", random.random())
    print("Numpy seed:", np.random.random())
    print("Torch seed:", torch.rand(1).item())
    print("Loading dataset...")
    train_dataset = tonic.datasets.NMNIST(save_to="/andromeda/datasets/DVSGesture/NMNIST", train=True)
    example, target = train_dataset[0]
    example = example[:1024]
    plot_frames(example, train_dataset, name="example", event_count=96)

