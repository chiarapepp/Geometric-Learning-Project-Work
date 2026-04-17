import numpy as np
import torch


class Compose:
    """
    Compose multiple event transforms.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, events):
        for transform in self.transforms:
            if len(events) == 0:
                break
            events = transform(events)
        return events

    def __repr__(self):
        lines = [self.__class__.__name__ + "("]
        for transform in self.transforms:
            lines.append(f"    {transform}")
        lines.append(")")
        return "\n".join(lines)


class Identity:
    def __call__(self, events):
        return events

    def __repr__(self):
        return "Identity()"


class CopyEvents:
    """
    Explicitly copy structured event array.
    Useful before in-place transforms.
    """

    def __call__(self, events):
        return events.copy()

    def __repr__(self):
        return "CopyEvents()"


class Denoise:
    """
    Remove isolated events based on a simple 4-neighbour temporal rule.

    Similar in spirit to tonic's denoise utility.
    """

    def __init__(self, filter_time=10000):
        self.filter_time = filter_time

    def __call__(self, events):
        if len(events) == 0:
            return events

        assert "x" in events.dtype.names
        assert "y" in events.dtype.names
        assert "t" in events.dtype.names

        events = events.copy()

        events_copy = np.zeros_like(events)
        copy_index = 0
        width = int(events["x"].max()) + 1
        height = int(events["y"].max()) + 1
        timestamp_memory = np.zeros((width, height)) + self.filter_time

        for event in events:
            x = int(event["x"])
            y = int(event["y"])
            t = event["t"]
            timestamp_memory[x, y] = t + self.filter_time

            if (
                (x > 0 and timestamp_memory[x - 1, y] > t)
                or (x < width - 1 and timestamp_memory[x + 1, y] > t)
                or (y > 0 and timestamp_memory[x, y - 1] > t)
                or (y < height - 1 and timestamp_memory[x, y + 1] > t)
            ):
                events_copy[copy_index] = event
                copy_index += 1

        return events_copy[:copy_index]

    def __repr__(self):
        return f"Denoise(filter_time={self.filter_time})"


class RefractoryPeriod:
    """
    Keep only events that are sufficiently separated in time at the same pixel.
    """

    def __init__(self, refractory_period=1000):
        self.refractory_period = refractory_period

    def __call__(self, events):
        if len(events) == 0:
            return events

        assert "x" in events.dtype.names
        assert "y" in events.dtype.names
        assert "t" in events.dtype.names

        events = events.copy()

        events_copy = np.zeros_like(events)
        copy_index = 0
        width = int(events["x"].max()) + 1
        height = int(events["y"].max()) + 1
        timestamp_memory = np.zeros((width, height)) - self.refractory_period

        for event in events:
            x = int(event["x"])
            y = int(event["y"])
            t = event["t"]
            time_since_last_spike = t - timestamp_memory[x, y]

            if time_since_last_spike > self.refractory_period:
                events_copy[copy_index] = event
                copy_index += 1

            timestamp_memory[x, y] = t

        return events_copy[:copy_index]

    def __repr__(self):
        return f"RefractoryPeriod(refractory_period={self.refractory_period})"


class SpatialJitter:
    """
    Add Gaussian spatial noise to x and y.
    """

    def __init__(
        self,
        sensor_size,
        var_x=1.0,
        var_y=1.0,
        sigma_xy=0.0,
        clip_outliers=True,
    ):
        self.sensor_size = sensor_size
        self.var_x = var_x
        self.var_y = var_y
        self.sigma_xy = sigma_xy
        self.clip_outliers = clip_outliers

    def __call__(self, events):
        if len(events) == 0:
            return events

        assert "x" in events.dtype.names
        assert "y" in events.dtype.names

        events = events.copy()

        shifts = np.random.multivariate_normal(
            mean=[0.0, 0.0],
            cov=[[self.var_x, self.sigma_xy], [self.sigma_xy, self.var_y]],
            size=len(events),
        )

        events["x"] = np.round(events["x"] + shifts[:, 0]).astype(events["x"].dtype)
        events["y"] = np.round(events["y"] + shifts[:, 1]).astype(events["y"].dtype)

        if self.clip_outliers:
            mask = (
                (events["x"] >= 0)
                & (events["x"] < self.sensor_size[0])
                & (events["y"] >= 0)
                & (events["y"] < self.sensor_size[1])
            )
            events = events[mask]

        return events

    def __repr__(self):
        return (
            "SpatialJitter("
            f"sensor_size={self.sensor_size}, "
            f"var_x={self.var_x}, var_y={self.var_y}, "
            f"sigma_xy={self.sigma_xy}, clip_outliers={self.clip_outliers})"
        )


class TimeJitter:
    """
    Add Gaussian noise to timestamps.
    """

    def __init__(self, std=1.0, clip_negative=True, sort_timestamps=True):
        self.std = std
        self.clip_negative = clip_negative
        self.sort_timestamps = sort_timestamps

    def __call__(self, events):
        if len(events) == 0:
            return events

        assert "t" in events.dtype.names

        events = events.copy()
        shifts = np.random.normal(loc=0.0, scale=self.std, size=len(events))
        events["t"] = events["t"] + shifts

        if self.clip_negative:
            events = events[events["t"] >= 0]

        if self.sort_timestamps and len(events) > 0:
            events = events[np.argsort(events["t"])]

        return events

    def __repr__(self):
        return (
            f"TimeJitter(std={self.std}, clip_negative={self.clip_negative}, "
            f"sort_timestamps={self.sort_timestamps})"
        )


class TimeSkew:
    """
    Global affine transform on timestamps: t' = coefficient * t + offset.
    Useful to test time scaling sensitivity.
    """

    def __init__(self, coefficient=1.0, offset=0.0):
        self.coefficient = coefficient
        self.offset = offset

    def __call__(self, events):
        if len(events) == 0:
            return events

        assert "t" in events.dtype.names

        events = events.copy()

        coefficient = self.coefficient
        offset = self.offset

        if isinstance(coefficient, tuple):
            coefficient = np.random.uniform(coefficient[0], coefficient[1])
        if isinstance(offset, tuple):
            offset = np.random.uniform(offset[0], offset[1])

        events["t"] = events["t"] * coefficient + offset

        if len(events) > 0:
            events = events[np.argsort(events["t"])]

        return events

    def __repr__(self):
        return f"TimeSkew(coefficient={self.coefficient}, offset={self.offset})"


class TemporalShuffle:
    """
    Shuffle timestamps for a fraction of events.

    This is useful for controlled corruption along the temporal axis.
    It does not move x/y, only alters temporal ordering.

    Parameters
    ----------
    fraction : float
        Fraction of events whose timestamps are shuffled.
    sort_timestamps : bool
        If True, re-sort the structured array by timestamp after corruption.
    """

    def __init__(self, fraction=1.0, sort_timestamps=True):
        if not (0.0 <= fraction <= 1.0):
            raise ValueError("fraction must be in [0, 1]")
        self.fraction = fraction
        self.sort_timestamps = sort_timestamps

    def __call__(self, events):
        if len(events) <= 1:
            return events

        assert "t" in events.dtype.names

        events = events.copy()
        n = len(events)
        k = max(1, int(round(self.fraction * n)))

        idx = np.random.choice(n, size=k, replace=False)
        shuffled_t = events["t"][idx].copy()
        np.random.shuffle(shuffled_t)
        events["t"][idx] = shuffled_t

        if self.sort_timestamps:
            events = events[np.argsort(events["t"])]

        return events

    def __repr__(self):
        return f"TemporalShuffle(fraction={self.fraction}, sort_timestamps={self.sort_timestamps})"


class AddUniformNoise:
    """
    Add random noise events uniformly distributed over x, y, p, and t.
    """

    def __init__(self, sensor_size, n=100):
        self.sensor_size = sensor_size
        self.n = n

    def __call__(self, events):
        if len(events) == 0:
            return events

        noise_events = np.zeros(self.n, dtype=events.dtype)

        for channel in events.dtype.names:
            if channel == "x":
                low, high = 0, self.sensor_size[0]
            elif channel == "y":
                low, high = 0, self.sensor_size[1]
            elif channel == "p":
                low, high = 0, self.sensor_size[2]
            elif channel == "t":
                low, high = events["t"].min(), events["t"].max()
            else:
                continue

            values = np.random.uniform(low=low, high=high, size=self.n)

            if np.issubdtype(events.dtype[channel], np.integer) or events.dtype[channel] == np.bool_:
                values = np.floor(values)

            noise_events[channel] = values.astype(events.dtype[channel])

        noisy_events = np.concatenate((events, noise_events))
        noisy_events = noisy_events[np.argsort(noisy_events["t"])]

        return noisy_events

    def __repr__(self):
        return f"AddUniformNoise(sensor_size={self.sensor_size}, n={self.n})"


class DropEventRandom:
    """
    Randomly drop a fraction of events.
    """

    def __init__(self, drop_probability=0.1):
        if not (0.0 <= drop_probability < 1.0):
            raise ValueError("drop_probability must be in [0, 1)")
        self.drop_probability = drop_probability

    def __call__(self, events):
        if len(events) == 0:
            return events

        n_events = len(events)
        n_drop = int(self.drop_probability * n_events + 0.5)
        if n_drop == 0:
            return events

        dropped_indices = np.random.choice(n_events, n_drop, replace=False)
        return np.delete(events, dropped_indices, axis=0)

    def __repr__(self):
        return f"DropEventRandom(drop_probability={self.drop_probability})"


class DropEventByTime:
    """
    Drop events in one random temporal interval.
    """

    def __init__(self, duration_ratio=0.2):
        self.duration_ratio = duration_ratio

    def __call__(self, events):
        if len(events) == 0:
            return events

        assert "t" in events.dtype.names

        duration_ratio = self.duration_ratio
        if isinstance(duration_ratio, tuple):
            duration_ratio = np.random.uniform(duration_ratio[0], duration_ratio[1])

        t_start = float(events["t"].min())
        t_end = float(events["t"].max())
        total_duration = t_end - t_start

        if total_duration <= 0:
            return events

        drop_duration = total_duration * duration_ratio
        if drop_duration <= 0:
            return events

        drop_start = np.random.uniform(t_start, t_end - drop_duration)
        mask = ~(
            (events["t"] >= drop_start) &
            (events["t"] <= drop_start + drop_duration)
        )
        return events[mask]

    def __repr__(self):
        return f"DropEventByTime(duration_ratio={self.duration_ratio})"


class Decimate:
    """
    Keep 1 event every n for each pixel location.
    """

    def __init__(self, n=2):
        if n <= 0:
            raise ValueError("n must be > 0")
        self.n = n

    def __call__(self, events):
        if len(events) == 0:
            return events

        assert "x" in events.dtype.names

        max_x = int(np.max(events["x"]))
        output_events = []

        if "y" in events.dtype.names:
            max_y = int(np.max(events["y"]))
            memory = np.zeros((max_x + 1, max_y + 1), dtype=np.int32)

            for event in events:
                x, y = int(event["x"]), int(event["y"])
                memory[x, y] += 1
                if memory[x, y] >= self.n:
                    memory[x, y] = 0
                    output_events.append(event)
        else:
            memory = np.zeros(max_x + 1, dtype=np.int32)
            for event in events:
                x = int(event["x"])
                memory[x] += 1
                if memory[x] >= self.n:
                    memory[x] = 0
                    output_events.append(event)

        if len(output_events) == 0:
            return events[:0]

        return np.array(output_events, dtype=events.dtype)

    def __repr__(self):
        return f"Decimate(n={self.n})"


class CropEvents:
    """
    Random crop in space.
    """

    def __init__(self, sensor_size, target_size):
        self.sensor_size = sensor_size
        self.target_size = target_size

    def __call__(self, events):
        if len(events) == 0:
            return events

        assert "x" in events.dtype.names
        assert "y" in events.dtype.names

        if self.target_size[0] > self.sensor_size[0] or self.target_size[1] > self.sensor_size[1]:
            raise ValueError("target_size must be <= sensor_size")

        events = events.copy()

        x_start = int(np.random.rand() * (self.sensor_size[0] - self.target_size[0] + 1))
        y_start = int(np.random.rand() * (self.sensor_size[1] - self.target_size[1] + 1))
        x_end = x_start + self.target_size[0]
        y_end = y_start + self.target_size[1]

        mask = (
            (events["x"] >= x_start) &
            (events["x"] < x_end) &
            (events["y"] >= y_start) &
            (events["y"] < y_end)
        )

        events = events[mask]
        if len(events) == 0:
            return events

        events["x"] -= x_start
        events["y"] -= y_start

        return events

    def __repr__(self):
        return f"CropEvents(sensor_size={self.sensor_size}, target_size={self.target_size})"


class EventsToXYTP:
    """
    Convert structured event array to dense float32 array with columns [x, y, t, p].
    """

    def __call__(self, events):
        if len(events) == 0:
            return np.zeros((0, 4), dtype=np.float32)

        assert events.dtype.names is not None
        assert "x" in events.dtype.names
        assert "y" in events.dtype.names
        assert "t" in events.dtype.names

        x = events["x"].astype(np.float32, copy=False)
        y = events["y"].astype(np.float32, copy=False)
        t = events["t"].astype(np.float32, copy=False)

        if "p" in events.dtype.names:
            p = events["p"].astype(np.float32, copy=False)
        else:
            p = np.zeros_like(x, dtype=np.float32)

        return np.stack([x, y, t, p], axis=1)

    def __repr__(self):
        return "EventsToXYTP()"


class NormalizeXYT:
    """
    Normalize x, y, t and optionally scale time with temporal_weight.

    Output stays dense array.
    Expected input shape: (N, 4) or (N, 3)

    x -> [0,1]
    y -> [0,1]
    t -> [0,1] then multiplied by temporal_weight

    If sensor_size is None, x and y are normalized per sample with min/max.
    """

    def __init__(self, sensor_size, temporal_weight=1.0):
        self.sensor_size = sensor_size
        self.temporal_weight = temporal_weight

    def __call__(self, points):
        points = np.asarray(points, dtype=np.float32).copy()

        if points.ndim != 2 or points.shape[1] < 3:
            raise ValueError(f"Expected shape (N, >=3), got {points.shape}")

        if len(points) == 0:
            return points

        if self.sensor_size is None:
            for dim in (0, 1):
                dim_min = float(points[:, dim].min())
                dim_max = float(points[:, dim].max())
                dim_range = dim_max - dim_min
                if dim_range > 1e-12:
                    points[:, dim] = (points[:, dim] - dim_min) / dim_range
                else:
                    points[:, dim] = 0.0
        else:
            width = self.sensor_size[0]
            height = self.sensor_size[1]

            if width > 1:
                points[:, 0] = points[:, 0] / float(width - 1)
            else:
                points[:, 0] = 0.0

            if height > 1:
                points[:, 1] = points[:, 1] / float(height - 1)
            else:
                points[:, 1] = 0.0

        t_min = float(points[:, 2].min())
        t_max = float(points[:, 2].max())
        dt = t_max - t_min

        if dt > 1e-12:
            points[:, 2] = (points[:, 2] - t_min) / dt
        else:
            points[:, 2] = 0.0

        points[:, 2] *= self.temporal_weight

        return points

    def __repr__(self):
        return f"NormalizeXYT(sensor_size={self.sensor_size}, temporal_weight={self.temporal_weight})"


class PolarityToMinusOnePlusOne:
    """
    Convert polarity channel from {0,1} to {-1,+1}.
    Expected dense array shape (N,4).
    """

    def __call__(self, points):
        points = np.asarray(points, dtype=np.float32).copy()
        if points.ndim != 2 or points.shape[1] < 4:
            raise ValueError(f"Expected shape (N, >=4), got {points.shape}")

        unique_vals = np.unique(points[:, 3])
        if np.all(np.isin(unique_vals, [0.0, 1.0])):
            points[:, 3] = 2.0 * points[:, 3] - 1.0

        return points

    def __repr__(self):
        return "PolarityToMinusOnePlusOne()"


class DropPolarity:
    """
    Convert [x,y,t,p] -> [x,y,t]
    """

    def __call__(self, points):
        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] < 3:
            raise ValueError(f"Expected shape (N, >=3), got {points.shape}")

        return points[:, :3].copy()

    def __repr__(self):
        return "DropPolarity()"


class SamplePoints:
    """
    Return exactly num_points points.

    Modes:
    - random
    - uniform
    - first

    Pad modes:
    - repeat
    - zeros
    """

    def __init__(self, num_points=2048, mode="random", pad_mode="repeat"):
        if num_points <= 0:
            raise ValueError("num_points must be > 0")
        if mode not in {"random", "uniform", "first"}:
            raise ValueError("mode must be one of {'random', 'uniform', 'first'}")
        if pad_mode not in {"repeat", "zeros"}:
            raise ValueError("pad_mode must be one of {'repeat', 'zeros'}")

        self.num_points = num_points
        self.mode = mode
        self.pad_mode = pad_mode

    def __call__(self, points):
        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 2:
            raise ValueError(f"Expected shape (N, D), got {points.shape}")

        n, d = points.shape

        if n == self.num_points:
            return points

        if n > self.num_points:
            if self.mode == "first":
                return points[:self.num_points]

            if self.mode == "uniform":
                idx = np.linspace(0, n - 1, self.num_points, dtype=np.int64)
                return points[idx]

            idx = np.random.choice(n, size=self.num_points, replace=False)
            return points[idx]

        # n < num_points
        if n == 0:
            return np.zeros((self.num_points, d), dtype=np.float32)

        if self.pad_mode == "zeros":
            pad = np.zeros((self.num_points - n, d), dtype=np.float32)
            return np.concatenate([points, pad], axis=0)

        extra_idx = np.random.choice(n, size=self.num_points - n, replace=True)
        pad = points[extra_idx]
        return np.concatenate([points, pad], axis=0)

    def __repr__(self):
        return (
            f"SamplePoints(num_points={self.num_points}, "
            f"mode='{self.mode}', pad_mode='{self.pad_mode}')"
        )


class ShufflePoints:
    """
    Shuffle rows of dense point array.
    Useful after sampling.
    """

    def __call__(self, points):
        points = np.asarray(points, dtype=np.float32).copy()
        if len(points) > 1:
            idx = np.random.permutation(len(points))
            points = points[idx]
        return points

    def __repr__(self):
        return "ShufflePoints()"


class ToTensor:
    """
    Convert numpy array to torch.FloatTensor.
    """

    def __call__(self, array):
        if isinstance(array, torch.Tensor):
            return array.float()
        return torch.from_numpy(np.asarray(array)).float()

    def __repr__(self):
        return "ToTensor()"
