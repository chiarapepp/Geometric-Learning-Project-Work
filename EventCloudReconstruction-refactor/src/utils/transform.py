import numpy as np
import random
from tonic.transforms import Compose, NumpyAsType


class TaskSpecificCompose(Compose):
    """
        Create a transform function comprised of multiple task specific transformations, that can be applied at the same time.
    """
    def __init__(self, task_transformations, fixed_transformations=None):
        """
        Args:
            task_transformations: List of tuples (task_id, transformation).
            fixed_transformations: List of fixed transformations applied before task transformations.
        """
        self.task_transformations = task_transformations if len(task_transformations) > 0 else None
        self.fixed_transformations = fixed_transformations if len(fixed_transformations) > 0 else None

    def __call__(self, events):
        """
        Apply fixed transformations first, then randomly choose one task transformation.
        Args:
            events: Input events data.
        Returns:
            transformed_events, task_id
        """
        if self.fixed_transformations is not None:
            for transform in self.fixed_transformations:
                events = transform(events)

        if self.task_transformations is not None:
            task_id, task_transform = random.choice(self.task_transformations)
            og_events, events = task_transform(events)
        else:
            task_id = -42
            og_events = events

        return og_events, events, task_id



class RandomExclusiveTransform:
    """
        Create a transform function comprised of multiple task specific transformations that are applied mutually exclusively, meaning at max once at time.
    """
    def __init__(self, transforms):
        """
        Args:
            transforms (list): List of transformations to apply. Each transformation should be a callable that takes events as input.
        """
        self.transforms = transforms

    def __call__(self, x):
        transform = random.choice(self.transforms)  # Pick one transform at random
        return transform(x)




class Downsampling:
    """
        Downsample the events by a spatial factor.
    """
    def __init__(self, spatial_factor):
        """
            Args:
            spatial_factor (int): Factor by which to downsample the spatial dimensions of the events.
        """
        self.spatial_factor = spatial_factor

    def __call__(self, events):
        return events, self.downsample(events, self.spatial_factor)

    @staticmethod
    def downsample(events, factor):
        out_events = events.copy()
        # out_events['x'] = (events['x'] * factor)
        # out_events['y'] = (events['y'] * factor)
        out_events[:, 0] = (events[:, 0] * factor)
        out_events[:, 1] = (events[:, 1] * factor)
        return out_events




class ReorderEvents:
    """
        Reorder the events by slicing them into n slices and shuffling them randomly.
    """
    def __init__(self, slices):
        """
        Args:
            slices (int): Number of slices to split the events into and reorder.
        """
        self.slices = slices

    def __call__(self, events):
        return events, self.reorder_events(events, self.slices)

    @staticmethod
    def reorder_events(events, slices):
        # take n slices from the points and reorder them randomly based on time
        slices = np.array_split(events, slices)
        # shuffle the slices
        np.random.shuffle(slices)
        events = np.concatenate(slices)

        return events




class AddNoiseEvents:
    """
        Add noise events to the original events based on a specified noise level and sensor size.
    """
    def __init__(self, noise_level, sensor_size):
        """
        Args:
            noise_level (float): Percentage of events to replace with noise.
            sensor_size (tuple): Size of the sensor in pixels (width, height).
        """
        self.noise_level = noise_level
        self.sensor_size = sensor_size

    def __call__(self, events):
        return self.add_noise(events, self.noise_level, self.sensor_size)

    @staticmethod
    def add_noise(events, noise_level, sensor_size):
        og_events = events[:int(len(events) * noise_level)].copy()
        noisy_events = og_events
        # take noise_level percentage of events from the original events and add random noise at their place
        noise_array = events[:int(len(events) * noise_level)]
        # generate random noise events
        new_len = len(events) - len(noise_array)
        noise_x = np.random.rand(new_len) * sensor_size[0]
        noise_y = np.random.rand(new_len) * sensor_size[1]
        noise_p = np.random.randint(0, 1, new_len).astype(np.float32)
        noise_ts = np.random.randint(min(events[:int(len(events) * noise_level)][:, 3]),
                                     max(events[:int(len(events) * noise_level)][:, 3]), new_len).astype(np.float32)

        # now stack the noise events to obtain new events
        new_noise_array = np.column_stack((noise_x, noise_y, noise_ts, noise_p))

        # concatenate and reorder the events
        noisy_events = np.concatenate((noise_array, new_noise_array)).astype(np.float32)
        noisy_events = noisy_events[noisy_events[:, 2].argsort()]
        # add padding to og_events
        og_events = np.concatenate((og_events, np.zeros((len(noisy_events) - len(og_events), 4)))).astype(np.float32)
        return og_events, noisy_events


class RemoveLastEvents:
    """
        Remove the last events based on a specified percentage or event count.
    """
    def __init__(self, percentage=None, event_count=None):
        """
        Args:
            percentage (float, optional): Percentage of events to remove from the end. Must be between 0 and 1.
            event_count (int, optional): Number of events to remove from the end. If both are provided, raises an error.
        """
        self.percentage = percentage
        self.event_count = event_count

    def __call__(self, events):
        return events, self.remove_last_events(events, self.percentage, self.event_count)

    @staticmethod
    def remove_last_events(events, percentage=None, event_count=None):
        if percentage is None and event_count is None:
            raise ValueError("Either percentage or event_count must be provided.")
        if percentage is not None and event_count is not None:
            raise ValueError("Only one of percentage or event_count must be provided.")

        masked_events = events.copy()
        if percentage is not None:
            # remove the last percentage of events
            masked_events[int(len(events) * (1 - percentage)):, 0] = -1
            masked_events[int(len(events) * (1 - percentage)):, 1] = -1
        else:
            # remove the last event_count of events
            masked_events[-event_count:, 0] = -1
            masked_events[-event_count:, 1] = -1

        return masked_events


class FixColumnsOrder:
    """
        Fix the order of columns in the events data from (x, y, p, t) to (x, y, t, p).
    """
    def __call__(self, events):
        out_events = events[:, [0, 1, 3, 2]]
        return out_events


class ToNumpyArray:
    """
        Convert the events data to a numpy array.
    """
    def __call__(self, events):
        return np.array(events)
