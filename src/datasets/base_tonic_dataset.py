import os.path
from collections.abc import Callable
from torch.utils.data import Dataset


class BaseTonicDataset(Dataset):
    """
    Minimal base class for neuromorphic datasets.

    Child classes should:
    - populate self.data and self.targets
    - implement __getitem__
    - implement _check_exists
    - optionally implement download
    """

    sensor_size = None
    dtype = None
    ordering = None

    def __init__(
        self,
        save_to: str,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        transforms: Callable | None = None,
    ):
        super().__init__()
        self.location_on_system = os.path.join(
            os.path.expanduser(save_to), self.__class__.__name__
        )
        self.transform = transform
        self.target_transform = target_transform
        self.transforms = transforms

        self.data = []
        self.targets = []
        self.folder_name = ""

    def __repr__(self):
        return self.__class__.__name__

    def _check_exists(self):
        raise NotImplementedError

    def download(self):
        raise NotImplementedError
