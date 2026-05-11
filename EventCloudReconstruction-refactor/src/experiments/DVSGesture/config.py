import json
from types import SimpleNamespace
import sys
import socket
import operator

class DefaultArgs(SimpleNamespace):
    """
    Default arguments for the training configuration.
    """
    def __init__(self):
        """
        Initializes the default arguments for training, dataset, model, and logging.
        The arguments are organized into four categories: training_args, dataset_args, model_args, and logging_args.
        Each category is a dictionary containing relevant parameters.
        """
        self.training_args = {
            'device': 'cuda:0',
            'num_epochs': 500,
            'lr': 5e-5,
            'losses': ['chamfer'],
            'loss_weights': [1],
            'normalizer': 'minmax',
            'train_batch_size': 40,
            'test_batch_size': 4,
            'test_every_n_epoch': 1,
        }

        self.dataset_args = {
            'dataset': 'DVSGesture',
            'slice_size': 4096, # number of events in each sample
            'stride': -1, # stride for slicing the video; -1 indicates no overlap
            'num_workers': 4 if sys.gettrace() is None else 0,
            'use_polarity': False,
        }

        self.model_args = {
            'model_structure': ['PointNetEncoder', 'MLPDecoder'],
            'model_dims': [4 if self.dataset_args['use_polarity'] else 3,
                           (self.dataset_args['slice_size'], 4 if self.dataset_args['use_polarity'] else 3)],
            'latent_size': 512,
        }

        self.logging_args = {
            'wandb': 'online' if sys.gettrace() is None else 'disabled',
            'hostname': socket.gethostname(),
            'wandb_entity': 'eventFM',
            'wandb_project': 'refactor', #f'pretrain_{self.dataset_args["dataset"]}',
            'log_every': 1,
            'save_path': 'experiments/',
            'metric_for_best': ('chamfer_loss', operator.lt),
            'save_every_n_epoch': 10,
        }

        # check if all the keys are unique
        assert len(set(self.training_args.keys()).intersection(self.dataset_args.keys())) == 0
        assert len(set(self.training_args.keys()).intersection(self.model_args.keys())) == 0
        assert len(set(self.dataset_args.keys()).intersection(self.model_args.keys())) == 0
        assert len(set(self.training_args.keys()).intersection(self.logging_args.keys())) == 0
        assert len(set(self.dataset_args.keys()).intersection(self.logging_args.keys())) == 0
        assert len(set(self.model_args.keys()).intersection(self.logging_args.keys())) == 0

        self.__dict__.update(self.training_args)
        self.__dict__.update(self.dataset_args)
        self.__dict__.update(self.model_args)
        self.__dict__.update(self.logging_args)

    def add_args(self, args):
        self.__dict__.update(args)

    def add_arg(self, key, value):
        self.__dict__[key] = value

    def to_json(self):
        return json.dumps(self.__dict__)

    def print_args(self):
        args_keys_checklist = list(self.__dict__.keys())
        args_keys_checklist.remove('training_args')
        args_keys_checklist.remove('dataset_args')
        args_keys_checklist.remove('model_args')
        print('========================')
        print('---- Training args ----')
        for key in self.training_args.keys():
            print(f'{key}: {getattr(self, key)}')
            args_keys_checklist.remove(key)
        print('========================')
        print('---- Dataset args ----')
        for key in self.dataset_args.keys():
            print(f'{key}: {getattr(self, key)}')
            args_keys_checklist.remove(key)
        print('========================')
        print('---- Model args ----')
        for key in self.model_args.keys():
            print(f'{key}: {getattr(self, key)}')
            args_keys_checklist.remove(key)
        print('========================')
        print('---- Logging args ----')
        for key in self.logging_args.keys():
            print(f'{key}: {getattr(self, key)}')
            args_keys_checklist.remove(key)
        print('========================')
        print('---- Remaining args ----')
        for key in args_keys_checklist:
            print(f'{key}: {getattr(self, key)}')
        print('========================')

    def serialize_args(self, file_path):
        with open(file_path, 'w') as f:
            f.write(self.to_json())

    def __repr__(self):
        return self.to_json()


if __name__ == '__main__':
    args = DefaultArgs()
    print(args)
    args.print_args()