import random
from datasets.dataset_factory import get_dataset
import torch
from models.model_factory import build_model
from utils.wandb_handler import WandbHandler
from loss.loss_factory import build_composite_loss
from normalizers.normalizer_factory import get_normalizer
from train.train_functions import do_epoch, do_epoch_downstream
from collections import defaultdict
import os
from datetime import datetime
import numpy as np


class Experiment:
    """
    Base class for experiments. It handles the initialization, training, evaluation and saving of the model.
    """
    def __init__(self, modality, save_path_root, config=None, hash=None):
        """
        modality: str, experiment modality. Either 'pretrain', 'finetune', 'eval'
        config: the configuration object
        hash: str, hash of the experiment. 
        """
        assert modality in ['pretrain', 'finetune', 'eval']
        if modality in ['finetune', 'eval']:
            # When finetuning or evaluating, the hash of a previous experiment must be provided
            assert hash is not None
        self.timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.modality = modality
        self.save_path_root = save_path_root
        self.config = config

        # if hash is provided, load config and model, otherwise reinitialize everything
        if not hash:
            assert config is not None # if hash is not provided, config must be provided
            self.hash = self.generate_hash()
            self.model = build_model(self.config.model_structure, self.config.model_dims, self.config.latent_size)
        else:    
            assert config is None # if hash is provided, config will be overwritten
            self.hash = hash
            self.config = self.load_config(self.hash)
            self.model = self.load_model(hash)
        
        self.dataset = self.config.dataset
        self.wandb_handler = WandbHandler(self.config)      

        self.save_path = os.path.join(self.save_path_root, str(self))
        self.stats = defaultdict(lambda: []) # store the statistics of the experiment
        self.best_metric = None

        # load train dataset only if needed
        if modality in ['pretrain', 'finetune']:
            self.dataset_train = get_dataset(dataset_name=self.config.dataset,
                            train=True,
                            N=self.config.slice_size,
                            stride=self.config.stride,
                            use_polarity=self.config.use_polarity)
            
            self.train_loader = torch.utils.data.DataLoader(self.dataset_train,
                                            batch_size=self.config.train_batch_size,
                                            shuffle=True,
                                            num_workers=self.config.num_workers)
        
        # we always need a test dataset    
        self.dataset_test = get_dataset(dataset_name=self.config.dataset,
                        train=False,
                        N=self.config.slice_size,
                        stride=self.config.stride,
                        use_polarity=self.config.use_polarity)
        
        self.test_loader = torch.utils.data.DataLoader(self.dataset_test,
                                            batch_size=self.config.test_batch_size,
                                            shuffle=False,
                                            num_workers=self.config.num_workers)

        # create optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)

        # loss
        self.loss_fn = build_composite_loss(self.config.losses, self.config.loss_weights)

        # device
        self.device = torch.device(self.config.device)
        self.model = self.model.to(self.device)

        # normalizer
        self.normalizer = get_normalizer(self.config.normalizer)
        self.save_config()

    
    def generate_hash(self):
        """
        Generate a hash for the experiment
        """
        hash = random.getrandbits(128)
        print(f'Generated hash: {hash}')
        return hash
    
    def run(self):
        """
        Run the experiment
        """
        if self.modality == 'pretrain':
            self.pretrain()
        elif self.modality == 'finetune':
            self.finetune()
        elif self.modality == 'eval':
            self.eval()

    def do_phase(self, phase, epoch):
        """
        Either train or test
        """
        if self.modality == 'finetune':
            epoch_fn = do_epoch_downstream
        else:
            epoch_fn = do_epoch
        loss_dict = epoch_fn(getattr(self, f'{phase}_loader'), 
                                           self.model,
                                           self.loss_fn,
                                           self.optimizer,
                                           self.normalizer,
                                           self.device,
                                           epoch,
                                           train= phase == 'train',
                                           logger=self.wandb_handler)
        print(f'Epoch {epoch}, {phase} losses {loss_dict}')
        self.stats[f'{self.modality}_{phase}_metrics'].append(loss_dict)

        # establish if the model is the best till now
        if phase == 'test' and self.config.metric_for_best[0] in loss_dict:
            # metric_for_best is a tuple containing a metric and an operator to find the best value, e.g. ('chamfer', operator.lt) or ('accuracy', operator.gt)
            if self.best_metric is None:
                # first result
                self.best_metric = loss_dict[self.config.metric_for_best[0]]
                self.save_model('best')
            if self.config.metric_for_best[1](loss_dict[self.config.metric_for_best[0]], self.best_metric):
                # if result is better than best, update the best
                self.best_metric = loss_dict[self.config.metric_for_best[0]]
                self.save_model('best')
            if epoch % self.config.save_every_n_epoch == 0 and epoch > 0:
                self.save_model(epoch)

    def train(self):
        """
        Train the model
        """
        for epoch in range(self.config.num_epochs):
            self.do_phase('train', epoch)
            if epoch % self.config.test_every_n_epoch == 0 and epoch > 0:
                self.do_phase('test', epoch)
            self.save_model(epoch)

    def pretrain(self):
        """
        Pretrain the model
        """
        self.train()

    def finetune(self):
        """
        Finetune the model
        """
        self.train()

    def eval(self):
        """
        Evaluate the model
        """
        self.do_phase('test', 0)
        self.save()

    def load_model(self, hash):
        """
        Load the model from the experiment folder
        """
        print('Loading model...')
        return self.load_asset(hash, 'model')

    def load_config(self, hash):
        """
        Load the configuration from the experiment folder
        """
        print('Loading config...')
        return self.load_asset(hash, 'config')
    
    def load_asset(self, hash, asset_name):
        """
        Load the configuration from the experiment folder
        """
        # find correct folder
        print(self.save_path_root)
        folder = [f for f in os.listdir(self.save_path_root) if hash in f]
        print(self.save_path_root)
        print(folder)
        assert len(folder) > 0
        if self.modality == 'eval':
            target_string = 'finetune'
        elif self.modality == 'finetune':
            target_string = 'pretrain'
        else:
            raise NotImplementedError # TODO: handle case where you want to resume pre-training
        folder = list(filter(lambda x: target_string in x, folder))
        assert len(folder) == 1
        folder = folder[0]
        if asset_name == 'model':
            asset = torch.load(os.path.join(self.save_path_root, folder, 'model_best.pth'))
        else:
            asset = np.load(os.path.join(self.save_path_root, folder, f'{asset_name}.npy'), allow_pickle=True)[None][0]
        return asset

    def save_config(self):
        """
        Save the config to the experiment folder
        """
        # create folder if it does not exist
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        np.save(os.path.join(self.save_path, 'config.npy'), self.config)

    def save_model(self, epoch):
        """
        Save the model to the experiment folder
        """
        # create folder if it does not exist
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        model_save_path = os.path.join(self.save_path, f'model_{epoch}.pth')
        torch.save(self.model, model_save_path)
        stats = {}
        for k,v in self.stats.items():
            stats[k] = v
        np.save(os.path.join(self.save_path, 'stats.npy'), stats)
        self.wandb_handler.run.log_model(path=model_save_path, name=f"model_{epoch}")

    def parse_experiment_folder(self, folder):
        """
        Parse the experiment folder to get the hash
        """
        dataset, hash, modality, timestamp = folder.split('_')
        return hash

    def __str__(self):
        return f'{self.dataset}_{self.hash}_{self.modality}_{self.timestamp}'
    
    def __repr__(self):
        return self.__str__()
    
    def __hash__(self):
        return self.hash