sys.path.append('src')

import comet_ml
import torch
import random
import numpy as np
import argparse
import sys
import os
import datetime
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
from torch import nn

import socket

# for run
from models.simpleCoder import PointCloudAE
from models.model_downstream_classification import model_downstream_classification_transformer
from utils.normalizers import Normalizer

from datasets.dataset_event import EventDataset, collate_events_raw
from datasets.sliced_dataset import SlicedDatasetNpy, DatasetNpy

# disable scientfici notation
np.set_printoptions(suppress=True)
torch.set_printoptions(sci_mode=False)


# create class
class Trainer():
    """
    Trainer class for training and validating the model on a downstream classification task.
    """
    def __init__(self, args):
        self.args = args
        self.set_experiment()
        print('loading dataset...')
        self.set_dataset()
        print('dataset loaded')
        print('setting normalizer...')
        self.set_normalizer()
        print('setting model...')
        self.set_model()
        print('model set')
        self.set_optimizer()
        print('optimizer set')
        self.set_loss()
        print('loss set')

    def set_normalizer(self):
        self.normalizer = Normalizer(self.args)
        self.normalizer.to(self.device)

    def set_experiment(self):

        date_time = str(datetime.datetime.now())[:19]
        self.folder_test = f'TRAININGS/DOWNSTREAM/CLASSIFICATION/{self.args.dataset}/{args.info}/'
        if not os.path.exists(self.folder_test):
            os.makedirs(self.folder_test)

        if sys.gettrace() is not None:
            disable_comet = True
            self.num_workers = 0
        else:
            disable_comet = False
            self.num_workers = 8
            if socket.gethostname() == 'tatooine':
                self.num_workers = 0
        # open txt file to get comet key
        with open(args.comet_key, 'r') as file:
            comet_key = file.read().replace('\n', '')

        name_project = 'downstream_classification/' + self.args.dataset
        if socket.gethostname() == 'tatooine':
            name_project = f'prove_beca'
        self.logging = args.log
        if self.logging:
            self.log_writer = comet_ml.Experiment(api_key=comet_key, project_name=name_project,
                                                  workspace="event-workspace", display_summary_level=0,
                                                  disabled=disable_comet,
                                                  log_code=True, auto_metric_logging=True)
            info_name = f'{self.args.info}'
            self.log_writer.set_name(info_name)
            self.log_writer.log_parameters(self.args)
            self.log_writer.log_code(folder="src")
        else:
            self.log_writer = None
        self.device = args.device
        self.use_amp = self.args.use_amp
        self.global_iteration = 0
        self.global_metric = 0.0
        # self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.save_plots = args.save_plots
        self.dataset_name = args.dataset
        self.temporal_unnorm = args.temporal_unnorm
        if self.save_plots:
            if not os.path.exists(f'{self.folder_test}/plots'):
                os.makedirs(f'{self.folder_test}/plots')

    def set_loss(self):
        loss_name = args.loss
        if loss_name == 'CE':
            self.loss = nn.CrossEntropyLoss()
        self.loss.to(self.device)

    def set_model(self):
        model = args.model
        # checkpoint_backbone = torch.load(args.backbone_checkpoint)
        #backbone = PointCloudAE(args, self.normalizer)
        backbone = PointCloudAE(args, self.normalizer)
        # backbone.load_state_dict(checkpoint_backbone['model'])

        if model == 'downstream_classification':
            #self.model = model_downstream_classification(self.args, backbone, self.normalizer)
            self.model = model_downstream_classification_transformer(self.args, backbone, self.normalizer)
        else:
            raise ValueError(f'Model {model} not implemented')

        # freeze backbone
        for param in self.model.backbone.parameters():
           param.requires_grad = False


        self.model.to(self.device)

    def set_dataset(self):
        if self.args.fast_loader:
            self.data_train = DatasetNpy(self.args, N=args.slicing_time_window, train=True)
            self.data_val = DatasetNpy(self.args,N=args.slicing_time_window, train=False)
            self.loader_train = DataLoader(self.data_train, batch_size=self.args.batch_size, shuffle=True, num_workers=self.num_workers)
            self.loader_val = DataLoader(self.data_val, batch_size=self.args.batch_size, shuffle=False, num_workers=self.num_workers)
        else:
            self.data_train = EventDataset(self.args, split='train', dataset_name=self.dataset_name)
            self.data_val = EventDataset(self.args, split='val', dataset_name=self.dataset_name)
            self.loader_train = DataLoader(self.data_train, batch_size=self.args.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=lambda batch: collate_events_raw(batch, max_length=self.args.slicing_time_window))
            self.loader_val = DataLoader(self.data_val, batch_size=self.args.batch_size, shuffle=False, num_workers=self.num_workers)
        self.dataset_dimension = self.data_train.dimensions
        self.plot_dimension = tuple(float(num) - 1.0 for num in self.dataset_dimension)[:2]
        self.args.frame_dim = self.plot_dimension

        if self.dataset_name == 'NMNIST':
            self.num_classes = 10
        elif self.dataset_name == 'DVSGesture':
            self.num_classes = 11
        elif self.dataset_name == 'ASLDVS':
            self.num_classes = 24
        else:
            raise ValueError(f'Dataset {self.dataset_name} not implemented')
        self.args.num_classes = self.num_classes


        #print len of dataset
        if self.logging:
            self.log_writer.log_other('len_train', len(self.data_train))
            self.log_writer.log_other('len_val', len(self.data_val))
        print(f'len train: {len(self.data_train)}')
        print(f'len val: {len(self.data_val)}')

    def set_optimizer(self):
        self.opt = Adam(self.model.parameters(), lr=self.args.lr)

    def train(self):
        print('start training!')
        for epoch in range(self.args.num_epochs):
            # VALIDATION PHASE
            if epoch > 0:
                self.val(epoch)

            # TRAINING PHASE
            self.model.train()
            for i, batch in tqdm(enumerate(self.loader_train), total=len(self.loader_train),
                                 desc=f'Training - Epoch {epoch}', leave=False):
                self.opt.zero_grad()
                # self.opt.zero_grad(set_to_none=True)
                dict_input = self.prepare_batch(batch)
                # check for nan

                out = self.model(dict_input)
                loss = self.loss(out, dict_input['target'])
                loss.backward()
                nn.utils.clip_grad_value_(self.model.parameters(), clip_value=0.5)

                self.opt.step()

                if self.global_iteration % 100 == 0:
                    if self.logging:
                        self.log_writer.log_metric('train_loss', loss.item(), step=self.global_iteration)
                    print(f'Epoch {epoch} - Iteration {self.global_iteration} - Loss: {loss.item()}')
                self.global_iteration += 1

    def prepare_batch(self, batch):
        tonic = False
        if self.args.fast_loader == False:
            idx = batch[0]
            event = batch[1].to(self.device).float()
            og_event = batch[2].to(self.device)
            target = batch[2].to(self.device)
        else:
            idx = batch[0]
            event = batch[1].to(self.device).float()
            og_event = batch[2].to(self.device)
            target = batch[3].to(self.device)
        #mask = batch[4].to(self.device)
        dict_batch = {
            'idx': idx,
            'events': event[:, :, :].contiguous(),
            'target': target,
            'mask': torch.tensor(0)# mask
        }
        return dict_batch


    def val(self, epoch):
        self.model.eval()
        loss_total = 0.0
        out_total = []
        gt_total = []
        target_total = []
        for i, batch in tqdm(enumerate(self.loader_val), total=len(self.loader_val), desc=f'Validation - Epoch {epoch}',
                             leave=False):
            with torch.no_grad():
                dict_input = self.prepare_batch(batch)
                out = self.model(dict_input)
                # loss = self.loss(out, dict_input['target'])

                # collect val data
                out_pred = out.argmax(dim=1).median()
                out_total.append(out_pred.item())
                target_total.append(dict_input['target'].item())

            # loss_total += loss.item()

        loss_total /= len(self.loader_val)
        if self.logging:
            self.log_writer.log_metric('val_loss', loss_total, epoch=epoch)

        # out_total = torch.cat(out_total, dim=0)
        # target_total = torch.cat(target_total, dim=0)

        # compute metrics: accuracy
        print(out_total)
        print(target_total)

        # conf matrix
        import sklearn.metrics as metrics
        conf_matrix = metrics.confusion_matrix(target_total, out_total)
        print(conf_matrix)

        accuracy = np.sum(np.array(out_total) == np.array(target_total))/len(target_total) * 100
        metric = accuracy

        if self.logging:
            self.log_writer.log_metric('val_accuracy', accuracy, epoch=epoch)

        if metric > self.global_metric:
            print('saving best model')
            print(f'Epoch {epoch} - METRIC: {metric}')
            self.global_metric = metric
            dict_save = {
                'args': self.args,
                'model': self.model.state_dict(),
                'epoch': epoch,
                'metric': metric,
            }
            torch.save(dict_save, f'{self.folder_test}/model_best.pth')




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # GENERAL
    parser.add_argument('--info', type=str, default='debug', help='name of comet experiment')
    parser.add_argument('--debug',  action='store_true', default=False, help='iif run is in debug mode')
    parser.add_argument('--seed', type=int, default=632, help='seed')
    parser.add_argument('--use_amp', action='store_true', default=True)
    parser.add_argument('--comet_key', type=str, default='comet_key.txt', help='comet key')
    parser.add_argument('--save_plots', action='store_true', default=True, help='save plots during training')
    parser.add_argument('--log', action='store_true', default=True, help='log')
    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--sampler', action='store_true', default=False, help='sampler')
    # TRAINING
    parser.add_argument('--phase', type=str, default='downstream', choices=['pretrain', 'downstream'], help='num_epochs')
    parser.add_argument('--num_epochs', type=int, default=50, help='num_epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
    parser.add_argument('--lr', type=float, default=1e-4, help='lr')
    parser.add_argument('--dataset', type=str, default='DVSGesture', help='dataset', choices=['DVSGesture', 'NMNIST'])
    parser.add_argument('--model', type=str, default='downstream_classification', help='model', choices=['downstream_classification'])
    parser.add_argument('--loss', type=str, default='CE', help='loss', choices=['CE'])
    parser.add_argument('--temporal_unnorm', default=False, help='temporal unnorm')
    parser.add_argument('--slicing_time_window', type=int, default=409600*2, help='slicing_time_window')
    parser.add_argument('--normalize_type', type=str, default='mean_variance_local', help='normalizer',
                        choices=['zero_one', 'mean_variance_local'])
    parser.add_argument('--weighted_components', action='store_true', default=False, help='weighted_components')
    parser.add_argument('--metric_losses', action='store_true', default=False, help='compute additional metric losses')

    # DATASET
    parser.add_argument('--input_dim', type=int, default=4096, help='input_dim')  #NMNIST: *16   -   DVS: 3115

    # MODEL
    parser.add_argument('--backbone_checkpoint', type=str, default='TRAININGS/DVSGesture/prova/model_best.pth', help='backbone_checkpoint')
    parser.add_argument('--num_heads', type=int, default=1, help='num heads')
    parser.add_argument('--num_layers', type=int, default=2, help='num layers')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout')
    parser.add_argument('--dim_embed', type=int, default=256, help='dim_embed')
    parser.add_argument('--latent_size', type=int, default=256, help='latent size')
    parser.add_argument('--onlyfirst', type=bool, default=True, help='only one feature for sequence')
    parser.add_argument('--out_size', type=int, default=4096, help='out_size')
    parser.add_argument('--get_event_hist', type=bool, default=False, help='')
    parser.add_argument('--fast_loader', type=bool, default=True, help='')



    args = parser.parse_args()
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    DEBUG = (sys.gettrace() is not None)
    trainer = Trainer(args)
    trainer.model.load_state_dict(torch.load('/media/becattini/HD8TB/workspace/eventFM/model_best_downstream.pth')['model'])
    trainer.val(0)