from experiments.experiment import Experiment
import os
import torch
import numpy as np
import operator
from loss.loss_factory import build_composite_loss
from train.train_functions import train_class_incremental

#exp = Experiment('finetune', save_path_root='experiments/', config=None, hash='253468894529340366591182026341267843850')
exp = Experiment('finetune', save_path_root='exp_logs/', config=None, hash='176659180239605430982828111881820356742')
exp.config.losses = ['cross_entropy']
exp.config.loss_weights = [1]
exp.loss_fn = build_composite_loss(exp.config.losses, exp.config.loss_weights)
exp.config.metric_for_best = ('accuracy_majority', operator.gt)
exp.device = torch.device('cuda:1')

exp.model[-1] = torch.nn.Linear(512, 11).to(exp.device)


exp.model = exp.model.to(exp.device)

for param in exp.model[:-1].parameters():
    param.requires_grad = False
exp.optimizer = torch.optim.Adam(exp.model.parameters(), lr=1e-11)

# Pretrain the model
train_class_incremental(exp.train_loader, exp.model, exp.loss_fn, exp.optimizer, exp.normalizer, exp.device, 0, model_name='incremental', train=True, logger=None)