import sys
sys.path.append('src')
from config.config import DefaultArgs
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from glob import glob
from tqdm import tqdm
from datasets.dataset_factory import get_dataset
from models.model_factory import build_model
from loss.loss_factory import get_loss, build_composite_loss
from normalizers.normalizer_factory import get_normalizer
from train_functions import do_epoch
from utils.wandb_handler import WandbHandler

args = DefaultArgs()
args.print_args()
wandb_handler = WandbHandler(args)

# ------------- DATASET -------------
dataset_train = get_dataset(dataset_name=args.dataset,
                            train=True,
                            N=args.slice_size,
                            stride=args.stride,
                            use_polarity=args.use_polarity)

dataset_test = get_dataset(dataset_name=args.dataset,
                            train=False,
                            N=args.slice_size,
                            stride=args.stride,
                            use_polarity=args.use_polarity)

print(f'Train dataset length: {len(dataset_train)}')
print(f'Test dataset length: {len(dataset_test)}')

# ------------- DATALOADER -------------
train_loader = torch.utils.data.DataLoader(dataset_train,
                                            batch_size=args.train_batch_size,
                                            shuffle=True,
                                            num_workers=args.num_workers)

test_loader = torch.utils.data.DataLoader(dataset_test,
                                            batch_size=args.test_batch_size,
                                            shuffle=False,
                                            num_workers=args.num_workers)

# ------------- MODEL -------------
model = build_model(args.model_structure, args.model_dims, args.latent_size)
print(model)

# ------------- OPTIMIZER -------------
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# ------------- LOSS -------------
loss_fn = build_composite_loss(args.losses, args.loss_weights)

# ------------- TRAINING -------------
device = torch.device(args.device)
model = model.to(device)

normalizer = get_normalizer(args.normalizer)

for epoch in range(args.num_epochs):
    train_loss_dict = do_epoch(train_loader, model, loss_fn, optimizer, normalizer, device, epoch, train=True, logger=wandb_handler)
    print(f'Epoch {epoch}, train losses {train_loss_dict}')

    # ------------- EVALUATION -------------
    test_loss_dict = do_epoch(test_loader, model, loss_fn, optimizer, normalizer, device, epoch, train=False, logger=wandb_handler)
    print(f'Epoch {epoch}, test losses {test_loss_dict}')