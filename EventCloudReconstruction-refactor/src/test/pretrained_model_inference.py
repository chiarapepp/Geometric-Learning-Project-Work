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
from train.train_functions import do_epoch
from utils.wandb_handler import WandbHandler
from utils.point_cloud_utils import to_frame
import cv2
import tonic

args = DefaultArgs()
args.print_args()
args.wandb='disabled'
wandb_handler = WandbHandler(args)
args.device='cuda:1'

# ------------- DATASET -------------
dataset_test = get_dataset(dataset_name=args.dataset,
                            train=False,
                            N=-1,#args.slice_size,
                            stride=args.stride,
                            use_polarity=args.use_polarity)

print(f'Test dataset length: {len(dataset_test)}')

# ------------- DATALOADER -------------
test_loader = torch.utils.data.DataLoader(dataset_test,
                                            batch_size=1, #args.test_batch_size,
                                            shuffle=True,
                                            num_workers=args.num_workers)

# ------------- MODEL -------------
# model = build_model(args.model_structure, args.model_dims, args.latent_size)
# print(model)
model = torch.load('model_170.pth', weights_only=False)

# ------------- INFERENCE -------------
device = torch.device(args.device)
model = model.to(device)

normalizer = get_normalizer(args.normalizer)

save_video = True
visualize_video = False

if save_video:
    writer = cv2.VideoWriter(f'out_frames/out_video_montage_input.avi', cv2.VideoWriter_fourcc(*'XVID'), 10, (128, 128))

counter = 0
for batch in test_loader:
    out_frames = []
    sample_id = batch[0][0]
    num_slices = batch[1].shape[-2]/args.slice_size
    for i in tqdm(range(int(num_slices))):
        cur_slice = batch[1][:,0,i*args.slice_size:(i+1)*args.slice_size]
        cur_slice = cur_slice.to(device)
        cur_slice = normalizer(cur_slice)
        with torch.no_grad():
            output = model(cur_slice)
            # out_frames.append(to_frame(output[0].cpu().numpy(), (128, 128, 3)))
            out_frames.append(to_frame(cur_slice[0].cpu().numpy(), (128, 128, 3)))
    # ------------- VISUALIZATION -------------
    for i, frame in enumerate(out_frames):
        if visualize_video:
            cv2.imshow('frame', frame)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
        if save_video:
            writer.write((frame*255).astype(np.uint8))
    counter +=1
    if counter == 10:
        break
    # break
if save_video:
    writer.release()