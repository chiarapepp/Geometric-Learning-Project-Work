from config.config import DefaultArgs
import torch
from datasets.dataset_factory import get_dataset
from models.model_factory import build_model
from loss.loss_factory import get_loss, build_composite_loss
from normalizers.normalizer_factory import get_normalizer
from train.train_functions import do_epoch, do_epoch_downstream
from utils.wandb_handler import WandbHandler

args = DefaultArgs()
args.print_args()
args.device='cuda:1'
args.wandb='disabled'
wandb_handler = WandbHandler(args)

# ------------- DATASET -------------
dataset_test = get_dataset(dataset_name='MICCGesture',#args.dataset,
                            train=False,
                            N=args.slice_size,
                            stride=args.stride,
                            use_polarity=args.use_polarity)

print(f'Test dataset length: {len(dataset_test)}')

# ------------- DATALOADER -------------
test_loader = torch.utils.data.DataLoader(dataset_test,
                                            batch_size=128,#args.test_batch_size,
                                            shuffle=False,
                                            num_workers=args.num_workers)

# ------------- MODEL -------------
#model = build_model(args.model_structure, args.model_dims, args.latent_size)
model = torch.load('downstream_model_70.pth', weights_only=False)
model.eval()

# ------------- TEST -------------
device = torch.device(args.device)
model = model.to(device)

normalizer = get_normalizer(args.normalizer)

with torch.no_grad():
        test_loss_dict = do_epoch_downstream(test_loader, model, None, None, normalizer, device, 0, train=False, logger=wandb_handler)
print(f'Test losses {test_loss_dict}')