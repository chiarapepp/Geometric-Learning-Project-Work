from experiments.experiment import Experiment
import torch
import operator
from loss.loss_factory import build_composite_loss

#exp = Experiment('finetune', save_path_root='experiments/', config=None, hash='253468894529340366591182026341267843850')
exp = Experiment('finetune', save_path_root='exp_logs/', config=None, hash='176659180239605430982828111881820356742')
exp.config.losses = ['cross_entropy']
exp.config.loss_weights = [1]
exp.loss_fn = build_composite_loss(exp.config.losses, exp.config.loss_weights)
exp.config.metric_for_best = ('accuracy_majority', operator.gt)
exp.device = torch.device('cuda:1')

#exp.model[-1] = torch.nn.Linear(512, 11).to(exp.device)
exp.model[-1] = torch.nn.Sequential(
    torch.nn.Linear(512, 256),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.5),
    torch.nn.Linear(256, 11)
).to(exp.device)

exp.model = exp.model.to(exp.device)

for param in exp.model[:-1].parameters():
    param.requires_grad = False
exp.optimizer = torch.optim.Adam(exp.model.parameters(), lr=5e-5)

# Pretrain the model
exp.run()