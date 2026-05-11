from experiments.experiment import Experiment
from config import DefaultArgs
import os
import torch
import numpy as np

config = DefaultArgs()
exp = Experiment('pretrain', save_path_root='exp_logs/', config=config)

# Pretrain the model
exp.run()