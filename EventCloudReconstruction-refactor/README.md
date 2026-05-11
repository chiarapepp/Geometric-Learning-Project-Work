# eventFM

## Installation

`conda env create -f newEventFM.yml`

## Datasets
The file `src/datasets/base_dataset.py` is supposed to contain base classes of datasets using different file types (raw, npy, dat, ...).
To implement a new dataset, extend the base class in a new file named "XxxxDataset", where "Xxxx" is CamelCase name of the dataset.
Check the base dataset to see wich methods should be implemented.

Do not add dependencies to other files in the dataset and do not pass a generic "args" variable to the dataset. Pass arguments explicitly.

### Data organization
Do not use absolute paths. Create a symlink to your dataset folder in the `data` folder.
For example, for DVSGesture:
```ln -s /path/to/DVSGesture data/```

## Models
Models are divided into `encoders` and `decoders`. To define a model you should define a stucture as a list of blocks (usually an encoder followed by a decoder). More complex models can be defined as a single block.

## Loss
Each loss is implemented in a separate file in the `loss` folder. Each loss should take as input the output of the model (`y_hat`) and the ground truth (`y`). Ideally nothing else should be passed to a loss.

## Training
From the src folder run `train/train.py`. You can customize parameters in `src/config/config.py` or by adding parameters directly in you `train.py` script.

## TODO
- optimizer factory (for now adam is used as default)
- an argparser?
- wandb