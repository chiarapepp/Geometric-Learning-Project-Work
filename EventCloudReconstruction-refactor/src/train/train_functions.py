from tqdm import tqdm
from collections import defaultdict
import torch

def do_epoch(loader, model, loss_fn, optimizer, normalizer, device, epoch, train=True, logger=None):
    """
    Perform a single epoch of training or validation.
    Args:
        loader: DataLoader for the dataset.
        model: The model to train or evaluate.
        loss_fn: Loss function to compute the loss.
        optimizer: Optimizer for training.
        normalizer: Normalizer function to preprocess the data.
        device: Device to run the computations on (CPU or GPU).
        epoch: Current epoch number.
        train: Boolean indicating whether it's training or validation.
        logger: Logger for logging metrics and visualizations.
    """
    if train:
        model.train()
    else:
        model.eval()
    loss_dict_epoch = defaultdict(lambda: 0)
    for i, (sample_idx, data, target_cls) in enumerate(tqdm(loader)):
        global_step = i + epoch * len(loader)
        data = data.squeeze(1).to(device) # x, y, t, p
        data = normalizer(data, axis=1)
        optimizer.zero_grad()
        output = model(data)    
        loss, loss_dict = loss_fn(output, data)
        loss_dict_epoch = {k: v + loss_dict_epoch[k] for k, v in loss_dict.items()}

        if logger and epoch > 0 and epoch % logger.config.log_every == 0 and i == 0:
            logger.log_point_cloud(data[0].cpu().numpy(), 'input', global_step, caption=f'Epoch {epoch}')
            output = output[0]
            if output.shape[1] == 4:
                # polarity thresholding
                polarity_mask = output[:,3]>0.5
                output[polarity_mask, 3] = 1
                output[~polarity_mask, 3] = 0
            #else:
            #    output = output[:, [0, 1, 2]]
            logger.log_point_cloud(output.cpu().detach().float().numpy(), 'output', global_step, caption=f'Epoch {epoch}')
        if train:
            loss.backward()
            optimizer.step()
    loss_dict_epoch = {k: v / len(loader) for k, v in loss_dict_epoch.items()}
    if logger is not None:
        for k, v in loss_dict_epoch.items():
            logger.log({f'{'train' if train else 'val'}/{k}': v,
                        'epoch': epoch})
        #if epoch > 0 and epoch % 10 == 0 and not train:
            #torch.save(model, f'model_{epoch}.pth')
            #logger.run.log_model(path=f'model_{epoch}.pth', name=f"model_{epoch}")
    return loss_dict_epoch


def do_epoch_downstream(loader, model, loss_fn, optimizer, normalizer, device, epoch, model_name=None, train=True, logger=None):
    """
    Perform a single epoch of training or validation for downstream tasks.
    Args:
        loader: DataLoader for the dataset.
        model: The model to train or evaluate.
        loss_fn: Loss function to compute the loss.
        optimizer: Optimizer for training.
        normalizer: Normalizer function to preprocess the data.
        device: Device to run the computations on (CPU or GPU).
        epoch: Current epoch number.
        model_name: Name of the model for saving purposes.
        train: Boolean indicating whether it's training or validation.
        logger: Logger for logging metrics and visualizations.
    """
    if train:
        model.train()
    else:
        model.eval()
    loss_dict_epoch = defaultdict(lambda: 0)
    correct = 0
    total = 0
    results = defaultdict(list)
    video_targets = defaultdict(list)
    for i, (sample_idx, data, target_cls) in enumerate(tqdm(loader)):
        global_step = i + epoch * len(loader)
        data = data.squeeze(1).to(device) # x, y, t, p
        target_cls = target_cls.to(device)
        data = normalizer(data, axis=1)
        if train:
            optimizer.zero_grad()
        output = model(data)    
        correct += (output.argmax(1) == target_cls).sum().item()
        total += target_cls.size(0)
        if train:
            loss, loss_dict = loss_fn(output, target_cls)
            loss_dict_epoch = {k: v + loss_dict_epoch[k] for k, v in loss_dict.items()}

        for iii, (batch_sample_idx, batch_output) in enumerate(zip(sample_idx, output)):
            video_id = loader.dataset.video_chunks_ids[batch_sample_idx.item()][0]
            results[video_id].append(batch_output.argmax().item())
            video_targets[video_id].append(target_cls[iii].item())

        if train:
            loss.backward()
            optimizer.step()
    loss_dict_epoch = {k: v / len(loader) for k, v in loss_dict_epoch.items()}
    loss_dict_epoch['accuracy'] = correct/len(loader.dataset)
    
    # majority voting
    for k, v in results.items():
        results[k] = max(set(v), key=v.count)
    for k, v in video_targets.items():
        video_targets[k] = max(set(v), key=v.count)
    loss_dict_epoch['accuracy_majority'] = sum([1 for k, v in results.items() if v == video_targets[k]]) / len(results)


    if logger is not None:
        for k, v in loss_dict_epoch.items():
            logger.log({f'{'train' if train else 'val'}/{k}': v,
                        'epoch': epoch})
        if epoch > 0 and epoch % 10 == 0 and not train:
            if model_name is not None:
                torch.save(model, f'{model_name}_{epoch}.pth')
                logger.run.log_model(path=f'{model_name}_{epoch}.pth', name=f"{model_name}_{epoch}")
            else:
                torch.save(model, f'downstream_model_{epoch}.pth')
                logger.run.log_model(path=f'downstream_model_{epoch}.pth', name=f"model_{epoch}")
    return loss_dict_epoch


def train_class_incremental(loader, model, loss_fn, optimizer, normalizer, device, epoch, model_name=None, train=True, logger=None):
    """
    Train the model in a class-incremental manner, where each class is trained sequentially. No idea where is used though!!!
    """
    model.train()
    loss_dicts = []
    total = [0 for _ in range(11)]
    results = [defaultdict(list) for _ in range(11)]
    video_targets = [defaultdict(list) for _ in range(11)]

    class_to_idx = {i: [] for i in range(11)}
    for i, (sample_idx, data, target_cls) in enumerate(tqdm(loader.dataset)):
        class_to_idx[target_cls].append(sample_idx)
    for i in range(11):
        for sample_id in class_to_idx[i]:
            idx, data, target_cls = loader.dataset[sample_id]
            data = data.to(device)
            target_cls = torch.tensor(target_cls)[None,...].to(device)
            data = normalizer(data, axis=1)
            optimizer.zero_grad()
            output = model(data)

            loss, loss_dict = loss_fn(output, target_cls)
            loss.backward()
            optimizer.step()
    
        loss_dict = do_epoch_downstream(loader, model, loss_fn, optimizer, normalizer, device, epoch, model_name=model_name, train=False, logger=None)
        print(f'Epoch {epoch}, {i} losses {loss_dict}')
        loss_dicts.append(loss_dict)
    return loss_dicts
