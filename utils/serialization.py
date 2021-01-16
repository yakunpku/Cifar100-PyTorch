import os
import torch
import torch.nn as nn


def save_checkpoint(network, network_label, iter_label, checkpoint_store_path, logger=None):
    checkpoint_filename = 'checkpoint_{:03d}_{}.pth'.format(iter_label, network_label)
    os.makedirs(checkpoint_store_path, exist_ok=True)

    save_path = os.path.join(checkpoint_store_path, checkpoint_filename)
    if isinstance(network, nn.parallel.DataParallel) or isinstance(network, nn.parallel.DistributedDataParallel):
        network = network.module
    state_dict = network.state_dict()
    for key, param in state_dict.items():
        state_dict[key] = param.cpu()
    torch.save(state_dict, save_path)
    if logger is not None:
        logger.info('Epoch: {}, save checkpoint: {} successful!'.format(iter_label, save_path))


def load_checkpoint(tgt_network, checkpoint_path, logger=None):
    tgt_network.load_state_dict(torch.load(checkpoint_path))
    if logger is not None:
        logger.info('Load checkpoint: {} successful!'.format(checkpoint_path))
    return tgt_network