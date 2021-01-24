import os
import torch
import torch.nn as nn
import shutil


def save_checkpoint(network, network_label, epoch, test_top1, best_acc, optimizer, checkpoint_store_path, is_best=False, logger=None):
    checkpoint_filename = 'checkpoint_{:03d}_{}.pth'.format(epoch, network_label)
    os.makedirs(checkpoint_store_path, exist_ok=True)

    checkpoint_save_path = os.path.join(checkpoint_store_path, checkpoint_filename)
    if isinstance(network, nn.parallel.DataParallel) or isinstance(network, nn.parallel.DistributedDataParallel):
        network = network.module
    model_state_dict = network.state_dict()
    for key, param in model_state_dict.items():
        model_state_dict[key] = param.cpu()
    
    checkpoint_dict = {
        'epoch': epoch,
        'state_dict': model_state_dict,
        'acc': test_top1,
        'best_acc': best_acc,
        'optimizer': optimizer.state_dict()
    }

    torch.save(checkpoint_dict, checkpoint_save_path)
    if is_best:
        shutil.copyfile(checkpoint_save_path, os.path.join(checkpoint_store_path, 'checkpoint_best.pth'))
    if logger is not None:
        logger.info('Epoch: {}, save checkpoint: {} successful!'.format(epoch, checkpoint_save_path))


def load_checkpoint(checkpoint_path, logger=None):
    checkpoint = torch.load(checkpoint_path)
    if logger is not None:
        logger.info('Load checkpoint: {} successful!'.format(checkpoint_path))
    return checkpoint