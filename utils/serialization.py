import os
import torch
import torch.nn as nn
import shutil


def save_checkpoint(network, arch, block_name, epoch, test_top1, best_acc, optimizer, checkpoint_store_path, is_best=False, logger=None):
    checkpoint_filename = 'checkpoint_{:03d}_{}_{}.pth'.format(epoch, arch, block_name)

    checkpoint_save_path = os.path.join(checkpoint_store_path, checkpoint_filename)
    if isinstance(network, nn.parallel.DataParallel) or isinstance(network, nn.parallel.DistributedDataParallel):
        network = network.module

    network_state_dict = network.state_dict()
    for key, param in network_state_dict.items():
        network_state_dict[key] = param.cpu()

    checkpoint_dict = {
        'arch': arch,
        'block_name': block_name,
        'epoch': epoch,
        'state_dict': network_state_dict,
        'acc': test_top1,
        'best_acc': best_acc,
        'optimizer': optimizer.state_dict()
    }

    torch.save(checkpoint_dict, checkpoint_save_path)
    if is_best:
        shutil.copyfile(checkpoint_save_path, os.path.join(checkpoint_store_path, 'checkpoint_best.pth'))
    if logger is not None:
        logger.info('Epoch: {}, save checkpoint: {} successfully!'.format(epoch, checkpoint_save_path))


def load_checkpoint(checkpoint_path, logger=None):
    checkpoint = torch.load(checkpoint_path)
    if logger is not None:
        logger.info('Load checkpoint: {} successfully!'.format(checkpoint_path))
    return checkpoint