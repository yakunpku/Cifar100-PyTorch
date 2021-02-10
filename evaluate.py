import os
import random
import numpy as np
import logging
import torch
from config import setup_logger
from config import Config as cfg
from utils.serialization import load_checkpoint
from datasets import create_dataloader 
from models import define_net
from evaluator.evaluators import Evaluator
from ptflops import get_model_complexity_info


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='The args for training the classifier on cifar-100 datasets.')
    parser.add_argument('--arch', type=str,
                        required=True, 
                        choices=['resnet-20', 'resnet-56', 'resnet-110'],
                        help='the architecture of classifier network, which is only support : [resnet-20, resnet-56, resnet-110] currently!')
    parser.add_argument('--block-name', type=str, default='BasicBlock',
                    help='the building block for resnet : BasicBlock, Bottleneck (default: Basicblock for cifar10/cifar100)')
    parser.add_argument('--checkpoint-path', type=str,
                        required=True, 
                        help='the pretrained classification model path')
    parser.add_argument('--num-classes', type=int,
                        default=100,
                        help='the number of classes in the classification dataset')
    parser.add_argument('--gpu', type=int, 
                        default=0, 
                        help='to assign the gpu to train the network')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    setup_logger('base', 'test')
    logger = logging.getLogger('base')

    device = "cuda:{}".format(args.gpu)
    network = define_net(args.arch, args.block_name, args.num_classes).to(device)
    checkpoint = load_checkpoint(args.checkpoint_path, logger)
    network.load_state_dict(checkpoint['state_dict'])

    logger.info('Best Acc: {:.3f}'.format(float(checkpoint['best_acc'].numpy())))
    test_dataloader = create_dataloader(cfg.test_image_dir, cfg.test_image_list, phase='test')

    macs, params = get_model_complexity_info(network, (3, 32, 32), as_strings=True,
                                                print_per_layer_stat=False, verbose=True)
    
    logger.info('Network: {}'.format(args.arch))
    logger.info('{} {}'.format('Computational Complexity: ', macs))
    logger.info('{} {}'.format('Number of Parameters: ', params))
    top1, top5, _ = Evaluator.eval(network, device, test_dataloader)
    logger.info('Evaluate Acc Top1: {0:.3f}%, Acc Top5: {1:.3f}%'.format(top1, top5))


if __name__ == '__main__':
    main()