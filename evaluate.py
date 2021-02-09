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
                        help='the classifier network')
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
    network = define_net(args.arch, pretrained=False, num_classes=args.num_classes).to(device)
    checkpoint = load_checkpoint(args.checkpoint_path, logger)
    network.load_state_dict(checkpoint['state_dict'])

    logger.info('best acc: {:.3f}'.format(float(checkpoint['best_acc'].numpy())))
    test_dataloader = create_dataloader(cfg.test_image_dir, cfg.test_image_list, phase='test')

    macs, params = get_model_complexity_info(network, (3, 32, 32), as_strings=True,
                                                print_per_layer_stat=False, verbose=True)
    
    logger.info('Network: {}'.format(args.arch))
    logger.info('{} {}'.format('Computational complexity: ', macs))
    logger.info('{} {}'.format('Number of parameters: ', params))
    top1, top5, _ = Evaluator.eval(network, device, test_dataloader)
    logger.info('Evaluate top1: {0:.3f}, top5: {1:.3f}'.format(top1, top5))


if __name__ == '__main__':
    main()