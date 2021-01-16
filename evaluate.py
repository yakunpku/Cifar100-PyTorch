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
    parser = argparse.ArgumentParser(description='The args for training the classifier on pedestrian trajectory datasets.')
    parser.add_argument('--which-model', type=str,
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
    tgt_network = define_net(args.which_model, pretrained=False, num_classes=args.num_classes).to(device)
    tgt_network = load_checkpoint(tgt_network, args.checkpoint_path, logger)

    test_dataloader = create_dataloader(cfg.test_image_dir, cfg.test_image_list, phase='test')

    macs, params = get_model_complexity_info(tgt_network, (3, 32, 32), as_strings=True,
                                           print_per_layer_stat=False, verbose=True)
    
    logger.info('Network: {}'.format(args.which_model))
    logger.info('{} {}'.format('Computational complexity: ', macs))
    logger.info('{} {}'.format('Number of parameters: ', params))
    accuracy, _ = Evaluator.eval(tgt_network, device, test_dataloader)
    logger.info('Evaluate accuacy: {:.3f}'.format(accuracy))


if __name__ == '__main__':
    main()