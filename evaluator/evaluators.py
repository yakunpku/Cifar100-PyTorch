import numpy as np
import time
from utils.eval import accuracy
from utils.meters import AverageMeter
import torch


class Evaluator(object):
    """ To evaluate the target model on dataset.
    """
    @staticmethod
    def eval(tgt_model, device, dataloader, loss_func=None):
        top1 = AverageMeter()
        top5 = AverageMeter()
        losses = None if loss_func is None else AverageMeter()
        
        tgt_model.eval()
        with torch.no_grad():
            for data_iter in dataloader:
                inputs = data_iter['inputs'].to(device)
                targets = data_iter['targets'].to(device, dtype=torch.int64)

                outputs = tgt_model(inputs)
                prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
                top1.update(prec1, inputs.size(0))
                top5.update(prec5, inputs.size(0))

                if loss_func is not None:
                    loss = loss_func(outputs, targets)
                    losses.update(loss.item(), inputs.size(0))

        tgt_model.train()
        return top1.avg, top5.avg, losses.avg if losses is not None else None