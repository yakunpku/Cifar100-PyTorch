import numpy as np
import time
import torch
from utils.meters import AverageMeter, SumMeter


class Evaluator(object):
    """ To evaluate the target model on dataset.
    """
    @staticmethod
    def eval(tgt_model, device, dataloader, loss_func=None):
        accuracy_meter = SumMeter()
        loss_meter = None if loss_func is None else AverageMeter()
        
        tgt_model.eval()
        with torch.no_grad():
            for data_iter in dataloader:
                inputs = data_iter['inputs'].to(device)
                labels = data_iter['labels']

                outputs = tgt_model(inputs)
                preds = torch.max(outputs, 1)[1].cpu()

                accuracy_meter.update((preds == labels).sum().cpu().numpy(), inputs.size(0))
                if loss_func is not None:
                    loss = loss_func(outputs, labels.to(device, dtype=torch.int64))
                    loss_meter.update(loss.item())

        tgt_model.train()

        accuracy = accuracy_meter.avg
        return accuracy, loss_meter.avg if loss_meter is not None else None