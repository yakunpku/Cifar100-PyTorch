import tqdm
import numpy as np
import time
from utils.eval import accuracy
from utils.meters import AverageMeter
import torch
import torch.nn.functional as F


class Evaluator(object):
    """ To evaluate the target model on dataset.
    """
    @staticmethod
    def eval(network, 
            device, 
            dataloader, 
            loss_func=None):
        top1 = AverageMeter()
        top5 = AverageMeter()
        losses = None if loss_func is None else AverageMeter()
        
        network.eval()
        with torch.no_grad():
            for data_iter in dataloader:
                inputs = data_iter['inputs'].to(device)
                targets = data_iter['targets'].to(device, dtype=torch.int64)

                outputs = network(inputs)

                prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
                top1.update(prec1, inputs.size(0))
                top5.update(prec5, inputs.size(0))

                if loss_func is not None:
                    loss = loss_func(outputs, targets)
                    losses.update(loss.item(), inputs.size(0))

        network.train()
        return top1.avg, top5.avg, losses.avg if losses is not None else None

    @staticmethod
    def extract_embedding(network,
                        device,
                        dataloader):
        tgt_layer = 'avgpool'
        assert tgt_layer in network._modules, "The target layer: {} is not in the network. The network modules: [{}]".format(tgt_layer, network._modules)
        embeddings = []
        network.eval()
        with torch.no_grad():
            pbar = tqdm.tqdm(total=len(dataloader.dataset), unit='extracted data batch')
            for idx, data_iter in enumerate(dataloader):
                x = data_iter['inputs'].to(device)
                for name, module in network._modules.items():
                    x = module(x)
                    if name == tgt_layer:
                        x = torch.squeeze(x)
                        if x.dim == 1:
                            x = torch.unsqueeze(x, dim=0)
                        x = x.float().cpu()
                        embeddings.append(x)
                        break
                pbar.update(x.shape[0])
                pbar.set_description("batch index: {}".format(idx))
        return embeddings