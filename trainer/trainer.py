import os
import torch
import numpy as np
import time
from utils.meters import AverageMeter
from utils.eval import accuracy
from utils.serialization import save_checkpoint
from evaluator.evaluators import Evaluator


class Trainer(object):
    def __init__(self, 
                args, 
                device, 
                start_epoch, 
                network, 
                metric_fc,
                optimizer, 
                lr_scheduler, 
                loss_func, 
                train_dataloader, 
                test_dataloader, 
                model_store_path, 
                logger):
        self.args = args
        self.device = device
        self.start_epoch = start_epoch
        self.network = network
        self.metric_fc = metric_fc
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_func = loss_func
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.model_store_path = model_store_path
        self.logger = logger

    def train(self):
        self.network.train()
        self.metric_fc.train()

        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        valid_logs = []
        best_acc = 0
        for epoch in range(self.start_epoch, self.args.num_epochs):
            start_time = time.time()
            for data_iter in self.train_dataloader:
                inputs = data_iter['inputs'].to(self.device)
                targets = data_iter['targets'].to(self.device, dtype=torch.int64)

                ## compute outputs
                features = self.network(inputs)
                outputs = self.metric_fc(features, targets)
                loss = self.loss_func(outputs, targets)

                ## measure accuracy and record loss
                prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
                losses.update(loss.item(), inputs.size(0))
                top1.update(prec1, inputs.size(0))
                top5.update(prec5, inputs.size(0))

                ## optimize parameters
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            test_top1, test_top5, test_loss = Evaluator.eval(self.network, self.metric_fc, self.device, self.test_dataloader, self.loss_func)
            current_lr = self.optimizer.param_groups[0]['lr']

            valid_logs.append((epoch, test_top1, test_top5))

            end_time = time.time()
            cost_time = end_time - start_time
            batch_time.update(cost_time)
            self.logger.info("Epoch: {0} / {1}, Cost Time: {2:.3f} s, ETA: {3:.3f} h, LR: {4:.5f}" 
            " Train Loss: {5:.3f}, Train Top1: {6:.3f}%, Train Top5: {7:.3f}%," 
            " Test Loss: {8:.3f}, Test Top1: {9:.3f}%, Test Top5: {10:.3f}%".format(
                epoch, self.args.num_epochs, cost_time, (batch_time.avg*(self.args.num_epochs-epoch-1)) / 3600, current_lr, 
                losses.avg, top1.avg, top5.avg, 
                test_loss, test_top1, test_top5))
            
            if epoch % self.args.checkpoint_cycle == 0:
                best_acc = max(best_acc, test_top1)
                save_checkpoint(self.network, self.metric_fc, self.args.arch, epoch, test_top1, best_acc, self.optimizer, 
                    self.model_store_path, is_best=(best_acc == test_top1), logger=self.logger)
            
            self.lr_scheduler.step()
        
        np.savetxt(os.path.join(self.model_store_path, 'valid_logs.list'), valid_logs, fmt='%f')