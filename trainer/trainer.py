import torch
import time
from utils.meters import SumMeter, AverageMeter
from utils.serialization import save_checkpoint
from evaluator.evaluators import Evaluator


class Trainer(object):
    def __init__(self, args, device, network, optimizer, lr_scheduler, loss_func, train_dataloader, test_dataloader, model_store_path, logger):
        self.args = args
        self.device = device
        self.network = network
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_func = loss_func
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.model_store_path = model_store_path
        self.logger = logger

    def train(self):
        self.network.train()

        time_meter = AverageMeter()
        accuracy_meter = SumMeter()
        loss_meter = AverageMeter()
        for epoch in range(self.args.num_epochs):
            start_time = time.time()
            for data_iter in self.train_dataloader:
                inputs = data_iter['inputs'].to(self.device)
                labels = data_iter['labels'].to(self.device, dtype=torch.int64)
                
                ## optimize parameters
                self.optimizer.zero_grad()
                outputs = self.network(inputs)
                loss = self.loss_func(outputs, labels)
                preds = torch.max(outputs, 1)[1]
                loss.backward()
                self.optimizer.step()

                accuracy_meter.update((preds == labels).sum().cpu().numpy(), inputs.size(0))
                loss_meter.update(loss.item())

            train_accuracy = accuracy_meter.avg
            train_loss = loss_meter.avg
            test_accuracy, test_loss = Evaluator.eval(self.network, self.device, self.test_dataloader, self.loss_func)

            end_time = time.time()
            cost_time = end_time - start_time
            time_meter.update(cost_time)
            self.logger.info('Epoch: {0} / {1}, cost time: {2:.3f} s, ETA: {3:.3f} h, train loss: {4:.3f}, train accuracy: {5:.3f}%, test loss: {6:.3f}, test accuracy: {7:.3f}%'.format(epoch, self.args.num_epochs, cost_time, (time_meter.avg*(self.args.num_epochs-epoch-1)) / 3600, train_loss, train_accuracy * 100, test_loss, test_accuracy * 100))

            if epoch % self.args.checkpoint_cycle == 0:
                save_checkpoint(self.network, self.args.which_model, epoch, self.model_store_path, self.logger)
            
            self.lr_scheduler.step()
