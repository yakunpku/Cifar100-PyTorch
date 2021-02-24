import math
from bisect import bisect_right
import torch 
from torch.optim.lr_scheduler import _LRScheduler


def _get_warmup_factor_at_iter(
        method:str,
        iter:int,
        warmup_iters:int,
        warmup_factor:float,
    ):
    if iter >= warmup_iters:
        return 1.0
    if method == "constant":
        return warmup_factor
    elif method == "linear":
        alpha = iter / warmup_iters
        return warmup_factor * (1 - alpha) + alpha
    else:
        raise NotImplementedError('The warmup method: {} is not implemented.'.format(method))


class WarmUpMultiStepLR(_LRScheduler):
    def __init__(self, 
                optimizer, 
                milestones:list,
                gamma:float=0.1,
                warmup_factor:float=1.0e-4,
                warmup_iters:int=2,
                warmup_method:str='linear',
                last_epoch:int=-1,
        ):
        if not (list(milestones) == sorted(milestones)):
            raise ValueEroor("Milestones should be a list of increasing intergers. Got {}".format(milestones))
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmUpMultiStepLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        warmup_factor = _get_warmup_factor_at_iter(self.warmup_method, self.last_epoch, self.warmup_iters, self.warmup_factor)
        return [
            base_lr 
            * warmup_factor 
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]


class WarmUpCosineLR(_LRScheduler):
    def __init__(self, 
                optimizer, 
                max_iters:int,
                warmup_factor:float=1.0e-3,
                warmup_iters:int= 2,
                warmup_method:str='linear',
                last_epoch:int=-1,
        ):
        self.max_iters = max_iters
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmUpCosineLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        warmup_factor = _get_warmup_factor_at_iter(self.warmup_method, self.last_epoch, self.warmup_iters, self.warmup_factor)
        return [
            base_lr 
            * warmup_factor 
            * 0.5
            * (1.0 + math.cos(math.pi * self.last_epoch  / self.max_iters))
            for base_lr in self.base_lrs
        ]