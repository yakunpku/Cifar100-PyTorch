
class AverageMeter(object):
    """To store the current value and compute the average
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        self._val = 0.
        self._avg = 0.
        self._sum = 0.
        self._count = 0
    
    def update(self, val, n=1):
        self._val = val
        self._sum += val * n
        self._count += n
        self._avg = self._sum / self._count
    
    @property
    def avg(self):
        return self._avg

    @property
    def count(self):
        return self._count
