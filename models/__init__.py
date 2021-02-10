import logging
import models.backbone.resnet_cifar as resnet
logger = logging.getLogger('base')

def define_net(arch, block_name, num_classes=100, pretrained=False):
    if arch.startswith("resnet"):
        depth = int(arch.split('-')[-1])
        net = resnet.ResNet(depth=depth, num_classes=num_classes, block_name=block_name)
    else:
        raise NotImplementedError('The model [{}] is not implemented.'.format(arch))
    logger.info('The network [{}] has created.'.format(arch))
    return net