import logging
import models.backbone.resnet as resnet
logger = logging.getLogger('base')

def define_net(which_model, pretrained, num_classes=2):
    if which_model == "resnet18":
        net = resnet.resnet18(pretrained=pretrained, num_classes=num_classes)
    elif which_model == "resnet34":
        net = resnet.resnet34(pretrained=pretrained, num_classes=num_classes)
    elif which_model == "resnet50":
        net = resnet.resnet50(pretrained=pretrained, num_classes=num_classes)
    elif which_model == "resnet101":
        net = resnet.resnet101(pretrained=pretrained, num_classes=num_classes)
    elif which_model == "resnet152":
        net = resnet.resnet152(pretrained=pretrained, num_classes=num_classes)
    else:
        raise NotImplementedError('The model [{}] is not implemented.'.format(which_model))
    logger.info('The network [{}] has created.'.format(which_model))
    return net