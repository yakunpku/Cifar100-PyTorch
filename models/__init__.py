import logging
import models.backbone.resnet as resnet
import models.backbone.shufflenetv2 as shufflenetv2
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
    elif which_model == "shufflenet_v2_x0_5":
        net = shufflenetv2.shufflenet_v2_x0_5(pretrained=pretrained, num_classes=num_classes)
    elif which_model == "shufflenet_v2_x1_0":
        net = shufflenetv2.shufflenet_v2_x1_0(pretrained=pretrained, num_classes=num_classes)
    elif which_model == "shufflenet_v2_x1_5":
        net = shufflenetv2.shufflenet_v2_x1_5(pretrained=pretrained, num_classes=num_classes)
    elif which_model == "shufflenet_v2_x2_0":
        net = shufflenetv2.shufflenet_v2_x2_0(pretrained=pretrained, num_classes=num_classes)
    else:
        raise NotImplementedError('The model [{}] is not implemented.'.format(which_model))
    logger.info('The network [{}] has created.'.format(which_model))
    return net