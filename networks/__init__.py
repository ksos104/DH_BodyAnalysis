from __future__ import absolute_import

from networks.AugmentCE2P import resnet101, resnet50, resnet18
# from networks.AugmentCE2P_t2t_vit import t2tnet

__factory = {
    'resnet101': resnet101,
    'resnet50': resnet50,
    'resnet18': resnet18
}


def init_model(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown model arch: {}".format(name))
    return __factory[name](*args, **kwargs)