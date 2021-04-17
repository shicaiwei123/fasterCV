'''
https://github.com/gpleiss/efficient_densenet_pytorch
'''

import torchvision.models as tm


def densenet121(pretrained=False, progress=True, **kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """

    eff_densenet121 = tm.DenseNet(32, (6, 12, 24, 16), 64, memory_efficient=True)

    if pretrained:
        densenet121 = tm.densenet121(pretrained=True)
        state_dict_origin = densenet121.state_dict()
        eff_dict = eff_densenet121.state_dict()

        pretrained_dict = {k: v for k, v in state_dict_origin.items() if k in eff_dict}
        eff_dict.update(pretrained_dict)
        eff_densenet121.load_state_dict(eff_dict)

        print({k: v for k, v in pretrained_dict.items() if k not in eff_dict})
    else:
        return eff_densenet121


def densenet169(pretrained=False, progress=True, **kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """

    eff_densenet169 = tm.DenseNet(32, (6, 12, 48, 32), 64, memory_efficient=True)

    if pretrained:
        densenet169 = tm.densenet169(pretrained=True)
        state_dict_origin = densenet169.state_dict()
        eff_dict = eff_densenet169.state_dict()

        pretrained_dict = {k: v for k, v in state_dict_origin.items() if k in eff_dict}
        eff_dict.update(pretrained_dict)
        eff_densenet169.load_state_dict(eff_dict)

        print({k: v for k, v in pretrained_dict.items() if k not in eff_dict})
    else:
        return eff_densenet169


def densenet201(pretrained=False, progress=True, **kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """

    eff_densenet201 = tm.DenseNet(32, (6, 12, 64, 48), 64, memory_efficient=True)

    if pretrained:
        densenet201 = tm.densenet201(pretrained=True)
        state_dict_origin = densenet201.state_dict()
        eff_dict = eff_densenet201.state_dict()

        pretrained_dict = {k: v for k, v in state_dict_origin.items() if k in eff_dict}
        eff_dict.update(pretrained_dict)
        eff_densenet201.load_state_dict(eff_dict)

        print({k: v for k, v in pretrained_dict.items() if k not in eff_dict})
    else:
        return eff_densenet201
