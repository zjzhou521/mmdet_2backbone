import warnings

from mmcv.utils import Registry, build_from_cfg
from torch import nn

BACKBONES = Registry('backbone')
NECKS = Registry('neck')
ROI_EXTRACTORS = Registry('roi_extractor')
SHARED_HEADS = Registry('shared_head')
HEADS = Registry('head')
LOSSES = Registry('loss')
DETECTORS = Registry('detector')

def build(cfg, registry, default_args=None):
    """Build a module.

    Args:
        cfg (dict, list[dict]): The config of modules, is is either a dict
            or a list of configs.
        registry (:obj:`Registry`): A registry the module belongs to.
        default_args (dict, optional): Default arguments to build the module.
            Defaults to None.

    Returns:
        nn.Module: A built nn module.
    """
    if isinstance(cfg, list):
        modules = [build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg]
        return nn.Sequential(*modules)
    else:
        # here runs 11 times
        sub_module = build_from_cfg(cfg, registry, default_args)
        # print("sub_module = ",sub_module)
        return sub_module


def build_backbone(cfg):
    """Build backbone."""
    backbone = build(cfg, BACKBONES)
    # print("backbone = ",backbone)
    return backbone


def build_neck(cfg):
    """Build neck."""
    return build(cfg, NECKS)


def build_roi_extractor(cfg):
    """Build roi extractor."""
    return build(cfg, ROI_EXTRACTORS)


def build_shared_head(cfg):
    """Build shared head."""
    return build(cfg, SHARED_HEADS)


def build_head(cfg):
    """Build head."""
    return build(cfg, HEADS)


def build_loss(cfg):
    """Build loss."""
    return build(cfg, LOSSES)


def build_detector(cfg, train_cfg=None, test_cfg=None):
    """Build detector."""
    if train_cfg is not None or test_cfg is not None:
        warnings.warn(
            'train_cfg and test_cfg is deprecated, '
            'please specify them in model', UserWarning)
    assert cfg.get('train_cfg') is None or train_cfg is None, \
        'train_cfg specified in both outer field and model field '
    assert cfg.get('test_cfg') is None or test_cfg is None, \
        'test_cfg specified in both outer field and model field '

    ret_model = build(cfg, DETECTORS, dict(train_cfg=train_cfg, test_cfg=test_cfg))
    # print("ret_model = ",ret_model)
    # for i, sub_module in enumerate(ret_model):
    #     if(i==0):
    #         print("sub_module[0] = ",sub_module)
    
    # print("type(ret_model) = ",type(ret_model)) # <class 'mmdet.models.detectors.faster_rcnn.FasterRCNN'>
    # import sys
    # sys.exit()
    return ret_model
