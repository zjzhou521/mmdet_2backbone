from ..builder import DETECTORS
from .two_stage import TwoStageDetector
from .two_stage_2bone import TwoStageDetector_2bone


@DETECTORS.register_module()
# derived from "two_stage.py"
# class FasterRCNN(TwoStageDetector):
class FasterRCNN(TwoStageDetector_2bone):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None):
        super(FasterRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)


