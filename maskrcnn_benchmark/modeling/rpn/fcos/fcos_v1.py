import math
import torch
import torch.nn.functional as F
from torch import nn

from .inference import make_fcos_postprocessor
from .loss import make_fcos_loss_evaluator

from maskrcnn_benchmark.layers import Scale
from .bbox_module import make_bbox_tower, make_layer, make_final_layers

class FCOSHead(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(FCOSHead, self).__init__()
        # TODO: Implement the sigmoid version first.
        num_classes = cfg.MODEL.FCOS.NUM_CLASSES - 1
        num_level = len(cfg.MODEL.FCOS.FPN_STRIDES)

        cls_tower = []
        for i in range(cfg.MODEL.FCOS.NUM_CONVS):
            cls_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            cls_tower.append(nn.GroupNorm(32, in_channels))
            cls_tower.append(nn.ReLU())
        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        
        self.bbox_channels_prebranch = cfg.MODEL.FCOS.BBOX_TOWER.NUM_CHANNELS_PERBRANCH
        """
                self.transition_layer = nn.Sequential(
            nn.Conv2d(in_channels, self.bbox_channels_prebranch*4,
                        1, 1, 0, bias=False),
            nn.BatchNorm2d(regression_channels),
            #GroupBN: GroupBN always used in Det
            nn.ReLU(True))
        """

        self.bbox_tower = make_bbox_tower(cfg.MODEL.FCOS.BBOX_TOWER)
        self.bbox_pred = make_final_layers(cfg.MODEL.FCOS.BBOX_TOWER)

        self.cls_logits = nn.Conv2d(
            in_channels, num_classes, kernel_size=3, stride=1,
            padding=1
        )
        self.centerness = nn.Conv2d(
            in_channels, 1, kernel_size=3, stride=1,
            padding=1
        )

        # initialization
        for modules in [self.cls_tower, self.bbox_tower,
                        self.cls_logits, self.bbox_pred,
                        self.centerness]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    if l.bias is not None:
                        torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.FCOS.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(num_level)])

    def forward(self, x):
        logits = []
        bbox_reg = []
        centerness = []
        for l, feature in enumerate(x):
            cls_tower = self.cls_tower(feature)
            logits.append(self.cls_logits(cls_tower))

            # NOTE: Apply centerness branch on box tower lead to 0.5% improvement.
            #       Just uncomment the line 91-93 and comment line 94-97.
            # bbox_tower = self.bbox_tower(feature)
            # bbox_reg.append(torch.exp(self.scales[l](self.bbox_pred(bbox_tower))))
            # centerness.append(self.centerness(bbox_tower))
            centerness.append(self.centerness(cls_tower))            

            # ???The meaning for 4 channels. 
            # l, r, t, b
            # ???Disentangle should consider the physical meaning.
            # bbox_feature = self.transition_layer(feature)

            bbox = []
            for i in range(4):
                bbox.append(self.bbox_pred[i](\
                    self.bbox_tower[i](\
                        feature[:,i*self.bbox_channels_prebranch:\
                            (i+1)*self.bbox_channels_prebranch])))

            left = bbox[0]
            right = bbox[1]
            top = bbox[2]
            bottom = bbox[3]

            bbox[0], bbox[1], bbox[2], bbox[3] = \
            (left+right)/2, (top+bottom)/2, bottom-top, right-left

            bbox = torch.cat(bbox, dim=1)
            bbox_reg.append(torch.exp(self.scales[l](bbox)))
            
        return logits, bbox_reg, centerness


class FCOSModule(torch.nn.Module):
    """
    Module for FCOS computation. Takes feature maps from the backbone and
    FCOS outputs and losses. Only Test on FPN now.
    """

    def __init__(self, cfg, in_channels):
        super(FCOSModule, self).__init__()

        head = FCOSHead(cfg, in_channels)
        box_selector_test = make_fcos_postprocessor(cfg)

        loss_evaluator = make_fcos_loss_evaluator(cfg)
        self.head = head
        self.box_selector_test = box_selector_test
        self.loss_evaluator = loss_evaluator
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES

    def forward(self, images, features, targets=None):
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (list[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (list[BoxList): ground-truth boxes present in the image (optional)

        Returns:
            boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                image.
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        box_cls, box_regression, centerness = self.head(features)
        locations = self.compute_locations(features)
 
        if self.training:
            return self._forward_train(
                locations, box_cls, 
                box_regression, 
                centerness, targets
            )
        else:
            return self._forward_test(
                locations, box_cls, box_regression, 
                centerness, images.image_sizes
            )

    def _forward_train(self, locations, box_cls, box_regression, centerness, targets):
        loss_box_cls, loss_box_reg, loss_centerness = self.loss_evaluator(
            locations, box_cls, box_regression, centerness, targets
        )
        losses = {
            "loss_cls": loss_box_cls,
            "loss_reg": loss_box_reg,
            "loss_centerness": loss_centerness
        }
        return None, losses

    def _forward_test(self, locations, box_cls, box_regression, centerness, image_sizes):
        boxes = self.box_selector_test(
            locations, box_cls, box_regression, 
            centerness, image_sizes
        )
        return boxes, {}

    def compute_locations(self, features):
        locations = []
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            locations_per_level = self.compute_locations_per_level(
                h, w, self.fpn_strides[level],
                feature.device
            )
            locations.append(locations_per_level)
        return locations

    def compute_locations_per_level(self, h, w, stride, device):
        shifts_x = torch.arange(
            0, w * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shifts_y = torch.arange(
            0, h * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
        return locations

def build_fcos(cfg, in_channels):
    return FCOSModule(cfg, in_channels)
