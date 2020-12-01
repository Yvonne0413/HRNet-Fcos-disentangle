import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

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
        
       
        self.bbox_channels_perbranch = cfg.MODEL.FCOS.BBOX_TOWER.NUM_CHANNELS_PERBRANCH
        bbox_tower = make_bbox_tower(cfg.MODEL.FCOS.BBOX_TOWER)
        self.add_module('bbox_tower', nn.Sequential(*bbox_tower))

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

    def forward(self, x, offset_idx):
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

            # bbox_pred_candidates = self.bbox_pred(self.bbox_tower(feature))
            # bbox_pred_candidates_new = wo_grad_identity(bbox_pred_candidates)
            
            # bbox_pred_candidates = bbox_pred_candidates * 0 + torch.ones(bbox_pred_candidates.shape, dtype=torch.float).cuda()

            # bbox_pred_candidates = torch.tensor(bbox_pred_candidates.detach().cpu().numpy()).cuda()

            bbox_pred_candidates_list = []
            for i in range(4):
                bbox_pred_candidates_list.append(self.bbox_pred[i](\
                    self.bbox_tower[i](\
                        feature[:,i*self.bbox_channels_perbranch:\
                            (i+1)*self.bbox_channels_perbranch])))
            
            # bbox_pred_candidates_list = torch.split(bbox_pred_candidates, 1, 1)
            assert len(bbox_pred_candidates_list) == 4, 'bbox_pred_candidates_list length error'
            
            temp_list = []
            print('offset idx = {}'.format(offset_idx))
            for idx in range(4):
                if idx == offset_idx:
                    temp_list.append(w_grad_identity(bbox_pred_candidates_list[idx]))
                else:
                    temp_list.append(wo_grad_identity(bbox_pred_candidates_list[idx]))
            bbox_pred_candidates = torch.cat(temp_list, dim=1)


            centerness.append(self.centerness(cls_tower))
            bbox_reg.append(torch.exp(self.scales[l](
                bbox_pred_candidates
            )))
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

    def forward(self, images, features, targets=None, offset_idx=None):
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
        box_cls, box_regression, centerness = self.head(features, offset_idx)
        locations = self.compute_locations(features)
        # origin: if self.training
        if 1:
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

@torch.no_grad()
def wo_grad_identity(tensor):
    output_tensor = torch.tensor(tensor.detach().cpu().numpy()).cuda()
    return output_tensor

def w_grad_identity(tensor):
    output_tensor = tensor
    return output_tensor

# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensor = tensor.detach()
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output