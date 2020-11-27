import math
import torch
import torch.nn.functional as F
from torch import nn

from .inference import make_fcos_postprocessor
from .loss import make_fcos_loss_evaluator

from maskrcnn_benchmark.layers import Scale
from .bbox_module import make_bbox_tower, make_layer, make_final_layers, make_cls_tower

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
        self.cfg = cfg

        if cfg.MODEL.FCOS.CLS_TOWER.APPLY:
            cls_tower = make_cls_tower(cfg.MODEL.FCOS, in_channels)
        else:
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
                cls_tower.append(nn.GroupNorm(cfg.MODEL.GROUP_NORM.NUM_GROUPS, in_channels))
                cls_tower.append(nn.ReLU())

        if cfg.MODEL.FCOS.MULTI_BRANCH_REG:
            bbox_tower = make_bbox_tower(cfg.MODEL.FCOS.BBOX_TOWER)     
            self.bbox_channels_perbranch = cfg.MODEL.FCOS.BBOX_TOWER.NUM_CHANNELS_PERBRANCH
            if self.bbox_channels_perbranch * cfg.MODEL.FCOS.BBOX_TOWER.NUM_BRANCHES != cfg.MODEL.HRNET.FPN.OUT_CHANNEL:
                self.transition_layer = nn.Sequential(
                    nn.Conv2d(in_channels, self.bbox_channels_perbranch*4,
                                1, 1, 0, bias=False),
                    nn.GroupNorm(self.bbox_channels_perbranch//2, self.bbox_channels_perbranch*4),
                    #GroupBN: GroupBN always used in Det
                    nn.ReLU())
            self.bbox_pred = make_final_layers(cfg.MODEL.FCOS.BBOX_TOWER)
        else:
            bbox_tower = []
            for i in range(cfg.MODEL.FCOS.NUM_CONVS):
                bbox_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                    )
                )
                bbox_tower.append(nn.GroupNorm(cfg.MODEL.GROUP_NORM.NUM_GROUPS, in_channels))
                bbox_tower.append(nn.ReLU())
            self.bbox_pred = nn.Conv2d(
                in_channels, 4, kernel_size=3, stride=1,
                padding=1
            )
        
        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('bbox_tower', nn.Sequential(*bbox_tower))
        self.cls_logits = nn.Conv2d(
            in_channels, num_classes, kernel_size=3, stride=1,
            padding=1
        )
        
        self.centerness = nn.Conv2d(
            in_channels, 1, kernel_size=3, stride=1,
            padding=1
        )

        # initialization
        for modules in [self.cls_tower,
                        self.cls_logits, self.bbox_pred,
                        self.centerness]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    if l.bias is not None:
                        torch.nn.init.constant_(l.bias, 0)
        
        for modules in [self.bbox_tower]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.001)
                    for name, _ in l.named_parameters():
                        if name in ['bias']:
                            torch.nn.init.constant_(l.bias, 0)
                elif isinstance(l, nn.BatchNorm2d):
                    torch.nn.init.constant_(l.weight, 1)
                    torch.nn.init.constant_(l.bias, 0)
                elif isinstance(l, nn.ConvTranspose2d):
                    torch.nn.init.normal_(l.weight, std=0.001)
                    for name, _ in l.named_parameters():
                        if name in ['bias']:
                            torch.nn.init.constant_(l.bias, 0)
        
            for l in self.modules():
                if hasattr(l, 'conv_bboxtower_def'):
                    torch.nn.init.constant_(l.conv_bboxtower_def.weight, 0)
                    if hasattr(l, 'bias'):
                        torch.nn.init.constant_(l.conv_offset.bias, 0)
                if hasattr(l, 'conv1_bboxtower_def'):
                    torch.nn.init.constant_(l.conv1_bboxtower_def.weight, 0)
                    if hasattr(l, 'bias'):
                        torch.nn.init.constant_(l.conv_offset_1.bias, 0)
                if hasattr(l, 'conv2_bboxtower_def'):
                    torch.nn.init.constant_(l.conv2_bboxtower_def.weight, 0)
                    if hasattr(l, 'bias'):
                        torch.nn.init.constant_(l.conv_offset_2.bias, 0)

                if hasattr(l, 'conv_transform_matrix_transstn'):
                    torch.nn.init.constant_(l.conv_transform_matrix_transstn.weight, 0)
                if hasattr(l, 'conv1_transform_matrix_transstn'):
                    torch.nn.init.constant_(l.conv1_transform_matrix_transstn.weight, 0)
                if hasattr(l, 'conv2_transform_matrix_transstn'):
                    torch.nn.init.constant_(l.conv2_transform_matrix_transstn.weight, 0)


                if hasattr(l, 'conv_translation_transstn'):
                    torch.nn.init.constant_(l.conv_translation_transstn.weight, 0)
                    if hasattr(l, 'bias'):
                        torch.nn.init.constant_(l.conv_translation_transstn.bias, 0)   
                if hasattr(l, 'conv1_translation_transstn'):
                    torch.nn.init.constant_(l.conv1_translation_transstn.weight, 0)
                    if hasattr(l, 'bias'):
                        torch.nn.init.constant_(l.conv1_translation_transstn.bias, 0)
                if hasattr(l, 'conv2_translation_transstn'):
                    torch.nn.init.constant_(l.conv2_translation_transstn.weight, 0)
                    if hasattr(l, 'bias'):
                        torch.nn.init.constant_(l.conv2_translation_transstn.bias, 0)

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.FCOS.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(num_level)])

    def forward(self, x):
        logits = []
        bbox_reg = []
        centerness = []
        # print("input x", len(x), x)
        if len(x) > 1:
            for l, feature in enumerate(x):
                # print("l and feature", l, feature.size(), type(feature))
                cls_tower = self.cls_tower(feature)
                logits.append(self.cls_logits(cls_tower))

                # NOTE: Apply centerness branch on box tower lead to 0.5% improvement.
                #       Just uncomment the line 91-93 and comment line 94-97.
                # bbox_tower = self.bbox_tower(feature)
                # bbox_reg.append(torch.exp(self.scales[l](self.bbox_pred(bbox_tower))))
                # centerness.append(self.centerness(bbox_tower))
                        
                # bbox_feature = self.transition_layer(feature)            
                
                if self.cfg.MODEL.FCOS.MULTI_BRANCH_REG:
                    if self.bbox_channels_perbranch * self.cfg.MODEL.FCOS.BBOX_TOWER.NUM_BRANCHES != self.cfg.MODEL.HRNET.FPN.OUT_CHANNEL:
                        feature = self.transition_layer(feature)
                    bbox = []
                    for i in range(self.cfg.MODEL.FCOS.BBOX_TOWER.NUM_BRANCHES):
                        bbox.append(self.bbox_pred[i](\
                            self.bbox_tower[i](\
                                feature[:,i*self.bbox_channels_perbranch:\
                                    (i+1)*self.bbox_channels_perbranch])))
                    bbox = torch.cat(bbox, dim=1)
                    bbox_reg.append(torch.exp(self.scales[l](bbox)))
                else:
                    bbox_tower = self.bbox_tower(feature)
                    bbox_reg.append(torch.exp(self.scales[l](self.bbox_pred(bbox_tower))))

                centerness.append(self.centerness(cls_tower))
        else:
            l = 0
            feature = x[l].unsqueeze(0)
            cls_tower = self.cls_tower(feature)
            logits.append(self.cls_logits(cls_tower))
            if self.cfg.MODEL.FCOS.MULTI_BRANCH_REG:
                if self.bbox_channels_perbranch * self.cfg.MODEL.FCOS.BBOX_TOWER.NUM_BRANCHES != self.cfg.MODEL.HRNET.FPN.OUT_CHANNEL:
                    if self.cfg.MODEL.FCOS.BBOX_TOWER.CHANNEL_OPTION == 0:
                        feature = self.transition_layer(feature)
                    else:

                bbox = []
                for i in range(self.cfg.MODEL.FCOS.BBOX_TOWER.NUM_BRANCHES):
                    bbox.append(self.bbox_pred[i](\
                        self.bbox_tower[i](\
                            feature[:,i*self.bbox_channels_perbranch:\
                                (i+1)*self.bbox_channels_perbranch])))
                bbox = torch.cat(bbox, dim=1)
                bbox_reg.append(torch.exp(self.scales[l](bbox)))
            else:
                bbox_tower = self.bbox_tower(feature)
                bbox_reg.append(torch.exp(self.scales[l](self.bbox_pred(bbox_tower))))

            centerness.append(self.centerness(cls_tower))

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
            targets (list[BoxList): ground-truth boxes present in the image (optional if for inference)

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


'''
class STRANSSTNBLOCK(nn.Module):
    expansion = 1

    def __init__(self, inplanes, outplanes, stride=1, 
    downsample=None, dilation=1, deformable_groups=1):
        super(STRANSSTNBLOCK, self).__init__()
        regular_matrix = torch.tensor(np.array([[-1, -1, -1, 0, 0, 0, 1, 1, 1], \
                                                [-1, 0, 1, -1 ,0 ,1 ,-1, 0, 1]]))
        self.register_buffer('regular_matrix', regular_matrix.float())
        self.downsample = downsample
        self.transform_matrix_conv1 = nn.Conv2d(inplanes, 4, 3, 1, 1, bias=True)
        self.translation_conv1 = nn.Conv2d(inplanes, 2, 3, 1, 1, bias=True)
        self.stn_conv1 = DeformConv(
            inplanes,
            outplanes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            deformable_groups=deformable_groups)
        self.bn1 = nn.BatchNorm2d(outplanes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        residual = x
        (N,C,H,W) = x.shape
        transform_matrix1 = self.transform_matrix_conv1(x)
        translation1 = self.translation_conv1(x)
        transform_matrix1 = transform_matrix1.permute(0,2,3,1).reshape((N*H*W,2,2))
        offset1 = torch.matmul(transform_matrix1, self.regular_matrix)
        offset1 = offset1-self.regular_matrix
        offset1 = offset1.transpose(1,2)
        offset1 = offset1.reshape((N,H,W,18)).permute(0,3,1,2)
        offset1[:,0::2,:,:] += translation1[:,0:1,:,:]
        offset1[:,1::2,:,:] += translation1[:,1:2,:,:]
        out = self.stn_conv1(x, offset1)
        out = self.bn1(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
'''