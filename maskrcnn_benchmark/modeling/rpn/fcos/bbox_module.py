import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

from .dcn import DeformConv, ModulatedDeformConv

BN_MOMENTUM = 0.1
# logger = logging.getLogger(__name__)


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, bias=False, dilation=dilation)

class FcosBlock(nn.Module):
    # the same as FCOS bbox tower conv
    expansion = 1

    def __init__(self, group_num, inplanes, planes, stride=1, 
            downsample=None, dilation=1):
        super(FcosBlock, self).__init__()
        self.conv_bboxtower_fcos = nn.Conv2d(
                    inplanes,
                    planes,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
        self.gn_bboxtower_fcos = nn.GroupNorm(group_num, planes)
        self.relu_bboxtower_fcos = nn.ReLU()

    def forward(self, x):
        out = self.conv_bboxtower_fcos(x)
        out = self.gn_bboxtower_fcos(out)
        out = self.relu_bboxtower_fcos(out)

        return out

class FcosBnBlock(nn.Module):
    # the same as FCOS bbox tower conv but replace gn with bn
    expansion = 1

    def __init__(self, group_num, inplanes, planes, stride=1, 
            downsample=None, dilation=1):
        super(FcosBnBlock, self).__init__()
        self.conv_bboxtower_fcosbn = nn.Conv2d(
                    inplanes,
                    planes,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
        self.bn_bboxtower_fcosbn = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu_bboxtower_fcosbn = nn.ReLU()

    def forward(self, x):
        out = self.conv_bboxtower_fcosbn(x)
        out = self.bn_bboxtower_fcosbn(out)
        out = self.relu_bboxtower_fcosbn(out)

        return out

class FcosSkipBlock(nn.Module):
    # the same as FCOS bbox tower conv but add residual
    expansion = 1

    def __init__(self, group_num, inplanes, planes, stride=1, 
            downsample=None, dilation=1):
        super(FcosSkipBlock, self).__init__()
        self.conv1_bboxtower_fcosskip = nn.Conv2d(
                    inplanes,
                    planes,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
        self.gn1_bboxtower_fcosskip = nn.GroupNorm(group_num, planes)
        self.relu1_bboxtower_fcosskip = nn.ReLU()

        self.conv2_bboxtower_fcosskip = nn.Conv2d(
                    planes,
                    planes,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
        self.gn2_bboxtower_fcosskip = nn.GroupNorm(group_num, planes)
        self.relu2_bboxtower_fcosskip = nn.ReLU()


    def forward(self, x):
        residual = x

        out = self.conv1_bboxtower_fcosskip(x)
        out = self.gn1_bboxtower_fcosskip(out)
        out = self.relu1_bboxtower_fcosskip(out)

        out = self.conv2_bboxtower_fcosskip(out)
        out = self.gn2_bboxtower_fcosskip(out)

        out += residual
        out = self.relu2_bboxtower_fcosskip(out)

        return out

class BasicBlock(nn.Module):
    # Different from Pose BasicBlock: 
    # 1. GroupNorm for det - BatchNorm for pose
    # 2. No skip conn for det - skip conn for pose
    # 3. Initialization
    expansion = 1

    def __init__(self, group_num, inplanes, planes, stride=1, 
            downsample=None, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1_bboxtower_basic = conv3x3(inplanes, planes, stride, dilation=dilation)
        self.bn1_bboxtower_basic = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu_bboxtower_basic = nn.ReLU(inplace=True)
        self.conv2_bboxtower_basic = conv3x3(planes, planes, dilation=dilation)
        self.bn2_bboxtower_basic = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample_bboxtower_basic = downsample
        self.stride_bboxtower_basic = stride


    def forward(self, x):
        residual = x

        out = self.conv1_bboxtower_basic(x)
        out = self.bn1_bboxtower_basic(out)
        out = self.relu_bboxtower_basic(out)

        out = self.conv2_bboxtower_basic(out)
        out = self.bn2_bboxtower_basic(out)

        if self.downsample_bboxtower_basic is not None:
            residual = self.downsample_bboxtower_basic(x)

        out += residual
        out = self.relu_bboxtower_basic(out)

        return out


class DeformableBlock(nn.Module):
    # Different from Pose basicblock: 
    # 1. GroupNorm for det - BatchNorm for pose
    # 2. No skip conn for det - skip conn for pose
    expansion = 1

    def __init__(self, group_num, inplanes, outplanes, stride=1, 
            downsample=None, deformable_groups=1, dilation=1):
        super(DeformableBlock, self).__init__()
        self.conv_offset = nn.Conv2d(inplanes, 18, 3, 1, 1, bias=True)

        self.conv_bboxtower_def = DeformConv(
            inplanes,
            outplanes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            deformable_groups=deformable_groups)
        self.gn_bboxtower_def = nn.GroupNorm(group_num, outplanes)
        self.relu_bboxtower_def = nn.ReLU()

        self.downsample = downsample

    def forward(self, x):

        offset = self.conv_offset(x)
        out = self.conv_bboxtower_def(x, offset)
        out = self.gn_bboxtower_def(out)
        out = self.relu_bboxtower_def(out)

        return out


class DeformableSkipBlock(nn.Module):
    # Different from Pose basicblock: 
    # 1. GroupNorm for det - BatchNorm for pose
    # 2. No skip conn for det - skip conn for pose
    expansion = 1

    def __init__(self, group_num, inplanes, outplanes, stride=1, 
            downsample=None, deformable_groups=1, dilation=1):
        super(DeformableSkipBlock, self).__init__()
        self.conv_offset_1 = nn.Conv2d(inplanes, 18, 3, 1, 1, bias=True)
        self.conv_offset_2 = nn.Conv2d(inplanes, 18, 3, 1, 1, bias=True)

        self.conv1_bboxtower_def = DeformConv(
            inplanes,
            outplanes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            deformable_groups=deformable_groups)
        self.gn1_bboxtower_def = nn.GroupNorm(group_num, outplanes)
        self.relu1_bboxtower_def = nn.ReLU()

        self.conv2_bboxtower_def = DeformConv(
            outplanes,
            outplanes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            deformable_groups=deformable_groups)
        self.gn2_bboxtower_def = nn.GroupNorm(group_num, outplanes)
        self.relu2_bboxtower_def = nn.ReLU()

        self.downsample = downsample

    def forward(self, x):
        residual = x
        offset = self.conv_offset_1(x)
        out = self.conv1_bboxtower_def(x, offset)
        out = self.gn1_bboxtower_def(out)
        out = self.relu1_bboxtower_def(out)

        offset_2 = self.conv_offset_2(out)
        out = self.conv2_bboxtower_def(out, offset_2)
        out = self.gn2_bboxtower_def(out)

        out += residual
        out = self.relu2_bboxtower_def(out)

        return out


class TRANSSTNBLOCK(nn.Module):
    # Different from Pose basicblock: 
    # 1. GroupNorm for det - BatchNorm for pose
    # 2. No skip conn for det - skip conn for pose
    expansion = 1

    def __init__(self, group_num, inplanes, outplanes, stride=1, 
            downsample=None, dilation=1, deformable_groups=1):
        super(TRANSSTNBLOCK, self).__init__()
        regular_matrix = torch.tensor(np.array([[-1, -1, -1, 0, 0, 0, 1, 1, 1], \
                                                [-1, 0, 1, -1 ,0 ,1 ,-1, 0, 1]]))
        self.register_buffer('regular_matrix', regular_matrix.float())
        self.downsample = downsample
        self.conv_transform_matrix_transstn = nn.Conv2d(inplanes, 4, 3, 1, 1, bias=True)
        self.conv_translation_transstn = nn.Conv2d(inplanes, 2, 3, 1, 1, bias=True)
        self.conv_transstn = DeformConv(
            inplanes,
            outplanes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            deformable_groups=deformable_groups)
        self.gn_bboxtower_transstn = nn.GroupNorm(group_num, outplanes)
        self.relu_bboxtower_transstn = nn.ReLU()
 
    def forward(self, x):
        residual = x
        (N,C,H,W) = x.shape
        transform_matrix = self.conv_transform_matrix_transstn(x)
        translation = self.conv_translation_transstn(x)
        transform_matrix = transform_matrix.permute(0,2,3,1).reshape((N*H*W,2,2))
        offset = torch.matmul(transform_matrix, self.regular_matrix)
        offset = offset-self.regular_matrix
        offset = offset.transpose(1,2)
        offset = offset.reshape((N,H,W,18)).permute(0,3,1,2)
        offset[:,0::2,:,:] += translation[:,0:1,:,:]
        offset[:,1::2,:,:] += translation[:,1:2,:,:]
        out = self.conv_transstn(x, offset)
        out = self.gn_bboxtower_transstn(out)
        out += residual
        out = self.relu_bboxtower_transstn(out)
        return out


class TransstnSkipBlock(nn.Module):
    # Different from Pose basicblock: 
    # 1. GroupNorm for det - BatchNorm for pose
    # 2. No skip conn for det - skip conn for pose
    expansion = 1
    
    def __init__(self, group_num, inplanes, outplanes, stride=1, 
            downsample=None, dilation=1, deformable_groups=1):
        super(TransstnSkipBlock, self).__init__()
        regular_matrix = torch.tensor(np.array([[-1, -1, -1, 0, 0, 0, 1, 1, 1], \
                                                [-1, 0, 1, -1 ,0 ,1 ,-1, 0, 1]]))
        self.register_buffer('regular_matrix', regular_matrix.float())
        self.downsample = downsample
        self.conv1_transform_matrix_transstn = nn.Conv2d(inplanes, 4, 3, 1, 1, bias=True)
        self.conv1_translation_transstn = nn.Conv2d(inplanes, 2, 3, 1, 1, bias=True)
        self.conv1_transstn = DeformConv(
            inplanes,
            outplanes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            deformable_groups=deformable_groups)
        self.gn1_bboxtower_transstn = nn.GroupNorm(group_num, outplanes)
        self.relu1_bboxtower_transstn = nn.ReLU()
 
        self.conv2_transform_matrix_transstn = nn.Conv2d(outplanes, 4, 3, 1, 1, bias=True)      
        self.conv2_translation_transstn = nn.Conv2d(outplanes, 2, 3, 1, 1, bias=True)      
        self.conv2_transstn = DeformConv(
            outplanes,
            outplanes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            deformable_groups=deformable_groups)
        self.gn2_bboxtower_transstn = nn.GroupNorm(group_num, outplanes)
        self.relu2_bboxtower_transstn = nn.ReLU()
 
    def forward(self, x):
        residual = x
        (N,C,H,W) = x.shape
        transform_matrix1 = self.conv1_transform_matrix_transstn(x)
        translation1 = self.conv1_translation_transstn(x)
        transform_matrix1 = transform_matrix1.permute(0,2,3,1).reshape((N*H*W,2,2))
        offset1 = torch.matmul(transform_matrix1, self.regular_matrix)
        offset1 = offset1-self.regular_matrix
        offset1 = offset1.transpose(1,2)
        offset1 = offset1.reshape((N,H,W,18)).permute(0,3,1,2)
        offset1[:,0::2,:,:] += translation1[:,0:1,:,:]
        offset1[:,1::2,:,:] += translation1[:,1:2,:,:]
 
        out = self.conv1_transstn(x, offset1)
        out = self.gn1_bboxtower_transstn(out)
        out = self.relu1_bboxtower_transstn(out)

        transform_matrix2 = self.conv2_transform_matrix_transstn(out)
        translation2 = self.conv2_translation_transstn(out)
        transform_matrix2 = transform_matrix2.permute(0,2,3,1).reshape((N*H*W,2,2))
        offset2 = torch.matmul(transform_matrix2, self.regular_matrix)
        offset2 = offset2-self.regular_matrix
        offset2 = offset2.transpose(1,2)
        offset2 = offset2.reshape((N,H,W,18)).permute(0,3,1,2)
        offset2[:,0::2,:,:] += translation2[:,0:1,:,:]
        offset2[:,1::2,:,:] += translation2[:,1:2,:,:]
 
        out = self.conv2_transstn(out, offset2)
        out = self.gn2_bboxtower_transstn(out)
        out += residual
        out = self.relu2_bboxtower_transstn(out)
 
        return out

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

class STNBLOCK(nn.Module):
    # Different from Pose basicblock: 
    # 1. GroupNorm for det - BatchNorm for pose
    # 2. No skip conn for det - skip conn for pose
    expansion = 1
    def __init__(self, group_num, inplanes, outplanes, stride=1, 
            downsample=None, dilation=1, deformable_groups=1):
        super(STNBLOCK, self).__init__()
        regular_matrix = torch.tensor(np.array([[-1, -1, -1, 0, 0, 0, 1, 1, 1], \
                                                [-1, 0, 1, -1 ,0 ,1 ,-1, 0, 1]]))
        self.register_buffer('regular_matrix', regular_matrix.float())
        self.downsample = downsample
        self.conv_transform_matrix_stn = nn.Conv2d(inplanes, 4, 3, 1, 1, bias=True)
        self.conv_bboxtower_stn = DeformConv(
            inplanes,
            outplanes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            deformable_groups=deformable_groups)
        self.gn_bboxtower_stn = nn.GroupNorm(group_num, outplanes)
        self.relu_bboxtower_stn = nn.ReLU()
 
    def forward(self, x):
        (N,C,H,W) = x.shape
        transform_matrix = self.conv_transform_matrix_stn(x)
        transform_matrix = transform_matrix.permute(0,2,3,1).reshape((N*H*W,2,2))
        offset = torch.matmul(transform_matrix, self.regular_matrix)
        offset = offset-self.regular_matrix
        offset = offset.transpose(1,2)
        offset = offset.reshape((N,H,W,18)).permute(0,3,1,2)
 
        out = self.conv_bboxtower_stn(x, offset)
        out = self.gn_bboxtower_stn(out)
        out = self.relu_bboxtower_stn(out)
 
        return out



blocks_dict = {
    'FCOS': FcosBlock,
    'FCOSBN': FcosBnBlock,
    'FCOSSKIP': FcosSkipBlock,
    'BASIC': BasicBlock,
    'STNBLOCK': STNBLOCK,
    'TRANSSTNBLOCK': TRANSSTNBLOCK,
    'DEFORMABLE': DeformableBlock,
    'DEFSKIP': DeformableSkipBlock,
    'TRANSSKIP': TransstnSkipBlock
}

def make_cls_layer(block, pos_tuple, group_num, inplanes, planes, blocks, dilation=1, stride=1):
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        # what is block.expansion? the component set in block class.
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes * block.expansion,
                        kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
        )

    layers = []
    if 0 in pos_tuple:
        layers.append(block(group_num, inplanes, planes, 
                stride, downsample, dilation=dilation))
        inplanes = planes * block.expansion
    else:
        layers.append(FcosBlock(group_num, inplanes, planes, 
                stride, downsample, dilation=dilation))
        inplanes = planes * FcosBlock.expansion

    
    for i in range(1, blocks):
        if i in pos_tuple:
            layers.append(block(group_num, inplanes, planes, dilation=dilation))
        else:
            layers.append(FcosBlock(group_num, inplanes, planes, dilation=dilation))

    return nn.Sequential(*layers)

def make_layer(in_channels, option, block, pos_tuple, group_num, inplanes, planes, blocks, dilation=1, stride=1):
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        # what is block.expansion? the component set in block class.
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes * block.expansion,
                        kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
        )

    layers = []
    if in_channels != inplanes:
        if option == 1:
            transition_layer = nn.Sequential(
                nn.Conv2d(in_channels, inplanes,
                            1, 1, 0, bias=False),
                nn.GroupNorm(inplanes//8, inplanes),
                #GroupBN: GroupBN always used in Det
                nn.ReLU())
            layers.append(transition_layer)
        elif option == 3:
            inplanes = in_channels
    

    if 0 in pos_tuple:
        layers.append(block(group_num, inplanes, planes, 
                stride, downsample, dilation=dilation))
        inplanes = planes * block.expansion
    else:
        layers.append(FcosBlock(group_num, inplanes, planes, 
                stride, downsample, dilation=dilation))
        inplanes = planes * FcosBlock.expansion

    
    for i in range(1, blocks):
        if i in pos_tuple:
            layers.append(block(group_num, inplanes, planes, dilation=dilation))
        else:
            layers.append(FcosBlock(group_num, inplanes, planes, dilation=dilation))

    return nn.Sequential(*layers)


def make_final_layers(layer_config):
    final_layers = []

    for i in range(layer_config['NUM_BRANCHES']):
        final_layers.append(
            nn.Conv2d(
                in_channels=layer_config['NUM_CHANNELS_PERBRANCH'],
                out_channels=layer_config['OUT_CHANNELS_PERBRANCH'],
                kernel_size=3,
                stride=1,
                padding=1
                )
            )

    return nn.ModuleList(final_layers)


def make_bbox_tower(layer_config, in_channels):
    multi_branches = []

    # print(layer_config['BLOCK'])
    # print(layer_config['NUM_CHANNELS_PERBRANCH'])
    # print(layer_config['NUM_BLOCKS'])
    # print(layer_config['DILATION_RATE'])

    for i in range(layer_config['NUM_BRANCHES']):
        multi_branches.append(
            make_layer(
                in_channels,
                layer_config['CHANNEL_OPTION'],
                blocks_dict[layer_config['BLOCK']],
                layer_config['POS_SPECIAL_BLOCKS'],
                layer_config['GROUP_NORM_NUM'],
                layer_config['NUM_CHANNELS_PERBRANCH'],
                layer_config['NUM_CHANNELS_PERBRANCH'],
                layer_config['NUM_BLOCKS'],
                layer_config['DILATION_RATE']
            )
        )

    return nn.ModuleList(multi_branches)


def make_cls_tower(fcos_config, in_channels):

    cls_tower = make_cls_layer(
                blocks_dict[fcos_config.BBOX_TOWER['BLOCK']],
                fcos_config.BBOX_TOWER['POS_SPECIAL_BLOCKS'],
                fcos_config.CLS_TOWER['GROUP_NORM_NUM'],
                in_channels,
                in_channels,
                fcos_config['NUM_CONVS'],
                fcos_config.BBOX_TOWER['DILATION_RATE']
            )
    return nn.Sequential(*cls_tower)

    