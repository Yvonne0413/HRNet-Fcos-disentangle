# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import stat
import pprint

import cv2
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms
import torch.multiprocessing
from tqdm import tqdm
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

import _init_paths
import models

from config import cfg
from config import check_config
from config import update_config
from dataset import make_test_dataloader
from utils.utils import create_logger
from utils.utils import get_model_summary
from utils.transforms import resize_align_multi_scale
from utils.transforms import get_multi_scale_size

color = [(255,255,0), (229,131,8),(252,176,243),(169, 209, 142),(255,255,0), (229,131,8),(252,176,243),(255,255,0), (169, 209, 142) ,(0,176,240), (220,87,18), (244,208,0), (138,151,123), (254,67,101), (114,83,52),(178,200,187),(69,137,148),(161,47,47),(30,41,61)]
#53626: (255,255,0), (229,131,8),(252,176,243),(169, 209, 142) 
#132931: (185,12,226),(255,255,0)
#301563: (255,255,0),(169, 209, 142) ,(252,176,243)
#312489: (229,131,8) ,(0,0,255)

torch.multiprocessing.set_sharing_strategy('file_system')
#show = [290, 997, 1353, 1556, 1637, 1765, 2116, 2162, 2541, \
#        615, 285, 2670, 2220, 2112, 2051, 1496, 1297, 1285, 963, 820, 712]
show = [36539, 43816, 53626, 132931]

def parse_args():
    parser = argparse.ArgumentParser(description='Test keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    return args


# markdown format output
def _print_name_value(logger, name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
         ' |'
    )


def main():
    args = parse_args()
    update_config(cfg, args)
    check_config(cfg)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'valid'
    )

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=False
    )

    if cfg.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=True)
    else:
        model_state_file = os.path.join(
            final_output_dir, 'model_best.pth.tar'
        )
        logger.info('=> loading model from {}'.format(model_state_file))
        model.load_state_dict(torch.load(model_state_file))

    #dump_input = torch.rand(
    #    (1, 3, cfg.DATASET.INPUT_SIZE, cfg.DATASET.INPUT_SIZE)
    #)
    #logger.info(get_model_summary(model, dump_input, verbose=cfg.VERBOSE))

    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()
    model.eval()

    data_loader, test_dataset = make_test_dataloader(cfg)

    if cfg.MODEL.NAME == 'pose_hourglass':
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
            ]
        )
    else:
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ]
        )
    
    show_image = []
    ori_image = []
    ori_joints = []
    show_image_name = []
    pbar = tqdm(total=len(test_dataset)) if cfg.TEST.LOG_PROGRESS else None
    select_image = 0
    for i, (images, image_names, joints, masks, areas) in enumerate(data_loader):
        #if select_image > show[-1]:
        #    break
        #if i > 1:
        #    break
        #if len(joints[0]) <= 1 or len(joints[0]) >= 5:
        #    continue
        assert 1 == images.size(0), 'Test batch size should be 1'
        image = images[0].cpu().numpy()
        joints = joints[0].cpu().numpy()
        mask = masks[0].cpu().numpy()
        area = areas[0].cpu().numpy()
        img_name = image_names[0]

        if int(img_name) in show:
            show_image_name.append(img_name)
            show_image.append(image)
            base_size, center, scale = get_multi_scale_size(
                image, cfg.DATASET.INPUT_SIZE, 1.0, 1.0
            )
            
            image_resized, joints_resized, _, center, scale = resize_align_multi_scale(
                image, joints, mask, cfg.DATASET.INPUT_SIZE, 1.0, 1.0
            )
            ori_image.append(image_resized)
            ori_joints.append(joints_resized)
            image_resized = transforms(image_resized)
            image_resized = image_resized.unsqueeze(0).cuda()

            STN_OFFSET = model(image_resized)

        if cfg.TEST.LOG_PROGRESS:
            pbar.update()
            select_image += 1

    for i in range(len(ori_image)):
        for branch in range(17):
            if branch not in [0, 5, 13, 15]:
                continue
            temp = show_image[i][:, :, :].copy()
            w,h = temp.shape[0],temp.shape[1]
            fig = plt.figure(figsize=(h/100, w/100), dpi=100)
            ax = plt.subplot(1,1,1)
            bk = plt.imshow(temp)
            bk.set_zorder(-1)
            scale_x = (show_image[i].shape[0])*1.0/ori_image[i].shape[0]
            scale_y = (show_image[i].shape[1])*1.0/ori_image[i].shape[1]
            offset = STN_OFFSET[i*34+2*branch]
            translation = STN_OFFSET[i*34+2*branch+1]
            joints = ori_joints[i]
            print(i)
            for j in range(len(joints)):
                joint_loc = joints[j][-1]
                y, x = max(int(joint_loc[0]/4), 0), max(int(joint_loc[1]/4), 0)
                y, x = min(int(joint_loc[0]/4), offset.shape[2]-1), min(int(joint_loc[1]/4), offset.shape[3]-1)
                offset_this_joint = offset[0,:,x,y]
                translation_this_joint = translation[0,:,x,y]
                regular_matrix = np.array([[-1, -1, -1, 0, 0, 0, 1, 1, 1], \
                                        [-1, 0, 1, -1 ,0 ,1 ,-1, 0, 1]])
                offset_this_joint = offset_this_joint.reshape((9,2))+regular_matrix.transpose(1,0)
                print(translation_this_joint.shape)
                offset_this_joint[:,0] = 4*(offset_this_joint[:,0]+translation_this_joint[0])
                offset_this_joint[:,1] = 4*(offset_this_joint[:,1]+translation_this_joint[1])
                #offset_this_joint[:,0] = 4*(offset_this_joint[:,0]-translation_this_joint[1])
                #offset_this_joint[:,1] = 4*(offset_this_joint[:,1]-translation_this_joint[0])                
                
                offset_this_joint[:,0] = (offset_this_joint[:,0] + joint_loc[1])*scale_x
                offset_this_joint[:,1] = (offset_this_joint[:,1] + joint_loc[0])*scale_y
                
                if np.max(offset_this_joint[:,0]) > w or np.min(offset_this_joint[:,0]) < 0 \
                or np.max(offset_this_joint[:,1]) > h or np.min(offset_this_joint[:,1]) < 0:
                    continue
                
                personcolor = color[j]
                line = mlines.Line2D(
                        np.array([offset_this_joint[0,1],
                                    offset_this_joint[2,1]]),
                        np.array([offset_this_joint[0,0],
                                    offset_this_joint[2,0]]),
                        ls='-', lw=3, alpha=1, color=tuple(np.array(personcolor)/255.),)
                line.set_zorder(0)
                ax.add_line(line)

                line = mlines.Line2D(
                        np.array([offset_this_joint[0,1],
                                    offset_this_joint[6,1]]),
                        np.array([offset_this_joint[0,0],
                                    offset_this_joint[6,0]]),
                        ls='-', lw=3, alpha=1, color=tuple(np.array(personcolor)/255.),)
                line.set_zorder(0)
                ax.add_line(line)

                line = mlines.Line2D(
                        np.array([offset_this_joint[6,1],
                                    offset_this_joint[8,1]]),
                        np.array([offset_this_joint[6,0],
                                    offset_this_joint[8,0]]),
                        ls='-', lw=3, alpha=1, color=tuple(np.array(personcolor)/255.),)
                line.set_zorder(0)
                ax.add_line(line)

                line = mlines.Line2D(
                        np.array([offset_this_joint[2,1],
                                    offset_this_joint[8,1]]),
                        np.array([offset_this_joint[2,0],
                                    offset_this_joint[8,0]]),
                        ls='-', lw=3, alpha=1, color=tuple(np.array(personcolor)/255.),)
                line.set_zorder(0)
                ax.add_line(line)

                line = mlines.Line2D(
                        np.array([offset_this_joint[2,1],
                                    offset_this_joint[6,1]]),
                        np.array([offset_this_joint[2,0],
                                    offset_this_joint[6,0]]),
                        ls='-', lw=3, alpha=1, color=tuple(np.array(personcolor)/255.),)
                line.set_zorder(0)
                ax.add_line(line)

                for point in range(9):
                    x, y = int(offset_this_joint[point,0]), int(offset_this_joint[point,1])
                    if x >= 0 and x < temp.shape[0] and y > 0 and y < temp.shape[1]:
                        if point != 4:
                            circle = mpatches.Circle((y,x), 
                                                    radius=2, 
                                                    ec='black', 
                                                    fc=tuple(np.array(personcolor)/255.), 
                                                    alpha=1, 
                                                    linewidth=0.5)
                        else:
                            circle = mpatches.Circle((y,x), 
                                                    radius=2, 
                                                    ec='black', 
                                                    fc=tuple(np.array((255, 255, 255))/255.), 
                                                    alpha=1, 
                                                    linewidth=0.5)
                        circle.set_zorder(1)
                        ax.add_patch(circle)

                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())
                plt.axis('off')
                plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)        
                plt.margins(0,0)
                #plt.savefig('stnfinal/'+show_image_name[i]+'_'+str(branch)+'n.png', 
                #            format='png', bbox_inckes='tight', dpi=100)
                plt.savefig('stnfinal/'+show_image_name[i]+'_'+str(branch)+'.pdf', format='pdf', 
                            bbox_inckes='tight', dpi=100)

    if cfg.TEST.LOG_PROGRESS:
        pbar.close()

if __name__ == '__main__':
    main()
