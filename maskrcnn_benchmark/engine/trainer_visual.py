# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time
import cv2
import os
import torch
import torch.distributed as dist

from maskrcnn_benchmark.utils.comm import get_world_size
from maskrcnn_benchmark.utils.metric_logger import MetricLogger
from maskrcnn_benchmark.structures.image_list import to_image_list
from vis.vis import save_batch_maps


def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor
def do_train(
    model,
    data_loader,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    arguments,
):
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("I am in do_train!!!!")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    start_iter = arguments["iteration"]

    unnormalize = UnNormalize([102.9801, 115.9465, 122.7717], [1, 1, 1])

    print("I am in do_train!!!!")
    model.eval()
    start_training_time = time.time()
    end = time.time()
    for iteration, (images, targets, _) in enumerate(data_loader, start_iter):
        # print("images!!!!", images)
        if iteration == 1000:
            break
        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        scheduler.step()

        images = images.to(device)
        targets = [target.to(device) for target in targets]

        grad_map_save = []
        print("I am going to seperate offset!!!!")
        for offset_idx in range(4):
            loss_dict = model(images, targets, offset_idx)
            print(loss_dict)
            loss_dict['loss_cls'] = loss_dict['loss_cls'] * 0
            loss_dict['loss_centerness'] = loss_dict['loss_centerness'] * 0
            losses = sum(loss for loss in loss_dict.values())

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_loss_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            meters.update(loss=losses_reduced, **loss_dict_reduced)

            # origin image
            image_save = unnormalize(to_image_list(images).tensors.squeeze(0).cpu())[None,:,:,:]

            optimizer.zero_grad()
            losses.backward()
            # print(model.rpn.head.bbox_pred.weight.grad)
            optimizer.step()


            gradients_as_arr = model.gradients.data.cpu().numpy()[0]
            # no gradient
            if gradients_as_arr.max() == 0:
                grad_map_save.append(torch.zeros(1,1,image_save.shape[-2], image_save.shape[-1]).float())
                continue
            from vis.vanilla_backprop import save_gradient_depu
            save_gradient_depu(gradients_as_arr, 'visual_output/vis_temp_v1/vis_vbp_idx{}_offset{}'.format(iteration, offset_idx))

            gradient = cv2.imread('visual_output/vis_temp_v1/vis_vbp_idx{}_offset{}_Vanilla_BP_gray.jpg'.format(iteration, offset_idx), cv2.IMREAD_GRAYSCALE)
            grad_map_save.append(torch.tensor(gradient)[None,None,:,:])
            print(len(grad_map_save))
        
        print("I am going save grad!!!!")
        for grad_index in range(4):
            grad_map_save[grad_index] = grad_map_save[grad_index].float()/torch.max(grad_map_save[grad_index])

        grad_map_save = torch.cat(grad_map_save, dim=1)
        # vis_v3 change to another names
        save_path = 'visual_output/vis_test'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        save_batch_maps(image_save, grad_map_save, None, save_path + '/grad_map_{}.jpg'.format(iteration))
        # loss_dict = model(images, targets)

        # losses = sum(loss for loss in loss_dict.values())

        # # reduce losses over all GPUs for logging purposes
        # loss_dict_reduced = reduce_loss_dict(loss_dict)
        # losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        # meters.update(loss=losses_reduced, **loss_dict_reduced)

        # optimizer.zero_grad()
        # losses.backward()
        # optimizer.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 20 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
        if iteration % checkpoint_period == 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )
