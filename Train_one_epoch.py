# -*- coding: utf-8 -*-
# @Time    : 2021/6/19 2:14 下午
# @Author  : Haonan Wang
# @File    : Train_one_epoch.py
# @Software: PyCharm
import torch.optim
import os
import time
from utils import *
import Config as config
import warnings
import torch.nn as kn
warnings.filterwarnings("ignore")  # 过滤一些警告
from utils import DiceLoss

#  打印模型的相关参数
def print_summary(epoch, i, nb_batch, loss, loss_name, batch_time,
                  average_loss, average_time,
                  iou, average_iou,
                  dice, average_dice,
                  acc, average_acc,
                  auc, average_auc,
                  mode, lr, logger):
    '''
        mode = Train or Test
    '''
    summary = '   [' + str(mode) + '] Epoch: [{0}][{1}/{2}]  '.format(
        epoch, i, nb_batch)
    string = ''
    string += 'Loss:{:.3f} '.format(loss)
    string += '(Avg {:.4f}) '.format(average_loss)
    string += 'IoU:{:.3f} '.format(iou)
    string += '(Avg {:.4f}) '.format(average_iou)
    string += 'Dice:{:.4f} '.format(dice)
    string += '(Avg {:.4f}) '.format(average_dice)
    string += 'Acc:{:.3f} '.format(acc)
    string += '(Avg {:.4f}) '.format(average_acc)
    string += 'Auc:{:.3f} '.format(auc)
    string += '(Avg {:.4f}) '.format(average_auc)
    if mode == 'Train':
        string += 'LR {:.2e}   '.format(lr)
    # string += 'Time {:.1f} '.format(batch_time)
    string += '(AvgTime {:.1f})   '.format(average_time)
    summary += string
    logger.info(summary)
    # print summary


##################################################################################
# =================================================================================
#          Train One Epoch   单次epoch遍历训练集训练策略定义
# =================================================================================
##################################################################################
def train_one_epoch(loader, model, criterion, optimizer, writer, epoch, lr_scheduler, model_type, logger):
    logging_mode = 'Train' if model.training else 'Val'  # 记录当前logging_mode是属于训练还是验证模型
    end = time.time()  # 获取初试时间
    time_sum, loss_sum = 0, 0  # 初始化： 时间总和 =0   总损失=0
    dice_sum, iou_sum, acc_sum, auc_sum = 0.0, 0.0, 0.0, 0.0   # 初始化： 总dice =0 总iou=0  总acc=0

    ce_loss = nn.CrossEntropyLoss
    dice_loss = DiceLoss(1)


    dices = []  # dice列表
    for i, (sampled_batch, names) in enumerate(loader, 1):
        try:
            loss_name = criterion._get_name()
        except AttributeError:
            loss_name = criterion.__name__

        # Take variable and put them to GPU
        images, masks = sampled_batch['image'], sampled_batch['label']  # 获取图像和标签
        images, masks = images.cuda(), masks.cuda()                     # 将imag和mask加载到GPU上




        # ====================================================
        #             Compute loss  及计算损失
        # ====================================================

        if config.model_name == 'MSESNet':
            preds, d1,d2,d3,d4,d5 = model(images)
            out_loss0 = criterion(preds, masks.float())  # 将预测Mask和pre_mask计算Loss
            out_loss1 = criterion(d1, masks.float())
            out_loss2 = criterion(d2, masks.float())
            out_loss3 = criterion(d3, masks.float())
            out_loss4 = criterion(d4, masks.float())
            out_loss5 = criterion(d5, masks.float())
            out_loss = (out_loss0 + out_loss1 + out_loss2 + out_loss3 + out_loss4 + out_loss5)/6

        if config.model_name == 'MFCNet':
            preds, d1,d2,d3,d4 = model(images)
            out_loss0 = criterion(preds, masks.float())  # 将预测Mask和pre_mask计算Loss
            out_loss1 = criterion(d1, masks.float())
            out_loss2 = criterion(d2, masks.float())
            out_loss3 = criterion(d3, masks.float())
            out_loss4 = criterion(d4, masks.float())
            out_loss = (out_loss0 + out_loss1 + out_loss2 + out_loss3 + out_loss4)/5

        elif config.model_name == 'MGCC':
            preds, d1, d2, d3 = model(images)
            out_loss0 = criterion(preds, masks.float())  # 将预测Mask和pre_mask计算Loss
            out_loss1 = criterion(d1, masks.float())
            out_loss2 = criterion(d2, masks.float())
            out_loss3 = criterion(d3, masks.float())
            out_loss = (out_loss0 + out_loss1 + out_loss2 + out_loss3)/4

        elif config.model_name == 'BUSNet':
            preds,d1,d2= model(images)
            # out_loss0 = criterion(preds, masks.float())  # 将预测Mask和pre_mask计算Loss
            out_loss1 = criterion(d1, masks.float())
            out_loss2 = criterion(d2, masks.float())
            out_loss = (out_loss1 + out_loss2)/2

        elif config.model_name == 'PFD_Net':
            preds, d1, d2, d3 = model(images)
            out_loss0 = criterion(preds, masks.float())  # 将预测Mask和pre_mask计算Loss
            out_loss1 = criterion(d1, masks.float())
            out_loss2 = criterion(d2, masks.float())
            out_loss3 = criterion(d3, masks.float())
            out_loss = (out_loss0 + out_loss1 + out_loss2 + out_loss3) / 4

        elif config.model_name == 'CASFNet':
            preds, d1, d2= model(images)
            out_loss0 = criterion(preds, masks.float())  # 将预测Mask和pre_mask计算Loss
            out_loss1 = criterion(d1, masks.float())
            out_loss2 = criterion(d2, masks.float())
            out_loss = (out_loss0 + out_loss1 + out_loss2)/3

        elif config.model_name == 'BASNet':
            preds = model(images)
            out_loss = criterion(preds, masks.float())  # 将预测Mask和pre_mask计算Loss

        elif config.model_name == 'CFATransUnet':
            preds, preds2 = model(images)
            out_loss = criterion(preds, masks.float())  # 将预测Mask和pre_mask计算Loss

        elif config.model_name == 'DSTransUNet':
            preds, loss1, loss2 = model(images)  # 预测图像的pre_mask
            loss1 = criterion(preds, masks.float())  # 没问题
            out_loss =loss1 # 0.6 * loss1 + 0.2 * loss2 + 0.2 * loss3
        else:
            preds= model(images)  # 预测图像的pre_mask
            out_loss = criterion(preds, masks.float())  # 将预测Mask和pre_mask计算Loss

        if model.training:         # 如果是训练模型
            optimizer.zero_grad()  # 初始训练梯度清0
            out_loss.backward()    # 计算所有的参数相关梯度
            optimizer.step()       # 反向更新参数


        train_iou = iou_on_batch(masks, preds)  # IOU
        train_dice = criterion._show_dice(preds, masks.float())  # DICE
        train_acc = acc_on_batch(masks, preds)  # ACC
        train_auc = auc_on_batch(masks, preds)  # AUC
        batch_time = time.time() - end

        # if epoch % config.vis_frequency == 0 and logging_mode is 'Val':  # 可视化频率：每10次，并且当前是在验证模型
        #
        #     # 可视化化路径：MoNuSeg/UCTransNet_pretrain/Test_session_11.27_11h44/visualize_val/str(epoch)/
        #     vis_path = config.visualize_path + str(epoch) + '/'
        #     if not os.path.isdir(vis_path):  # 如果可视化路径不存在，则创建以可视化目录
        #         os.makedirs(vis_path)  # 创建可视化目录
        #     save_on_batch(images, masks, preds, names, vis_path)  # 将可视化结果保存到对应的可视化路径中去
        dices.append(train_dice)  # 将dice记录到dices[]列表中去

        time_sum += len(images) * batch_time
        loss_sum += len(images) * out_loss
        acc_sum += len(images) * train_acc
        auc_sum += len(images) * train_auc
        iou_sum += len(images) * train_iou
        dice_sum += len(images) * train_dice

        if i == len(loader):
            average_loss = loss_sum / (config.batch_size * (i - 1) + len(images))  # Loss
            average_time = time_sum / (config.batch_size * (i - 1) + len(images))
            train_iou_average = iou_sum / (config.batch_size * (i - 1) + len(images))
            train_acc_average = acc_sum / (config.batch_size * (i - 1) + len(images))
            train_auc_average = auc_sum / (config.batch_size * (i - 1) + len(images))
            train_dice_avg = dice_sum / (config.batch_size * (i - 1) + len(images))
        else:
            average_loss = loss_sum / (i * config.batch_size)  # Loss
            average_time = time_sum / (i * config.batch_size)
            train_iou_average = iou_sum / (i * config.batch_size)  # IOU
            train_acc_average = acc_sum / (i * config.batch_size)  # ACC
            train_auc_average = auc_sum / (i * config.batch_size)  # AUC
            train_dice_avg = dice_sum / (i * config.batch_size)  # DICE

        end = time.time()
        torch.cuda.empty_cache()

        if i % config.print_frequency == 0:
            print_summary(epoch + 1, i, len(loader), out_loss, loss_name, batch_time,
                          average_loss, average_time,
                          train_iou, train_iou_average,
                          train_dice, train_dice_avg,
                          train_acc, train_acc_average,
                          train_auc, train_auc_average,
                          logging_mode,lr=min(g["lr"] for g in optimizer.param_groups), logger=logger)

        if config.tensorboard:
            step = epoch * len(loader) + i
            writer.add_scalar(logging_mode + '_' + loss_name, out_loss.item(), step)

            # plot metrics in tensorboard
            writer.add_scalar(logging_mode + '_iou', train_iou, step)
            writer.add_scalar(logging_mode + '_acc', train_acc, step)
            writer.add_scalar(logging_mode + '_auc', train_auc, step)
            writer.add_scalar(logging_mode + '_dice', train_dice, step)

        torch.cuda.empty_cache()
    if lr_scheduler is not None:
        lr_scheduler.step()

    # if epoch + 1 > 10: # Plateau
    #     if lr_scheduler is not None:
    #         lr_scheduler.step(train_dice_avg)

    return average_loss, train_dice_avg   # 返回平均损失，训练的平均DICE
