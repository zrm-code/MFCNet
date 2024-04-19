# -*- coding: utf-8 -*-
# @Time    : 2021/7/8 8:59 上午
# @Author  : Haonan Wang
# @File    : train.py
# @Software: PyCharm

# ----------------------完全阅读完这个项目的策略（那么自已的项目也能够很快的建立起来，CTrans模块太难的话，可以不用那么细致的看）----------------#

import torch.optim
from tensorboardX import SummaryWriter
import os
import numpy as np
import random
from torch.backends import cudnn
from Load_Dataset import RandomGenerator, ValGenerator, ImageToImage2D  # 从模块中导入需要的函数

from torch.utils.data import DataLoader
import logging
from Train_one_epoch import train_one_epoch
import Config as config
from torchvision import transforms
from nets.TransUNet import CONFIGS as CONFIGS_ViT_seg
from utils import CosineAnnealingWarmRestarts, WeightedDiceBCE

import nets.HiFormer_configs as configs

from nets.UNet import UNet                    # UNet
from nets.MultiResUNet import MultiResUnet    # MultiResUNet
from nets.AttenUNet import AttUNet            # Attention-UNet
from nets.MTUNet import MTUNet                # MTUNet-UNet
from nets.SwinUNet import SwinUNet            # SwinUNet
from nets.TransUNet import TransNet            # TransNet
from nets.MSESNet import MSESNet              # MSESNet
from nets.MFCNet import MFCNet              # MSESNet
from nets.UCTransNet import UCTransNet        # UCTransNet
from nets.MedT_axialnet import MedT           # MedT
from nets.CAFNet import CAFNet           # BASNet
from nets.CASFNet import CASFNet           # CASFNet
from nets.NestedUNet  import NestedUNet          # MedT
# from nets.DSTransUNet  import DSTransUNet          # DSTransUNet
from nets.CANet  import LUNet          # CANet
from nets.DCSAU_Net  import DCSAU_Net
from nets.DCFNet  import BUSNet
from nets.BASNet  import BASNet
from nets.H2Former  import H2Former
from nets.HiFormer  import HiFormer
from nets.CFATransUnet import CFATransUnet
from nets.SASAN import SASAN
from nets.BEFUnet import BEFUnet
from nets.PFD_Net import PFD_Net
# =======================肿瘤专门网络========================== #
from nets.CMUNet  import CMUNet
from nets.MGCC  import MGCC
from nets.MDA_Net  import MDA_Net
from nets.SegNet  import SegNet
from nets.DAF  import DAF
from nets.MSNet  import M2SNet

# 日志文件函数,这个函数并不重要，参数：日志的保存路径
def logger_config(log_path):
    loggerr = logging.getLogger()
    loggerr.setLevel(level=logging.INFO)
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    loggerr.addHandler(handler)
    loggerr.addHandler(console)
    return loggerr


# 保存节点的一些信息
def save_checkpoint(state, save_path):
    '''
        Save the current model.
        If the model is the best model since beginning of the training
        it will be copy
        (保存当前模型。如果模型是自训练开始以来的最佳模型它将被复制保存)
    '''
    logger.info('\t Saving to {}'.format(save_path))  # 打印模型保存的路径
    if not os.path.isdir(save_path):  # 判断路径中是否有该目录路径
        os.makedirs(save_path)  # 创建多层目录

    epoch = state['epoch']  # 获取是哪一次epoch
    best_model = state['best_model']  # bool  判断是否是best_model最好的模型
    model = state['model']  # model type  判断是什么样的模型类型：UCTransNet 还是 UCTransNet_Pretrain

    if best_model:  # （best_model = True）则保存当前最好的模型
        filename = save_path + '/' + 'best_model-{}.pth.tar'.format(model)  # 将当前最好的模型进行保存
    else:
        filename = save_path + '/' + 'model-{}-{:02d}.pth.tar'.format(model, epoch)  # 保存普通模型，并不是最好的
    torch.save(state, filename)  # torch.save保存模型


def worker_init_fn(worker_id):
    random.seed(config.seed + worker_id)


##################################################################################
# =================================================================================
#          Main Loop: load model,加载模型
# =================================================================================
##################################################################################

# 参数设置：    batch_size=4（config中设置的，GPU上的内存量不足，最好设置成3），
# 模型是：     UCTransNet_pretrain或者UCTransNet ,
# 可视化工具：  tensorboard=True
def main_loop(batch_size=config.batch_size, model_type='', tensorboard=True):
    # Load train and val data --加载训练和验证集的一些数据
    # transforms.Compose：将多个图像变换步骤整合到一起，对数据集进行随机变换，增强模型的泛化能力
    train_tf = transforms.Compose(
        [RandomGenerator(output_size=[config.img_size, config.img_size])])  # 训练数据的随机增强生成数据集:大小244*244
    val_tf = ValGenerator(output_size=[config.img_size, config.img_size])  # 验证集数据的生成器:大小244*244

    # 将图像转为2D序列，参数：train_dataset-训练数据集路径，train_tf-训练数据集增强配置，输入图像大小image_size==config.img_size：224
    train_dataset = ImageToImage2D(config.train_dataset, train_tf, image_size=config.img_size)
    # 将图像转为2D序列，参数：val_dataset-验证数据集路径，val_tf-验证数据集增强配置，输入图像大小image_size==config.img_size：224
    val_dataset = ImageToImage2D(config.val_dataset, val_tf, image_size=config.img_size)

    # 训练集数据加载块，Dataset类对象, 决定数据从哪读取以及如何读取
    train_loader = DataLoader(train_dataset,  # 训练数据集
                              batch_size=config.batch_size,  # 单次训练batch_size=4
                              shuffle=True,  # 每个epoch是否乱序，将输入数据的顺序打乱，是为了使数据更有独立性，但如果数据是有序列特征的，就不要设置成True了
                              worker_init_fn=worker_init_fn,
                              num_workers=8,  # 多进程读取数据的线程数
                              pin_memory=True
                              # ,drop_last = True
                              )  # 内存寄存，默认为False。在数据返回前，是否将数据复制到CUDA内存中。
    # 验证集数据加载块，Dataset类对象, 决定数据从哪读取以及如何读取
    val_loader = DataLoader(val_dataset,  # 验证数据集
                            batch_size=config.batch_size,  # # 单次训练batch_size=4
                            shuffle=True,  # 每个 epoch 是否乱序
                            worker_init_fn=worker_init_fn,
                            num_workers=8,  # 这个参数太大，可以考虑设置小一点，设置成0
                            pin_memory=True
                            # ,drop_last = True
                            )  # 内存寄存，默认为False。在数据返回前，是否将数据复制到CUDA内存中。

    lr = config.learning_rate  # 设置学习率，通常使用默认的学习率
    logger.info(model_type)  # 记录模型


    if model_type == "UCTransNet":  # 如果模型类型是UCTransNet
        config_vit = config.get_CTranS_config()  # 获得相应的参数配置
        logger.info('transformer head num: {}'.format(config_vit.transformer.num_heads))
        logger.info('transformer layers num: {}'.format(config_vit.transformer.num_layers))
        logger.info('transformer expand ratio: {}'.format(config_vit.expand_ratio))
        model = UCTransNet(config_vit, n_channels=config.n_channels, n_classes=config.n_labels)  # 加载UCTransNet模型（非预训练）

    elif model_type == 'UNet':  # UNet
        model = UNet()

    elif model_type == 'MultiResUnet':
        model = MultiResUnet()

    elif model_type == 'AttUNet':
        model = AttUNet()

    elif model_type == 'MTUNet':
        model = MTUNet()

    elif model_type == 'SwinUNet':
        model = SwinUNet()

    elif model_type == 'SASAN':
        model = SASAN()

    elif model_type == 'NestedUNet':
        model = NestedUNet()

    elif model_type == 'MGCC':
        model = MGCC()

    elif model_type == 'CASFNet':
        model = CASFNet()

    elif model_type == 'CFATransUnet':
        model = CFATransUnet()

    elif model_type == 'BEFUnet':
        model = BEFUnet()

    elif model_type == 'PFD_Net':
        model = PFD_Net()

    elif model_type == 'BUSNet':
        config_vit = config.get_CTranS_config()  # 获得相应的参数配置
        logger.info('transformer head num: {}'.format(config_vit.transformer.num_heads))
        logger.info('transformer layers num: {}'.format(config_vit.transformer.num_layers))
        logger.info('transformer expand ratio: {}'.format(config_vit.expand_ratio))
        model = BUSNet(config_vit, 128, n_channels=config.n_channels, n_classes=config.n_labels)

    elif model_type == 'BASNet':
        config_vit = config.get_CTranS_config()  # 获得相应的参数配置
        logger.info('transformer head num: {}'.format(config_vit.transformer.num_heads))
        logger.info('transformer layers num: {}'.format(config_vit.transformer.num_layers))
        logger.info('transformer expand ratio: {}'.format(config_vit.expand_ratio))
        model = BASNet(config_vit, n_channels=config.n_channels, n_classes=config.n_labels)


    # elif model_type == 'DSTransUNet':
    #     model = DSTransUNet(128, 1)

    elif model_type == 'MedT':
        model = MedT(img_size = 224, imgchan = 3)

    elif model_type == 'CANet':
        model = LUNet(3, 1)

    elif model_type == 'M2SNet':
        model = M2SNet()

    elif model_type == 'MGCC':
        model = MGCC()

    elif model_type == 'CMUNet':
        model = CMUNet(img_ch=3, output_ch=1, l=7, k=7)

    elif model_type == 'MDANet':
        model = MDA_Net()

    elif model_type == 'SegNet':
        model = SegNet(3,1)

    elif model_type == 'DAF':
        model = DAF()

    elif model_type == 'H2Former':
        model = H2Former(224,1)

    elif model_type == 'HiFormer':
        model = HiFormer(config=configs.get_hiformer_b_configs(), img_size=224, n_classes=1)

    elif model_type == 'DCSAU_Net':
        model = DCSAU_Net(img_channels=3, n_classes=1)

    elif model_type == 'MSESNet':
        config_vit = config.get_CTranS_config()  # 获得相应的参数配置
        logger.info('transformer head num: {}'.format(config_vit.transformer.num_heads))  # 打印transformer的头部数目
        logger.info('transformer layers num: {}'.format(config_vit.transformer.num_layers))  # 打印transformer的层数目
        logger.info('transformer expand ratio: {}'.format(config_vit.expand_ratio))  # MLP通道尺寸扩展率
        model = MSESNet(config_vit, n_channels=config.n_channels, n_classes=config.n_labels)

    elif model_type == 'MFCNet':
        config_vit = config.get_CTranS_config()  # 获得相应的参数配置
        logger.info('transformer head num: {}'.format(config_vit.transformer.num_heads))  # 打印transformer的头部数目
        logger.info('transformer layers num: {}'.format(config_vit.transformer.num_layers))  # 打印transformer的层数目
        logger.info('transformer expand ratio: {}'.format(config_vit.expand_ratio))  # MLP通道尺寸扩展率
        model = MFCNet(config_vit, n_channels=config.n_channels, n_classes=config.n_labels)

    elif model_type == 'TransNet':
        config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
        config_vit.n_classes = 1
        config_vit.n_skip = 3
        if 'R50-ViT-B_16'.find('R50') != -1:
            config_vit.patches.grid = (
            int(224 / 16), int(224 / 16))
        model = TransNet(config_vit, img_size=224, num_classes=config_vit.n_classes)

    else:
        raise TypeError('Please enter a valid name for the model type')

    if torch.cuda.is_available():
        model = model.cuda()  # 将模型加载到GPU上运行，这里会占用一些显存的内存

    # if torch.cuda.device_count() > 1:
    #     print ("Let's use {0} GPUs!".format(torch.cuda.device_count()))
    # model = nn.DataParallel(model, device_ids=[0])

    criterion = WeightedDiceBCE(dice_weight=0.5, BCE_weight=0.5)  # 用于计算损失

    # lambda函数用于指定过滤列表元素的条件。例如filter(lambda x: x % 3 == 0, [1, 2, 3])
    # 指定将列表[1,2,3]中能够被3整除的元素过滤出来，其结果是[3]。
    # requires_grad: 如果需要为张量计算梯度，则为True，否则为False。
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)  # Choose optimize，选择优化器


    # 感觉这部分没什么用，下回直接设置成固定的学习率
    if config.cosineLR is True:  # （是否使用余弦LR--learn rate）
        lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-4)  # 为优化器设置学习速率调度
    else:
        lr_scheduler = None  # 不设置学习速率调度

    if tensorboard:  # 设置是否可视化
        log_dir = config.tensorboard_folder  # 可视化文件夹
        logger.info('log dir: '.format(log_dir))  # 打印可视化文件夹的路径
        if not os.path.isdir(log_dir):  # 判断是否存在可视化文件夹路径
            os.makedirs(log_dir)  # 创建可视化文件夹目录
        writer = SummaryWriter(log_dir)  # 打印可视化文件

    else:  # 不设置可视化
        writer = None

    max_dice = 0.0  # 初始化最大的max_dice
    best_epoch = 1  # 用于记录最好的DICE，

    # ------------多次遍历数据集训练------------
    for epoch in range(config.epochs):
        logger.info('\n========= Epoch [{}/{}] ========='.format(epoch + 1, config.epochs + 1))
        logger.info(config.session_name)  # 训练的模型分支名，打印到训练日志中

        model.train(True)
        logger.info('Training with batch size : {}'.format(batch_size))  # 训练的batchsize，打印到训练日志中

        train_one_epoch(train_loader, model, criterion, optimizer, writer, epoch, None, model_type, logger)

        # -----------------evaluate on validation set，在验证集上进行模型验证---------------------------------#
        logger.info('Validation')  # 打印“Validation”，打印到训练日志中
        with torch.no_grad():  # 不计算梯度
            model.eval()       # 模型验证，获得模型的损失+Dice
            val_loss, val_dice = train_one_epoch(val_loader, model, criterion,optimizer, writer, epoch, lr_scheduler, model_type, logger)

        # =======================================================================================
        #  -----------------Save best model--保存最好的模型（每次训练得到的模型使用验证集验证，保存验证集上损失最小模型）--------------------------#
        # =======================================================================================
        if val_dice > max_dice:  # 如果验证集的val_Dice大于max_dice
            if epoch + 1 > 5:  # 至少4次循环数据集过后，才开始记录最好的模型
                logger.info('\t Saving best model, mean dice increased from: {:.4f} to {:.4f}'.format(max_dice,
                                                                                                      val_dice))  # 输出记录语句
                max_dice = val_dice  # 将验证集的最好泛化的Val_Dice重新赋值给max_dice
                best_epoch = epoch + 1  # best_epoch的值为记录哪次的epoch循环

                # 设置保存节点的字典，参数1：字典，参数2：模型保存的路径
                save_checkpoint({'epoch': epoch,  # 保存 epoch
                                 'best_model': True,  # best_model设置为最好
                                 'model': model_type,  # 模型的类型：UCTransNet 还是 UCTransNet_Pretrain
                                 'state_dict': model.state_dict(),  # pytorch 中的 state_dict 是一个简单的python的字典对象,
                                 # 将每一层与它的对应参数建立映射关系.(如model的每一层的weights及偏置等等) ，以字典的格式存储,然后以字典的格式被加载
                                 'val_loss': val_loss,  # 模型的验证损失
                                 'optimizer': optimizer.state_dict()},
                                config.model_path)  # 模型的保存路径

        else:  # 如果泛化能力不好，则打印语句，表明平均的Dice并没有增加，最好的模型仍然是上一次保存的最好模型
            logger.info(
                '\t Mean dice:{:.4f} does not increase, ''the best is still: {:.4f} in epoch {}'.format(val_dice,
                                                                                                        max_dice,
                                                                                                        best_epoch))

        early_stopping_count = epoch - best_epoch + 1  # 判断提前结束训练依据，本次遍历数据集次数epoch距离上一次记录的最好模型best_epoch大于50，则立马停止训练
        logger.info('\t early_stopping_count: {}/{}'.format(early_stopping_count, config.early_stopping_patience))
        if early_stopping_count > config.early_stopping_patience:  # 判断是否提前结束训练,（epoch-best_epoch）>50
            logger.info('\t early_stopping!')  # 输出提前结束语句
            break
    return model  # 返回最终的模型


if __name__ == '__main__':  # 主函数
    deterministic = True  # 确定性=True
    if not deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(config.seed)  # 从config配置中获取随机种子
    np.random.seed(config.seed)  # 从config配置中获取随机种子
    torch.manual_seed(config.seed)  # 从config配置中获取随机种子
    torch.cuda.manual_seed(config.seed)  # 从config配置中获取随机种子
    torch.cuda.manual_seed_all(config.seed)  # 从config配置中获取随机种子

    # 判断是否已经存在对应的工作目录，如果没有那么创造工作目录
    if not os.path.isdir(config.save_path):  # save_path：MoNuSeg/UCTransNet_pretrain/Test_session_11.27_11h44/
        os.makedirs(config.save_path)  # 创建对应的数据集的工作目录，

    # logger_path(日志文件)路径保存：MoNuSeg/UCTransNet_pretrain/Test_session_11.27_11h44/Test_session_11.27_11h44.log/
    logger = logger_config(log_path=config.logger_path)  # 正常执行

    # 参数：模型类型UCTransNet_pretrain或者UCTransNet, 可视化工具：tensorboard=True
    model = main_loop(model_type=config.model_name, tensorboard=True)  # 开始循环训练模型，最后是返回一个性能最优的模型
    os.system("shutdown")  #关机
