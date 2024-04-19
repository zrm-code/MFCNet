# -*- coding: utf-8 -*-
# @Time    : 2021/6/19 2:44 下午
# @Author  : Haonan Wang
# @File    : Config.py
# @Software: PyCharm
import os
import torch
import time
import ml_collections

#  ---------------parameters of the model(模型参数)----------------------
save_model = True  # 是否保存模型（OK）
# Tensorboard原本是Google TensorFlow的可视化工具，可以用于记录训练数据、评估数据、网络结构、图像等，并且可以在web上展示，对于观察神经网络的过程非常有帮助。
tensorboard = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
use_cuda = torch.cuda.is_available()
seed = 666
os.environ['PYTHONHASHSEED'] = str(seed)
cosineLR = True  # whether use cosineLR or not（是否使用余弦LR）----动态的学习速率参数调整
n_channels = 3
n_labels = 1
epochs = 2000
img_size = 224
print_frequency = 1
save_frequency = 5000
vis_frequency = 1
early_stopping_patience = 110
pretrain = False

# task_name = 'MoNuSeg'
# task_name = 'GlaS'
# task_name = 'ChaoSheng'
# task_name = 'CXR'
task_name = 'KvasirCapsule'
# task_name = 'KvasirSessile'

learning_rate = 1e-3
batch_size = 4

# model_name = 'UCTransNet'
# model_name = 'MultiResUnet'
# model_name = 'AttUNet'
# model_name = 'MTUNet'
# model_name = 'SwinUNet'
# model_name = 'MSESNet'
# model_name = 'MedT'
# model_name = 'TransNet'
# model_name = 'DCSAU_Net'
# model_name = 'CANet'
# model_name = 'CASFNet'
# model_name = 'H2Former'
# model_name = 'HiFormer'
# model_name = 'CFATransUnet'
# model_name = 'SASAN'
# model_name = 'BASNet'
# model_name = 'BEFUnet'
# model_name = 'vmunet'
# model_name = 'PFD_Net'
model_name = 'MFCNet'   # new model

# model_name = 'DAF'
# model_name = 'CMUNet'
# model_name = 'MGCC'
# model_name = 'SegNet'
# model_name = 'MDANet'
# model_name = 'BUSNet'
# model_name = 'M2SNet'

train_dataset = './datasets/' + task_name + '/Train_Folder/'

val_dataset = './datasets/' + task_name + '/Val_Folder/'

test_dataset = './datasets/' + task_name + '/Test_Folder/'

session_name = 'Test_session' + '_' + time.strftime('%m.%d_%Hh%M')

save_path = task_name + '/' + model_name + '/' + session_name + '/'

model_path = save_path + 'models/'

tensorboard_folder = save_path + 'tensorboard_logs/'

logger_path = save_path + session_name + ".log"

visualize_path = save_path + 'visualize_val/'


def get_CTranS_config():  # 获取CTrans相关配置
    config = ml_collections.ConfigDict()
    config.transformer = ml_collections.ConfigDict()
    config.KV_size = 960 # KV_size = Q1 + Q2 + Q3 + Q4
    config.transformer.num_heads = 4
    config.transformer.num_layers = 4
    config.expand_ratio = 4
    config.transformer.embeddings_dropout_rate = 0.1
    config.transformer.attention_dropout_rate = 0.1
    config.transformer.dropout_rate = 0
    config.patch_sizes = [16, 8, 4, 2]
    config.base_channel = 64
    config.n_classes = 1
    return config


# used in testing phase, copy the session name in training phase
test_session = "Test_session_04.18_14h08"
