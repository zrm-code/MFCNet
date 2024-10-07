
import os
import torch
import time
import ml_collections

#  ---------------parameters of the model----------------------
save_model = True 
tensorboard = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
use_cuda = torch.cuda.is_available()
seed = 666
os.environ['PYTHONHASHSEED'] = str(seed)
cosineLR = True  # whether use cosineLR or not
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
task_name = 'KvasirCapsule'


learning_rate = 1e-3
batch_size = 4

model_name = 'MFCNet'

train_dataset = './datasets/' + task_name + '/Train_Folder/'

val_dataset = './datasets/' + task_name + '/Val_Folder/'

test_dataset = './datasets/' + task_name + '/Test_Folder/'

session_name = 'Test_session' + '_' + time.strftime('%m.%d_%Hh%M')

save_path = task_name + '/' + model_name + '/' + session_name + '/'

model_path = save_path + 'models/'

tensorboard_folder = save_path + 'tensorboard_logs/'

logger_path = save_path + session_name + ".log"

visualize_path = save_path + 'visualize_val/'


def get_CTranS_config():  
    config = ml_collections.ConfigDict()
    config.transformer = ml_collections.ConfigDict()
    config.KV_size = 960 # KV_size = Q1 + Q2 + Q3 + Q4
    config.transformer.num_heads = 2
    config.transformer.num_layers = 2
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
