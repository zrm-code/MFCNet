import torch.optim
from Load_Dataset import ValGenerator, ImageToImage2D
from torch.utils.data import DataLoader
import warnings

warnings.filterwarnings("ignore")
import Config as config
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import nets.HiFormer_configs as configs

from utils import *
import cv2
from nets.TransUNet import CONFIGS as CONFIGS_ViT_seg
from nets.UNet import UNet
from nets.MultiResUNet import MultiResUnet
from nets.AttenUNet import AttUNet  # Attention-UNet
from nets.MTUNet import MTUNet  # MTUNet-UNet
from nets.SwinUNet import SwinUNet  # SwinUNet
from nets.MSESNet import MSESNet  # MSESNet
from nets.UCTransNet import UCTransNet  # UCTransNet
from nets.TransUNet import TransNet  # TransNet
from nets.MedT_axialnet import MedT  # MedT
from nets.CAFNet import CAFNet  # BASNet
from nets.NestedUNet import NestedUNet  # UNet++
from nets.CANet import LUNet  # CANet
from nets.DCSAU_Net import DCSAU_Net  # DCSAU_Net
# from nets.DSTransUNet import DSTransUNet  # DSTransUNet
from nets.DCFNet import BUSNet
from nets.BASNet import BASNet
from nets.CASFNet import CASFNet           # CASFNet
from nets.H2Former  import H2Former
from nets.HiFormer  import HiFormer
from nets.CFATransUnet import CFATransUnet
from nets.PFD_Net import PFD_Net
from nets.MFCNet import MFCNet
# =======================肿瘤专门网络========================== #
from nets.DAF import DAF
from nets.CMUNet import CMUNet
from nets.MGCC import MGCC
from nets.MDA_Net import MDA_Net
from nets.SegNet import SegNet
from nets.MSNet import M2SNet


def show_image_with_dice(predict_save, labs, save_path):
    tmp_lbl = (labs).astype(np.float32)
    tmp_3dunet = (predict_save).astype(np.float32)
    dice_pred = 2 * np.sum(tmp_lbl * tmp_3dunet) / (np.sum(tmp_lbl) + np.sum(tmp_3dunet) + 1e-5)
    # dice_show = "%.3f" % (dice_pred)
    iou_pred = jaccard_score(tmp_lbl.reshape(-1), tmp_3dunet.reshape(-1))
    # fig, ax = plt.subplots()
    # plt.gca().add_patch(patches.Rectangle(xy=(4, 4),width=120,height=20,color="white",linewidth=1))
    if config.task_name is "MoNuSeg":
        predict_save = cv2.pyrUp(predict_save, (448, 448))
        predict_save = cv2.resize(predict_save, (2000, 2000))
        # kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32) #定义一个核
        # predict_save = cv2.filter2D(predict_save, -1, kernel=kernel)
        cv2.imwrite(save_path, predict_save * 255)
    else:
        cv2.imwrite(save_path, predict_save * 255)
    # plt.imshow(predict_save * 255,cmap='gray')
    # plt.text(x=10, y=24, s="Dice:" + str(dice_show), fontsize=5)
    # plt.axis("off")
    # remove the white borders
    # height, width = predict_save.shape
    # fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())
    # plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    # plt.margins(0, 0)
    # plt.savefig(save_path, dpi=2000)
    # plt.close()
    return dice_pred, iou_pred


def vis_and_save_heatmap(model, input_img, img_RGB, labs, vis_save_path, dice_pred, dice_ens):
    model.eval()
    if config.model_name == 'MSESNet':
        output, d1, d2, d3, d4, d5 = model(input_img.cuda())

    elif config.model_name == 'MFCNet':
        output, d2, d3, d4, d5 = model(input_img.cuda())

    elif config.model_name == 'BASNet':
        output = model(input_img.cuda())

    elif model_type == 'DSTransUNet':
        output, l1, l2 = model(input_img.cuda())

    elif model_type == 'MGCC':
        output, l1, l2, l3 = model(input_img.cuda())

    elif model_type == 'PFD_Net':
        output, l1, l2, l3 = model(input_img.cuda())

    elif model_type == 'CFATransUnet':
        output, l1 = model(input_img.cuda())

    elif config.model_name == 'DAF':
        output, d1, d2, d3, d4, d5, d6, d7, d8 = model(input_img.cuda())

    elif model_type == 'BUSNet':
        output, d1, d2 = model(input_img.cuda())
    else:
        output = model(input_img.cuda())
    pred_class = torch.where(output > 0.5, torch.ones_like(output), torch.zeros_like(output))
    predict_save = pred_class[0].cpu().data.numpy()
    predict_save = np.reshape(predict_save, (config.img_size, config.img_size))
    dice_pred_tmp, iou_tmp = show_image_with_dice(predict_save, labs,
                                                  save_path=vis_save_path + '_predict' + model_type + '.jpg')
    return dice_pred_tmp, iou_tmp


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    test_session = config.test_session  # 配置test_session=test_session = "Test_session_07.03_20h39"
    if config.task_name is "GlaS":
        test_num = 80
        model_type = config.model_name
        model_path = "./GlaS/" + model_type + "/" + test_session + "/models/best_model-" + model_type + ".pth.tar"

    elif config.task_name is "MoNuSeg":
        test_num = 14
        model_type = config.model_name
        model_path = "./MoNuSeg/" + model_type + "/" + test_session + "/models/best_model-" + model_type + ".pth.tar"

    elif config.task_name is "ChaoSheng":
        test_num = 43
        model_type = config.model_name
        model_path = "./ChaoSheng/" + model_type + "/" + test_session + "/models/best_model-" + model_type + ".pth.tar"

    elif config.task_name is "KvasirSessile":
        test_num = 60
        model_type = config.model_name
        model_path = "./KvasirSessile/" + model_type + "/" + test_session + "/models/best_model-" + model_type + ".pth.tar"

    elif config.task_name is "KvasirCapsule":
        test_num = 11
        model_type = config.model_name
        model_path = "./KvasirCapsule/" + model_type + "/" + test_session + "/models/best_model-" + model_type + ".pth.tar"


    elif config.task_name is "KvasirSEG":
        test_num = 250
        model_type = config.model_name
        model_path = "./KvasirSEG/" + model_type + "/" + test_session + "/models/best_model-" + model_type + ".pth.tar"

    save_path = config.task_name + '/' + model_type + '/' + test_session + '/'
    vis_path = "./" + config.task_name + '_visualize_test/'
    if not os.path.exists(vis_path):
        os.makedirs(vis_path)

    checkpoint = torch.load(model_path, map_location='cuda')

    if model_type == 'UCTransNet':  # 如果模型类型是UCTransNet
        config_vit = config.get_CTranS_config()
        model = UCTransNet(config_vit, n_channels=config.n_channels, n_classes=config.n_labels)  # 加载UCTransNet模型（非预训练）

    elif model_type == 'BASNet':
        config_vit = config.get_CTranS_config()  # 获得相应的参数配置
        model = BASNet(config_vit, n_channels=config.n_channels, n_classes=config.n_labels)

    elif model_type == 'UNet':  # 如果模型类型是UCTransNet
        model = UNet()

    elif model_type == 'CFATransUnet':
        model = CFATransUnet()

    elif model_type == 'CANet':
        model = LUNet(3, 1)

    elif model_type == 'DCSAU_Net':
        model = DCSAU_Net(img_channels=3, n_classes=1)

    elif model_type == 'NestedUNet':
        model = NestedUNet()

    elif model_type == 'DAF':
        model = DAF()

    elif model_type == 'MGCC':
        model = MGCC()

    elif model_type == 'CASFNet':
        model = CASFNet()

    elif model_type == 'CMUNet':
        model = CMUNet()

    elif model_type == 'MDANet':
        model = MDA_Net()

    elif model_type == 'M2SNet':
        model = M2SNet()

    elif model_type == 'PFD_Net':
        model = PFD_Net()

    elif model_type == 'H2Former':
        model = H2Former(224,1)

    elif model_type == 'HiFormer':
        model = HiFormer(config=configs.get_hiformer_b_configs(), img_size=224, n_classes=1)

    elif model_type == 'SegNet':
        model = SegNet(3, 1)

    elif model_type == 'CAFNet':
        config_vit = config.get_CTranS_config()
        model = CAFNet(config_vit, n_channels=config.n_channels, n_classes=config.n_labels)  # 定义网络

    elif model_type == 'BUSNet':
        config_vit = config.get_CTranS_config()
        model = BUSNet(config_vit, 128, n_channels=config.n_channels, n_classes=config.n_labels)

    elif model_type == 'MedT':  # 如果模型类型是UCTransNet
        model = MedT(img_size=224, imgchan=3)

    elif model_type == 'MSESNet':  # 如果模型类型是UCTransNet
        config_vit = config.get_CTranS_config()
        model = MSESNet(config_vit, n_channels=config.n_channels, n_classes=config.n_labels)

    elif model_type == 'MFCNet':  # 如果模型类型是UCTransNet
        config_vit = config.get_CTranS_config()
        model = MFCNet(config_vit, n_channels=config.n_channels, n_classes=config.n_labels)

    elif model_type == 'TransNet':
        config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
        config_vit.n_classes = 1
        config_vit.n_skip = 3
        if 'R50-ViT-B_16'.find('R50') != -1:
            config_vit.patches.grid = (
                int(224 / 16), int(224 / 16))
        model = TransNet(config_vit, img_size=224, num_classes=config_vit.n_classes)

    elif model_type == 'MultiResUnet':  # 如果模型类型是UCTransNet
        model = MultiResUnet()

    elif model_type == 'AttUNet':  # 如果模型类型是UCTransNet
        model = AttUNet()

    elif model_type == 'MTUNet':  # 如果模型类型是UCTransNet
        model = MTUNet()

    elif model_type == 'SwinUNet':  # 如果模型类型是UCTransNet
        model = SwinUNet()

    # elif model_type == 'DSTransUNet':
    #     model = DSTransUNet(128, 1)

    else:
        raise TypeError('Please enter a valid name for the model type')

    model = model.cuda()
    if torch.cuda.device_count() > 1:
        print("Let's use {0} GPUs!".format(torch.cuda.device_count()))
        model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    model.load_state_dict(checkpoint['state_dict'])
    print('Model loaded !')
    tf_test = ValGenerator(output_size=[config.img_size, config.img_size])
    test_dataset = ImageToImage2D(config.test_dataset, tf_test, image_size=config.img_size)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    dice_pred = 0.0
    iou_pred = 0.0
    dice_ens = 0.0

    with tqdm(total=test_num, desc='Test visualize', unit='img', ncols=70, leave=True) as pbar:
        for i, (sampled_batch, names) in enumerate(test_loader, 1):
            test_data, test_label = sampled_batch['image'], sampled_batch['label']
            arr = test_data.numpy()
            arr = arr.astype(np.float32())
            lab = test_label.data.numpy()
            img_lab = np.reshape(lab, (lab.shape[1], lab.shape[2])) * 255
            fig, ax = plt.subplots()
            plt.imshow(img_lab, cmap='gray')
            plt.axis("off")
            height, width = config.img_size, config.img_size
            fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.savefig(vis_path + str(i) + "_lab.jpg", dpi=300)
            plt.close()
            input_img = torch.from_numpy(arr)
            dice_pred_t, iou_pred_t = vis_and_save_heatmap(model, input_img, None, lab,
                                                           vis_path + str(i),
                                                           dice_pred=dice_pred, dice_ens=dice_ens)
            dice_pred += dice_pred_t
            iou_pred += iou_pred_t
            torch.cuda.empty_cache()
            pbar.update()
    print("dice_pred", dice_pred / test_num)
    print("iou_pred", iou_pred / test_num)
