# -*- coding: utf-8 -*-
# @Time    : 2021/6/19 11:30 上午
# @Author  : Haonan Wang
# @File    : Load_Dataset.py
# @Software: PyCharm
import numpy as np
import torch
import random
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F
from typing import Callable
import os
import cv2
from scipy import ndimage

# 定义变换函数，参数：imag，label
def random_rot_flip(image, label):
    k = np.random.randint(0, 4)  # K：返回一个(0, 4)随机整型数
    image = np.rot90(image, k)  #将img按逆时针方向旋转 k*90 度，其中 k 是一个整数。
    label = np.rot90(label, k)  #将mask按逆时针方向旋转 k*90 度，其中 k 是一个整数。
    axis = np.random.randint(0, 2)  # axis：返回一个(0, 2)随机整型数
    image = np.flip(image, axis=axis).copy()  #image：flip()函数翻转一个二维的矩阵，包含垂直翻转，水平翻转，以及垂直水平翻转，
    label = np.flip(label, axis=axis).copy()  #mask：flip()函数翻转一个二维的矩阵，包含垂直翻转，水平翻转，以及垂直水平翻转，
    return image, label  # 返回增强过后的图像和标签

# 定义变换函数，参数：imag，label
def random_rotate(image, label):
    angle = np.random.randint(-20, 20)  # 返回一个(-20, 20)随机整型数
    image = ndimage.rotate(image, angle, order=0, reshape=False)  #image：2D转换之旋转
    label = ndimage.rotate(label, angle, order=0, reshape=False)  #mask：2D转换之旋转
    return image, label  # 返回增强过后的图像和标签

# -------验证集数据生成器---------- #
class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size  # 输出的形状尺寸

    def __call__(self, sample):
        image, label = sample['image'], sample['label']   # 获取图像和标签
        image, label = F.to_pil_image(image), F.to_pil_image(label)   # 将 张量 或 ndarray 转换为 PIL 图像
        x, y = image.size   # 获取图像的尺寸
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)   # 变换

        elif random.random() < 0.5:
            image, label = random_rotate(image, label)   # 变换

        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = F.to_tensor(image)  # to_tensor()：接受PILImage或numpy.ndarray，将其先由HWC转置为CHW格式，再转为float后每个像素除以255.
        label = to_long_tensor(label)
        sample = {'image': image, 'label': label}  # 设置字典，image和mask的字典
        return sample  # 返回image和mask的字典

# -------验证集数据生成器---------- #
class ValGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size  # 输出的形状尺寸

    def __call__(self, sample):
        image, label = sample['image'], sample['label']   # 获取图像和标签
        image, label = F.to_pil_image(image), F.to_pil_image(label)   # 将 张量 或 ndarray 转换为 PIL 图像
        x, y = image.size   # 获取图像的尺寸

        # 如果 H不等于规定输出的H，W不等于规定输出的W，
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = F.to_tensor(image)  # to_tensor()：接受PILImage或numpy.ndarray，将其先由HWC转置为CHW格式，再转为float后每个像素除以255.
        label = to_long_tensor(label)
        sample = {'image': image, 'label': label}  # 设置字典，image和mask的字典
        return sample  # 返回image和mask的字典


def to_long_tensor(pic):
    # handle numpy array 处理numpy数组
    img = torch.from_numpy(np.array(pic, np.uint8))
    # backward compatibility 向后兼容性
    return img.long()


# 依次对每个图片进行尺寸校正，参数是：image 和 mask（不需要阅读，知道功能就行）
def correct_dims(*images):
    corr_images = []  # 校正图像
    # print(images)
    for img in images:  # 循环对图像进行校正
        if len(img.shape) == 2:
            corr_images.append(np.expand_dims(img, axis=2))
        else:
            corr_images.append(img)

    if len(corr_images) == 1:
        return corr_images[0]
    else:
        return corr_images


#  图像转化成2D图像序列（阅读OK）
class ImageToImage2D(Dataset):
    """
    Reads the images and applies the augmentation transform on them.
    （读取图像并对其应用增强转换）

    Usage（用法）:
        1. If used without the unet.model.Model wrapper, an instance of this object should be passed to
           torch.utils.data.DataLoader. Iterating through this returns the tuple of image, mask and image
           filename.
           （如果在没有 unet.model.Model包装器的情况下使用，则应将此对象的实例传递给 torch.utils.data.DataLoader。
            遍历此将返回图像、掩码和图像文件名的元组。）

        2. With unet.model.Model wrapper, an instance of this object should be passed as train or validation
           datasets.
           （使用 unet.model.Model包装器，此对象的实例应作为训练或验证传递数据。）

    Args:
        （1）dataset_path: path to the dataset. Structure of the dataset should be:
        （dataset_path：数据集的路径。数据集的结构应为）：
            dataset_path（数据集路径）
              |-- images
                  |-- img001.png
                  |-- img002.png
                  |-- ...
              |-- masks
                  |-- img001.png
                  |-- img002.png
                  |-- ...

        joint_transform: augmentation transform, an instance of JointTransform2D. If bool(joint_transform)
            evaluates to False, torchvision.transforms.ToTensor will be used on both image and mask.
        one_hot_mask: bool, if True, returns the mask in one-hot encoded form.
        （2）joint_transform：增强变换，JointTransform2D的一个实例。如果布尔（joint_transform）评估为 False，torchvision.transforms.ToTensor 将同时用于图像和Mask版。
       （3）one_hot_mask：bool，如果为 True，则以独热编码形式返回掩码。
    """

    def __init__(self, dataset_path: str, joint_transform: Callable = None, one_hot_mask: int = False,
                 image_size: int = 224) -> None:
        self.dataset_path = dataset_path  # 数据集的路径
        self.image_size = image_size  # 图像的大小
        self.input_path = os.path.join(dataset_path, 'img')  # 获取数据集所有图像的路径：dataset_path/img
        self.output_path = os.path.join(dataset_path, 'labelcol')  # 获取数据集所有mask图像的路径： dataset_path/labelcol
        self.images_list = os.listdir(self.input_path)  # 所有图像的列表
        self.one_hot_mask = one_hot_mask  # 获取独热编码

        if joint_transform:  # 如果joint_transform=True
            self.joint_transform = joint_transform   # self.joint_transform = True

        else:  # 如果joint_transform=False，增强变换，torchvision.transforms.ToTensor 将同时用于图像和Mask版。

            # to_tensor()函数看到，函数接受PILImage或numpy.ndarray，将其先由HWC转置为CHW格式，再转为float后每个像素除以255.
            # 将numpy的ndarray或PIL.Image读的图片转换成形状为(C,H, W)的Tensor格式，且/255归一化到[0, 1.0]之间
            # 通道的具体顺序与cv2读的还是PIL.Image读的图片有关系cv2:(B,G,R)  PIL.Image:(R, G, B)
            to_tensor = T.ToTensor()
            self.joint_transform = lambda x, y: (to_tensor(x), to_tensor(y))   # lambda x, y: x*y；函数输入是x和y，输出是它们的积x*y

    def __len__(self):
        return len(os.listdir(self.input_path))  # 返回列表下有多少张图片

    def __getitem__(self, idx):

        image_filename = self.images_list[idx]  # 所有图像名列表
        # print(image_filename[: -3])
        # read image
        # print(os.path.join(self.input_path, image_filename))
        # print(os.path.join(self.output_path, image_filename[: -3] + "png"))
        # print(os.path.join(self.input_path, image_filename))
        image = cv2.imread(os.path.join(self.input_path, image_filename))  # 将加载所有的图像
        # print("img",image_filename)
        # print("1",image.shape)
        image = cv2.resize(image, (self.image_size, self.image_size))  # 所有图像大小调整到：(224*224)
        # print(np.max(image), np.min(image))
        # print("2",image.shape)
        # read mask image
        mask = cv2.imread(os.path.join(self.output_path, image_filename[: -3] + "png"), 0)  # 加载所有的mask图像，后缀名.png
        # print("mask",image_filename[: -3] + "png")
        # print(np.max(mask), np.min(mask))
        mask = cv2.resize(mask, (self.image_size, self.image_size))  # 所有mask图像大小调整到：(224*224)
        # print(np.max(mask), np.min(mask))
        mask[mask <= 0] = 0  # <0的像素全部置为0
        # (mask == 35).astype(int)
        mask[mask > 0] = 1  # >0的像素全部置为1
        # print("11111",np.max(mask), np.min(mask))

        # correct dimensions if needed(如果需要的话，校正尺寸)
        image, mask = correct_dims(image, mask)  # 校正尺寸
        # image, mask = F.to_pil_image(image), F.to_pil_image(mask)
        # print("11",image.shape)
        # print("22",mask.shape)
        sample = {'image': image, 'label': mask}  # 设置字典，image和mask的字典

        if self.joint_transform:   # self.joint_transform=ture
            sample = self.joint_transform(sample)  # 进行图像增强变换
        # sample = {'image': image, 'label': mask}
        # print("2222",np.max(mask), np.min(mask))

        if self.one_hot_mask:  # true 则以独热编码形式返回掩码。
            assert self.one_hot_mask > 0, 'one_hot_mask must be nonnegative'
            mask = torch.zeros((self.one_hot_mask, mask.shape[1], mask.shape[2])).scatter_(0, mask.long(), 1)
        # mask = np.swapaxes(mask,2,0)
        # print(image.shape)
        # print("mask",mask)
        # mask = np.transpose(mask,(2,0,1))
        # image = np.transpose(image,(2,0,1))
        # print(image.shape)
        # print(mask.shape)
        # print(sample['image'].shape)

        return sample, image_filename  # 返回image和mask的字典，以及对应的图像名
