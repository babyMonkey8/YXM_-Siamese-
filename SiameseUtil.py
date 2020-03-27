import torchvision
import torchvision.datasets as dataset
from torch.utils.data import DataLoader,Dataset
import torch.nn as nn  # nn表示神经网络
import numpy as np
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps   
import matplotlib.pyplot as plt
import torchvision.utils

# ## 帮助函数

def imshow(img,text=None,should_save=False):  #显示图片
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()    

def show_plot(iteration,loss):
    plt.plot(iteration,loss)
    plt.show()

#%% [markdown]
# ## 用于配置的帮助类

#%%
class Config():  # 配置类
    training_dir = "data/faces/training/s1" # 训练文件
    testing_dir = "data/faces/testing/" # 测试文件
    train_batch_size = 64 # 每次训练的batch
    train_number_epochs = 100 # epoch数量


# ## 定制 Dataset 类
# 这个类用于产生一对图片. 
# 不能使用pychor中自带的读取文件的工具类，因为在本项目中我们并不关心类别的标签，而是判断两个图片是否类似，所以我继承工具类进而自定义一个类
# 每一次都读取两张图片
class SiameseNetworkDataset(Dataset):

    def __init__(self,imageFolderDataset,transform=None, keepOrder=False):
        self.imageFolderDataset = imageFolderDataset    
        self.transform = transform
        self.keepOrder = keepOrder
        
    def __getitem__(self,index):
        if self.keepOrder:
            img_tuple = self.imageFolderDataset.imgs[index] # 读取图片文件路径
            img = Image.open(img_tuple[0]) # 读取图片
            # 图像转为黑白灰度图
            img = img.convert("L") # 将值转换为适当的类型
            if self.transform is not None:
                img = self.transform(img)
        
            return img, img, torch.from_numpy(np.array([int(0)],dtype=np.float32)), img_tuple[0], img_tuple[0]

        img0_tuple = random.choice(self.imageFolderDataset.imgs) # 随机选择一个图像
        # 使得50%的训练数据为一对图像属于同一类别
        should_get_same_class = random.randint(0,1)  # 0 表示不同的类别图片；1表示相同的类别图片
        if should_get_same_class:
            while True:
                # 循环直到一对图像属于同一类别
                img1_tuple = random.choice(self.imageFolderDataset.imgs) 
                if img0_tuple[1]==img1_tuple[1]:
                    break
        else:
            while True:
                # 循环直到一对图像属于不同的类别
        
                img1_tuple = random.choice(self.imageFolderDataset.imgs) 
                if img0_tuple[1] !=img1_tuple[1]:
                    break

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])
        # 图像转为黑白灰度图
        img0 = img0.convert("L")
        img1 = img1.convert("L")

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        # torch.from_numpy(np.array([int(img1_tuple[1] != img0_tuple[1])] 表示两个图片是否为一个类别
        # img0_tuple[0]表示图片1的路径 ； img1_tuple[0]表示图片2的路径
        return img0, img1, torch.from_numpy(np.array([int(img1_tuple[1]!=img0_tuple[1])],dtype=np.float32)), img0_tuple[0], img1_tuple[0]

    def __len__(self):
        return len(self.imageFolderDataset.imgs)

#%% [markdown]
# ## 定义Siamese神经网络
#%%
class SiameseNetwork(nn.Module):  # 继承与nn.Module模块
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = nn.Sequential(
            # 卷积层1
            nn.ReflectionPad2d(1), # pytorch常用的padding函数:https://www.cnblogs.com/wanghui-garcia/p/11265843.html
            nn.Conv2d(1, 4, kernel_size=3), # 输入是一个通道（黑白图），输出是4个
            nn.ReLU(input(True)),
            nn.BatchNorm2d(4),
            # 卷积层2
            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            # 卷积层3
            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
        )
        # 全链接层
        self.fc1 = nn.Sequential(
            # 全链接层1
            nn.Linear(8 * 100 * 100, 500),
            nn.ReLU(inplace=True),
            # 全链接层2
            nn.Linear(500, 500),
            nn.ReLU(inplace= True),
            # 全链接层3
            nn.Linear(500, 5) # 输入是500，输出是5
            )
    # 前向传播
    def forward_once(self, x):
        output = self.cnn1(x) # 卷积层
        output = output.view(output.size()[0], -1) # 类似于Keras中的展平层
        output = self.fc1(output) # 全链接层
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1) # 图片1
        output2 = self.forward_once(input2) # 图片2
        return output1, output2