# # One Shot Learning with Siamese Networks

import torchvision
import torchvision.datasets as dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps    
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from tensorflow_tanxinkeji_works.Preject7_基于人脸的门禁.代码数据分享.SiameseUtil import *


# ## 使用文件夹数据集
folder_dataset = dataset.ImageFolder(root=Config.training_dir) # 将数据按照固定格式放好，就可以调用pytchor中自带的函数读取数据
siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset,
                                        transform=transforms.Compose([transforms.Resize((100,100)), # 调用pytchor.transforms中自带的函数对图像做转换
                                                                      transforms.ToTensor() # 将图片数据类型转化为Tensor
                                                                      ]))

                                                        
# ## 可视化
# 每一列的两张图像属于一对训练图像
# 1 表示他们属于不同的类别, 反之为0
vis_dataloader = DataLoader(siamese_dataset,  # DataLoader类似于Keras中的ImageGenerater
                        shuffle=True,
                        num_workers=8, # 进程数为8
                        batch_size=8)  # 每次读取的照片数目
dataiter = iter(vis_dataloader)


example_batch = next(dataiter)
concatenated = torch.cat((example_batch[0],example_batch[1]),0)
imshow(torchvision.utils.make_grid(concatenated))
print(example_batch[2].numpy())

# ## 定义损失函数
class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        '''
        Args:
            output1: 图片1(经过网路层的)
            output2: 图片2(经过网路层的)
            label: 表示是否为同一个类别，1 表示他们属于不同的类别, 反之为0
        Returns: 返回损失

        '''
        euclidean_distance = F.pairwise_distance(output1, output2) # 计算距离
        # 计算损失：当为同一类别时，我们希望距离小，那么计算出的损失小；当不为同一个类别的时候，我们希望距离大，距离小
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


        return loss_contrastive

# ## 训练
train_dataloader = DataLoader(siamese_dataset,
                        shuffle=True,
                        num_workers=8,
                        batch_size=Config.train_batch_size)

net = SiameseNetwork() # 网络结构
criterion = ContrastiveLoss() # 损失函数
optimizer = optim.Adam(net.parameters(),lr = 0.0005 ) # 指定优化器，与keras优化器不同点：指定优化器时需要将网络结构中的参数传给优化器


counter = []
loss_history = [] 
iteration_number= 0
for epoch in range(0,Config.train_number_epochs):
    for i, data in enumerate(train_dataloader,0):
        img0, img1 , label, _, _ = data
        img0, img1 , label = img0, img1, label
        optimizer.zero_grad() #　梯度清零
        output1, output2 = net(img0, img1) # 经过双子网络
        loss_contrastive = criterion(output1, output2, label) # 计算损失
        loss_contrastive.backward() # 损失误差后向传播
        optimizer.step() # 迭代一次，更新参数
        # 每遍历10次，看一下loss
        if i %10 == 0 :
            print("Epoch number {}\n Current loss {}\n".format(epoch, loss_contrastive.item()))
            iteration_number +=10
            counter.append(iteration_number)
            loss_history.append(loss_contrastive.item())
show_plot(counter,loss_history)

# 保存模型
torch.save(net.state_dict(), "best.siamese.ph")
