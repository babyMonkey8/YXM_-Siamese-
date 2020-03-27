"""
训练一个图片分类器
我们要按顺序做这几个步骤：

使用torchvision来读取并预处理CIFAR10数据集
定义一个卷积神经网络
定义一个损失函数
在神经网络中训练训练集数据
使用测试集数据测试神经网络
"""

"""
1. 读取并预处理CIFAR10
使用torchvision读取CIFAR10相当的方便。
"""

import torchvision # 这个库提供了一些计算机视觉处理的常用库
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable


# torchvision数据集的输出是在[0, 1]范围内的PILImage图片。
# 我们此处使用归一化的方法将其转化为Tensor，数据范围为[-1, 1]

transform=transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                             ])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, 
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, 
                                          shuffle=False, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
'''注：这一部分需要下载部分数据集 因此速度可能会有一些慢 同时你会看到这样的输出

Downloading http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz to ./data/cifar-10-python.tar.gz
Extracting tar file
Done!
Files already downloaded and verified
'''

# functions to show an image
import matplotlib.pyplot as plt
import numpy as np
'''%matplotlib inline'''
def imshow(img):
    img = img / 2 + 0.5 # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()

# show some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s'%classes[labels[j]] for j in range(4)))
'''
2. 定义一个卷积神经网络
'''
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool  = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
'''
3. 定义代价函数和优化器
'''
criterion = nn.CrossEntropyLoss() # 使用 Cross-Entropy 损失
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

'''
4. 训练网络
我们只需一轮一轮迭代然后不断通过输入来进行参数调整就行了。
'''
for epoch in range(2): # 迭代 2 次
    
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # 得到图像和对应的标签
        inputs, labels = data
        
        # 把输入变为变量
        inputs, labels = Variable(inputs), Variable(labels)
        
        # 将梯度置零
        optimizer.zero_grad()
        
        # 前向传播 + 计算损失 + 误差后传 + 优化权值
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()        
        optimizer.step()
        
        # 打印统计结果
        running_loss += loss.data.item()# loss.data[0]
        if i % 2000 == 1999: # 每 2000 个图像计算一次
            print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, running_loss / 2000))
            running_loss = 0.0
print('Finished Training')
'''这部分的输出结果为
[1,  2000] loss: 2.212
[1,  4000] loss: 1.892
[1,  6000] loss: 1.681
[1,  8000] loss: 1.590
[1, 10000] loss: 1.515
[1, 12000] loss: 1.475
[2,  2000] loss: 1.409
[2,  4000] loss: 1.394
[2,  6000] loss: 1.376
[2,  8000] loss: 1.334
[2, 10000] loss: 1.313
[2, 12000] loss: 1.264
Finished Training
'''
'''
通过对比神经网络给出的分类和已知的类别结果，可以得出正确与否，
如果预测的正确，我们可以将样本加入正确预测的结果的列表中。

好的第一步，让我们展示几张照片来熟悉一下。
'''
dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s'%classes[labels[j]] for j in range(4)))
'''
看看神经网络如何预测这几个照片的标签。
'''
outputs = net(Variable(images))

# 输出 10 个类别的概率. 值越高, 属于某个类别可能性越大. 
# 得到每个图片预测的10个类别概率中值最高的那一个的index
_, predicted = torch.max(outputs.data, 1)

print('Predicted: ', ' '.join('%5s'% classes[predicted[j][0]] for j in range(4)))

'''输出结果为
Predicted:    cat plane   car plane
'''
'''
结果看起来挺好。看看神经网络在整个数据集上的表现结果如何。
'''
correct = 0
total = 0
for data in testloader:
    images, labels = data
    outputs = net(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

'''输出结果为
Accuracy of the network on the 10000 test images: 54 %
'''
'''
输出的结果比随机整的要好，随机选择的话从十个中选择一个出来，准确率大概只有10%。
看上去神经网络学到了点东西。
那么到底哪些类别表现良好又是哪些类别不太行呢？
'''
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
for data in testloader:
    images, labels = data
    outputs = net(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    c = (predicted == labels).squeeze()
    for i in range(4):
        label = labels[i]
        class_correct[label] += c[i]
        class_total[label] += 1

for i in range(10):
    print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))

'''输出结果为
Accuracy of plane : 73 %
Accuracy of   car : 70 %
Accuracy of  bird : 52 %
Accuracy of   cat : 27 %
Accuracy of  deer : 34 %
Accuracy of   dog : 37 %
Accuracy of  frog : 62 %
Accuracy of horse : 72 %
Accuracy of  ship : 64 %
Accuracy of truck : 53 %
'''
'''
好吧，接下来该怎么搞了？
我们该如何将神经网络运行在GPU上呢？
在GPU上进行训练, 把Tensor传递给GPU，也可以将神经网络传递给GPU。
'''
net.cuda()

'''输出结果为
Net (
  (conv1): Conv2d(3, 6, kernel_size=(5, 5), stride=(1, 1))
  (pool): MaxPool2d (size=(2, 2), stride=(2, 2), dilation=(1, 1))
  (conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
  (fc1): Linear (400 -> 120)
  (fc2): Linear (120 -> 84)
  (fc3): Linear (84 -> 10)
)
'''

"""
记住，每一步都需要把输入和目标传给GPU。

   inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
"""