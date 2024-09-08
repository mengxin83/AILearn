import os
import torch
import torchvision.models as models
from torchvision import datasets, transforms

import matplotlib.pyplot as plt

resnet18 = models.resnet18(pretrained=True)
print(resnet18)

# 在训练集上：扩充、归一化
# 在验证集上：归一化
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
#数据集存放路径
data_dir = 'D://code//vsworkspace//AILearn//data//hymenoptera'
#使用torchvision.datasets.ImageFolder类快速封装数据集
#此处使用了lambda语法
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}
#读取数据集的数目
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
#读取数据集中的图像种类
class_names = image_datasets['train'].classes

def imshow(inp, title=None): 
    # 可视化一组 Tensor 的图片
    inp = inp.numpy().transpose((1, 2, 0)) #转换图片维度
    mean = np.array([0.485, 0.456, 0.406]) 
    std = np.array([0.229, 0.224, 0.225]) 
    inp = std * inp + mean #反向操作
    inp = np.clip(inp, 0, 1) 
    plt.imshow(inp) 
    if title is not None: 
        plt.title(title) 
    plt.pause(0.001) # 暂停一会儿，为了将图片显示出来
# 获取一批训练数据
inputs, classes = next(iter(dataloaders['train'])) 
# 批量制作网格
out = torchvision.utils.make_grid(inputs) 
imshow(out, title=[class_names[x] for x in classes]) 