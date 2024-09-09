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

def train_model(model, criterion, optimizer, scheduler, num_epochs=25): 
    """ 训练模型，并返回在验证集上的最佳模型和准确率 
    Args: 
    - model(nn.Module): 要训练的模型 
    - criterion: 损失函数 
    - optimizer(optim.Optimizer): 优化器 
    - scheduler: 学习率调度器 
    - num_epochs(int): 最大 epoch 数 
    Return: 
    - model(nn.Module): 最佳模型 
    - best_acc(float): 最佳准确率 
    """
    since = time.time()

    # 保存模型权重
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # 训练集和验证集交替进行前向传播
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 设置为训练模式，可以更新网络参数
            else:
                model.eval()   # 设置为预估模式，不可更新网络参数

            running_loss = 0.0
            running_corrects = 0

            # 遍历数据集
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 清空梯度，避免累加了上一次的梯度
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    # 正向传播
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 反向传播且仅在训练阶段进行优化
                    if phase == 'train':
                        loss.backward() # 反向传播
                        optimizer.step()

                # 统计loss、准确率
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # 发现了更优的模型，记录起来
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # 加载训练的最好的模型
    model.load_state_dict(best_model_wts)
    return model

model = models.resnet18(pretrained=True) # 加载预训练模型
num_ftrs = model.fc.in_features # 获取低级特征维度 
model.fc = nn.Linear(num_ftrs, 2) # 替换新的输出层 
model = model.to(device) 
# 交叉熵作为损失函数 
criterion = nn.CrossEntropyLoss() 
# 所有参数都参加训练 
optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9) 
# 每过 7 个 epoch 将学习率变为原来的 0.1 
scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
model_conv = train_model(model_conv, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=5)

def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}')
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)
visualize_model(model_conv)

plt.ioff()
plt.show()

model_conv = torchvision.models.resnet18(pretrained=True) # 加载预训练模型
for param in model_conv.parameters(): # 锁定模型所有参数
    param.requires_grad = False

num_ftrs = model_conv.fc.in_features  # 获取低级特征维度
model_conv.fc = nn.Linear(num_ftrs, 2) # 替换新的输出层

model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()

# 只有最后一层全连接层fc，参加训练
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

# 每过 7 个 epoch 将学习率变为原来的 0.1 
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
model_conv = train_model(model_conv, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=5)
visualize_model(model_conv)

plt.ioff()
plt.show()