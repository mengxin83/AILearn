import torch
import torchvision.datasets
import torch
from torch.utils import data
from torchvision import transforms
from torch import nn

# ToTensor类，实例化一个ToTensor工具
# 将数据集转换为Tensor张量
trans = transforms.ToTensor() 
 
# 训练集 
mnist_train = torchvision.datasets.MNIST(
    root='D://code//vsworkspace//AILearn//data//mnist',train=True,transform=trans,download=True
)
# 测试集
mnist_test = torchvision.datasets.MNIST(
    root='D://code//vsworkspace//AILearn//data//mnist',train=False,transform=trans,download=True
)

# 每批次64个样本
batch_size = 64
train_dataloader = data.DataLoader(mnist_train,batch_size=batch_size)
test_dataloader = data.DataLoader(mnist_test,batch_size=batch_size)
 
# 迭代测试数据集，格式(sample,label)
for X,y in test_dataloader:
    print("Shape of X [N,C,H,W]:",X.shape)
    print("Shape of y:",y.shape,y.dtype)
    break

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork,self).__init__()
        # 多维张量展评为一维
        self.flatten = nn.Flatten()
        # 定义模型各层
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28,
                      ),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,10),
            nn.ReLU()
        )
 
    def forward(self, X):
        X = self.flatten(X)
        logits = self.linear_relu_stack(X)
        return logits
    
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

model = NeuralNetwork().to(device=0)
print(model)

# 交叉熵损失,定义损失函数
loss_fn = nn.CrossEntropyLoss()
# 梯度下降
optimizer = torch.optim.SGD(model.parameters(),lr = 1e-3)

def train(dataloader,model,loss_fn,optimizer):
    # 训练集大小
    size = len(dataloader.dataset)
    for batch,(X,y) in enumerate(dataloader):
        X,y = X.to(0),y.to(0)
 
        pred = model(X)
        loss = loss_fn(pred,y)
 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
 
        if batch % 100 == 0:
            loss,current = loss.item(), batch * len(X)
            print(f"loss:{loss:>7f}[{current:>5d}/{size:>5d}]")

def test(dataloader,model):
    size = len(dataloader.dataset)
    model.eval()
    test_loss,correct = 0,0
    with torch.no_grad():
        for X,y in dataloader:
            X,y = X.to(0),y.to(0)
            pred = model(X)
            # 计算损失值
            test_loss += loss_fn(pred,y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= size
        correct /=size
        print(f"Test Error:\n Accuracy:{100*correct:>0.1f}%,Avg loss:{test_loss:>8f} \n")

# 训练轮次
epochs = 50
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------")
    train(train_dataloader,model,loss_fn,optimizer)
    test(test_dataloader,model)
print("Done!")

