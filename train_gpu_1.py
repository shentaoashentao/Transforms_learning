import torch.optim
import torchvision
import time
from torch import nn

from torch.utils.tensorboard import SummaryWriter
from torch.nn import ReLU
from torch.utils.data import DataLoader

from model import Shentao
#准备数据集
#训练数据集
train_data = torchvision.datasets.CIFAR10(root = "../data_gpu", train=True, transform=torchvision.transforms.ToTensor(), download = True)
#测试数据集
test_data = torchvision.datasets.CIFAR10(root= "../data_gpu", train=False, transform=torchvision.transforms.ToTensor(), download = True)

#看看数据集大小
train_data_size = len(train_data)
test_data_size = len(test_data)

print("训练数据集长度为：{}".format(train_data_size))
print("测试数据集长度为：{}".format(test_data_size))

#利用DataLoader来加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

#搭网络
class Shentao(nn.Module):
    #初始化
    def __init__(self):
        #父类初始化
        super(Shentao, self).__init__()
        self.model = nn.Sequential(
            #卷积第一步  对input进行卷积 到达池化层
            nn.Conv2d(3, 32, 5, 1, 2),
            #2*2 kernel池化
            nn.MaxPool2d(2),
            #nn.Conv2d(3, 32, 5, 1, 2)  in_channel = 32, out_channel = 32, kernel = 5, stride = 1, padding = 2
            #再卷积
            nn.Conv2d(32, 32, 5, 1, 2),
            #池化
            nn.MaxPool2d(2),
            #卷积
            nn.Conv2d(32, 64, 5, 1, 2),
            # 池化
            nn.MaxPool2d(2),
            #展平
            nn.Flatten(),

            nn.Linear(64*4*4, 64),
            nn.Linear(64, 10)

        )
    def forward(self, x):
        x = self.model(x)
        return x

# 创建网络模型
shentao = Shentao()
shentao = shentao.cuda()
#损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.cuda()
#优化器
learning_rate=0.01
optimizer = torch.optim.SGD(shentao.parameters(), lr=learning_rate)

#设置训练网络的参数
#记录训练次数
total_train_step = 0
#记录测试次数
total_test_step = 0
#训练的轮数
epoch = 5

writer = SummaryWriter("train_gpu_1")

start_time = time.time()
for i in range(epoch):
    print("-----------第{}轮训练开始------------".format(i+1))
    #训练开始
    shentao.train()
    for data in train_dataloader:
        imgs, tragets = data

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        imgs = imgs.cuda()
        targets = tragets.cuda()
        tragets = tragets.to(device)

        outputs = shentao(imgs)
        loss = loss_fn(outputs, tragets)#(预测的输出，真实)

        #优化器优化模型
        optimizer.zero_grad()#梯度清零
        loss.backward()#损失进行反向传播
        optimizer.step()#参数进行优化

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            end_time = time.time()
            print(end_time - start_time)
            print("训练次数：{},loss:{}".format(total_train_step, loss))
            writer.add_scalar("train_loss", loss.item(), total_train_step)


    #测试开始
    shentao.eval()## 将模型设置成验证状态
    total_test_loss = 0
    total_accurary = 0

    with torch.no_gard():## 对梯度清零，防止上一轮的梯度影响下一轮的学习
        for data in test_dataloader:
            imgs, tragets = data

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            imgs = imgs.cuda()
            targets = tragets.cuda()
            tragets = tragets.to(device)
            outputs = shentao(imgs)

            loss = loss_fn(outputs, tragets)
            total_test_loss = total_test_loss + loss.item()

            accurary = (outputs.argmax(1) == tragets).sum()
            total_accurary = total_accurary + accurary

    print("整体测试集上的Loss:{}".format(total_test_loss))
    print("整体测试集上的准确率:{}".format(total_accurary/test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accurary", total_accurary/test_data_size, total_test_step)
    total_test_step  = total_test_step + 1

    torch.save(shentao, "shentao_{}.pth".format(i))
    print("模型已保存")

writer.close()