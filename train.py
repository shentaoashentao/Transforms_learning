import torch.optim
import torchvision
from torch import nn
from torch.nn import ReLU

#准备数据集
#训练数据集
from torch.utils.data import DataLoader

from model import Shentao

train_data = torchvision.datasets.CIFAR10(root = "../data", train=True, transform=torchvision.transforms.ToTensor(), download=True)
#测试数据集
test_data = torchvision.datasets.CIFAR10(root= "../data", train=False, transform=torchvision.transforms.ToTensor(), download=True)

#看看数据集大小
train_data_size = len(train_data)
test_data_size = len(test_data)

print("训练数据集长度为：{}".format(train_data_size))
print("测试数据集长度为：{}".format(test_data_size))

#利用DataLoader来加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

#创建网络模型
shentao = Shentao()

#损失函数
loss_fn = nn.CrossEntropyLoss()

#优化器
learning_rate=0.01
optimizer = torch.optim.SGD(shentao.parameters(), lr=learning_rate, )

#设置训练网络的参数
#记录训练次数
total_train_step = 0
#记录测试次数
total_test_step =0
#训练的轮数
epoch = 10

for i in range(epoch):
    print("-----------第{}轮训练开始------------".format(i+1))
    #训练开始
    for data in train_dataloader:
        imgs, tragets = data
        outputs = shentao(imgs)
        loss = loss_fn(outputs,tragets)

        #优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        print("训练次数：{},loss:{}".format(total_train_step, loss))

    #测试开始
    total_test_loss = 0
    with torch.no_gard():
        for data in test_dataloader:
            imgs,tragets = data
            outputs = shentao(imgs)
            loss = loss_fn(outputs,tragets)
            total_test_loss = total_test_loss + loss.item()

        print("整体测试集上的Loss:{}".format(total_test_loss))
        writer.add_scalar("test_loss", total_test_loss, total_test_step)
        total_test_step  = total_test_step + 1

        torch.save(shentao,"shentao_{}.pth".format(i))
        print("模型已保存")

writer.close()