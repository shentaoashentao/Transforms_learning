import torchvision

#下载数据集到./dataset下
train_set = torchvision.datasets.CIFAR10(root = "./dataset" ,  train=True ,download=True)
test_set = torchvision.datasets.CIFAR10(root = "./dataset" ,  train=False ,download=True)

#看数据集的第一个
print(train_set[0])