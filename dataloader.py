import torchvision

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_data = torchvision.datasets.CIFAR10(root = "./dataset" ,  train = False , transform = torchvision.transforms.ToTensor(),download=True)

test_loader = DataLoader(dataset = test_data ,batch_size=4, shuffle=True,num_workers=0,drop_last=False)

#测试数据集中第一张图片及target

img,target = test_data[0]

print(img.shape)
print(target)

writer = SummaryWriter("dataloader")
step = 0
for data in test_loader:
    imgs , targets = data
    #print(img.shape)
    #rint(target)
    writer.add_images("test_data",imgs,step)
    step = step+1

writer.close()