from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter("logs")
img  = Image.open("E:\\reggie\\reggie-master\\imgs\9a851fbe-5f29-4ec2-b795-bb4553086b4d.jpg")
print(img)

#ToTensor
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image("Totensor", img_tensor)
#在pytorch下 当前project下 tensorboard --logdir=logs
#归一化  正则化
print(img_tensor[0][0][0])
#前面的（0.5，0.5，0.5） 是 R G B 三个通道上的均值， 后面(0.5, 0.5, 0.5)是三个通道的标准差，
trans_norm = transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])

writer.add_image("Normalize", img_norm,1)
writer.close()