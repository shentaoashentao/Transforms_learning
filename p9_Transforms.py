from torch.utils.tensorboard import SummaryWriter
from torchvision import  transforms
from PIL import Image
import cv2
img_path="E:\\reggie\\reggie-master\\imgs\9a851fbe-5f29-4ec2-b795-bb4553086b4d.jpg"
img = Image.open(img_path)
#print(img)
writer = SummaryWriter("logs")

#把图片转成tensor型
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)

#在pytorch下 当前project下 tensorboard --logdir=logs
#归一化  正则化
print(tensor_img[0][0][0])
#前面的（0.5，0.5，0.5） 是 R G B 三个通道上的均值， 后面(0.5, 0.5, 0.5)是三个通道的标准差，
trans_norm = transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
img_norm = trans_norm(tensor_img)

print(img_norm[0][0][0])
writer.add_image("Tensor_img", img_norm)
writer.close()
#print(tensor_img)

