from torch.utils.data import Dataset
from PIL import Image
import os
class MyData(Dataset):
    root_dir="E:\\reggie\\reggie-master\\imgs\9a851fbe-5f29-4ec2-b795-bb4553086b4d.jpg"
    img_path="E:\\reggie\\reggie-master\\imgs\9a851fbe-5f29-4ec2-b795-bb4553086b4d.jpg"
    img=Image.open(img_path)
    print(img.size)
    img.show()

    def __init__(self, root_dir, label_dir):
        self.root_dir=root_dir
        self.label_dir=label_dir
        self.path=os.path.join(self.root_dir,self.label_dir)
        self.img_path=os.listdir(self.path)

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir,self.label_dir,img_name)
        img = Image.open(img_item_path)
        label=self.label_dir
        return img,label

    def __len__(self):
        return len(self.img_path)
    print(__len__(img))