import torch
from torch import nn
#搭建神经网络
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

if __name__ == '__main__':
    shentao = Shentao()
    input = torch.ones((63, 3, 32, 32))
    output = shentao(input)
    print(output.shape)