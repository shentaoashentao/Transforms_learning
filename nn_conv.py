#卷积
import torch
import torch.nn.functional as F
input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]])

kernel = torch.tensor([[1, 2, 1],
                       [0, 1, 0],
                       [2, 1, 0]])
#通道数是1  batch为1  数据维度5*5
input= torch.reshape(input, (1, 1, 5, 5))
kernel = torch.reshape(kernel, (1, 1, 3, 3))
print(input.shape)
print(kernel.shape)
#stride=1 表示一格一格移
output1 = F.conv2d(input, kernel, stride=1)
print(output1)
#stride=1 表示两格两个移
output2 = F.conv2d(input, kernel, stride=2)
print(output2)
#padding=1表示在矩形外侧再添一圈0
output3 = F.conv2d(input, kernel, stride=1, padding=1)
print(output3)
