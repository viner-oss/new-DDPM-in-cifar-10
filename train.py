import math

import torchvision
import torch
from torch.nn import MSELoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms

import Model

"""
--- Config ---
"""
batch_size = 64
height = 32
width = 32
T = 300            # 总时间步数
beta_begin = 0.0001
beta_end = 0.02
train_epoch = 150        # 训练轮数
warmup_epoch = 25
num_classes = 10 # 类别个数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 图像预处理
mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)
transform = torchvision.transforms.Compose([
        transforms.RandomHorizontalFlip(),       # 水平翻转（概率0.5）
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
])

"""
--- adding noise part ---
"""
Model.coefficient_init(T)
ddpm = Model.Diffusion()
Time_Embeddings = Model.embedding_utils(string="time", dim=512, device=device, T=T)

# 准备训练集与加载数据集
train_set = torchvision.datasets.CIFAR10("../data",train=True,transform=transform,
                                         download=False)
train_len = len(train_set)
TrainLoader = DataLoader(dataset=train_set, batch_size=batch_size, drop_last=True)
print(f"The Length of Train Set:{train_len}")

# 学习率设置
learning_rate_min = 0.00002
learning_rate_max = 0.0001

# warmup与余弦退火
def lr_lambda(current_epoch:int):
    """
    epoch record from 0, if current_epoch is smaller than warmup_epoch, take warmup measure, else take cosine measure
    :param current_epoch:
    :return:
    """
    if current_epoch < warmup_epoch:
        return ((learning_rate_min + (learning_rate_max - learning_rate_min) * float(current_epoch+1) / warmup_epoch)
                / learning_rate_max)
    else:
        progress = float(current_epoch - warmup_epoch) / float(max(1, train_epoch - warmup_epoch))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        lr = learning_rate_min + (learning_rate_max - learning_rate_min) * cosine_decay
        return lr / learning_rate_max


# 模型 损失函数 优化器 训练技巧
model = Model.Unet().to(device)
criterion = MSELoss().to(device)
optimizer = AdamW(model.parameters(),lr=learning_rate_max)
scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

for epoch in range(train_epoch):
    model.train()
    running_loss = 0.0
    for inputs, _ in TrainLoader:
        inputs = inputs.to(device)
        optimizer.zero_grad()
        random_time_steps = torch.randint(0, T, [batch_size,], dtype=torch.long)
        images, real_noise = ddpm.adding_noise(x_0=inputs, random_time_steps=random_time_steps,
                                               device=device)

        predict_noise = model(images, Time_Embeddings[random_time_steps])
        loss = criterion(predict_noise, real_noise)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
    scheduler.step()

    epoch_loss = running_loss / len(TrainLoader.dataset)
    current_lr = optimizer.param_groups[0]['lr']
    if epoch % 5 == 0:
        torch.save(model.state_dict(), fr"D:\python\DDPM_for_Classify\Parameters\Parameter{epoch}")
    print(f"Epoch [{epoch+1:02d}/{train_epoch:02d}]"
          f"Loss: {epoch_loss:.8f}"
          f"LR: {current_lr:.6f}")
