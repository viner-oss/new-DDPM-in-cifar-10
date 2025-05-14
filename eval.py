import random
import torchvision
import torch
from torch.nn import MSELoss, CrossEntropyLoss
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms



"""
基本参数设置 Config
"""
batch_size = 64
height = 32
width = 32
T = 1000            # 总时间步数
beta_begin = 0.0001
beta_end = 0.02
test_epcho = 100        # 训练轮数
num_classes = 10 # 类别个数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")