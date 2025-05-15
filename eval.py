import torch
from torchvision.utils import save_image

from Model import coefficient_dict
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

# 初始化时间步编码张量
time_embeddings = Model.embedding_utils("time", dim=512, device=device, T=T)

# 初始化betas, alphas以及相关张量
Model.coefficient_init(T)

# 模型加载
model = Model.Unet().to(device).eval()
pretrained_dict = torch.load(fr"path.pt")
model.load_state_dict(pretrained_dict)

ddpm = Model.Diffusion()
samples = ddpm.sample(model=model,
            batch_size=batch_size,
            image_size=32,
            device=device,
            time_embeddings=time_embeddings,
            betas=coefficient_dict["betas"],
            alphas_cumprod=coefficient_dict["cumprod_alphas"],
            sqrt_recip_alphas=coefficient_dict["sqrt_recip_alphas"],
            sqrt_one_minus_alphas_cumprod=coefficient_dict["sqrt_one_minus_cumprod_alphas"],
            posterior_variance=coefficient_dict["posterior_variance"],
            T=T)

x = samples.clamp(-1., 1.)
x01 = (x + 1.) / 2.
save_image(x01, r"D:\python\DDPM_for_Classify\generative_image")




