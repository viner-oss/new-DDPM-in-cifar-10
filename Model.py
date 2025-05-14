import torch
import math
from torch import nn

"""
beta alpha dictionary
"""
coefficient_dict = {}
def coefficient_init(T:int, beta_src:float=0.0001, beta_cls:float=0.02):
    betas = torch.linspace(start=beta_src, end=beta_cls, steps=T, dtype=torch.float32)   # β[]
    alphas = 1 - betas   # α[]
    cumprod_alphas = torch.cumprod(alphas, dim=0)
    sqrt_one_minus_cumprod_alphas = torch.sqrt(1 - cumprod_alphas)
    sqrt_cumprod_alphas = torch.sqrt(cumprod_alphas)
    coefficient_dict["betas"] = betas
    coefficient_dict["alphas"] = alphas
    coefficient_dict["cumprod_alphas"] = cumprod_alphas
    coefficient_dict["sqrt_one_minus_cumprod_alphas"] = sqrt_one_minus_cumprod_alphas
    coefficient_dict["sqrt_cumprod_alphas"] = sqrt_cumprod_alphas

"""
func:get the time_embedding ranging from 0 to T
embedding_utils
"""
def embedding_utils(string:str, dim:int, device, num_classes:int=0, T:int=0):
    div = torch.exp(torch.arange(0, dim, 2).float() * -(math.log(10000.0) / dim))
    assert string == "time" or string == "label", "error:invalid iput"
    if string == "time":
        position = torch.arange(T, dtype=torch.float32).unsqueeze(1)
        embedding = torch.zeros([T, dim])
        embedding[:,0::2] = torch.sin(position * div)
        embedding[:,1::2] = torch.cos(position * div)
        embedding = embedding.to(device=device)
        return embedding
    elif string == "label":
        position = torch.arange(num_classes, dtype=torch.float32).unsqueeze(1)
        embedding = torch.zeros([num_classes, dim])
        embedding[:, 0::2] = torch.sin(position * div)
        embedding[:, 1::2] = torch.cos(position * div)
        embedding = embedding.to(device=device)
        return embedding

"""
func:time_step_embedding with features
"""
def time_embedding_util(x, time_embedding, target_dim:int, embedding_dim:int=512):
    MLP = nn.Linear(in_features=embedding_dim, out_features=target_dim)
    time_vector = MLP(time_embedding)
    time_vector = time_vector.unsqueeze(2).unsqueeze(2)
    return x + time_vector
"""
Multihead_Attn
"""
class Multihead_Attn(nn.Module):
    def __init__(self, in_channels:int, hidden_dim:int):
        super(Multihead_Attn, self).__init__()
        self.W_K = nn.Linear(in_features=in_channels, out_features=hidden_dim)
        self.W_Q = nn.Linear(in_features=in_channels, out_features=hidden_dim)
        self.W_V = nn.Linear(in_features=in_channels, out_features=hidden_dim)
        self.Multihead_Attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, dropout=0.2)

    def forward(self, x):
        batch, channel, height, width = x.shape
        vector = x.view([batch, channel, height*width]).permute([2,0,1])
        Q = self.W_Q(vector)
        K = self.W_K(vector)
        V = self.W_V(vector)
        attn_output, attn_weight = self.Multihead_Attn(Q,K,V)
        new_channel = attn_output.shape[2]
        attn_output = attn_output.view([height, width, batch, new_channel]).permute([2,3,0,1])
        return attn_output

"""
Mobile Inverted Convolution
"""
class MobileBlock(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, expansion_factor:int):
        super(MobileBlock, self).__init__()
        self.target_dim = in_channels
        self.expansion_channels = in_channels * expansion_factor

        # expansion layer
        self.Expansion = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=self.expansion_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(8,self.expansion_channels),
            nn.SiLU(inplace=True)
        )

        # depth_wise convolution
        self.Depth_wise_conv = nn.Sequential(
            nn.Conv2d(in_channels=self.expansion_channels, out_channels=self.expansion_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8,self.expansion_channels),
            nn.SiLU(inplace=True)
        )

        # channel_weight
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # SE_net
        self.SE = nn.Sequential(
            nn.Linear(in_features=self.expansion_channels, out_features=self.expansion_channels//16),
            nn.ReLU6(inplace=True),
            nn.Linear(in_features=self.expansion_channels//16, out_features=self.expansion_channels),
            nn.Sigmoid()
        )

        # project layer
        self.Projection = nn.Sequential(
            nn.Conv2d(in_channels=self.expansion_channels, out_channels=out_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(8,out_channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, x, time_embedding, res:bool):
        y = time_embedding_util(x, time_embedding, self.target_dim)
        y = self.Expansion(x)
        y = self.Depth_wise_conv(y)
        batch, channel, _, _ = y.shape
        channel_weight = self.avg_pool(y).view([batch, channel])
        channel_weight = self.SE(channel_weight).view([batch, channel, 1, 1])
        y = channel_weight*y
        y = self.Projection(y)
        if res:
            y += x
            return y
        else:
            return y

class Encoder(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, expansion_factor:int):
        super(Encoder, self).__init__()
        self.MBConv_1 = MobileBlock(in_channels=in_channels, out_channels=out_channels,
                                    expansion_factor=expansion_factor)
        self.MBConv_2 = MobileBlock(in_channels=out_channels, out_channels=out_channels,
                                    expansion_factor=expansion_factor)
        self.max_pool = nn.MaxPool2d(2)

    def forward(self, x, time_embedding):
        y = self.MBConv_1(x, time_embedding, False)
        y = self.MBConv_2(y, time_embedding, True)
        skip_image = y
        y = self.max_pool(y)
        return y, skip_image

class BottleNeck(nn.Module):
    def __init__(self, in_channels:int, hidden_dim:int):
        super(BottleNeck, self).__init__()
        self.attn = Multihead_Attn(in_channels=in_channels, hidden_dim=hidden_dim)
        self.SE_net = MobileBlock(in_channels=hidden_dim, out_channels=hidden_dim, expansion_factor=3)

    def forward(self, x, time_embedding):
        attn_output = self.attn(x)
        y = self.SE_net(attn_output, time_embedding, True)
        return y

class Decoder(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, expansion_factor:int):
        super(Decoder, self).__init__()
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                            kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU6(inplace=True)
        )
        self.MBConv_1 = MobileBlock(in_channels=in_channels, out_channels=out_channels,
                                    expansion_factor=expansion_factor)
        self.MBConv_2 = MobileBlock(in_channels=out_channels, out_channels=out_channels,
                                    expansion_factor=expansion_factor)

    def forward(self,x, skip_image, time_embedding):
        y = self.upsample(x)
        y = torch.cat([y, skip_image], dim=1)
        y = self.MBConv_1(y, time_embedding, False)
        y = self.MBConv_2(y, time_embedding, True)
        return y

"""
unet-predict noise
"""
class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        self.Encoder_layer1 = Encoder(3,32,16)
        self.Encoder_layer2 = Encoder(32,64,16)
        self.Encoder_layer3 = Encoder(64,128,16)
        self.BottleNeck = BottleNeck(in_channels=128, hidden_dim=256)
        self.Decoder_layer3 = Decoder(in_channels=256, out_channels=128, expansion_factor=16)
        self.Decoder_layer2 = Decoder(in_channels=128, out_channels=64, expansion_factor=16)
        self.Decoder_layer1 = Decoder(in_channels=64, out_channels=32, expansion_factor=16)
        self.output_layer = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=3, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(3),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1,  padding=1)
        )



    def forward(self, x, time_embedding):
        y, skip_image1 = self.Encoder_layer1(x, time_embedding)
        y, skip_image2 = self.Encoder_layer2(y, time_embedding)
        y, skip_image3 = self.Encoder_layer3(y, time_embedding)
        y = self.BottleNeck(y, time_embedding)
        y = self.Decoder_layer3(y, skip_image3, time_embedding)
        y = self.Decoder_layer2(y, skip_image2, time_embedding)
        y = self.Decoder_layer1(y, skip_image1, time_embedding)
        y = self.output_layer(y)
        return y

class Diffusion:
    def adding_noise(self, x_0, random_time_steps, device):
        random_noise = torch.randn(x_0.shape)
        batch_sqrt_cumprod_alphas = (coefficient_dict["sqrt_cumprod_alphas"][random_time_steps].
                                     view([x_0.shape[0],1,1,1])).to(device)
        batch_sqrt_one_minus_cumprod_alphas = (coefficient_dict["sqrt_one_minus_cumprod_alphas"][random_time_steps].
                                               view([x_0.shape[0],1,1,1])).to(device)
        return x_0*batch_sqrt_cumprod_alphas+random_noise*batch_sqrt_one_minus_cumprod_alphas, random_noise



if __name__ == "__main__":
    """
    config
    """
    batch = 64
    height = 32
    width = 32
    channels = 3
    T = 100
    num_classes = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    simulate_image = torch.randint(-3,3, size=[batch,channels,height,width], dtype=torch.float32).to(device)
    random_time_steps = torch.randint(0, T, [batch,], dtype=torch.long)

    coefficient_init(T)     # betas, alphas参数
    time_embedding = embedding_utils("time", 512, device, T=T)      # 2 dim tensor
    label_embedding = embedding_utils("label", 512, device, num_classes=num_classes)    # 2 dim tensor


    DIFF = Diffusion()
    image, real_noise = DIFF.adding_noise(simulate_image, random_time_steps, device)

    Model = Unet().to(device)
    print(Model)
    output = Model(image, time_embedding[random_time_steps])
    print(output.shape)
    print(output)











