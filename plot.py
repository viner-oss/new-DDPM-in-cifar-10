import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from PIL import Image
from matplotlib import animation

# 基本参数
mean = torch.tensor([0.4914, 0.4822, 0.4465], dtype=torch.float32)
std = torch.tensor([0.2023, 0.1994, 0.2010], dtype=torch.float32)

def Gaussian(x, mean, std):
    return (1/std*np.sqrt(2*np.pi))*np.exp(-(x-mean)**2/2*(std**2))

"""
每批次都选择第一张图来判断效果
显示预测噪声 与 真实噪声的概率密度曲线
训练阶段可以尝试使用
"""
def graph_show(real_noise, predict_noise, train_step):
    predict_np_noise = predict_noise.cpu().detach().numpy()[0,:,:,:]
    real_np_noise = real_noise.cpu().numpy()[0,:,:,:]

    plt.figure(num=1, label='real', figsize=(6,6))
    real_mean = np.mean(real_np_noise[:,:,:])
    predict_mean = np.mean(predict_np_noise[:,:,:])
    real_std = np.std(real_np_noise[:,:,:])
    predict_std = np.std(predict_np_noise[:,:,:])

    x_ch1 = np.linspace(-5, 5, 1000)
    y_real_ch1 = Gaussian(x_ch1, real_mean, real_std)
    y_predict_ch1 = Gaussian(x_ch1, predict_mean, predict_std)

    # 单独显示real
    plt.subplot(2,1,1)
    plt.title('real')
    plt.plot(x_ch1, y_real_ch1)
    ax1 = plt.gca()
    ax1.spines['top'].set_color('none')
    ax1.spines['right'].set_color('none')
    ax1.xaxis.set_ticks_position('bottom')
    ax1.yaxis.set_ticks_position('left')
    ax1.spines['bottom'].set_position(('data', 0))
    ax1.spines['left'].set_position(('data', -5))
    plt.xticks(np.arange(-5, 5, 1))           # 修改x轴的精度
    plt.xlim(-5, 5)
    plt.yticks(np.arange(0, 5, 1))
    plt.ylim(0, 4)

    plt.scatter(real_mean, Gaussian(real_mean, mean=real_mean, std=real_std),
                s=50, c='red')
    plt.plot([real_mean, real_mean],
             [Gaussian(real_mean, mean=real_mean, std=real_std), 0],
             linestyle='--', color='k')

    plt.annotate(text='(%.4f,%.4f)' % (real_mean, Gaussian(real_mean, mean=real_mean, std=real_std)),
                 xy=(real_mean, Gaussian(real_mean, mean=real_mean, std=real_std)),
                 xytext=(+30, 0), xycoords='data', textcoords='offset points')
    plt.savefig(fr'D:\python\DDPM_for_Classify\Denoising_distribution\real\{train_step}')
    plt.close(1)


    # 单独显示predict
    plt.figure(num=2, label='predict', figsize=(6,6))
    plt.subplot(2,1,2)
    plt.title('predict_noise')
    plt.plot(x_ch1, y_predict_ch1)
    ax2 = plt.gca()
    ax2.spines['top'].set_color('none')
    ax2.spines['right'].set_color('none')
    ax2.xaxis.set_ticks_position('bottom')
    ax2.yaxis.set_ticks_position('left')
    ax2.spines['left'].set_position(('data', -5))
    ax2.spines['bottom'].set_position(('data', 0))
    plt.xticks(np.arange(-5, 5, 1))
    plt.xlim(-5 ,5)
    plt.yticks(np.arange(0, 5, 1))
    plt.ylim(0, 4)

    plt.scatter(predict_mean, Gaussian(predict_mean, mean=predict_mean, std=predict_std),
                s=50, c='red')
    plt.plot([predict_mean, predict_mean],
                [Gaussian(predict_mean, mean=predict_mean, std=predict_std), 0],
             linestyle='--', color='k')

    plt.annotate(text='(%.4f,%.4f)' %(predict_mean, Gaussian(predict_mean, mean=predict_mean, std=predict_std)),
                xy=(predict_mean, Gaussian(predict_mean, mean=predict_mean, std=predict_std)),
                 xytext=(+30,0), xycoords='data', textcoords='offset points')
    plt.savefig(fr'D:\python\DDPM_for_Classify\Denoising_distribution\predict\{train_step}')
    plt.close(2)


    # 同时显示对比
    plt.figure(num=3, label='real && predict', figsize=(6,6))
    plt.subplot(1,1,1)
    plt.title('real && predict')
    plt.plot(x_ch1, y_real_ch1, label='real', color='blue')
    plt.plot(x_ch1, y_predict_ch1, label='predict', color='red')
    ax3 = plt.gca()
    ax3.spines['right'].set_color('none')
    ax3.spines['top'].set_color('none')
    ax3.xaxis.set_ticks_position('bottom')
    ax3.yaxis.set_ticks_position('left')
    ax3.spines['left'].set_position(('data', -5))
    ax3.spines['bottom'].set_position(('data', 0))
    plt.xticks(np.arange(-5, 5, 0.5))
    plt.xlim(-5, 5)
    plt.yticks(np.arange(0, 5, 1))
    plt.ylim(0, 4)

    plt.scatter(real_mean, Gaussian(real_mean, mean=real_mean, std=real_std),
                s=50, color='red')
    plt.plot([real_mean, real_mean],
             [Gaussian(real_mean, mean=real_mean, std=real_std), 0],
             linestyle='--', color='k', linewidth=1)
    plt.annotate(text='(%.4f,%.4f)' % (real_mean, Gaussian(real_mean, mean=real_mean, std=real_std)),
                 xy=(real_mean, Gaussian(real_mean, mean=real_mean, std=real_std)),
                 xytext=(-150, 0), xycoords='data', textcoords='offset points',
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3, rad=.2'))

    plt.scatter(predict_mean, Gaussian(predict_mean, mean=predict_mean, std=predict_std),
                s=50, c='blue')
    plt.plot([predict_mean, predict_mean],
                [Gaussian(predict_mean, mean=predict_mean, std=predict_std), 0],
             linestyle='--', color='k', linewidth=1)

    plt.annotate(text='(%.4f,%.4f)' %(predict_mean, Gaussian(predict_mean, mean=predict_mean, std=predict_std)),
                xy=(predict_mean, Gaussian(predict_mean, mean=predict_mean, std=predict_std)),
                 xytext=(+60,0), xycoords='data', textcoords='offset points',
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3, rad=.2'))
    plt.legend()
    plt.savefig(fr'D:\python\DDPM_for_Classify\Denoising_distribution\dual\{train_step}')
    plt.close(3)

"""
显示还原过程的图像历程
预测过程可以尝试使用
"""
def imshow_image(x, loc, epcho):
    image = x[0,:,:,:]
    # 反归一化操作
    image = image * std.unsqueeze(1).unsqueeze(2) + mean.unsqueeze(1).unsqueeze(2)
    # 将像素值限制在0 - 1之间
    image = torch.clamp(image, 0, 1)
    # 将像素值从0 - 1转换为0 - 255
    image = (image * 255).byte()
    image_numpy = image.cpu().numpy()

    plt.subplot(4,4,loc)
    image = np.swapaxes(np.swapaxes(image_numpy[:,:,:], axis1=0, axis2=1), axis1=1, axis2=2)
    plt.imshow(image, cmap='bone', interpolation='nearest', origin='upper')
    os.makedirs(fr'D:\python\Segmentation\generative_image\epcho{epcho}', exist_ok=True)
    plt.imsave(fr'D:\python\Segmentation\generative_image\epcho{epcho}\0{loc}.png', arr=image)

def image_animation(path, size):
    channels = size[2]
    width = size[1]
    height = size[0]
    image_folder = path
    # sort是python的内置函数 用于整理文件路径并保存为list形式 找到以.png .jpg .jpeg结尾的文件并将路径打包组合
    # [] 是一个列表推导式
    # image_files是一个列表类型
    image_files = sorted(
        [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])

    figure, ax = plt.subplots()

    """
    返回值是一个matplotlib.image.AxesImage对象。
    该对象代表了显示在图像坐标轴上的图像，它包含了图像的相关属性和方法，可用来对显示的图像进行进一步操作和修改
    """
    image = ax.imshow(np.zeros((height, width, channels), dtype=np.uint8))

    """
    在使用matplotlib库的FuncAnimation创建动画时，animate函数（即用于更新每一帧动画的函数）的返回值有时会带上逗号，
    这通常是因为FuncAnimation要求返回一个可迭代对象（通常是一个元组），其中包含需要更新的绘图对象
    带上逗号后 返回值会从单个值变成一个元组
    """

    # 初始化图像信息
    def init():
        image.set_data(np.zeros((height, width, channels), dtype=np.uint8))
        return image,

    # 实时更新图像
    def animate(frame):
        image_path = image_files[frame]
        img = Image.open(image_path)
        image.set_data(np.array(img))
        return image,

    ani = animation.FuncAnimation(fig=figure, func=animate, frames=len(image_files), init_func=init, blit=True,
                                  interval=200)

    plt.show()

if __name__ == "__main__":
    # graph_show(real_np_noise=real, predict_np_noise=predict_noise)

    plt.figure(label='prediction')
    simulate_image = torch.randn([1,3,32,32], dtype=torch.float32)
    for i in range(16):
        imshow_image(simulate_image, i+1, 1)

