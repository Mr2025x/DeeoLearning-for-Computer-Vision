from __future__ import print_function

import torch
import torch.utils.data
from torch import nn, optim

NOISE_DIM = 96


def hello_gan():
    print("Hello from gan.py!")


def sample_noise(batch_size, noise_dim, dtype=torch.float, device="cpu"):
    """
    Generate a PyTorch Tensor of uniform random noise.

    Input:
    - batch_size: Integer giving the batch size of noise to generate.
    - noise_dim: Integer giving the dimension of noise to generate.

    Output:
    - A PyTorch Tensor of shape (batch_size, noise_dim) containing uniform
      random noise in the range (-1, 1).
    """
    noise = None
    ##############################################################################
    # TODO: Implement sample_noise.                                              #
    ##############################################################################
    # Replace "pass" statement with your code
    noise = torch.rand((batch_size, noise_dim), dtype=dtype, device=device)
    noise = noise * 2.0 - 1.0

    ##############################################################################
    #                              END OF YOUR CODE                              #
    ##############################################################################

    return noise


def discriminator():
    """
    Build and return a PyTorch nn.Sequential model implementing the architecture
    in the notebook.
    """
    model = None
    ############################################################################
    # TODO: Implement discriminator.                                           #
    ############################################################################
    # Replace "pass" statement with your code
    model = nn.Sequential(
        nn.Flatten(), # 将二维图片展平为一维向量
        nn.Linear(784, 256),
        nn.LeakyReLU(0.01, inplace=True), # 判别器常用 LeakyReLU
        nn.Linear(256, 256),
        nn.LeakyReLU(0.01, inplace=True),
        nn.Linear(256, 1) # 输出一个分数
    )
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    return model


def generator(noise_dim=NOISE_DIM):
    """
    Build and return a PyTorch nn.Sequential model implementing the architecture
    in the notebook.
    """
    model = None
    ############################################################################
    # TODO: Implement generator.                                               #
    ############################################################################
    # Replace "pass" statement with your code
    model = nn.Sequential(
        nn.Linear(noise_dim, 1024),
        nn.ReLU(inplace=True), # 生成器隐藏层常用 ReLU
        nn.Linear(1024, 1024),
        nn.ReLU(inplace=True),
        nn.Linear(1024, 784), # 输出图片的像素总数
        nn.Tanh() # 将输出压缩到 -1 到 1 之间
    )
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return model


def discriminator_loss(logits_real, logits_fake):
    """
    Computes the discriminator loss described above.

    Inputs:
    - logits_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Returns:
    - loss: PyTorch Tensor containing (scalar) the loss for the discriminator.
    """
    loss = None
    ##############################################################################
    # TODO: Implement discriminator_loss.                                        #
    ##############################################################################
    # Replace "pass" statement with your code
    true_labels = torch.ones_like(logits_real)
    fake_labels = torch.zeros_like(logits_fake)
    
    # 分别计算两种情况的 loss 并相加
    loss_real = nn.functional.binary_cross_entropy_with_logits(logits_real, true_labels)
    loss_fake = nn.functional.binary_cross_entropy_with_logits(logits_fake, fake_labels)
    
    loss = loss_real + loss_fake
    ##############################################################################
    #                              END OF YOUR CODE                              #
    ##############################################################################
    return loss


def generator_loss(logits_fake):
    """
    Computes the generator loss described above.

    Inputs:
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Returns:
    - loss: PyTorch Tensor containing the (scalar) loss for the generator.
    """
    loss = None
    ##############################################################################
    # TODO: Implement generator_loss.                                            #
    ##############################################################################
    # Replace "pass" statement with your code
    true_labels = torch.ones_like(logits_fake)
    
    # 计算被当做真图时的 loss
    loss = nn.functional.binary_cross_entropy_with_logits(logits_fake, true_labels)
    ##############################################################################
    #                              END OF YOUR CODE                              #
    ##############################################################################
    return loss


def get_optimizer(model):
    """
    Construct and return an Adam optimizer for the model with learning rate 1e-3,
    beta1=0.5, and beta2=0.999.

    Input:
    - model: A PyTorch model that we want to optimize.

    Returns:
    - An Adam optimizer for the model with the desired hyperparameters.
    """
    optimizer = None
    ##############################################################################
    # TODO: Implement optimizer.                                                 #
    ##############################################################################
    # Replace "pass" statement with your code
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.5, 0.999))
    ##############################################################################
    #                              END OF YOUR CODE                              #
    ##############################################################################
    return optimizer


def ls_discriminator_loss(scores_real, scores_fake):
    """
    Compute the Least-Squares GAN loss for the discriminator.

    Inputs:
    - scores_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """
    loss = None
    ##############################################################################
    # TODO: Implement ls_discriminator_loss.                                     #
    ##############################################################################
    # Replace "pass" statement with your code
    loss_real = 0.5 * torch.mean((scores_real - 1.0) ** 2)
    loss_fake = 0.5 * torch.mean((scores_fake - 0.0) ** 2)
    loss = loss_real + loss_fake
    ##############################################################################
    #                              END OF YOUR CODE                              #
    ##############################################################################
    return loss


def ls_generator_loss(scores_fake):
    """
    Computes the Least-Squares GAN loss for the generator.

    Inputs:
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """
    loss = None
    ##############################################################################
    # TODO: Implement ls_generator_loss.                                         #
    ##############################################################################
    # Replace "pass" statement with your code
    loss = 0.5 * torch.mean((scores_fake - 1.0) ** 2)
    ##############################################################################
    #                              END OF YOUR CODE                              #
    ##############################################################################
    return loss


def build_dc_classifier():
    """
    Build and return a PyTorch nn.Sequential model for the DCGAN discriminator
    implementing the architecture in the notebook.
    """
    model = None
    ############################################################################
    # TODO: Implement build_dc_classifier.                                     #
    ############################################################################
    # Replace "pass" statement with your code
    model = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
        nn.LeakyReLU(0.01, inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
        nn.LeakyReLU(0.01, inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(7 * 7 * 64, 1024),
        nn.LeakyReLU(0.01, inplace=True),
        nn.Linear(1024, 1)
    )
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return model


def build_dc_generator(noise_dim=NOISE_DIM):
    """
    Build and return a PyTorch nn.Sequential model implementing the DCGAN
    generator using the architecture described in the notebook.
    """
    model = None
    ############################################################################
    # TODO: Implement build_dc_generator.                                      #
    ############################################################################
    # Replace "pass" statement with your code
    model = nn.Sequential(
        # 1. 初始放大：将低维噪声映射到高维空间
        nn.Linear(noise_dim, 1024),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(1024),
        
        # 2. 映射到空间特征图的基础维度 (128个 7x7 的特征图)
        nn.Linear(1024, 7 * 7 * 128),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(7 * 7 * 128),
        
        # 3. 变形：将一维向量重新排布成三维的图像张量格式 (通道数, 高, 宽)
        nn.Unflatten(dim=1, unflattened_size=(128, 7, 7)),
        
        # 4. 第一次上采样：7x7 -> 14x14
        nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(64),
        
        # 5. 第二次上采样：14x14 -> 28x28，并压缩回 1 个通道 (灰度图)
        nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
        nn.Tanh() # 必须使用 Tanh，将输出像素值约束到 [-1, 1]
    )
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return model
