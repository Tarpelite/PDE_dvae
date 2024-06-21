import torch
import torch.nn as nn
import numpy as np
from src.models.components.fno import FNO2dDecoder, FNO2dEncoder
from src.models.components.quantizer import VectorQuantizer


class VQVAE(nn.Module):
    def __init__(self,
                 modes1,
                 modes2,
                 width, 
                 n_embeddings, embedding_dim, beta, save_img_embedding_map=False):
        super(VQVAE, self).__init__()
        # encode image into continuous latent space
        self.encoder = FNO2dEncoder(
            modes1=modes1,
            modes2=modes2,
            width=width,
            emb_dim=embedding_dim
        )

        self.decoder = FNO2dDecoder(
            modes1=modes1,
            modes2=modes2,
            emb_dim=embedding_dim

        )
        # self.pre_quantization_conv = nn.Conv2d(
        #     h_dim, embedding_dim, kernel_size=1, stride=1)
        # pass continuous latent vector through discretization bottleneck
        self.vector_quantization = VectorQuantizer(
            n_embeddings, embedding_dim, beta)
        # decode the discrete latent representation
        # self.decoder = Decoder(embedding_dim, h_dim, n_res_layers, res_h_dim)

        if save_img_embedding_map:
            self.img_to_embedding_map = {i: [] for i in range(n_embeddings)}
        else:
            self.img_to_embedding_map = None

    def forward(self, x, verbose=False):

        z_e = self.encoder(x)
        x_size = (x.shape[1], x.shape[2])
        # z_e = self.pre_quantization_conv(z_e)
        # print(z_e.shape)
        embedding_loss, z_q, perplexity, _, _ = self.vector_quantization(
            z_e)
        x_hat = self.decoder(z_q, x_size)

        if verbose:
            print('original data shape:', x.shape)
            print('encoded data shape:', z_e.shape)
            print('recon data shape:', x_hat.shape)
            assert False

        return embedding_loss, x_hat, perplexity

if __name__ == "__main__":
    # 创建模型实例
    model = VQVAE(
        modes1=16,
        modes2=16,
        width=32,
        n_embeddings=512, 
        embedding_dim=64,
        beta=0.25,
    )

    # 随机生成一个测试输入，形状为 [batch_size, height, width]
    input_data = torch.randn(32, 128, 128, 1) # 注意：默认数据是3维的。如果模型需要4维输入 (batch, channel, height, width)，请调整这里

    # 如果你的模型期望一个四维张量（Batch x Channel x Height x Width），那么你可能需要这样来准备输入：
    # input_data = input_data.unsqueeze(1)  # 添加通道维度，假设这是灰度图像的单通道。

    # 将数据传递给VQ-VAE
    device = torch.device("cuda")
    model.to(device)
    embedding_loss, reconstructed, perplexity = model(input_data.to(device))

    print('Embedding Loss:', embedding_loss.item())
    print('Reconstructed data shape:', reconstructed.shape)
    print('Perplexity:', perplexity.item())

    # 进一步验证是否形状正确（假设输出和输入有相同的维度）
    assert reconstructed.shape == input_data.shape