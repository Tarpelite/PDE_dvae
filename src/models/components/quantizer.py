import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VectorQuantizer(nn.Module):
    """
    Discretization bottleneck part of the VQ-VAE.

    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    """

    def __init__(self, n_e, e_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete 
        one-hot vector that is the index of the closest embedding vector e_j

        z (continuous) -> z_q (discrete)

        z.shape = (batch, channel, height, width)

        quantization pipeline:

            1. get encoder input (B, C, grid_size)
            2. flatten input to (B*grid_size,C)

        """
        # reshape z -> (batch, height, width, channel) and flatten
        # z = z.permute(0, 2, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e).to(device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        # compute loss for embedding
        loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
            torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        z_q = z_q.contiguous()

        return loss, z_q, perplexity, min_encodings, min_encoding_indices
    
if __name__ == "__main__":
    # 测试参数设置
    n_e = 128  # number of embeddings
    e_dim = 1024 # dimension of embedding
    beta = 0.25 

    # 创建VectorQuantizer实例
    vq_module = VectorQuantizer(n_e, e_dim, beta).to(device)

    # 创建一个示例输入张量，大小为(10, e_dim, 1024)，即batch_size=10
    z = torch.randn((10, e_dim, 1024), device=device, requires_grad=True)

    # 前向传播计算
    loss, z_q, perplexity, min_encodings, min_encoding_indices = vq_module(z)

    # 打印输出结果进行检查
    print("Loss:", loss.item())
    print("Perplexity:", perplexity.item())

    # 检查z_q的形状是否与输入相同
    assert z.shape == z_q.shape, "Shape of the quantized vector does not match input shape."
    print("Quantized vector shape matches input shape.")

    # 检查min_encoding_indices的形状，应当为(10 * grid_size, 1)，因为是每个像素的索引
    assert min_encoding_indices.shape == (z.shape[0] * z.shape[2], 1), "Shape of the encoding indices does not match expected shape."
    print("Encoding indices shape matches expected shape.")

    # 检查min_encodings形状，应当为(10 * grid_size, n_e)
    assert min_encodings.shape == (z.shape[0] * z.shape[2], n_e), "Shape of the encoding vectors does not match expected shape."
    print("Encoding vectors shape matches expected shape.")
