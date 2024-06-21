import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class FNO2dEncoder(nn.Module):
    def __init__(self, modes1=12, modes2=12, width=20, emb_dim=1024):
        
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.emb_dim = emb_dim

        self.in_channels = width
        self.out_channels = width

        self.scale = (1/(self.in_channels * self.out_channels))

        self.padding = 2
        self.fc0 = nn.Linear(1, self.width)
        self.weights1 = nn.Parameter(self.scale * torch.rand(self.in_channels, self.out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(self.in_channels, self.out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

        self.head = nn.Linear(self.width*modes1*modes2*2*2, self.emb_dim)

    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x, y), (in_channel, out_channel, x,y) -> (batch, out_channel, x, y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        # x dim = [b, x1, x2, 1]
        # grid dim = [b, x1, x2, 2]
        bsz = x.shape[0]
        # x = torch.cat(x, dim=-1) #[b, x1, x2, 1]
        x = self.fc0(x) #[..., width]
        x = x.permute(0, 3, 1, 2)

        x = F.pad(x, [0, self.padding, 0, self.padding])

        x_ft = torch.fft.rfft2(x)

        mode1 = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        mode2 =  self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        out = torch.cat([mode1.view(bsz, self.width, -1), mode2.view(bsz, self.width, -1)], dim=-1)
        out = torch.cat([out.real, out.imag])
        out = self.head(out.view(bsz, -1)) #[bsz, emb_dim]
        return out


class FNO2dDecoder(nn.Module):
     def __init__(self, modes1=12, modes2=12, width=20, emb_dim=1024):
        
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.emb_dim = emb_dim

        self.in_channels = width
        self.out_channels = width

        self.scale = (1/(self.in_channels * self.out_channels))

        self.padding = 2
        self.decode_head = nn.Linear(self.emb_dim, width*modes1*modes2*2*2)
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

     def forward(self, token, x_size):
        # x dim = [bsz, emb_dim]
        # out dim = [bsz, x_size_0, x_size_1]

        bsz = token.shape[0]
        latent = self.decode_head(token)

        out_ft_modes = latent.reshape(bsz, -1, 2)
        out_ft_modes = torch.complex(out_ft_modes[:, :, 0], out_ft_modes[:,:,1])

        # reshape latent to have a format of complex numbers (real and imaginary parts)
        out_ft_modes = latent.view(bsz, self.width, self.modes1, self.modes2, 2, 2)
        out_ft_modes_complex = torch.complex(out_ft_modes[..., 0], out_ft_modes[...,1])

        full_size = (x_size[0], x_size[1])
        out_ft = torch.zeros((bsz, self.width) + full_size, dtype=torch.cfloat, device=token.device)

        out_ft[:, :, :self.modes1, :self.modes2] = out_ft_modes_complex[:, :, :, :, 0]
        out_ft[:, :, -self.modes1:, :self.modes2] = out_ft_modes_complex[:, :, :, :, 1]

        x = torch.fft.irfft2(out_ft, s=full_size)

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x.squeeze(-1)

if __name__ == "__main__":
        # Do the Test
    test_x = torch.randn(2, 128, 128, 1)
    test_grid = torch.randn(2, 128, 128, 2)
    
    x_sizes = (128,128)

    encoder = FNO2dEncoder(emb_dim=1024)
    decoder = FNO2dDecoder(emb_dim=1024)

    print("encoder_total_params:{} Mb".format(sum(p.numel() for p in encoder.parameters()) /float(10e6)))
    print("decoder_total_params:{} Mb".format(sum(p.numel() for p in decoder.parameters()) /float(10e6)))

    token = encoder(test_x, test_grid)
    print(token.shape)

    out = decoder(token, x_sizes)
    print(out.shape)