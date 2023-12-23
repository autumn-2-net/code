import torch
import torch.nn as nn
import torch.nn.functional as F


class CodeBook(nn.Module):
    def __init__(self, code_num, code_dim, sc=0.001):
        super().__init__()
        code_weight = torch.rand(code_num, code_dim) * sc
        self.codebook = nn.Parameter(code_weight, requires_grad=True)

    def forward(self, x):
        """
        x B T logit
        return B T dim

        """
        return x @ torch.unsqueeze(self.codebook, 0)

def index_to_one_hot(index, num_classes):

    # 创建单位矩阵，并选择相应的行
    one_hot = torch.eye(num_classes,device=index.device,dtype=index.dtype)[index]

    return one_hot


class VQCodebook(nn.Module):
    def __init__(self, code_num, code_dim, sc=0.001, beta=1):
        super().__init__()


        self.beta = beta
        self.code_num=code_num
        code_weight = torch.randn(code_num, code_dim) * sc
        self.codebook = nn.Parameter(code_weight, requires_grad=True)


    def forward(self, z):
        # z = z.permute(0, 2, 3, 1).contiguous()
        # z_flattened = z.view(-1, self.latent_dim)
        # z_flattened=torch.transpose(z,1,2)
        z_flattened = z
        d = torch.sum(z_flattened**2, dim=-1, keepdim=True) + torch.sum(self.codebook**2, dim=1) - 2*(torch.matmul(z_flattened, self.codebook.t()))

        min_encoding_indices = torch.argmin(d, dim=-1)
        # z_q = self.embedding(min_encoding_indices).view(z.shape)
        z_q=index_to_one_hot(min_encoding_indices,self.code_num).to(z) @ torch.unsqueeze(self.codebook, 0)
        loss = torch.mean((z_q.detach() - z)**2) + self.beta * torch.mean((z_q - z.detach())**2)
        z_q = z + (z_q - z).detach()
        # z_q = z_q.permute(0, 3, 1, 2)

        return z_q, min_encoding_indices, loss




if __name__ == '__main__':
    codebooks = VQCodebook(2, 5)
    code = codebooks(torch.tensor(
        [[[1, 0, 0, 0, 0], [1, 0, 0, 0, 1], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0],
          [0, 0, 0, 0, 1]]]).float())
    pass
