
import torch
from torch import nn

class crowdFormerBlock(nn.Module):
    def __init__(self,in_c,out_c,num_heads=8,dropout=0.3):
        super(trans_FPN, self).__init__()
        #self.conv = nn.Conv2d(out_c, in_c, kernel_size=3, padding=1)
        self.in_dim = 9*in_c
        self.num_heads=num_heads
        self.dropout= dropout
        #self.weights = torch.rand(128,128,3,3)
        self.weights = nn.Parameter(torch.Tensor(out_c,in_c,3,3))
        #self.norm = nn.LayerNorm(self.in_dim)
        self.attn = Attention(self.in_dim, heads = self.num_heads, dropout = dropout)   # This layer is the Transformer layer e.g. swin transformer block or others
        #self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.SiLU()
    def forward(self,fea):
      
        _n,_c,_h,_w = fea.shape
        dim_h = _h//2
        dim_w = _w//2
        x = torch.nn.functional.unfold(fea,(3,3),padding=1,stride=2).transpose(1,2)
        #x = self.norm(x)
        x = self.attn(x) + x
        x = x.matmul(self.weights.view(self.weights.size(0), -1).t()).transpose(1, 2)
        x = torch.nn.functional.fold(x, (dim_h, dim_w), (1, 1))
        #x = self.bn(x)
        x = self.act(x)

        return x
