import torch
from torch import nn as nn
import torch.nn.functional as F

from einops import rearrange




class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x








class GDAU(nn.Module):
    def __init__(self, dim,num_heads):
        super(GDAU, self).__init__()

        self.norm1 = LayerNorm(dim)

        self.project_in = nn.Conv2d(dim, dim, kernel_size=1)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim, kernel_size=1)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        self.k_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        self.v_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)


        self.project_out = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        residual=x
        x=self.norm1(x)
        b,c,h,w = x.shape

        qkv = self.qkv(x)
        q= self.q_dwconv(qkv)
        k= self.k_dwconv(qkv)
        v= self.v_dwconv(qkv)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        x1 = self.project_in(x)
        x_ffn = self.dwconv(x1)
        x2 = out * F.gelu(x_ffn)
        end = self.project_out(x2)
        return end+residual





class GCAT(nn.Module):
    def __init__(self,
                 num_in_ch,
                 num_out_ch,
                 num_feat=64,
                 num_heads=4,
                 num_block=10,
                 upscale=4,):
        super(GCAT, self).__init__()


        self.scale = upscale
        self.num_block = num_block

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)



        self.layers = nn.ModuleList()

        for i_layer in range(self.num_block):
            layer = GDAU(num_feat,num_heads)
            self.layers.append(layer)





        self.conv_after_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1,groups=2)

        if self.scale == 4:
            self.upsapling = nn.Sequential(
                nn.Conv2d(num_feat, num_feat*4, 1, 1, 0),
                nn.PixelShuffle(2),
                nn.Conv2d(num_feat, num_feat*4, 1, 1, 0),
                nn.PixelShuffle(2)
            )
        else:
            self.upsapling = nn.Sequential(
                nn.Conv2d(num_feat, num_feat*self.scale*self.scale, 1, 1, 0),
                nn.PixelShuffle(self.scale)
            )

        # self.upsample = Upsample(upscale, num_feat)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.act = nn.LeakyReLU(0.2, inplace=True)


    def forward_features(self, x):
        for layer in self.layers:
            x = layer(x)

        return x
    def forward(self, x):
        x0=x

        x = self.conv_first(x)

        x1=self.forward_features(x)

        res = self.conv_after_body(x1)
        res += x

        x = self.conv_last(self.act(self.upsapling(res)))
        x_i = F.interpolate(x0, scale_factor=self.scale, mode='bilinear', align_corners=False)


        return x+x_i

