import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import AvgPool1d, Conv1d, Conv2d, ConvTranspose1d
from torch.nn.modules.module import T
from torch.nn.utils import remove_weight_norm, spectral_norm, weight_norm

from moduls.codebook import CodeBook, VQCodebook

LRELU_SLOPE = 0.1


class Encoder(nn.Module):
    def __init__(self, h,
                 ):
        super().__init__()

        self.h = h
        # h["inter_channels"]
        self.num_kernels = len(h["Eresblock_kernel_sizes"])
        # self.out_channels = h["num_mels"]
        self.num_downsamples = len(h["upsample_rates"])
        self.conv_pre = weight_norm(
            Conv1d(1, h["Eupsample_initial_channel"] // (2 ** len(h["upsample_rates"])), 7, 1, padding=3))
        resblock = ResBlock1 if h["resblock"] == '1' else ResBlock2
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(reversed(h["upsample_rates"]), reversed(h["upsample_kernel_sizes"]))):
            self.ups.append(weight_norm(
                Conv1d(h["Eupsample_initial_channel"] // (2 ** (len(h["upsample_rates"]) - i)),
                       h["Eupsample_initial_channel"] // (2 ** (len(h["upsample_rates"]) - i - 1)),
                       k, u, padding=(k - u + 1) // 2)))
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups), 0, -1):
            ch = h["Eupsample_initial_channel"] // (2 ** (i - 1))
            for j, (k, d) in enumerate(zip(h["Eresblock_kernel_sizes"], h["Eresblock_dilation_sizes"])):
                self.resblocks.append(resblock(h, ch, k, d))

        self.conv_post = weight_norm(Conv1d(ch, h["codenum"], 7, 1, padding=3))
        # self.conv_postres = weight_norm(Conv1d(ch, h["codedim"], 7, 1, padding=3))
        # self.conv_post = weight_norm(Conv1d(ch, h["codedim"], 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
        self.upp = np.prod(h["upsample_rates"])

    def forward(self, x):
        # x = x[:, None, :]
        x = self.conv_pre(x)
        for i in range(self.num_downsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = 0
            ssx=[]
            for j in range(self.num_kernels):
                # ssx.append(self.resblocks[i * self.num_kernels + j])
                # if xs is None:
                #     xs = self.resblocks[i * self.num_kernels + j](x)
                # else:
                xs =xs+ self.resblocks[i * self.num_kernels + j](x)

            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        # res=self.conv_postres(x)
        x = self.conv_post(x)
        # m, logs = torch.split(x, self.out_channels, dim=1)
        # z = m + torch.randn_like(m) * torch.exp(logs)
        return x


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


class ResBlock1(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.h = h
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlock2(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.h = h
        self.convs = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1])))
        ])
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)






class PixelShuffle1D(torch.nn.Module):
    """
    1D pixel shuffler. https://arxiv.org/pdf/1609.05158.pdf
    Upscales sample length, downscales channel length
    "short" is input, "long" is output
    """
    def __init__(self, upscale_factor):
        super(PixelShuffle1D, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        batch_size = x.shape[0]
        short_channel_len = x.shape[1]
        short_width = x.shape[2]

        long_channel_len = short_channel_len // self.upscale_factor
        long_width = self.upscale_factor * short_width

        x = x.contiguous().view([batch_size, self.upscale_factor, long_channel_len, short_width])
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(batch_size, long_channel_len, long_width)

        return x

class spcup(nn.Module):
    def __init__(self,indim,uprate,out_channels):
        super().__init__()
        # dct=indim//out_channels
        ssxdim=out_channels*uprate
        # if ssxdim%uprate !=0:
        #     ssxdim=ssxdim+ (uprate-ssxdim%uprate)
        self.spcc=weight_norm(nn.Conv1d(indim,ssxdim,kernel_size=5,padding=2))
        self.spc=PixelShuffle1D(upscale_factor=uprate)
        self.cpco=weight_norm(nn.Conv1d(ssxdim,out_channels,kernel_size=3,padding=1)) if out_channels!=(ssxdim//uprate) else nn.Identity()

    def forward(self, x):
        x=self.spcc(x)
        return self.cpco(self.spc(x))
class Generators(torch.nn.Module):
    def __init__(self, h):
        super().__init__()
        self.h = h

        self.num_kernels = len(h["resblock_kernel_sizes"])
        self.num_upsamples = len(h["upsample_rates"])
        self.conv_pre = weight_norm(Conv1d(h['codedim'], h["upsample_initial_channel"], 7, 1, padding=3))
        resblock = ResBlock1 if h["resblock"] == '1' else ResBlock2
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h["upsample_rates"], h["upsample_kernel_sizes"])):
            # self.ups.append(weight_norm(
            #     ConvTranspose1d(h["upsample_initial_channel"] // (2 ** i),
            #                     h["upsample_initial_channel"] // (2 ** (i + 1)),
            #                     k, u, padding=(k - u + 1) // 2)))
            self.ups.append(
                spcup(h["upsample_initial_channel"] // (2 ** i), u, h["upsample_initial_channel"] // (2 ** (i + 1))))
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h["upsample_initial_channel"] // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(h["resblock_kernel_sizes"], h["resblock_dilation_sizes"])):
                self.resblocks.append(resblock(h, ch, k, d))

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
        self.upp = np.prod(h["upsample_rates"])

    # def forward(self, x):
    #     x = self.conv_pre(x)
    #     for i in range(self.num_upsamples):
    #         x = F.leaky_relu(x, LRELU_SLOPE)
    #         x = self.ups[i](x)
    #         xs = None
    #         for j in range(self.num_kernels):
    #             if xs is None:
    #                 xs = self.resblocks[i * self.num_kernels + j](x)
    #             else:
    #                 xs += self.resblocks[i * self.num_kernels + j](x)
    #         x = xs / self.num_kernels
    #     x = F.leaky_relu(x)
    #     x = self.conv_post(x)
    #     x = torch.tanh(x)
    #
    #     return x
    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = 0
            for j in range(self.num_kernels):

                xs = xs+self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)
class ssc(nn.Module):
    def __init__(self,codenum,codedim):
        super().__init__()
        self.codec=nn.Conv1d(codenum,codedim,kernel_size=1)
    def forward(self, x):
        x=torch.transpose(x, 1, 2)
        x=self.codec(x)
        x = torch.transpose(x, 1, 2)
        return x


class HiFivae(torch.nn.Module):
    def __init__(self, h):
        super().__init__()
        self.Generator = Generators(h)
        # self.Generator2 = Generator(h)
        self.Encoder = Encoder(h)
        self.codebook = CodeBook(h['codenum'], h['codedim'])
        # self.codebook = VQCodebook(h['codenum'], h['codedim'])
        # self.codebook = ssc(h['codenum'], h['codedim'])
        self.TEM = 0.6
        self.resx=1

    def forward(self, x):
        # self.setTME(step)
        z = self.Encoder(x)
        p = torch.transpose(z, 1, 2)
        # p = F.softmax(p, dim=-1)
        # p=torch.exp(p)
        p = F.gumbel_softmax(logits=p, tau=self.TEM, hard=False, dim=-1)
        z = self.codebook(p)

        # p = F.leaky_relu(p, LRELU_SLOPE)
        # z,_,lossvq = self.codebook(p)
        lossvq=0
        # z=p
        # z = torch.transpose(z, 1, 2)+res*self.resx
        z = torch.transpose(z, 1, 2)
        x = self.Generator(z)
        # x1=self.Generator2(z)
        return x
        # return x
    def setTME(self,step):
        pass
        if self.TEM>0.05:
            self.TEM=np.cos((step/100000)*np.pi)-0.6
        if self.TEM<0.05:
            self.TEM=0.05
        if self.resx>0:
            self.resx = np.cos((step / 6000) * np.pi)*0.5+0.5
        if self.resx < 0:
            self.resx=0



if __name__ == '__main__':
    hh = {'codenum': 4, 'codedim': 1}

    hh.update({'upsample_rates': [8, 8, 2, 2, 2], 'upsample_kernel_sizes': [16, 16, 4, 4, 4],
               'upsample_initial_channel': 512})
    hh.update({'resblock_kernel_sizes': [3, 7, 11], 'resblock_dilation_sizes': [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
               'discriminator_periods': [3, 5, 7, 11, 17, 23, 37]})
    hh.update({'resblock': '1'})
    HV = HiFivae(hh)

    heex = HV(torch.randn(12, 1, 8192*4))
    pass
