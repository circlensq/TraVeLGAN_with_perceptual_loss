"""(Slightly modified) code from junyanz/pytorch-CycleGAN-and-pix2pix
repository for their CycleGAN and pix2pix paper implementation in PyTorch.
"""

import torch
import torch.nn as nn
import functools
from utils import spectral_normalization
from torchvision import models

from torchvision.models import vgg19
from collections import namedtuple

class Generator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, n_filters, num_downs,
                 dropout=False, sn=True, sa=False, norm_layer="bn"):
        super(Generator, self).__init__()

        norm_dict = {"bn": nn.BatchNorm2d, "none": nn.Identity}
        norm_layer = norm_dict[norm_layer]
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(n_filters * 8, n_filters * 8,
                                             input_nc=None,
                                             submodule=None,
                                             norm_layer=norm_layer,
                                             innermost=True,
                                             )
        # add intermediate layers with n_filters * 8 filters
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(n_filters * 8, n_filters * 8,
                                                 input_nc=None,
                                                 submodule=unet_block,
                                                 norm_layer=norm_layer,
                                                 use_dropout=dropout)
        # gradually reduce number of filters from n_filters * 8 to n_filters
        self.unet_block_1 = UnetSkipConnectionBlock(n_filters * 4, n_filters * 8,
                                             input_nc=None,
                                             submodule=unet_block,
                                             sa=sa,
                                             norm_layer=norm_layer)
        self.unet_block_2 = UnetSkipConnectionBlock(n_filters * 2, n_filters * 4,
                                             input_nc=None,
                                             submodule=self.unet_block_1,
                                             norm_layer=norm_layer,
                                             sa=sa)
        self.unet_block_3 = UnetSkipConnectionBlock(n_filters, n_filters * 2,
                                             input_nc=None,
                                             submodule=self.unet_block_2,
                                             norm_layer=norm_layer)
        self.unet_block_4 = UnetSkipConnectionBlock(output_nc, n_filters,
                                             input_nc=input_nc,
                                             submodule=self.unet_block_3,
                                             outermost=True,
                                             norm_layer=norm_layer)
        
        self.model = self.unet_block_4

        if sn:
            self.apply(spectral_normalization)

    def forward(self, input):
        """Standard forward"""
        out = self.model(input)
        return out





class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False,
                 norm_layer=nn.BatchNorm2d, sa=False, use_dropout=False, semantic_loss=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        self.semantic_loss = semantic_loss
        # print('self.outermost : ', self.outermost)
        # print('self.semantic_loss : ', self.semantic_loss)
        # print('')
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)
        upatt = Self_Attn(outer_nc) if sa else nn.Identity()

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
            
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm, upatt]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm, upatt]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)


class Discriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, n_filters, num_downs, sn=True,
                 sa=False, norm_layer="none"):
        super(Discriminator, self).__init__()

        norm_dict = {"bn": nn.BatchNorm2d, "none": nn.Identity}
        norm_layer = norm_dict[norm_layer]
        # no need to use bias as BatchNorm2d has affine parameters
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, n_filters, kernel_size=kw,
                              stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, num_downs):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(n_filters * nf_mult_prev, n_filters * nf_mult,
                          kernel_size=kw, stride=2, padding=padw,
                          bias=use_bias),
                norm_layer(n_filters * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        if sa:
            sequence.append(Self_Attn(n_filters * nf_mult))

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** num_downs, 8)
        sequence += [
            nn.Conv2d(n_filters * nf_mult_prev, n_filters * nf_mult,
                      kernel_size=kw, stride=2, padding=padw, bias=use_bias),
            norm_layer(n_filters * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        if sa:
            sequence.append(Self_Attn(n_filters * nf_mult))

        # output 1 channel prediction map
        sequence += [nn.Conv2d(n_filters * nf_mult, 1,
                               kernel_size=kw, stride=1, padding=0)]
        self.model = nn.Sequential(*sequence)
        if sn:
            self.apply(spectral_normalization)

    def forward(self, input):
        return self.model(input)


class Siamese(nn.Module):
    """Defines a Siamese Network """

    def __init__(self, input_nc, n_filters, num_downs, latent_dim,
                 sn=True, norm_layer="none"):
        super(Siamese, self).__init__()

        norm_dict = {"bn": nn.BatchNorm2d, "none": nn.Identity}
        norm_layer = norm_dict[norm_layer]
        # no need to use bias as BatchNorm2d has affine parameters
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, n_filters, kernel_size=kw,
                              stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, num_downs):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(n_filters * nf_mult_prev, n_filters * nf_mult,
                          kernel_size=kw, stride=2, padding=padw,
                          bias=use_bias),
                norm_layer(n_filters * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** num_downs, 8)
        sequence += [
            nn.Conv2d(n_filters * nf_mult_prev, n_filters * nf_mult,
                      kernel_size=kw, stride=2, padding=padw, bias=use_bias),
            norm_layer(n_filters * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(n_filters * nf_mult, latent_dim,
                               kernel_size=kw, stride=1, padding=0)]
        self.model = nn.Sequential(*sequence)
        if sn:
            self.apply(spectral_normalization)

    def forward(self, input):
        return self.model(input)


class Self_Attn(nn.Module):
    """ Self attention Layer (implementation from @heykeetae)"""

    def __init__(self, in_dim):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(
            m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(
            m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(
            m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out  # , attention


class VGG19(nn.Module):
    def __init__(self, init_weights=None, feature_mode=False, batch_norm=False, num_classes=1000):
        super(VGG19, self).__init__()
        self.cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
        self.init_weights = init_weights
        self.feature_mode = feature_mode
        self.batch_norm = batch_norm
        self.num_clases = num_classes
        # self.features = self.make_layers(self.cfg, batch_norm)
        features = list(vgg19(pretrained=True).features)[:33]
        self.features = nn.ModuleList(features).eval()

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if not init_weights == None:
            self.load_state_dict(torch.load(init_weights))

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
            
            
        # return nn.Sequential(*layers)
        return layers

    def forward(self, x):
        if self.feature_mode:
            # module_list = list(self.features.modules())
            # for l in module_list[1:27]:                 # conv4_4
                # x = l(x)
            results = []
            n = 0
            for ii, model in enumerate(self.features):
                x = model(x)
                if ii in {3, 7, 8, 15, 22}:    # relu1_2, ...  ,relu2_2, relu3_3, relu4_3
                    results.append(x)
                    # print('x-', n ,', ', x.shape)
                n += 1
            
            vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_1', 'relu2_2', 'relu3_3', 'relu4_3'])
            # print('output : ', vgg_outputs(*results))
            return vgg_outputs(*results)

        if not self.feature_mode:
            x = x.view(x.size(0), -1)
            x = self.classifier(x)

        return x
