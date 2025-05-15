import torch
import torch.nn as nn

from .common import Reshape, Trim

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, activation='gelu', norm=None):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = nn.BatchNorm1d(out_channels)

        self.activation = activation
        if self.activation == 'relu':
            self.act = nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = nn.Sigmoid()
        elif self.activation == 'gelu':
            self.act = nn.GELU()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)
        if self.activation is not None:
            return self.act(out)
        else:
            return out

class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=True, activation='gelu', norm=None):
        super(DeconvBlock, self).__init__()
        self.deconv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = nn.BatchNorm1d(out_channels)
        
        self.activation = activation
        if self.activation == 'relu':
            self.act = nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = nn.Sigmoid()
        elif self.activation == 'gelu':
            self.act = nn.GELU()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.deconv(x))
        else:
            out = self.deconv(x)
        if self.activation is not None:
            return self.act(out)
        else:
            return out

class UpBlock(nn.Module):
    def __init__(self, channels, kernel_size=8, stride=4, padding=2, bias=True, activation='gelu', norm=None):
        super(UpBlock, self).__init__()
        self.conv1 = DeconvBlock(channels, channels, kernel_size, stride, padding, bias, activation, norm)
        self.conv2 = ConvBlock(channels, channels, kernel_size, stride, padding, bias, activation, norm)
        self.conv3 = DeconvBlock(channels, channels, kernel_size, stride, padding, bias, activation, norm)

    def forward(self, x):
        h0 = self.conv1(x)
        l0 = self.conv2(h0)
        h1 = self.conv3(l0 - x)
        return h1 + h0
    
class DownBlock(nn.Module):
    def __init__(self, channels, kernel_size=8, stride=4, padding=2, bias=True, activation='gelu', norm=None):
        super(DownBlock, self).__init__()
        self.conv1 = ConvBlock(channels, channels, kernel_size, stride, padding, bias, activation, norm)
        self.conv2 = DeconvBlock(channels, channels, kernel_size, stride, padding, bias, activation, norm)
        self.conv3 = ConvBlock(channels, channels, kernel_size, stride, padding, bias, activation, norm)

    def forward(self, x):
        l0 = self.conv1(x)
        h0 = self.conv2(l0)
        l1 = self.conv3(h0 - x)
        return l1 + l0
    
class D_DownBlock(nn.Module):
    def __init__(self, channels, kernel_size=8, stride=4, padding=2, num_stages=1, bias=True, activation='gelu', norm=None):
        super(D_DownBlock, self).__init__()
        self.conv = ConvBlock(channels*num_stages, channels, 1, 1, 0, bias, activation, norm)
        self.down1 = ConvBlock(channels, channels, kernel_size, stride, padding, bias, activation, norm)
        self.down2 = DeconvBlock(channels, channels, kernel_size, stride, padding, bias, activation, norm)
        self.down3 = ConvBlock(channels, channels, kernel_size, stride, padding, bias, activation, norm)

    def forward(self, x):
        x = self.conv(x)
        l0 = self.down1(x)
        h0 = self.down2(l0)
        l1 = self.down3(h0 - x)
        return l1 + l0

class D_UpBlock(nn.Module):
    def __init__(self, channels, kernel_size=8, stride=4, padding=2, num_stages=1, bias=True, activation='gelu', norm=None):
        super(D_UpBlock, self).__init__()
        self.conv = ConvBlock(channels*num_stages, channels, 1, 1, 0, bias, activation, norm)
        self.up1 = DeconvBlock(channels, channels, kernel_size, stride, padding, bias, activation, norm)
        self.up2 = ConvBlock(channels, channels, kernel_size, stride, padding, bias, activation, norm)
        self.up3 = DeconvBlock(channels, channels, kernel_size, stride, padding, bias, activation, norm)

    def forward(self, x):
        x = self.conv(x)
        h0 = self.up1(x)
        l0 = self.up2(h0)
        h1 = self.up3(l0 - x)
        return h1 + h0
    
class IterativeBlock(nn.Module):
    def __init__(self, channels, out_channels, kernel, stride, padding, activation='gelu', input_shape_layout='bcs'):
        super(IterativeBlock, self).__init__()
        bias = False
        norm = "batch"
        self.input_shape_layout = input_shape_layout
        self.up1 = UpBlock(channels, kernel, stride, padding, bias=bias, activation=activation, norm=norm)
        self.down1 = DownBlock(channels, kernel, stride, padding, bias=bias, activation=activation, norm=norm)
        self.up2 = UpBlock(channels, kernel, stride, padding, bias=bias, activation=activation, norm=norm)
        self.down2 = D_DownBlock(channels, kernel, stride, padding, 2, bias=bias, activation=activation, norm=norm)
        self.up3 = D_UpBlock(channels, kernel, stride, padding, 2, bias=bias, activation=activation, norm=norm)
        self.down3 = D_DownBlock(channels, kernel, stride, padding, 3, bias=bias, activation=activation, norm=norm)
        self.up4 = D_UpBlock(channels, kernel, stride, padding, 3, bias=bias, activation=activation, norm=norm)
        self.down4 = D_DownBlock(channels, kernel, stride, padding, 4, bias=bias, activation=activation, norm=norm)
        self.up5 = D_UpBlock(channels, kernel, stride, padding, 4, bias=bias, activation=activation, norm=norm)
        # self.down5 = D_DownBlock(channels, kernel, stride, padding, 5, bias=bias, activation=activation, norm=norm)
        # self.up6 = D_UpBlock(channels, kernel, stride, padding, 5, bias=bias, activation=activation, norm=norm)
        self.out_conv = ConvBlock(5*channels, out_channels, 3, 1, 1, bias=bias, activation=activation, norm=norm)
        
    def forward(self, x):
        if self.input_shape_layout == 'bsc':
            x = x.permute(0, 2, 1)
        h1 = self.up1(x)
        l1 = self.down1(h1)
        h2 = self.up2(l1)
        
        concat_h = torch.cat((h2, h1), 1)
        l = self.down2(concat_h)
        
        concat_l = torch.cat((l, l1), 1)
        h = self.up3(concat_l)

        concat_h = torch.cat((h, concat_h), 1)
        l = self.down3(concat_h)

        concat_l = torch.cat((l, concat_l), 1)
        h = self.up4(concat_l)

        concat_h = torch.cat((h, concat_h), 1)
        l = self.down4(concat_h)

        concat_l = torch.cat((l, concat_l), 1)
        h = self.up5(concat_l)

        concat_h = torch.cat((h, concat_h), 1)
        # l = self.down5(concat_h)

        # concat_l = torch.cat((l, concat_l), 1)
        # h = self.up6(concat_l)

        # concat_h = torch.cat((h, concat_h), 1)
        out = self.out_conv(concat_h)
        if self.input_shape_layout == 'bsc':
            out = out.permute(0, 2, 1)

        return out
    
class D_DBPN(nn.Module):
    def __init__(self, decoder_config):
        super(D_DBPN, self).__init__()

        nbins = decoder_config.nbins
        latent_dim = decoder_config.latent_dim
        target_size = decoder_config.target_size
        initial_size = decoder_config.initial_size
        base_channels = 512
        kernel = 4
        stride = 2
        padding = 1
        
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512*16),
            nn.BatchNorm1d(512*16),
            nn.ReLU(True),
            # nn.PReLU(),
            Reshape(-1, 512, 16),
        )

        activation = 'prelu'

        self.conv0 = ConvBlock(512, base_channels, 3, 1, 1)

        # Back-projection stages
        self.up1 = IterativeBlock(base_channels, base_channels, kernel, stride, padding, activation=activation)
        self.up2 = IterativeBlock(base_channels, base_channels, kernel, stride, padding, activation=activation)
        self.up3 = IterativeBlock(base_channels, base_channels, kernel, stride, padding, activation=activation)
        self.up4 = IterativeBlock(base_channels, base_channels, kernel, stride, padding, activation=activation)
        self.up5 = IterativeBlock(base_channels, base_channels, kernel, stride, padding, activation=activation)
        self.up6 = IterativeBlock(base_channels, base_channels, kernel, stride, padding, activation=activation)
        
        # Reconstruction
        self.out_conv = ConvBlock(base_channels, nbins, 3, 1, 1, activation=None)
        self.trim = Trim(target_size)

    def forward(self, x):
        x = self.fc(x)
        x = self.conv0(x)

        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.up5(x)
        x = self.up6(x)
        x = self.out_conv(x)
        out = self.trim(x)
        return out