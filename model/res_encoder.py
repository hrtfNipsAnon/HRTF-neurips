import torch.nn as nn

from .common import initial_size_to_strides_map

class ResBlock(nn.Module):
    def __init__(self, in_channnels, out_channels, stride=1, expansion=1, identity_downsample=None):
        super(ResBlock, self).__init__()
        self.expansion = expansion
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channnels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm1d(out_channels),
        ) 
        self.conv2 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels * self.expansion, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels * self.expansion)
        )
        self.relu = nn.ReLU()
        self.prelu = nn.PReLU()
        self.leakyRelu = nn.LeakyReLU(0.2)
        self.identity_downsample = identity_downsample
        self.stride = stride

    def forward(self, x):
        identity = x.clone()
        x = self.conv1(x)
        x = self.prelu(x)
        x = self.conv2(x)
        
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.prelu(x)
        return x
    
class ResEncoder(nn.Module):
    def __init__(self, encoder_config):
        super(ResEncoder, self).__init__()
        num_blocks = 2
        self.expansion = 1
        self.in_channels = 256
        nbins = encoder_config.nbins
        latent_dim = encoder_config.latent_dim
        initial_size = encoder_config.initial_size
        block = ResBlock
        self.conv1 = nn.Sequential(
            nn.Conv1d(nbins, self.in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(self.in_channels),
            nn.PReLU(),
        )
        res_layers = []
        # strides for downsampling layers
        self.strides = initial_size_to_strides_map[initial_size]

        res_layers.append(self._make_layer(block, 256, num_blocks))
        for stride in self.strides:
            res_layers.append(self._make_layer(block, 512, num_blocks, stride=stride))
        self.res_layers = nn.Sequential(*res_layers)
        output_size = self._get_output_dim(initial_size)
        self.fc = nn.Sequential(nn.Linear(512*output_size, 512),
                                nn.BatchNorm1d(512),
                                nn.PReLU(),
                                nn.Linear(512, latent_dim))
    
    def _make_layer(self, block, out_channels, num_blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels * self.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels * self.expansion)
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, self.expansion, downsample))
        self.in_channels = out_channels * self.expansion

        for i in range(num_blocks-1):
            layers.append(block(self.in_channels, out_channels, expansion=self.expansion))
        return nn.Sequential(*layers)
    
    def _get_output_dim(self, size):
        # configuration for convolution layer
        kernel_size = 3
        padding = 1
        # compute the output shape
        for s in self.strides:
            size = (size + 2 * padding - kernel_size) // s + 1
        return size
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.res_layers(x)
        x = x.view(x.size(0), -1)
        z = self.fc(x)
        return z