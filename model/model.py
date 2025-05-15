import torch
import torch.nn as nn
import math
from .transformer import Encoder as TransformerLayer
from configs.model_config import ModelConfig
from .attention import ChannelAttention
from .res_encoder import ResBlock, ResEncoder
from .DBPN import IterativeBlock, D_DBPN
from .common import Reshape, Trim, initial_size_to_strides_map

class DownsampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super(DownsampleLayer, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        # self.gelu = nn.GELU()
        self.act = nn.PReLU()
        # self.norm = nn.LayerNorm(out_channels)
        # self.norm = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        # input shape: [batch_size, num_elements (coefficients or raw hrtf points), channels]
        x = x.permute(0, 2, 1) # adjust to [batch_size, channels, num_elements]
        x = self.conv(x)
        x = self.act(x)
        # x = self.norm(x)
        x = x.permute(0, 2, 1) # adjust back to [batch_size, num_elements, channels]
        # x = self.gelu(x)
        
        return x

class Encoder(nn.Module):
    def __init__(self, model_config: ModelConfig):
        super(Encoder, self).__init__()
        initial_size = model_config.initial_size
        assert initial_size in initial_size_to_strides_map, f"invalid initial size, should be one of {initial_size_to_strides_map.keys()}"

        # strides for downsampling layers
        self.strides = initial_size_to_strides_map[initial_size]
        in_channels = model_config.nbins
        # each layer of Encoder model is constructed by a transformer layer followed by a downsampling layer
        # except the last layer, which is only a transformer layer without downsampling
        # for example, if total number encoding layer is 5, the structure is as:
        # [Transofrmer, downsampling, transformer, downsampling, transformer, downsampling, transformer]
        # strides only indicate the stride used in each downsampling layer
        # therefore the total number of encoding layer is len(strides) + 1
        num_encoding_layer = len(self.strides)
        self.layers = nn.ModuleList()
        for i in range(num_encoding_layer):
            self.layers.append(TransformerLayer(emb_size=in_channels,
                                                hidden_size=model_config.hidden_size,
                                                num_layers=model_config.num_transformer_layers,
                                                num_heads=model_config.num_heads,
                                                num_groups=model_config.num_groups,
                                                norm_type=model_config.norm_type,
                                                activation=model_config.activation,
                                                dropout=model_config.dropout,
                                                target_size=model_config.target_size))
            
            # Add channel attention after transformer
            # self.layers.append(ChannelAttention(channels=in_channels))
            out_channels = min(in_channels * 2, 2048)
            self.layers.append(DownsampleLayer(in_channels=in_channels, out_channels=out_channels,
                                               stride=self.strides[i])) # downsamply by 2 if stride=2
            in_channels = out_channels

            # no downsampling for last layer
            # if i < num_encoding_layer - 1:
            #     out_channels = min(in_channels * 2, 2048)
            #     self.layers.append(DownsampleLayer(in_channels=in_channels, out_channels=out_channels,
            #                                        stride=self.strides[i])) # downsamply by 2 if stride=2
            #     in_channels = out_channels
        
        output_size = self._get_output_dim(initial_size)
        self.fc = nn.Sequential(nn.Linear(output_size * in_channels, 1024),
                                nn.BatchNorm1d(1024),
                                nn.PReLU(),
                                # nn.GELU(),
                                nn.Linear(1024, model_config.latent_dim))

    def _get_output_dim(self, size):
        # configuration for convolution layer
        kernel_size = 3
        padding = 1
        # compute the output shape
        for s in self.strides:
            size = (size + 2 * padding - kernel_size) // s + 1
        return size

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        # x = x.permute(0, 2, 1)
        # x = self.latent_conv(x)
        # x = x.view(x.shape[0], -1)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

class TrimLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        x = self.norm(x)
        return x

class UpsampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super(UpsampleLayer, self).__init__()
        self.conv_transpose = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1)
        self.gelu = nn.GELU()
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        # input shape: [batch_size, num_elements (coefficients or raw hrtf points), channels]
        x = x.permute(0, 2, 1) # adjust to [batch_size, channels, num_elements]
        x = self.conv_transpose(x)
        x = x.permute(0, 2, 1) # adjust back to [batch_size, num_elements, channels]
        x = self.gelu(x)
        x = self.norm(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self, model_config: ModelConfig):
        super(Decoder, self).__init__()
        in_channels = 512
        initial_size = model_config.initial_size
        self.fc = nn.Sequential(
            nn.Linear(model_config.latent_dim, initial_size*in_channels),
            nn.BatchNorm1d(initial_size * in_channels),
            # nn.GELU(),
            nn.PReLU(),
            Reshape(-1, initial_size, in_channels)
        )
        # self.conv0 = nn.Conv1d(model_config.latent_dim, in_channels, kernel_size=3, stride=1, padding=1)
        self.layers = nn.ModuleList()
        if model_config.apply_sht:
            # for SH coefficients: 4->8->16->32->64->128->256->512
            out_channels = [1024, 512, 512, 256, 256]
        else:
            # for raw hrtf points: 4->8->16->32->64->128->256->512->1024
            # out_channels = [1024, 1024, 512, 512, 512, 256, 256, 256]
            out_channels = [1024, 1024, 512, 512, 512, 512]
        # num_layers = len(out_channels) + 1

        num_layers = len(out_channels)

        for layer_index in range(num_layers):
            self.layers.append(TransformerLayer(emb_size=in_channels,
                                                hidden_size=model_config.hidden_size,
                                                num_layers=model_config.num_transformer_layers,
                                                num_heads=model_config.num_heads,
                                                num_groups=model_config.num_groups,
                                                norm_type=model_config.norm_type,
                                                activation=model_config.activation,
                                                dropout=model_config.dropout,
                                                target_size=model_config.target_size))
            self.layers.append(IterativeBlock(in_channels, out_channels[layer_index], kernel=4, stride=2, padding=1, activation='prelu', input_shape_layout='bsc'))
            in_channels = out_channels[layer_index]
            # self.layers.append(ChannelAttention(channels=in_channels))

            # if layer_index < num_layers - 1:
            #     # self.layers.append(UpsampleLayer(in_channels=in_channels,out_channels=out_channels[layer_index]))
            #     self.layers.append(IterativeBlock(in_channels, out_channels[layer_index], kernel=4, stride=2, padding=1, activation='prelu', input_shape_layout='bsc'))
            #     in_channels = out_channels[layer_index]
            # if layer_index == num_layers - 2:
            #     self.layers.append(Trim(model_config.target_size, dim=1))
        self.layers.append(Trim(model_config.target_size, dim=1))
        self.out_conv = nn.Conv1d(in_channels, model_config.nbins, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        # x = self.conv0(x)
        # x = x.permute(0, 2, 1)
        x = self.fc(x)
        for layer in self.layers:
            x = layer(x)
        x = x.permute(0, 2, 1)
        x = self.out_conv(x)
        return x

class HRTF_Transformer(nn.Module):
    def __init__(self, encoder_config, decoder_config) -> None:
        super(HRTF_Transformer, self).__init__()
        self.encoder = Encoder(encoder_config)
        self.decoder = Decoder(decoder_config)

    def forward(self, x):
        encoder_out = self.encoder(x)
        sr = self.decoder(encoder_out)
        return sr.permute(0, 2, 1)


class AutoEncoder(nn.Module):
    def __init__(self, encoder_cls, encoder_config, decoder_cls, decoder_config) -> None:
        super().__init__()
        self.encoder = encoder_cls(encoder_config)
        self.decoder = decoder_cls(decoder_config)
        self.init_parameters()
    
    def init_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):
                if hasattr(m, 'weight') and m.weight is not None and m.weight.requires_grad:
                    nn.init.kaiming_normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None and m.bias.requires_grad:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        if isinstance(self.encoder, ResEncoder):
            x = x.permute(0, 2, 1)
        z = self.encoder(x)
        out = self.decoder(z)
        return out
