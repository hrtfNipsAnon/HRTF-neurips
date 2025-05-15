import torch
import torch.nn as nn
import torch.nn.init as init
from .attention import GroupedQueryAttention
from .normalization import RMSNorm, CustomizedNormalization, TokenScaling

class TransformerBlock(nn.Module):
    def __init__(self, emb_size, hidden_size, num_heads, num_groups, norm_type="batch", activation="prelu", dropout=0., target_size=484):
        super(TransformerBlock, self).__init__()
        """
        Args:
            target_size: used for position embedding
        """
        self.attention = GroupedQueryAttention(emb_size, hidden_size, num_heads, num_groups, dropout, target_size)
        if norm_type == "rms_norm":
            self.norm1 = RMSNorm(emb_size)
            self.norm2 = RMSNorm(emb_size)
        elif norm_type == "layer_norm":
            self.norm1 = nn.LayerNorm(emb_size)
            self.norm2 = nn.LayerNorm(emb_size)
        elif norm_type == "token_scale":
            self.norm1 = TokenScaling()
            self.norm2 = TokenScaling()
        else:
            self.norm1 = CustomizedNormalization(norm_type, emb_size)
            self.norm2 = CustomizedNormalization(norm_type, emb_size)
        
        self.mlp = nn.Sequential(
            nn.Linear(emb_size, emb_size * 4),
            self._get_activation(activation),
            nn.Linear(emb_size * 4, emb_size)
        )

        self.dropout = nn.Dropout(dropout)

        # Initialize parameters
        # self._init_mlp_weights()
    
    def _init_mlp_weights(self):
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    init.zeros_(layer.bias)

    def _get_activation(self, activation):
        if activation == "prelu":
            return nn.PReLU()
        elif activation == "gelu":
            return nn.GELU()
        elif activation == "relu":
            return nn.ReLU()
        else:
            raise ValueError(f"Unknown activation function: {activation}")

    def forward(self, query, key, value, mask):
        attention = self.attention(query, key, value, mask)

        x = self.norm1(query + self.dropout(attention))
        # x = query + self.dropout(attention)
        mlp_out = self.mlp(x)
        out = self.norm2(x + self.dropout(mlp_out))
        # out = x + self.dropout(mlp_out)
        return out

class Encoder(nn.Module):
    def __init__(self, emb_size, hidden_size, num_layers, num_heads, num_groups, norm_type, activation, dropout, target_size) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    emb_size,
                    hidden_size,
                    num_heads,
                    num_groups,
                    norm_type,
                    activation,
                    dropout,
                    target_size
                )
                for _ in range(num_layers)
            ]
        )
    
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, x, x, mask)
        return x
    
class DecoderBlock(nn.Module):
    def __init__(self, emb_size, hidden_size, num_heads, num_groups, dropout, target_size):
        super(DecoderBlock, self).__init__()
        self.norm = RMSNorm(emb_size)
        self.self_attention = GroupedQueryAttention(emb_size, hidden_size, num_heads, num_groups, dropout, target_size)
        self.transfomer_block = TransformerBlock(
            emb_size, hidden_size, num_heads, num_groups, dropout, target_size
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key, value, lr_mask, hr_mask):
        self_attention = self.self_attention(x, x, x, hr_mask)
        query = self.dropout(self.norm(self_attention + x))
        out = self.transfomer_block(query, key, value, lr_mask)
        return out
    
class Decoder(nn.Module):
    def __init__(self, emb_size, hidden_size, num_layers, num_heads, num_groups, dropout, target_size) -> None:
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(
            [
                DecoderBlock(emb_size, hidden_size, num_heads, num_groups, dropout, target_size)
                for _ in range(num_layers)
            ]
        )
    
    def forward(self, x, enc_out, lr_mask, hr_mask):
        for layer in self.layers:
            x = layer(x, enc_out, enc_out, lr_mask, hr_mask)

        return x

class Transformer(nn.Module):
    def __init__(
            self,
            lr_pad_idx,
            emb_size=256,
            hidden_size=4096,
            num_layers=5,
            num_heads=32,
            num_groups=8,
            dropout=0,
            target_size=484,
            device="cpu",
    ):
        super(Transformer, self).__init__()

        self.encoder = Encoder(
            emb_size=emb_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            num_groups=num_groups,
            dropout=dropout,
            target_size=target_size
        )

        self.decoder = Decoder(
            emb_size=emb_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            num_groups=num_groups,
            dropout=dropout,
            target_size=target_size
        )

        self.lr_pad_idx = lr_pad_idx
        self.device = device

    def make_lr_mask(self, lr_sample):
        # currently the input sizes are determined by the number of initial points,
        # and therefore are uniform in each batch
        # and we don't actually need padding for low resolution inputs.
        # but in the future we might consider inputs with different upsampling ratio
        # and probably use padding to ensure equal squence length
        lr_mask = (lr_sample != self.lr_pad_idx).unsqueeze(1).unsqueeze(2)
        return lr_mask.to(self.device)
    
    def make_hr_mask(self, hr_coefficients):
        b, hr_len = hr_coefficients.shape[:2]
        hr_coef_mask = torch.tril(torch.ones((hr_len, hr_len))).expand(
            b, 1, hr_len, hr_len
        )
        return hr_coef_mask
    
    def forward(self, lr_sample, hr_sample):
        hr_mask = self.make_hr_mask(hr_sample)
        enc_lr = self.encoder(lr_sample, mask=None)
        out = self.decoder(hr_sample, enc_lr, lr_mask=None, hr_mask=hr_mask)
        return out

if __name__ == "__main__":
    batch_size = 2
    query_len = 84
    key_len = 120
    embed_dim = 256
    query = torch.randn(batch_size, query_len, embed_dim)
    key = torch.randn(batch_size, key_len, embed_dim)
    value = torch.randn(batch_size, key_len, embed_dim)
    hidden_size = 1024
    num_heads = 16
    num_groups = 4
    print("--------test TransformerBlock------")
    model = TransformerBlock(embed_dim, hidden_size, num_heads, num_groups)
    out = model(query, key, value, None)
    print(out.shape)

    print("----------test Encoder--------------")
    x = torch.randn(batch_size, query_len, embed_dim)
    num_layers = 2
    dropout = 0
    target_size = 484
    encoder = Encoder(embed_dim, hidden_size, num_layers, num_heads, num_groups, dropout, target_size)
    encoder_out = encoder(x, None)
    print(encoder_out.shape)

    print("------------test Decoder------------")
    y = torch.randn(batch_size, key_len, embed_dim)
    decoder = Decoder(embed_dim, hidden_size, num_layers, num_heads, num_groups, dropout, target_size)
    lr_mask = (encoder_out != 0).any(-1).unsqueeze(1).unsqueeze(2)
    hr_mask = torch.tril(torch.ones((key_len, key_len))).expand(batch_size, 1, key_len, key_len)
    decoder_out = decoder(y, encoder_out, lr_mask, hr_mask)
    print(decoder_out.shape)

    print("----------test Transformer---------")
    lr_sample = torch.randn(batch_size, query_len, embed_dim)
    hr_sample = torch.randn(batch_size, key_len, embed_dim)
    model = Transformer(lr_pad_idx=0, emb_size=embed_dim, hidden_size=hidden_size,
                        num_layers=num_layers, num_heads=num_heads, num_groups=num_groups,
                        dropout=dropout, target_size=target_size)
    upsampeld_sample = model(lr_sample, hr_sample)
    print(upsampeld_sample.shape)
