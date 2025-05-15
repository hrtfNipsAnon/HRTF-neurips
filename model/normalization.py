import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x.pow(2), dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)
    

class DualEarNormalization(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.left_norm = RMSNorm(dim // 2, eps)
        self.right_norm = RMSNorm(dim // 2, eps)

    def forward(self, x):
        left = self.left_norm(x[..., :x.shape[-1] // 2])
        right = self.right_norm(x[..., x.shape[-1] // 2:])
        return torch.cat([left, right], dim=-1)


class CustomizedNormalization(nn.Module):
    def __init__(self, norm_type, channels):
        super().__init__()
        if norm_type == "instance":
            self.norm = nn.InstanceNorm1d(channels)
        elif norm_type == "batch":
            self.norm = nn.BatchNorm1d(channels)
        else:
            raise ValueError(f"unrecognized normalization: {norm_type}")
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.norm(x)
        x = x.permute(0, 2, 1)
        return x


class TokenScaling(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        scale = torch.sqrt((x.pow(2).mean(dim=-1, keepdim=True) + self.eps))
        return x / scale

if __name__ == "__main__":
    x = torch.randn(1, 138, 128)
    norm = RMSNorm(128)
    x = norm(x)
    print(x.shape)

    x = torch.randn(2, 18, 256)
    dual_ear_norm = DualEarNormalization(256)
    x = dual_ear_norm(x)
    print(x.shape)

    x = torch.randn(3, 27, 128)
    norm_type = "instance"
    norm = CustomizedNormalization(norm_type, 128)
    x = norm(x)
    print(x.shape)

    norm_type = "batch"
    norm = CustomizedNormalization(norm_type, 128)
    x = norm(x)
    print(x.shape)

    x = torch.randn(4, 36, 128)
    token_scale = TokenScaling()
    x = token_scale(x)
    print(x.shape)