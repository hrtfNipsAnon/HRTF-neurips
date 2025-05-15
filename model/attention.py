import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from einops import einsum, rearrange
from .position_embedding import RotaryEmbedding, RelativePositionBias

class GroupedQueryAttention(nn.Module):
    def __init__(self, emb_size, hidden_size, num_heads, num_groups, dropout=0., target_size=484):
        super(GroupedQueryAttention, self).__init__()
        """
        num_groups:
            = 1 -> multi-query attention, all queies share 1 pair of key, value
            = num_heads -> multi-head attention, each query relates to an unique pair of key, value
        
        """
        self.use_rope = False
        self.hidden_size = hidden_size
        assert num_heads % num_groups == 0, "num_heads must be divisible by num_groups"
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.num_kv_heads = num_heads // num_groups
        self.head_dim = hidden_size // num_heads
        assert self.head_dim * num_heads == hidden_size, "hidden_size must be divisible by num_heads"

        self.query_proj = nn.Linear(emb_size, hidden_size, bias=False)
        self.key_proj = nn.Linear(emb_size, self.head_dim * num_groups, bias=False)
        self.value_proj = nn.Linear(emb_size, self.head_dim * num_groups, bias=False)

        if self.use_rope:
            # rotary position embedding
            self.query_rope = RotaryEmbedding(dim=self.head_dim, max_seq_len=target_size)
            self.key_rope = RotaryEmbedding(dim=self.head_dim, max_seq_len=target_size)
        else:
            # relative position bias
            self.relative_pos_bias = RelativePositionBias(num_heads, max_distance=target_size)

        self.out_proj = nn.Linear(hidden_size, emb_size, bias=False)
        self.dropout = nn.Dropout(dropout)

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, query, key, value, mask=None):
        b, q_len = query.shape[:2]
        k_len = key.shape[1]
        query = self.query_proj(query) # [b, q_len, hidden_size]
        query = rearrange(query, "b sq (n d) -> b n sq d", d=self.head_dim).contiguous()
        if self.use_rope:
            query = self.query_rope(query)
        query = rearrange(query, "b (g h) sq d -> b g h sq d", g=self.num_groups).contiguous()

        key = self.key_proj(key)
        key = rearrange(key, "b sk (g d) -> b g sk d", d=self.head_dim).contiguous()
        if self.use_rope:
            key = self.key_rope(key)
        value = self.value_proj(value)
        value = rearrange(value, "b sv (g d) -> b g sv d", d=self.head_dim).contiguous()
        scale = self.head_dim ** 0.5

        # calculate attention scores
        scores = einsum(query, key, "b g h sq d, b g sk d -> b g h sq sk")

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-1e20"))
        
        if self.use_rope:
            attention = torch.softmax(scores / (scale), dim=-1)
        else:
            # add relative position bias
            relative_pos_bias = self.relative_pos_bias(q_len, k_len).view(1, self.num_groups, self.num_kv_heads, q_len, k_len)
            attention = torch.softmax(scores / (scale) + relative_pos_bias, dim=-1) # [b, group, head_per_group, q_len, k_len]
        attention = self.dropout(attention)

        # out = einsum(attention, value, "b g h sq sk, b g sv d -> b sq g h d").reshape(b, q_len, -1) # sk = sv
        out = einsum(attention, value, "b g h sq sk, b g sv d -> b g h sq d")
        out = rearrange(out, "b g h sq d -> b sq (g h d)").contiguous()
        out = self.out_proj(out) # [b, q_len, emb_size]
        return out

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc1 = nn.Conv1d(channels, channels // reduction, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv1d(channels // reduction, channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = x.permute(0, 2, 1) # [batch_size, feature, channel] -> [batch_size, channel, feature]
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)
        out = avg_out + max_out
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        attention_weights = self.sigmoid(out)
        x = x * attention_weights
        x = x.permute(0, 2, 1)
        return x

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
    gqa = GroupedQueryAttention(embed_dim, hidden_size, num_heads, num_groups)
    out = gqa(query, key, value)
    print(out.shape)





