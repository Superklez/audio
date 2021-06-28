import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class Conformer(nn.Module):
    def __init__(self,
        num_classes : int,
        in_channels : int = 1,
        d_model : int = 64,
        num_conformer_blocks : int = 1,
        subsampling_kernel_size : int = 80,
        subsampling_stride : int = 4,
        subsampling_dropout : float = 0.1,
        ffn_expansion : int = 4,
        ffn_dropout : float = 0.1,
        mhsa_num_heads : int = 4,
        mhsa_max_len : int = 5000,
        mhsa_dropout : float = 0.1,
        conv_kernel_size : int = 7,
        conv_dropout : float = 0.1,
        lstm_num_layers : int = 1,
        lstm_dropout : float = 0.):
        super(Conformer, self).__init__()
        self.subsampling = nn.Sequential(
            SamePadding(subsampling_kernel_size, subsampling_stride),
            nn.Conv1d(in_channels, d_model, subsampling_kernel_size,
                subsampling_stride),
            # SamePadding(subsampling_kernel_size, subsampling_stride),
            # nn.Conv1d(d_model, d_model, subsampling_kernel_size,
            #     subsampling_stride),
            nn.Conv1d(d_model, d_model, 1),
            nn.Dropout(subsampling_dropout)
        )
        conformer_blocks = []
        for _ in range(num_conformer_blocks):
            conformer_blocks += [
                FeedForwardModule(d_model, ffn_expansion, ffn_dropout),
                MHSAModule(d_model, mhsa_num_heads, mhsa_max_len, mhsa_dropout),
                ConvModule(d_model, conv_kernel_size, conv_dropout),
                FeedForwardModule(d_model, ffn_expansion, ffn_dropout),
                nn.LayerNorm(d_model)
            ]
        self.conformer_blocks = nn.Sequential(*conformer_blocks)
        self.lstm = nn.LSTM(d_model, num_classes, lstm_num_layers,
            dropout=lstm_dropout, batch_first=True)

    def forward(self, x:Tensor):
        x = self.subsampling(x).permute(0, 2, 1)
        x = self.conformer_blocks(x).permute(0, 2, 1)
        x = F.avg_pool1d(x, x.size(2)).permute(0, 2, 1)
        x = self.lstm(x)[0]
        return x

class FeedForwardModule(nn.Module):
    def __init__(self, d_model:int, expansion:int=4, dropout:float=0.1):
        super(FeedForwardModule, self).__init__()
        self.ffn_block = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model*expansion),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(d_model*expansion, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x:Tensor):
        # input shape: [n, l, c]
        residual = x
        x = self.ffn_block(x)
        # output shape: [n, l, c]
        return x/2 + residual

class MHSAModule(nn.Module):
    def __init__(self, d_model:int, num_heads:int=4, max_len:int=5000,
        dropout:float=0.1):
        super(MHSAModule, self).__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.pe = PositionalEncoding(d_model, max_len)
        self.mha = MultiheadAttention(d_model, num_heads)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x:Tensor):
        # input shape: [n, l, c]
        residual = x
        x = self.layer_norm(x)
        x = self.pe(x)
        attn_mask = get_subsequent_mask(x)
        x = self.mha(x, x, x, attn_mask)[0]
        x = self.dropout(x)
        # output shape: [n, l, c]
        return x + residual

class ConvModule(nn.Module):
    def __init__(self, d_model:int, kernel_size:int, dropout:float=0.1):
        super(ConvModule, self).__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.conv_block = nn.Sequential(
            nn.Conv1d(d_model, d_model*2, 1),
            GLU(),
            SamePadding(kernel_size, 1),
            nn.Conv1d(d_model, d_model, kernel_size, groups=d_model),
            nn.BatchNorm1d(d_model),
            Swish(),
            nn.Conv1d(d_model, d_model, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x:Tensor):
        # input shape: [n, l, c]
        residual = x
        x = self.layer_norm(x).permute(0, 2, 1)
        x = self.conv_block(x).permute(0, 2, 1)
        # output shape: [n, l, c]
        return x + residual

class SamePadding(nn.Module):
    def __init__(self, kernel_size:int, stride:int, len_dim:int=-1):
        super(SamePadding, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.len_dim = len_dim

    def forward(self, x:Tensor):
        in_len = x.size(self.len_dim)
        out_len = math.ceil(in_len / self.stride)
        padding = (out_len - 1) * self.stride + self.kernel_size - in_len
        return F.pad(x, (padding, 0))

class GLU(nn.Module):
    def __init__(self, dim:int=1):
        super(GLU, self).__init__()
        self.dim = dim
    
    def forward(self, x:Tensor):
        x1, x2 = x.chunk(2, self.dim)
        return x1 * x2.sigmoid()

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x:Tensor):
        return x * x.sigmoid()

class PositionalEncoding(nn.Module):
    def __init__(self, d_model:int, max_len:int=5000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.register_buffer('pe', self._get_positional_encoding(d_model,
            max_len))

    def _get_positional_encoding(self, d_model:int, max_len:int):
        pe = torch.zeros((max_len, d_model), requires_grad=False)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * 
            -(math.log(10000.) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x:Tensor):
        # input shape: [b, l, c]
        if x.size(1) > self.max_len:
            self.max_len = x.size(1)
            self.register_buffer('pe', self._get_positional_encoding(
                self.d_model, self.max_len))
        return x + self.pe[:, :x.size(1)].to(x.device).clone().detach()

class MultiheadAttention(nn.Module):
    def __init__(self, d_model:int, n_heads:int, dropout:float=0.1,
        d_k:int=None, d_v:int=None):
        super(MultiheadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        assert (d_model // n_heads) * n_heads == d_model, \
        "d_model must be divisible by n_heads"

        self.d_k = d_k if d_k is not None else d_model // n_heads
        self.d_v = d_v if d_v is not None else d_model // n_heads

        self.w_q = nn.Linear(d_model, n_heads*self.d_k, bias=False)
        self.w_k = nn.Linear(d_model, n_heads*self.d_k, bias=False)
        self.w_v = nn.Linear(d_model, n_heads*self.d_v, bias=False)
        self.fc = nn.Linear(n_heads*self.d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=self.d_k**0.5)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q:Tensor, k:Tensor, v:Tensor, mask:Tensor=None):
        # input shape: [b, l, c]
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        residual = q

        q = self.w_q(q).view(sz_b, len_q, self.n_heads, self.d_k)
        k = self.w_k(k).view(sz_b, len_k, self.n_heads, self.d_k)
        v = self.w_v(v).view(sz_b, len_v, self.n_heads, self.d_v)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)

        q, attn = self.attention(q, k, v, mask=mask)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q = self.layer_norm(q + residual)

        return q, attn
    

class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature:float, attn_dropout:float=0.):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q:Tensor, k:Tensor, v:Tensor, mask=None):
        attn = torch.matmul(q/self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask==0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn
    
def get_subsequent_mask(seq):
    len_s = seq.size(1)
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1))
    return subsequent_mask
