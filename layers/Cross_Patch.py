__all__ = ['Cross_Patch']

from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
import math
from math import sqrt


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)

    
def get_activation_fn(activation):
    if callable(activation): return activation()
    elif activation.lower() == "relu": return nn.ReLU()
    elif activation.lower() == "gelu": return nn.GELU()
    raise ValueError(f'{activation} is not available. You can use "relu", "gelu", or a callable') 


def positional_encoding(pe, learn_pe, q_len, d_model):
    # Positional encoding
    if pe == None:
        W_pos = torch.empty((q_len, d_model))   # pe = None and learn_pe = False can be used to measure impact of pe
        nn.init.uniform_(W_pos, -0.02, 0.02)
        learn_pe = False
    elif pe == 'zero':
        W_pos = torch.empty((q_len, 1))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'zeros':
        W_pos = torch.empty((q_len, d_model))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'normal' or pe == 'gauss':
        W_pos = torch.zeros((q_len, 1))
        torch.nn.init.normal_(W_pos, mean=0.0, std=0.1)
    elif pe == 'uniform':
        W_pos = torch.zeros((q_len, 1))
        nn.init.uniform_(W_pos, a=0.0, b=0.1)
    elif pe == 'lin1d': W_pos = Coord1dPosEncoding(q_len, exponential=False, normalize=True)
    elif pe == 'exp1d': W_pos = Coord1dPosEncoding(q_len, exponential=True, normalize=True)
    elif pe == 'lin2d': W_pos = Coord2dPosEncoding(q_len, d_model, exponential=False, normalize=True)
    elif pe == 'exp2d': W_pos = Coord2dPosEncoding(q_len, d_model, exponential=True, normalize=True)
    elif pe == 'sincos': W_pos = PositionalEncoding(q_len, d_model, normalize=True)
    else: raise ValueError(f"{pe} is not a valid pe (positional encoder. Available types: 'gauss'=='normal', \
        'zeros', 'zero', uniform', 'lin1d', 'exp1d', 'lin2d', 'exp2d', 'sincos', None.)")
    return nn.Parameter(W_pos, requires_grad=learn_pe)

 
class Cross_Patch(nn.Module):
    def __init__(self, c_in:int, context_window:int, target_window:int, patch_len:int, stride:int, win_size:int,
                 max_seq_len:Optional[int]=1024, d_model=128, n_heads=16, d_k:Optional[int]=None, d_v:Optional[int]=None,
                 d_ff:int=256, norm:str='BatchNorm', attn_dropout:float=0., dropout:float=0., act:str="gelu", key_padding_mask:bool='auto',
                 padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, 
                 pe:str='zeros', learn_pe:bool=True, fc_dropout:float=0., head_dropout = 0, padding_patch = None,
                 head_type = 'flatten', revin = True, affine = True, subtract_last = False, verbose:bool=False, **kwargs):

        super().__init__()

        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        patch_num = int((context_window - patch_len)/stride + 1) 
        if padding_patch == 'end': 
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride)) # 一维复制填充层。对一维张量样本通过复制边缘值填充扩展长度。左填充0个，右填充8个
            patch_num += 1 

        win_num1 = int((patch_num - win_size)/stride + 1)
        win_num2 = int((win_num1 - win_size)/stride + 1)
        
        self.encoder_shared = Cross_Patch_Encoder(c_in, patch_num=patch_num, patch_len=patch_len, stride=stride, win_size=win_size, max_seq_len=max_seq_len, 
                                d_model=d_model, n_heads=n_heads, pe=pe, learn_pe=learn_pe, d_k=d_k, d_v=d_v, d_ff=d_ff,
                                attn_dropout=attn_dropout, dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var,
                                attn_mask=attn_mask, res_attention=res_attention, verbose=verbose, **kwargs)

        # Head
        self.head_nf = d_model * win_num2 
        self.head = Prediction_Head(self.head_nf, target_window, head_dropout=head_dropout)
    
    def forward(self, z):    # z: [bs x nvars x seq_len]

        # Patch Partition
        if self.padding_patch == 'end':
            z = self.padding_patch_layer(z)
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)   # z: [bs x nvars x patch_num x patch_len]
        # dim表明想要切分的维度，size表明切分块的尺寸，step表明切分的步长
        
        # Cross-Patch Encoder(shared) 
        z = self.encoder_shared(z)      # z: [bs x nvars x d_model x patch_num]

        # Prediction Head
        z = self.head(z)                # z: [bs x nvars x target_window]
        
        return z


class Prediction_Head(nn.Module):
    def __init__(self, nf, target_window, head_dropout=0):
        super().__init__()
        
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window) 
        self.dropout = nn.Dropout(head_dropout)
            
    def forward(self, x):               # x: [bs x nvars x d_model x patch_num]

        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Cross_Patch_Encoder(nn.Module):  
    def __init__(self, c_in, patch_num, patch_len, stride, win_size, 
                 max_seq_len=1024, d_model=128, n_heads=16, d_k=None, d_v=None,
                 d_ff=256, norm='BatchNorm', attn_dropout=0.1, dropout=0.2, act="gelu",
                 key_padding_mask='auto', padding_var=None, attn_mask=None, res_attention=True, 
                 pe='zeros', learn_pe=True, verbose=False, **kwargs):

        super().__init__()

        # Embedding
        q_len = patch_num 
        self.W_P = nn.Linear(patch_len, d_model)    # projection of feature vectors onto a d-dim vector space
        self.seq_len = q_len

        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, q_len, d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Sliding Window
        self.win_size = win_size
        self.stride = stride

        # Conv Layer
        self.conv1 = nn.Conv1d(in_channels=self.win_size * d_model, out_channels=d_model, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=self.win_size * d_model, out_channels=d_model, kernel_size=1)

        # Transformer Encoder Layer
        self.layer1 = EncoderLayer(d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                    attn_dropout=attn_dropout, dropout=dropout,
                                    activation=act, res_attention=res_attention) 
        self.layer2 = EncoderLayer(d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                    attn_dropout=attn_dropout, dropout=dropout,
                                    activation=act, res_attention=res_attention) 
        self.layer3 = EncoderLayer(d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                    attn_dropout=attn_dropout, dropout=dropout,
                                    activation=act, res_attention=res_attention) 
        
    def forward(self, x, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None) -> Tensor:    # x: [bs x nvars x patch_len x patch_num]
        
        n_vars = x.shape[1]
        # stage1
        x = self.W_P(x)                                                         # x: [bs x nvars x patch_num x d_model]
        u = torch.reshape(x, (x.shape[0]*x.shape[1],x.shape[2],x.shape[3]))     # u: [bs * nvars x patch_num x d_model]
        u = self.dropout(u + self.W_pos)                                        # u: [bs * nvars x patch_num x d_model]
        z = self.layer1(u, key_padding_mask=key_padding_mask, attn_mask=attn_mask)

        # stage2
        # Sliding Window
        z = z.unfold(dimension=1, size=self.win_size, step=self.stride) 
        z = torch.reshape(z, (z.shape[0], z.shape[1], z.shape[2] * z.shape[3]))   

        # Feature Fusion
        z = z.permute(0,2,1)  
        z = self.conv1(z)
        z = z.permute(0,2,1) 
        z = self.layer2(z, key_padding_mask=key_padding_mask, attn_mask=attn_mask)

        # Sliding Window
        z = z.unfold(dimension=1, size=self.win_size, step=self.stride) 
        z = torch.reshape(z, (z.shape[0], z.shape[1], z.shape[2] * z.shape[3])) 
        
        # Feature Fusion
        z = z.permute(0,2,1)  
        z = self.conv2(z)
        z = z.permute(0,2,1) 
        z = self.layer3(z, key_padding_mask=key_padding_mask, attn_mask=attn_mask)

        z = torch.reshape(z, (-1,n_vars,z.shape[-2],z.shape[-1]))               # z: [bs x nvars x patch_num x d_model]
        z = z.permute(0,1,3,2)                                                  # z: [bs x nvars x d_model x patch_num]

        return z    
                        

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, d_ff=256, 
                 norm='BatchNorm', attn_dropout=0.1, dropout=0., bias=True, activation="gelu", res_attention=False):
        super().__init__()
        assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads if d_k is None else d_k 
        d_v = d_model // n_heads if d_v is None else d_v 

        # Multi-Head Attention
        self.self_attn = _MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout)   # attn_dropout=0, proj_dropout=0.2

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
            # Sequential(
            #   (0): Transpose()
            #   (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            #   (2): Transpose()
            # )
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias)) 

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)


    def forward(self, src:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None) -> Tensor:

        src2, attn = self.self_attn(src, src, src, key_padding_mask=key_padding_mask, attn_mask=attn_mask)

        ## Add & Norm
        src = src + self.dropout_attn(src2)     # Add: residual connection with residual dropout
        src = self.norm_attn(src)               # Layernorm

        ## Position-wise Feed-Forward
        src2 = self.ff(src)

        ## Add & Norm
        src = src + self.dropout_ffn(src2)      # Add: residual connection with residual dropout
        src = self.norm_ffn(src)

        return src


class _MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, attn_dropout=0.1, proj_dropout=0.2, qkv_bias=True, lsa=False):
        """
        Input shape:
            Q:       [batch_size (bs) x max_q_len x d_model]
            K, V:    [batch_size (bs) x q_len x d_model]
            mask:    [q_len x q_len]
        """
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v

        self.inner_correlation = _ScaledDotProductAttention(d_model, n_heads)

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias) 
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        # Poject output
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))


    def forward(self, Q:Tensor, K:Optional[Tensor]=None, V:Optional[Tensor]=None, prev:Optional[Tensor]=None,
                key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):

        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q

        # Linear (+ split in multiple heads)
        queries = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,2)           # [bs x n_heads x max_q_len x d_k]
        keys = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0,2,3,1)            # [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        values = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1,2)            # [bs x n_heads x q_len x d_v]

        output, attn = self.inner_correlation(
            queries,
            keys,
            values,
            attn_mask
        )

        # back to the original inputs dimensions
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v)  # output: [bs x q_len x n_heads * d_v]
        output = self.to_out(output)

        return output, attn
    

class _ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model, n_heads, attn_dropout=0.1, lsa=False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa

    def forward(self, q:Tensor, k:Tensor, v:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        '''
        Input shape:
            q               : [bs x n_heads x max_q_len x d_k]
            k               : [bs x n_heads x d_k x seq_len]
            v               : [bs x n_heads x seq_len x d_v]
            prev            : [bs x n_heads x q_len x seq_len]
            key_padding_mask: [bs x seq_len]
            attn_mask       : [1 x seq_len x seq_len]
        Output shape:
            output:  [bs x n_heads x q_len x d_v]
            attn   : [bs x n_heads x q_len x seq_len]
            scores : [bs x n_heads x q_len x seq_len]
        '''

        # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        attn_scores = torch.matmul(q, k) * self.scale                 # attn_scores : [bs x n_heads x max_q_len x q_len]

        # Add pre-softmax attention scores from the previous layer (optional)
        if prev is not None: attn_scores = attn_scores + prev

        # Attention mask (optional)
        if attn_mask is not None:                                     # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        # Key padding mask (optional)
        if key_padding_mask is not None:                              # mask with shape [bs x q_len] (only when max_w_len == q_len)
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)

        # normalize the attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)                 # attn_weights: [bs x n_heads x max_q_len x q_len]
        attn_weights = self.attn_dropout(attn_weights)

        # compute the new values given the attention weights
        output = torch.matmul(attn_weights, v)                        # output: [bs x n_heads x max_q_len x d_v]

        return output, attn_weights