__all__ = ['CPAT']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from layers.Cross_Patch import Cross_Patch


class Model(nn.Module):
    def __init__(self, configs, max_seq_len:Optional[int]=1024, d_k:Optional[int]=None, d_v:Optional[int]=None, norm:str='BatchNorm', attn_dropout:float=0.,
                 act:str="gelu", key_padding_mask:bool='auto',padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, 
                 pre_norm:bool=False, store_attn:bool=False, pe:str='zeros', learn_pe:bool=True, pretrain_head:bool=False, head_type = 'flatten', verbose:bool=False, **kwargs):
        
        super().__init__()
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len

        # load parameters
        c_in = configs.enc_in 
        context_window = configs.seq_len 
        target_window = configs.pred_len 
        label_len = configs.label_len
        n_heads = configs.n_heads 
        d_model = configs.d_model 
        d_ff = configs.d_ff 
        dropout = configs.dropout 
        fc_dropout = configs.fc_dropout 
        head_dropout = configs.head_dropout 
    
        patch_len = configs.patch_len 
        stride = configs.stride 
        win_size = configs.win_size
        padding_patch = configs.padding_patch 
        revin = configs.revin 
        affine = configs.affine 
        subtract_last = configs.subtract_last 

        # model
        self.model = Cross_Patch(c_in=c_in, context_window=context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                win_size=win_size, max_seq_len=max_seq_len, d_model=d_model, label_len=label_len, 
                                n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                attn_mask=attn_mask, res_attention=res_attention, pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, 
                                head_dropout=head_dropout, padding_patch=padding_patch, head_type=head_type, revin=revin, affine=affine,
                                subtract_last=subtract_last, verbose=verbose, **kwargs)
        

    def forward(self, x):           # x: [Batch, Input length, Channel]

        # Normalization
        mean = x.mean(1, keepdim=True).detach() # B x 1 x E
        x = x - mean
        std = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach() # B x 1 x E
        x = x / std
    
        x = x.permute(0,2,1)    # x: [Batch, Channel, Input length]
        x = self.model(x)
        x = x.permute(0,2,1)    # x: [Batch, Input length, Channel]

        # De-normalization
        x = x * std + mean
        return x[:, -self.pred_len:, :]