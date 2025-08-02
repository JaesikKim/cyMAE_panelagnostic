import math
from functools import partial
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import drop_path

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement 
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x

    
# sin-cos position encoding
# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py#L31
def get_sinusoid_encoding_table(n_position, d_hid): 
    ''' Sinusoid position encoding table ''' 
    # TODO: make it with torch instead of numpy 
    def get_position_angle_vec(position): 
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)] 

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)]) 
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i 
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i+1 

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)



from collections import namedtuple
from dataclasses import dataclass
from functools import wraps

import torch
import torch.nn.functional as F
from einops import rearrange

from torch import Tensor, einsum, nn

from abc import abstractmethod


class BaseAttention(nn.Module):
    @abstractmethod
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x, context=None, mask=None):
        pass


# constants

EfficientAttentionConfig = namedtuple(
    "EfficientAttentionConfig",
    ["enable_flash", "enable_math", "enable_mem_efficient"],
)

# helpers


def exists(val):
    return val is not None


def once(fn):
    called = False

    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)

    return inner


print_once = once(print)

# main class


@dataclass
class Intermediates:
    """
    Dataclass to store intermediate tensors during attention computation.

    Args:
        qk_similarities (torch.Tensor): Tensor storing the similarities between query and key.
        pre_softmax_attn (torch.Tensor): Tensor storing the attention weights before softmax.
        post_softmax_attn (torch.Tensor): Tensor storing the attention weights after softmax.

    Methods:
        to_tuple(): Convert the Intermediates object to a tuple.

    """

    qk_similarities: Tensor = None
    pre_softmax_attn: Tensor = None
    post_softmax_attn: Tensor = None

    def to_tuple(self):
        """
        Convert the Intermediates object to a tuple.

        Returns:
            tuple: Tuple representation of the Intermediates object.
        """
        return (
            self.qk_similarities,
            self.pre_softmax_attn,
            self.post_softmax_attn,
        )


# class FlashAttention(BaseAttention):
#     def __init__(
#         self, causal: bool = False, dropout: float = 0.0, flash: bool = True
#     ):
#         """
#         FlashAttention module that performs attention computation.

#         Args:
#             causal (bool): Whether to apply causal masking (default: False).
#             dropout (float): Dropout probability (default: 0.).
#             flash (bool): Whether to use flash attention (default: True).

#         """
#         super().__init__()

#         self.dropout = dropout
#         self.attn_dropout = nn.Dropout(dropout)

#         self.causal = causal
#         self.flash = flash
#         # determine efficient attention configs for cuda and cpu

#         self.cpu_config = EfficientAttentionConfig(True, True, True)
#         self.cuda_config = None

#         if not torch.cuda.is_available() or not flash:
#             return

#         device_properties = torch.cuda.get_device_properties(
#             torch.device("cuda")
#         )

#         if device_properties.major == 8 and device_properties.minor == 0:
#             print_once(
#                 "A100 GPU detected, using flash attention if input tensor is on"
#                 " cuda"
#             )
#             self.cuda_config = EfficientAttentionConfig(True, False, False)
#         else:
#             print_once(
#                 "Non-A100 GPU detected, using math or mem efficient attention"
#                 " if input tensor is on cuda"
#             )
#             self.cuda_config = EfficientAttentionConfig(False, True, True)

#     def get_mask(self, i, j, device):
#         """
#         Generate a mask for attention computation.

#         Args:
#             i (int): Length of the query sequence.
#             j (int): Length of the key sequence.
#             device (torch.device): Device to place the mask tensor.

#         Returns:
#             torch.Tensor: Mask tensor of shape (i, j).

#         """
#         return torch.ones((i, j), device=device, dtype=torch.bool).triu(
#             j - i + 1
#         )

#     def flash_attn(self, q, k, v, mask=None, attn_bias=None):
#         """
#         Perform flash attention computation.

#         Args:
#             q (torch.Tensor): Query tensor of shape (batch, heads, q_len, dim).
#             k (torch.Tensor): Key tensor of shape (batch, heads, k_len, dim).
#             v (torch.Tensor): Value tensor of shape (batch, heads, v_len, dim).
#             mask (torch.Tensor): Mask tensor of shape (batch, heads, q_len, k_len) (default: None).
#             attn_bias (torch.Tensor): Attention bias tensor of shape (batch, heads, q_len, k_len) (default: None).

#         Returns:
#             torch.Tensor: Output tensor of shape (batch, heads, q_len, dim).

#         """
#         batch, heads, q_len, _, k_len, is_cuda, device = (
#             *q.shape,
#             k.shape[-2],
#             q.is_cuda,
#             q.device,
#         )

#         # Recommended for multi-query single-key-value attention by Tri Dao
#         # kv shape torch.Size([1, 512, 64]) -> torch.Size([1, 8, 512, 64])

#         if k.ndim == 3:
#             k = rearrange(k, "b ... -> b 1 ...").expand_as(q)

#         if v.ndim == 3:
#             v = rearrange(v, "b ... -> b 1 ...").expand_as(q)

#         # handle scale - by default they scale by dim_head ** -0.5, but need to take care if using cosine sim attention
#         # Check if mask exists and expand to compatible shape
#         # The mask is B L, so it would have to be expanded to B H N L

#         causal = self.causal

#         if exists(mask):
#             assert mask.ndim == 4
#             mask = mask.expand(batch, heads, q_len, k_len)

#             # manually handle causal mask, if another mask was given

#             if causal:
#                 causal_mask = self.create_causal_mask(
#                     q_len, k_len, device=device
#                 )
#                 mask = mask & ~causal_mask
#                 causal = False

#         # handle alibi positional bias
#         # convert from bool to float

#         if exists(attn_bias):
#             attn_bias = rearrange(attn_bias, "h i j -> 1 h i j").expand(
#                 batch, heads, -1, -1
#             )

#             # if mask given, the mask would already contain the causal mask from above logic
#             # otherwise, if no mask given but still causal, mask out alibi
#             # positional bias to a large negative number

#             mask_value = -torch.finfo(q.dtype).max

#             if exists(mask):
#                 attn_bias = attn_bias.masked_fill(~mask, mask_value // 2)
#             elif causal:
#                 causal_mask = self.create_causal_mask(
#                     q_len, k_len, device=device
#                 )
#                 attn_bias = attn_bias.masked_fill(causal_mask, mask_value // 2)
#                 causal = False

#             # scaled_dot_product_attention handles attn_mask either as bool or additive bias
#             # make it an additive bias here

#             mask = attn_bias

#         # Check if there is a compatible device for flash attention

#         config = self.cuda_config if is_cuda else self.cpu_config

#         # pytorch 2.0 flash attn: q, k, v, mask, dropout, causal, softmax_scale

#         with torch.backends.cuda.sdp_kernel(**config._asdict()):
#             out = F.scaled_dot_product_attention(
#                 q,
#                 k,
#                 v,
#                 attn_mask=mask,
#                 dropout_p=self.dropout if self.training else 0.0,
#                 is_causal=causal,
#             )

#             return out

#     def forward(self, q, k, v, mask=None, attn_bias=None):
#         """
#         Perform attention computation.

#         einstein notation
#         b - batch
#         h - heads
#         n, i, j - sequence length (base sequence length, source, target)
#         d - feature dimension

#         Args:
#             q (torch.Tensor): Query tensor of shape (batch, heads, q_len, dim).
#             k (torch.Tensor): Key tensor of shape (batch, heads, k_len, dim).
#             v (torch.Tensor): Value tensor of shape (batch, heads, v_len, dim).
#             mask (torch.Tensor): Mask tensor of shape (batch, heads, q_len, k_len) (default: None).
#             attn_bias (torch.Tensor): Attention bias tensor of shape (batch, heads, q_len, k_len) (default: None).

#         Returns:
#             torch.Tensor: Output tensor of shape (batch, heads, q_len, dim).

#         """

#         q_len, k_len, device = q.shape[-2], k.shape[-2], q.device

#         scale = q.shape[-1] ** -0.5

#         kv_einsum_eq = "b j d" if k.ndim == 3 else "b h j d"

#         if self.flash:
#             return self.flash_attn(q, k, v, mask=mask, attn_bias=attn_bias)

#         # similarity

#         sim = einsum(f"b h i d, {kv_einsum_eq} -> b h i j", q, k) * scale

#         # attention bias

#         if exists(attn_bias):
#             sim = sim + attn_bias

#         # causal mask

#         if self.causal:
#             causal_mask = self.get_mask(q_len, k_len, device)
#             sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

#         # attention

#         attn = sim.softmax(dim=-1)
#         attn = self.attn_dropout(attn)

#         # aggregate values

#         out = einsum(f"b h i j, {kv_einsum_eq} -> b h i d", attn, v)

#         return out




class FlashAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        attn_head_dim=None,
        causal=False
    ):
        """
        flash attention 모듈.
        
        Args:
            dim (int): 입력 feature 차원.
            num_heads (int): multi-head attention의 head 개수.
            qkv_bias (bool): qkv 프로젝션에 bias를 추가할지 여부.
            qk_scale (float, optional): Scaling factor (없으면 head_dim**-0.5 사용).
            attn_drop (float): attention dropout 확률.
            proj_drop (float): 최종 projection dropout 확률.
            attn_head_dim (int, optional): 각 head의 차원 (없으면 dim//num_heads 사용).
            causal (bool): causal attention 적용 여부.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = (dim // num_heads) if attn_head_dim is None else attn_head_dim
        self.scale = qk_scale or head_dim ** -0.5
        all_head_dim = head_dim * num_heads

        # qkv 프로젝션 (bias는 따로 처리)
        self.qkv = nn.Linear(dim, 3 * all_head_dim, bias=False)
        if qkv_bias:
            # q에만 bias 적용하고, k는 0, v에 대해서 별도 bias 적용
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.causal = causal

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 입력 텐서 (B, N, C)
        Returns:
            torch.Tensor: 출력 텐서 (B, N, C)
        """
        B, N, C = x.shape

        # qkv 프로젝션 bias 처리
        qkv_bias = None
        if self.q_bias is not None:
            # q_bias와 v_bias를 사용하고, k는 bias 없이 0값을 사용합니다.
            qkv_bias = torch.cat([
                self.q_bias, 
                torch.zeros_like(self.v_bias, requires_grad=False), 
                self.v_bias
            ])

        # (B, N, 3 * all_head_dim)
        qkv = F.linear(x, self.qkv.weight, bias=qkv_bias)
        # reshape: (B, N, 3, num_heads, head_dim) -> permute -> (3, B, num_heads, N, head_dim)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # 각 shape: (B, num_heads, N, head_dim)

        # q 스케일링
        q = q * self.scale

        # flash attention 연산: pytorch 내장 scaled_dot_product_attention 사용
        # dropout은 self.attn_drop의 확률(self.training 여부에 따라 적용)
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            is_causal=self.causal
        )
        # out shape: (B, num_heads, N, head_dim) -> 재배열하여 (B, N, all_head_dim)
        out = out.transpose(1, 2).reshape(B, N, -1)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out



class FlashAttnBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = FlashAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x