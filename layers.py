import math

import torch
# torch.backends.cudnn.enabled = False
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)

    
class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)
                
        self.a = nn.Parameter(torch.zeros(size=(1, 2*out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, input, adj):
        dv = 'cuda' if input.is_cuda else 'cpu'

        N = input.size()[0]
        edge = adj.nonzero().t()

        h = torch.mm(input, self.W)
        # h: N x out
        assert not torch.isnan(h).any()

        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        # edge: 2*D x E

        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N,1), device=dv))
        # e_rowsum: N x 1

        edge_e = self.dropout(edge_e)
        # edge_e: E

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out
        
        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()

        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class MultiHeadAttention(Module):
    """
    Simple attention layer

    """
    def __init__(self, hidden_size, num_heads, dropout_rate):
        super().__init__()
        self.num_heads = num_heads
        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5
        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.att_dropout = nn.Dropout(dropout_rate)
        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

    def forward(self, batch, attn_bias = None):
        orig_batch_size = batch.size()
        batch_size = batch.size(0)
        d_k = self.att_size
        d_v = self.att_size

        q = self.linear_q(batch).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_k(batch).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(batch).view(batch_size, -1, self.num_heads, d_v)

        q = q.transpose(1, 2)                  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)                  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]  QK/dk**0.5
        zero_vec = -9e15*torch.ones_like(x)
        x = torch.where(x == 0, zero_vec, x)
        if attn_bias is not None:
            x = x + attn_bias
        

        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.num_heads * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_batch_size
        return x



# ----------------------------------------------------------------------------

class HCoN(Module):

    def __init__(self, in_features, out_features, bias=True,device='cuda:0',args=None):
        super(HCoN, self).__init__()
        self.dv=device
        self.in_features = in_features
        self.out_features = out_features

        # ---------------------------------------
        self.Qv=nn.Linear(in_features,in_features).to(device)
        self.Qe=nn.Linear(in_features,in_features).to(device)
        self.Pe=nn.Linear(in_features,in_features).to(device)
        self.Pv=nn.Linear(in_features,in_features).to(device)
        self.alpha=args.alpha
        # self.alpha=self.alpha.to("cuda")
        self.beta=args.beta
        # self.beta=self.beta.to("cuda")
        # ---------------------------------------

        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()


    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, y, x,hg):
        # self.hcon1(x, y, hg)
        X=(hg.DHWD) @ (self.alpha * ((hg.HD) @ self.Qv(x)) + (1-self.alpha) * self.Qe(y) )
        # 5030*64                                        621*64                  621*64


        Y=(hg.BHUB) @ (self.beta  * ((hg.HB) @ self.Pe(y)) + (1-self.beta ) * self.Pv(x) )

        Y0= hg.HD @ self.Qv(x)
        de=hg.D_e.to(self.dv)
        Y1=de @ Y0
        X1= hg.DHWD @ Y0

        X2= hg.HB @ self.Pe(y)
        bv=hg.D_v.to(self.dv)
        X3=bv @ X2
        Y3=hg.BHUB @ X2

        return X,Y


    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


# ------------------------------------------------------------------------------------------------------------------------
class TimeEncode(torch.nn.Module):
  # Time Encoding proposed by TGAT
  def __init__(self, dimension):
    super(TimeEncode, self).__init__()

    self.dimension = dimension
    self.w = torch.nn.Linear(1, dimension)

    self.w.weight = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 1.5, dimension)))
                                       .float().reshape(dimension, -1))
    self.w.bias = torch.nn.Parameter(torch.zeros(dimension).float())

  def forward(self, t):
    # t has shape [batch_size, seq_len]
    # Add dimension at the end to apply linear layer --> [batch_size, seq_len, 1]
    t = torch.log(t + 1)
    t = t.unsqueeze(dim=1)

    # output has shape [batch_size, seq_len, dimension]
    output = torch.cos(self.w(t))

    return output

class MultiHeadAttention1(Module):
    """
    Simple attention layer

    """

    def __init__(self, hidden_size, num_heads, dropout_rate,device):
        super().__init__()
        self.device=device
        self.num_heads = num_heads
        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size + 64, num_heads * att_size) # 64 32 48
        self.linear_k = nn.Linear(hidden_size + 64, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size + 64, num_heads * att_size)

        self.att_dropout = nn.Dropout(dropout_rate)
        self.output_layer = nn.Linear(num_heads * att_size, hidden_size) #81-82-83
        self.time_encode = TimeEncode(64)

    def forward(self, batch, attn_bias=None):
        # [nodes_num, snap_len, hidden_size]
        ans = []
        orig_batch_size = batch.size()
        for i in range(10):
            x_slice = x_slice = batch[:, i, :]
            t = torch.tensor([10 - i]).to(self.device)
            t_encode = self.time_encode(t)
            t_encode = t_encode.expand(batch.size(0), -1)
            x_slice = torch.cat((x_slice, t_encode), dim=1)
            ans.append(x_slice)
        batch = torch.stack(ans, dim=1)

        batch_size = batch.size(0)
        d_k = self.att_size
        d_v = self.att_size

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(batch).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_k(batch).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(batch).view(batch_size, -1, self.num_heads, d_v)

        q = q.transpose(1, 2)  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        zero_vec = -9e15 * torch.ones_like(x)
        x = torch.where(x == 0, zero_vec, x)
        if attn_bias is not None:
            x = x + attn_bias

        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.num_heads * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_batch_size
        return x


class HiLo(nn.Module):
    """
    HiLo Attention

    Paper: Fast Vision Transformers with HiLo Attention
    Link: https://arxiv.org/abs/2205.13213
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., window_size=2, alpha=0.5):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        head_dim = int(dim/num_heads)
        self.dim = dim

        # self-attention heads in Lo-Fi
        self.l_heads = int(num_heads * alpha)
        # token dimension in Lo-Fi
        self.l_dim = self.l_heads * head_dim

        # self-attention heads in Hi-Fi
        self.h_heads = num_heads - self.l_heads
        # token dimension in Hi-Fi
        self.h_dim = self.h_heads * head_dim

        # local window size. The `s` in our paper.
        self.ws = window_size

        if self.ws == 1:
            # ws == 1 is equal to a standard multi-head self-attention
            self.h_heads = 0
            self.h_dim = 0
            self.l_heads = num_heads
            self.l_dim = dim

        self.scale = qk_scale or head_dim ** -0.5

        # Low frequence attention (Lo-Fi)
        if self.l_heads > 0:
            if self.ws != 1:
                self.sr = nn.AvgPool2d(kernel_size=(2, 3),
                                        stride=(2, 2),
                                        padding=(0, 0))
            self.l_q = nn.Linear(self.dim+64, self.l_dim, bias=qkv_bias)   # q/kv/ mlp？/
            self.l_kv = nn.Linear(self.dim+64, self.l_dim * 2, bias=qkv_bias)
            self.l_proj = nn.Linear(self.l_dim, self.l_dim)

        # High frequence attention (Hi-Fi)
        if self.h_heads > 0:
            self.h_qkv = nn.Linear(self.dim+64, self.h_dim * 3, bias=qkv_bias) #  qkv
            self.h_proj = nn.Linear(self.h_dim, self.h_dim) # mlp?

        self.time_encode = TimeEncode(64)

    def hifi(self, x):
        B, N, C = x.shape
        # h_group, w_group = H // self.ws, W // self.ws

        # total_groups = h_group * w_group
        total_groups = N // self.ws


        x = x.reshape(B, total_groups, self.ws, C)

        qkv = self.h_qkv(x).reshape(B, total_groups, -1, 3, self.h_heads, self.h_dim // self.h_heads).permute(3, 0, 1, 4, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B, hw, n_head, ws*ws, head_dim

        attn = (q @ k.transpose(-2, -1)) * self.scale  # B, hw, n_head, ws*ws, ws*ws
        attn = attn.softmax(dim=-1)
        attn = (attn @ v).transpose(2, 3).reshape(B, total_groups, self.ws, self.h_dim)
        x = attn.transpose(2, 3).reshape(B,total_groups*self.ws, self.h_dim)

        x = self.h_proj(x)
        return x

    def lofi(self, x):
        B, N, C = x.shape

        q = self.l_q(x).reshape(B, N, self.l_heads, self.l_dim // self.l_heads).permute(0, 2, 1, 3)

        if self.ws > 1:
            x_ = x.permute(0, 2,1)
            x_ = self.sr(x_)
            x_ = x_.reshape(B, C, -1).permute(0, 2, 1)
            kv = self.l_kv(x_).reshape(B, -1, 2, self.l_heads, self.l_dim // self.l_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.l_kv(x).reshape(B, -1, 2, self.l_heads, self.l_dim // self.l_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, self.l_dim)
        x = self.l_proj(x)
        return x

    def forward(self, x, H, W):
        ans = []
        orig_batch_size = x.size()
        for i in range(10):
            x_slice = x_slice = x[:, i, :]
            t = torch.tensor([10 - i]).to('cuda:0')
            t_encode = self.time_encode(t)
            t_encode = t_encode.expand(x.size(0), -1)
            x_slice = torch.cat((x_slice, t_encode), dim=1)
            ans.append(x_slice)
        x = torch.stack(ans, dim=1)

        B, N, C = x.shape

        # x = x.reshape(B, H, W, C)   # 64，14，14，384

        if self.h_heads == 0:
            x = self.lofi(x)
            return x.reshape(B, N, C)

        if self.l_heads == 0:
            x = self.hifi(x)
            return x.reshape(B, N, C)

        hifi_out = self.hifi(x)
        lofi_out = self.lofi(x)

        x = torch.cat((hifi_out, lofi_out), dim=-1)
        x = x.reshape(B, N, -1)

        return x

    def flops(self, H, W):
        # pad the feature map when the height and width cannot be divided by window size
        Hp = self.ws * math.ceil(H / self.ws)
        Wp = self.ws * math.ceil(W / self.ws)

        Np = Hp * Wp

        # For Hi-Fi
        # qkv
        hifi_flops = Np * self.dim * self.h_dim * 3
        nW = (Hp // self.ws) * (Wp // self.ws)
        window_len = self.ws * self.ws
        # q @ k and attn @ v
        window_flops = window_len * window_len * self.h_dim * 2
        hifi_flops += nW * window_flops
        # projection
        hifi_flops += Np * self.h_dim * self.h_dim

        # for Lo-Fi
        # q
        lofi_flops = Np * self.dim * self.l_dim
        kv_len = (Hp // self.ws) * (Wp // self.ws)
        # k, v
        lofi_flops += kv_len * self.dim * self.l_dim * 2
        # q @ k and attn @ v
        lofi_flops += Np * self.l_dim * kv_len * 2
        # projection
        lofi_flops += Np * self.l_dim * self.l_dim

        return hifi_flops + lofi_flops
