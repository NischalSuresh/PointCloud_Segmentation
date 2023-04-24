import numpy as np
import copy
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import pdb

class AttentionLayer(nn.Module):

    def __init__(self, embed_dim, dropout=0.1):       
        super().__init__()
        self.embed_dim = embed_dim
        self.query_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.key_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.value_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, attn_mask=None):
        N, S, D = query.shape
        N, T, D = value.shape
        query = self.query_proj(query)
        key = self.key_proj(key)
        value = self.value_proj(value)
        dot_product = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.embed_dim)
        if attn_mask is not None:
            additive_mask = (1 - attn_mask) * -1e9
            dot_product += additive_mask   
        y = torch.matmul(self.dropout(F.softmax(dot_product, dim=-1)), value)
        return y  

class MultiHeadAttentionLayer(AttentionLayer):

    def __init__(self, embed_dim, num_heads, dropout=0.1):     
        super().__init__(embed_dim, dropout)
        self.num_heads = num_heads
        self.head_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, query, key, value, attn_mask=None):
        H = self.num_heads
        N, S, D = query.shape
        N, T, D = value.shape
        query = self.query_proj(query).view(N, S, H, D // H).transpose(1,2)
        key = self.key_proj(key).view(N, T, H, D // H).transpose(1,2)
        value = self.value_proj(value).view(N, T, H, D // H).transpose(1,2)
        dot_product = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.embed_dim / H)
        if attn_mask is not None:
            additive_mask = (1 - attn_mask) * -1e9
            dot_product += additive_mask.to(query.device)      
        y = torch.matmul(self.dropout(F.softmax(dot_product, dim=-1)), value)
        output = self.head_proj(y.transpose(1,2).reshape(N, S, D))
        return output

class PositionalEncoding(nn.Module):

    def __init__(self, embed_dim, dropout=0.1, max_len=5000):
        super().__init__()
        self.encoding = nn.Embedding(max_len, embed_dim)
        self.dropout = nn.Dropout(dropout)
      
    def forward(self, x):
        N, S, D = x.shape
        embeddings = torch.arange(S).repeat(N, 1).to(x.device)
        output = x + self.encoding(embeddings)
        output = self.dropout(output)
        return output

class FeedForwardBlock(nn.Module):
    '''
    Input: Features with embedding size emb_dim
    expands and contracts the emd_dim
    Output: Features with embedding size emb_dim
    '''

    def __init__(self, input_dim, dim_feedforward=2048, dropout=0.1 ):
        super().__init__()
        lin1 = nn.Linear(input_dim, dim_feedforward)
        lin2 = nn.Linear(dim_feedforward, input_dim)
        self.mlp = nn.Sequential(
            lin1, nn.ReLU(), nn.Dropout(dropout), lin2
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(input_dim)
       

    def forward(self, seq):
        mlp_out = self.mlp(seq)
        x_resid = self.dropout(mlp_out)
        x_out = x_resid + seq
        return self.norm(x_out)

class CrossAttentionBlock(nn.Module):
    '''
      Takes as input the key from zxy, query and value from depth
      Returns the encoded feature
    '''
    def __init__(self, input_dim, num_heads, dropout=0.1):
        super().__init__()
        self.cross_attn = MultiHeadAttentionLayer(input_dim, num_heads)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(input_dim)
       
    def forward(self, seq, cond):
        attn_out = self.cross_attn(seq, cond, cond)
        x_resid = self.dropout(attn_out)
        x_out = x_resid + seq
        return self.norm(x_out)

class Attend(nn.Module):
  '''
  Complete Cross Attention block which performs both cross attention and feed forward
  Inputs: Two spatial inputs of 4 dimensions and of equal sizes
  Outputs: Features of same dimension as that of input with 32 channels to suit FPS Net
  '''

  def __init__(self, Height, Width):
    super().__init__()
    self.patch_dim = 16
    self.num_patches = (Height*Width)/(16*16)
    self.embed_dim = 32*4
    self.dim_feedforward = 128
    self.posEncode = PositionalEncoding(embed_dim = self.embed_dim, dropout = 0.1, max_len = int(self.num_patches))
    self.patch_embedding_1 = nn.Linear(3 * self.patch_dim * self.patch_dim, self.embed_dim)
    self.patch_embedding_2 = nn.Linear(self.patch_dim * self.patch_dim, self.embed_dim)
    self.layers = nn.ModuleList([])
    self.upsample = nn.UpsamplingBilinear2d(size= (64,2048))
    self.conv = nn.Conv2d(self.embed_dim, 32, 1, 1)

    num_layers = 6
    for _ in range(num_layers):
      self.layers.append(nn.ModuleList([
          CrossAttentionBlock(input_dim = self.embed_dim, num_heads = 4, dropout = 0.1),
          FeedForwardBlock(self.embed_dim, self.dim_feedforward, dropout=0.1)
      ]))
      #self.conv = nn.Conv2d()

  def patchify(self, images):
        '''
            Inputs: images: a FloatTensor of shape (N, C(3/1), H, W) giving a minibatch of images
            Returns: patches: a FloatTensor of shape (N, num_patches, patch_dim x patch_dim x 3) giving a minibatch of patches
        '''    
        N = images.shape[0]
        images = images.transpose(1, 3)
        images = images.reshape(N, int(self.num_patches), -1)
        return images

  def forward(self, seq, cond):
    B, _, H, W = seq.shape
    seq_patch = self.patchify(seq)
    cond_patch = self.patchify(cond)
    seq_emb = self.patch_embedding_1(seq_patch)
    cond_emb = self.patch_embedding_2(cond_patch)
    seq = self.posEncode(seq_emb)
    cond = self.posEncode(cond_emb)
    out = seq
    for cAttnBlk, ffn in self.layers:
      out = cAttnBlk(out, cond)
      out = ffn(out)
    out = out.transpose(-1,-2).reshape(B,self.embed_dim,int(H/self.patch_dim),int(W/self.patch_dim))
    out = self.upsample(out)
    out = F.relu(self.conv(out))
    return out