"""
Native port of Roblox/cube's shape tokenizer decode path (OneDAutoEncoder).

Reference: https://github.com/Roblox/cube  (cube3d/model/autoencoder/*).

Only the DECODE path is ported (token IDs -> latents -> occupancy grid -> mesh);
the point-cloud encoder is not needed for text-to-3D generation. Encoder weights in
the checkpoint are loaded with strict=False and ignored.

Module/parameter names mirror upstream so the checkpoint loads directly:
  embedder.weight
  bottleneck.block.{codebook, cb_weight, cb_bias, c_in, c_x, c_out, ...}
  decoder.{positional_encodings, blocks.N...}
  occupancy_decoder.{query_in, attn_out, ln_f, c_head}
"""

import logging
import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.ops
ops = comfy.ops.disable_weight_init


# ---------------------------------------------------------------------------
# Norms
# ---------------------------------------------------------------------------

class CubeLayerNorm(nn.Module):
    """LayerNorm upcasting to fp32. affine=False by default (no params)."""

    def __init__(self, dim, eps=1e-6, elementwise_affine=False, dtype=None, device=None):
        super().__init__()
        self.dim = (dim,)
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim, dtype=dtype, device=device))
            self.bias = nn.Parameter(torch.zeros(dim, dtype=dtype, device=device))
        else:
            self.weight = None
            self.bias = None

    def forward(self, x):
        w = self.weight.float() if self.weight is not None else None
        b = self.bias.float() if self.bias is not None else None
        y = F.layer_norm(x.float(), self.dim, w, b, self.eps)
        return y.type_as(x)


class CubeRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5, elementwise_affine=True, dtype=None, device=None):
        super().__init__()
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim, dtype=dtype, device=device))
        else:
            self.register_buffer("weight", torch.ones(dim), persistent=False)

    def forward(self, x):
        xf = x.float()
        out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (out * self.weight.float()).type_as(x)


# ---------------------------------------------------------------------------
# Fourier embedder
# ---------------------------------------------------------------------------

class PhaseModulatedFourierEmbedder(nn.Module):
    def __init__(self, num_freqs, input_dim=3, dtype=None, device=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(input_dim, num_freqs, dtype=dtype, device=device))
        carrier = (num_freqs / 8) ** torch.linspace(1, 0, num_freqs)
        carrier = (carrier + torch.linspace(0, 1, num_freqs)) * 2 * math.pi
        self.register_buffer("carrier", carrier, persistent=False)
        self.out_dim = input_dim * (num_freqs * 2 + 1)

    def forward(self, x):
        m = x.float().unsqueeze(-1)
        w = self.weight.float()
        carrier = self.carrier.float()
        fm = (m * w).view(*x.shape[:-1], -1)
        pm = (m * 0.5 * math.pi + carrier).view(*x.shape[:-1], -1)
        return torch.cat([x, fm.cos() + pm.cos(), fm.sin() + pm.sin()], dim=-1).type_as(x)


# ---------------------------------------------------------------------------
# Attention building blocks
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, embed_dim, hidden_dim, bias=True, dtype=None, device=None):
        super().__init__()
        self.up_proj = ops.Linear(embed_dim, hidden_dim, bias=bias, dtype=dtype, device=device)
        self.down_proj = ops.Linear(hidden_dim, embed_dim, bias=bias, dtype=dtype, device=device)
        self.act_fn = nn.GELU(approximate="none")

    def forward(self, x):
        return self.down_proj(self.act_fn(self.up_proj(x)))


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, bias=True, eps=1e-6, dtype=None, device=None):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.c_qk = ops.Linear(embed_dim, 2 * embed_dim, bias=bias, dtype=dtype, device=device)
        self.c_v = ops.Linear(embed_dim, embed_dim, bias=bias, dtype=dtype, device=device)
        self.c_proj = ops.Linear(embed_dim, embed_dim, bias=bias, dtype=dtype, device=device)
        self.q_norm = CubeRMSNorm(head_dim, dtype=dtype, device=device)
        self.k_norm = CubeRMSNorm(head_dim, dtype=dtype, device=device)

    def forward(self, x, attn_mask=None, is_causal=False):
        b, l, d = x.shape
        q, k = self.c_qk(x).chunk(2, dim=-1)
        v = self.c_v(x)
        q = self.q_norm(q.view(b, l, self.num_heads, -1).transpose(1, 2))
        k = self.k_norm(k.view(b, l, self.num_heads, -1).transpose(1, 2))
        v = v.view(b, l, self.num_heads, -1).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0,
                                           is_causal=is_causal and attn_mask is None)
        y = y.transpose(1, 2).contiguous().view(b, l, d)
        return self.c_proj(y)


class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, q_dim=None, kv_dim=None, bias=True, dtype=None, device=None):
        super().__init__()
        assert embed_dim % num_heads == 0
        q_dim = q_dim or embed_dim
        kv_dim = kv_dim or embed_dim
        self.c_q = ops.Linear(q_dim, embed_dim, bias=bias, dtype=dtype, device=device)
        self.c_k = ops.Linear(kv_dim, embed_dim, bias=bias, dtype=dtype, device=device)
        self.c_v = ops.Linear(kv_dim, embed_dim, bias=bias, dtype=dtype, device=device)
        self.c_proj = ops.Linear(embed_dim, embed_dim, bias=bias, dtype=dtype, device=device)
        self.num_heads = num_heads

    def forward(self, x, c, attn_mask=None):
        q, k, v = self.c_q(x), self.c_k(c), self.c_v(c)
        b, l, d = q.shape
        s = k.shape[1]
        q = q.view(b, l, self.num_heads, -1).transpose(1, 2)
        k = k.view(b, s, self.num_heads, -1).transpose(1, 2)
        v = v.view(b, s, self.num_heads, -1).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0)
        y = y.transpose(1, 2).contiguous().view(b, l, d)
        return self.c_proj(y)


class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, bias=True, eps=1e-6, dtype=None, device=None):
        super().__init__()
        self.ln_1 = CubeLayerNorm(embed_dim, eps=eps)
        self.attn = SelfAttention(embed_dim, num_heads, bias=bias, eps=eps, dtype=dtype, device=device)
        self.ln_2 = CubeLayerNorm(embed_dim, eps=eps)
        self.mlp = MLP(embed_dim, embed_dim * 4, bias=bias, dtype=dtype, device=device)

    def forward(self, x, attn_mask=None, is_causal=False):
        x = x + self.attn(self.ln_1(x), attn_mask=attn_mask, is_causal=is_causal)
        x = x + self.mlp(self.ln_2(x))
        return x


class EncoderCrossAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, q_dim=None, kv_dim=None, bias=True, eps=1e-6, dtype=None, device=None):
        super().__init__()
        q_dim = q_dim or embed_dim
        kv_dim = kv_dim or embed_dim
        self.attn = CrossAttention(embed_dim, num_heads, q_dim=q_dim, kv_dim=kv_dim, bias=bias, dtype=dtype, device=device)
        self.ln_1 = CubeLayerNorm(q_dim, eps=eps)
        self.ln_2 = CubeLayerNorm(kv_dim, eps=eps)
        self.ln_f = CubeLayerNorm(embed_dim, eps=eps)
        self.mlp = MLP(embed_dim, embed_dim * 4, bias=bias, dtype=dtype, device=device)

    def forward(self, x, c, attn_mask=None):
        x = x + self.attn(self.ln_1(x), self.ln_2(c), attn_mask=attn_mask)
        x = x + self.mlp(self.ln_f(x))
        return x


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim, embed_dim, bias=True, dtype=None, device=None):
        super().__init__()
        self.in_layer = ops.Linear(in_dim, embed_dim, bias=bias, dtype=dtype, device=device)
        self.silu = nn.SiLU()
        self.out_layer = ops.Linear(embed_dim, embed_dim, bias=bias, dtype=dtype, device=device)

    def forward(self, x):
        return self.out_layer(self.silu(self.in_layer(x)))


# ---------------------------------------------------------------------------
# Spherical VQ (decode-only parts)
# ---------------------------------------------------------------------------

class SphericalVectorQuantizer(nn.Module):
    def __init__(self, embed_dim, num_codes, width=None, dtype=None, device=None):
        super().__init__()
        self.num_codes = num_codes
        self.codebook = ops.Embedding(num_codes, embed_dim, dtype=dtype, device=device)
        width = width or embed_dim
        if width != embed_dim:
            self.c_in = ops.Linear(width, embed_dim, dtype=dtype, device=device)
            self.c_x = ops.Linear(width, embed_dim, dtype=dtype, device=device)
            self.c_out = ops.Linear(embed_dim, width, dtype=dtype, device=device)
        else:
            self.c_in = self.c_out = self.c_x = nn.Identity()
        self.norm = CubeRMSNorm(embed_dim, elementwise_affine=False, dtype=dtype, device=device)
        # "kl" codebook regularization (released config)
        self.cb_weight = nn.Parameter(torch.ones([embed_dim], dtype=dtype, device=device))
        self.cb_bias = nn.Parameter(torch.zeros([embed_dim], dtype=dtype, device=device))

    def cb_norm(self, x):
        return x * self.cb_weight + self.cb_bias

    def get_codebook(self):
        return self.norm(self.cb_norm(self.codebook.weight))

    def lookup_codebook(self, q):
        z_q = F.embedding(q, self.get_codebook())
        return self.c_out(z_q)


class OneDBottleNeck(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block


# ---------------------------------------------------------------------------
# Decoders
# ---------------------------------------------------------------------------

class OneDDecoder(nn.Module):
    def __init__(self, num_latents, width, num_heads, num_layers, eps=1e-6, dtype=None, device=None):
        super().__init__()
        self.register_buffer("query", torch.empty([0, width]), persistent=False)
        self.positional_encodings = nn.Parameter(torch.empty(num_latents, width, dtype=dtype, device=device))
        self.blocks = nn.ModuleList([
            EncoderLayer(width, num_heads, eps=eps, dtype=dtype, device=device)
            for _ in range(num_layers)
        ])

    def forward(self, z):
        h = z + self.positional_encodings[:z.shape[1]].unsqueeze(0).to(z.dtype)
        for block in self.blocks:
            h = block(h)
        return h


class OneDOccupancyDecoder(nn.Module):
    def __init__(self, embedder, out_features, width, num_heads, eps=1e-6, dtype=None, device=None):
        super().__init__()
        self.embedder = embedder
        self.query_in = MLPEmbedder(embedder.out_dim, width, dtype=dtype, device=device)
        self.attn_out = EncoderCrossAttentionLayer(width, num_heads, dtype=dtype, device=device)
        self.ln_f = CubeLayerNorm(width, eps=eps, elementwise_affine=True, dtype=dtype, device=device)
        self.c_head = ops.Linear(width, out_features, dtype=dtype, device=device)

    def forward(self, queries, latents):
        x = self.query_in(self.embedder(queries))
        x = self.attn_out(x, latents)
        return self.c_head(self.ln_f(x))


# ---------------------------------------------------------------------------
# Top-level shape VAE
# ---------------------------------------------------------------------------

def generate_dense_grid_points(bbox_min, bbox_max, resolution_base, indexing="ij"):
    length = bbox_max - bbox_min
    num_cells = np.exp2(resolution_base)
    x = np.linspace(bbox_min[0], bbox_max[0], int(num_cells) + 1, dtype=np.float32)
    y = np.linspace(bbox_min[1], bbox_max[1], int(num_cells) + 1, dtype=np.float32)
    z = np.linspace(bbox_min[2], bbox_max[2], int(num_cells) + 1, dtype=np.float32)
    xs, ys, zs = np.meshgrid(x, y, z, indexing=indexing)
    xyz = np.stack((xs, ys, zs), axis=-1).reshape(-1, 3)
    grid_size = [int(num_cells) + 1] * 3
    return xyz, grid_size, length


class CubeShapeVAE(nn.Module):
    """Decode-only OneDAutoEncoder. Encoder weights load with strict=False (ignored)."""

    # Fixed query bounds for the occupancy grid (upstream default).
    decode_bounds = (-1.05, -1.05, -1.05, 1.05, 1.05, 1.05)

    def __init__(self, num_encoder_latents=1024, embed_dim=32, width=768, num_heads=12,
                 num_freqs=128, num_decoder_layers=24, num_codes=16384, out_dim=1, eps=1e-6,
                 dtype=None, device=None):
        super().__init__()
        self.cfg_num_encoder_latents = num_encoder_latents
        self.cfg_num_codes = num_codes
        self.embedder = PhaseModulatedFourierEmbedder(num_freqs=num_freqs, input_dim=3, dtype=dtype, device=device)
        self.bottleneck = OneDBottleNeck(
            SphericalVectorQuantizer(embed_dim, num_codes, width, dtype=dtype, device=device)
        )
        self.decoder = OneDDecoder(num_encoder_latents, width, num_heads, num_decoder_layers,
                                   eps=eps, dtype=dtype, device=device)
        self.occupancy_decoder = OneDOccupancyDecoder(self.embedder, out_dim, width, num_heads,
                                                      eps=eps, dtype=dtype, device=device)

    @torch.no_grad()
    def decode(self, samples, resolution_base=8.0, chunk_size=100_000, **kwargs):
        """Token IDs -> occupancy grid logits. Entry point for comfy.sd.VAE.decode, which
        manages model loading/device/dtype. `samples` arrive as (B, 1, num_tokens) in the
        VAE working dtype on the load device. VAE.decode applies a trailing movedim(1, -1),
        so pre-invert it here to hand the node grid logits as (B, gx, gy, gz)."""
        ids = samples.reshape(samples.shape[0], -1)[:, :self.cfg_num_encoder_latents]
        ids = ids.round().long().clamp(0, self.cfg_num_codes - 1)
        latents = self.decode_indices(ids)
        grid_logits, _, _, _ = self.extract_geometry(
            latents, bounds=self.decode_bounds, resolution_base=resolution_base, chunk_size=chunk_size)
        return grid_logits.movedim(-1, 1)

    @torch.no_grad()
    def decode_indices(self, shape_ids):
        z_q = self.bottleneck.block.lookup_codebook(shape_ids)
        return self.decoder(z_q)

    @torch.no_grad()
    def query(self, queries, latents):
        return self.occupancy_decoder(queries, latents).squeeze(-1)

    @torch.no_grad()
    def extract_geometry(self, latents, bounds=(-1.05, -1.05, -1.05, 1.05, 1.05, 1.05),
                         resolution_base=8.0, chunk_size=100_000):
        bbox_min = np.array(bounds[0:3])
        bbox_max = np.array(bounds[3:6])
        bbox_size = bbox_max - bbox_min

        xyz, grid_size, _ = generate_dense_grid_points(bbox_min, bbox_max, resolution_base, indexing="ij")
        xyz = torch.from_numpy(xyz)
        batch_size = latents.shape[0]
        batch_logits = []
        for start in range(0, xyz.shape[0], chunk_size):
            queries = xyz[start:start + chunk_size, :]
            n = queries.shape[0]
            if start > 0 and n < chunk_size:
                queries = F.pad(queries, [0, 0, 0, chunk_size - n])
            bq = queries.unsqueeze(0).expand(batch_size, -1, -1).to(latents)
            batch_logits.append(self.query(bq, latents)[:, :n])

        grid_logits = torch.cat(batch_logits, dim=1).detach().view(
            batch_size, grid_size[0], grid_size[1], grid_size[2]).float()
        return grid_logits, grid_size, bbox_size, bbox_min


def grid_logits_to_mesh(grid_logit, grid_size, bbox_size, bbox_min, level=0.0):
    """Occupancy-logit grid -> mesh, using the vendored dependency-free marching cubes
    (classic Lorensen, same family as upstream cube's default warp backend). The vertex
    transform mirrors upstream: index coords -> bbox, with the same face winding flip."""
    from comfy.ldm.cube.marching_cubes import marching_cubes
    vertices, faces = marching_cubes(grid_logit, level)
    vertices = vertices / np.array(grid_size) * bbox_size + bbox_min
    faces = faces[:, [2, 1, 0]]
    return vertices.astype(np.float32), np.ascontiguousarray(faces)
