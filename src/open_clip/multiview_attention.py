"""Spatial attention fusion for independently encoded, ordered image views."""
import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class ViewAttentionBlock(nn.Module):
    def __init__(self, width, heads, mlp_ratio, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, int(width * mlp_ratio)), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(int(width * mlp_ratio), width),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, tokens):
        normalized = self.norm1(tokens)
        update = self.attention(normalized, normalized, normalized, need_weights=False)[0]
        tokens = tokens + self.drop(update)
        return tokens + self.drop(self.mlp(self.norm2(tokens)))


class SpatialViewAttention(nn.Module):
    """Fuse [B,V,C,H,W] maps while retaining a token for every view/grid cell.

    Self-attention operates over all V*H*W tokens. Learned queries subsequently
    pool this sequence; they are summary slots, not supervised lesion labels.
    Normalized sinusoidal coordinates support different inference resolutions
    without resizing a learned positional table.
    """
    def __init__(self, in_channels, num_views, width=256, layers=2, heads=8,
                 num_queries=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        for name, value in dict(width=width, layers=layers, heads=heads,
                                num_queries=num_queries, num_views=num_views).items():
            if not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if width % 4 or width % heads:
            raise ValueError('width must be divisible by 4 and by heads')
        if not math.isfinite(mlp_ratio) or int(width * mlp_ratio) < 1:
            raise ValueError('mlp_ratio must produce a positive hidden width')
        if not 0 <= dropout < 1:
            raise ValueError('dropout must be in [0, 1)')
        self.width = width
        self.num_views = num_views
        self.in_channels = in_channels
        self.grad_checkpointing = False
        self.input_proj = nn.Linear(in_channels, width)
        self.input_norm = nn.LayerNorm(width)
        self.view_embed = nn.Parameter(torch.empty(num_views, width))
        # Construct blocks independently rather than cloning identical initial weights.
        self.blocks = nn.ModuleList([
            ViewAttentionBlock(width, heads, mlp_ratio, dropout) for _ in range(layers)
        ])
        self.token_norm = nn.LayerNorm(width)
        self.queries = nn.Parameter(torch.empty(num_queries, width))
        self.query_norm = nn.LayerNorm(width)
        self.pool_attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.pool_norm = nn.LayerNorm(width)
        nn.init.normal_(self.view_embed, std=0.02)
        nn.init.normal_(self.queries, std=0.02)

    def position_encoding(self, height, width, device, dtype):
        # Compute in FP32 even under BF16 autocast, then match the token dtype.
        frequency = torch.exp(-math.log(10000.) *
                              torch.arange(self.width // 4, device=device, dtype=torch.float32) /
                              (self.width // 4))
        y = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) / height
        x = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) / width
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        phases = [coordinate.flatten()[:, None] * frequency * (2 * math.pi) for coordinate in (yy, xx)]
        return torch.cat([value for phase in phases for value in (phase.sin(), phase.cos())], -1).to(dtype)

    def encode_tokens(self, spatial):
        """Return [B,V*H*W,D] contextual tokens, in view-major, row-major order."""
        if spatial.ndim != 5 or spatial.shape[1:3] != (self.num_views, self.in_channels):
            raise ValueError(f'Expected [B,{self.num_views},{self.in_channels},H,W] spatial features')
        batch, views, _, height, width = spatial.shape
        tokens = self.input_norm(self.input_proj(spatial.permute(0, 1, 3, 4, 2)))
        position = self.position_encoding(height, width, tokens.device, tokens.dtype)
        tokens = tokens.reshape(batch, views, height * width, self.width)
        tokens = tokens + position[None, None] + self.view_embed.to(tokens.dtype)[None, :, None]
        tokens = tokens.flatten(1, 2)
        for block in self.blocks:
            if self.grad_checkpointing and self.training and torch.is_grad_enabled():
                tokens = checkpoint(block, tokens, use_reentrant=False)
            else:
                tokens = block(tokens)
        return self.token_norm(tokens)

    def pool_tokens(self, tokens):
        queries = self.queries.to(tokens.dtype)[None].expand(tokens.shape[0], -1, -1)
        update = self.pool_attention(self.query_norm(queries), tokens, tokens, need_weights=False)[0]
        return self.pool_norm(queries + update).mean(dim=1)

    def forward(self, spatial):
        return self.pool_tokens(self.encode_tokens(spatial))
