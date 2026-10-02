"""GLoRIA-style region/token alignment for native CLIP with PET ResNet views.

Uses contextual subword tokens from native CLIP or an HF text encoder
(not GLoRIA's explicit word aggregation).
Global fusion supports pooled concatenation or spatial attention. Local matching uses all
cross-rank negatives, with checkpointed chunks to bound attention memory.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .loss import ClipLoss, gather_features


def configure_region_branch(model, config):
    from .timm_model import TimmModel

    defaults = dict(dim=128, max_tokens=128, chunk_size=4, weight=1.0,
                    attention_temperature=0.1, word_temperature=0.1,
                    contrastive_temperature=0.07)
    unknown = set(config) - set(defaults)
    if unknown:
        raise ValueError(f'Unknown region_cfg options: {sorted(unknown)}')
    defaults.update(config)
    for key in ('dim', 'max_tokens', 'chunk_size'):
        if not isinstance(defaults[key], int) or defaults[key] < 1:
            raise ValueError(f'region_cfg {key} must be a positive integer')
    for key in ('weight', 'attention_temperature', 'word_temperature', 'contrastive_temperature'):
        if not math.isfinite(defaults[key]) or defaults[key] <= 0:
            raise ValueError(f'region_cfg {key} must be finite and positive')
    visual = model.visual
    if not (isinstance(visual, TimmModel) and visual.num_views == 2
            and visual.view_fusion in ('concat', 'attention') and hasattr(visual.trunk, 'layer4')):
        raise ValueError('region_cfg requires a two-view concat or attention timm ResNet')
    if hasattr(model, 'text'):
        from .hf_model import HFTextEncoder, MeanPooler
        if not isinstance(model.text, HFTextEncoder) or not isinstance(model.text.pooler, MeanPooler):
            raise ValueError('HF region alignment requires HFTextEncoder with mean pooling')
        if any(x is None for x in (model.text.pad_id, model.text.bos_id, model.text.eos_id)):
            raise ValueError('HF region alignment needs explicit pad/CLS/SEP ids in the text configuration')
        width = model.text.config.hidden_size
    else:
        if model.text_pool_type != 'argmax' or model.vocab_size != 49408:
            raise ValueError('Native region alignment requires the standard CLIP BPE tokenizer and EOT argmax pooling')
        width = model.token_embedding.embedding_dim
    model.region_cfg = defaults
    region_width = (visual.head.fusion.width if visual.view_fusion == 'attention'
                    else visual.trunk.num_features)
    model.region_image_projection = nn.Linear(region_width, defaults['dim'], bias=False)
    model.region_text_projection = nn.Linear(width, defaults['dim'], bias=False)


def encode_image_regions(model, image):
    """Return global embedding and [B, view, height, width, D] local features."""
    if image.ndim != 5 or image.shape[1] != 2:
        raise ValueError('Local PET encoding expects [B, 2, 1, H, W]')
    batch = image.shape[0]
    spatial = model.visual.trunk.forward_features(image.flatten(0, 1))
    if spatial.ndim != 4:
        raise ValueError('Local PET encoding requires NCHW spatial features')
    if model.visual.view_fusion == 'attention':
        global_features, regions = model.visual.fuse_spatial_features(spatial, return_tokens=True)
        regions = model.region_image_projection(regions)
    else:
        pooled = model.visual.trunk.forward_head(spatial)
        global_features = model.visual.head(pooled.reshape(batch, -1))
        regions = model.region_image_projection(spatial.permute(0, 2, 3, 1))
        regions = regions.reshape(batch, 2, *regions.shape[1:])
    return F.normalize(global_features, dim=-1), F.normalize(regions, dim=-1)


def select_local_tokens(model, hidden, token_ids, content_mask=None):
    """Deterministic uniform subword sampling; exclude SOT, EOT, and padding.

    At most max_tokens tokens span the entire report; the global branch still
    consumes the full context. Fixed output shape permits distributed gathering.
    Empty reports carry an all-false mask and are rejected by the training loss.
    """
    limit = model.region_cfg['max_tokens']
    slot = torch.arange(limit, device=token_ids.device)[None, :]
    if content_mask is None:
        length = (token_ids.argmax(dim=-1) - 1).clamp_min(0)
        content_positions = None
    else:
        length = content_mask.sum(dim=-1)
        indices = torch.arange(hidden.shape[1], device=hidden.device)[None, :].expand_as(content_mask)
        content_positions = indices.masked_fill(~content_mask, hidden.shape[1]).sort(dim=-1).values
    offsets = torch.div(slot * length[:, None], length.clamp(min=1, max=limit)[:, None],
                        rounding_mode='floor')
    valid = slot < length[:, None].clamp_max(limit)
    positions = (offsets + 1 if content_positions is None else
                 content_positions.gather(1, offsets.clamp_max(hidden.shape[1] - 1)))
    positions = positions.clamp_max(hidden.shape[1] - 1)
    selected = hidden.gather(1, positions[..., None].expand(-1, -1, hidden.shape[-1]))
    features = F.normalize(model.region_text_projection(selected), dim=-1)
    return features.masked_fill(~valid[..., None], 0), valid


class RegionClipLoss(nn.Module):
    """Global CLIP plus symmetric cross-example region/token matching loss."""
    def __init__(self, config, **clip_options):
        super().__init__()
        self.global_loss = ClipLoss(**clip_options)
        self.config = config
        self.rank = clip_options.get('rank', 0)
        self.world_size = clip_options.get('world_size', 1)

    def pair_scores(self, regions, tokens, valid):
        """All image/text combinations in a chunk, computed in FP32 for AMP stability."""
        with torch.autocast(device_type=regions.device.type, enabled=False):
            regions, tokens = regions.float(), tokens.float()
            # [image, report, region, token]. First normalize over valid words
            # per region (GLoRIA), then over regions per word.
            similarity = torch.einsum('ird,jwd->ijrw', regions, tokens)
            similarity = similarity.masked_fill(~valid[None, :, None, :], -torch.inf)
            word_normalized = similarity.softmax(dim=-1)
            attention = (word_normalized / self.config['attention_temperature']).softmax(dim=-2)
            context = torch.einsum('ijrw,ird->ijwd', attention, regions)
            cosine = F.cosine_similarity(context, tokens[None, :, :, :], dim=-1)
            scaled = (cosine / self.config['word_temperature']).masked_fill(~valid[None, :, :], -torch.inf)
            # Log-mean-exp removes report-length bias while retaining GLoRIA's
            # smooth maximum over word alignments. This is an explicit adaptation.
            scores = self.config['word_temperature'] * (
                torch.logsumexp(scaled, dim=-1) - valid.sum(-1).float().log()[None, :])
            return scores / self.config['contrastive_temperature']

    def score_matrix(self, regions, tokens, valid):
        size = self.config['chunk_size']
        rows = []
        for start in range(0, len(regions), size):
            cols = []
            for offset in range(0, len(tokens), size):
                args = (regions[start:start + size], tokens[offset:offset + size], valid[offset:offset + size])
                if torch.is_grad_enabled() and (args[0].requires_grad or args[1].requires_grad):
                    score = checkpoint(self.pair_scores, *args, use_reentrant=False)
                else:
                    score = self.pair_scores(*args)
                cols.append(score)
            rows.append(torch.cat(cols, dim=1))
        return torch.cat(rows, dim=0)

    def validation_loss(self, regions, tokens, valid):
        """Unweighted symmetric local loss within one validation batch.

        No distributed collectives: standard validation runs only on rank zero.
        Uses the training matching function, temperatures, and valid-token mask.
        """
        if not valid.any(dim=-1).all():
            raise ValueError('Region contrastive learning requires at least one content token per report')
        scores = self.score_matrix(regions, tokens, valid)
        labels = torch.arange(len(regions), device=regions.device)
        return (F.cross_entropy(scores, labels) + F.cross_entropy(scores.T, labels)) / 2

    def forward(self, image_features, text_features, logit_scale, region_features,
                local_text_features, local_text_valid, logit_bias=None, output_dict=False):
        global_loss = self.global_loss(image_features, text_features, logit_scale, logit_bias)
        regions, tokens, valid = region_features, local_text_features, local_text_valid
        if self.world_size > 1:
            # Equal local batch/token/region shapes are required (training drop_last).
            all_regions, all_tokens = gather_features(regions, tokens, local_loss=True,
                                                       gather_with_grad=True, rank=self.rank,
                                                       world_size=self.world_size)
            masks = [torch.empty_like(valid) for _ in range(self.world_size)]
            torch.distributed.all_gather(masks, valid.contiguous())
            all_valid = torch.cat(masks)
        else:
            all_regions, all_tokens, all_valid = regions, tokens, valid
        if not all_valid.any(dim=-1).all():
            raise ValueError('Region contrastive learning requires at least one content token per report')
        logits_image = self.score_matrix(regions, all_tokens, all_valid)
        logits_text = (logits_image.T if self.world_size == 1 else
                       self.score_matrix(all_regions, tokens, valid).T)
        labels = torch.arange(len(regions), device=regions.device) + len(regions) * self.rank
        local_loss = (F.cross_entropy(logits_image, labels) + F.cross_entropy(logits_text, labels)) / 2
        losses = {'contrastive_loss': global_loss, 'region_contrastive_loss': self.config['weight'] * local_loss}
        return losses if output_dict else sum(losses.values())
