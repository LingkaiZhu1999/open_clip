"""Cached accumulation equals differentiating one loss over concatenated microbatches."""
import copy
import math
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from open_clip.local_region import RegionClipLoss
from open_clip.loss import ClipLoss
from open_clip_train.train import _train_step_eager


class TinyPET(nn.Module):
    def __init__(self, local):
        super().__init__()
        self.local = local
        self.bn = nn.BatchNorm1d(6, momentum=None)
        self.dropout = nn.Dropout(0.35)
        self.image_proj = nn.Linear(6, 4)
        self.text_proj = nn.Linear(6, 4)
        self.region_proj = nn.Linear(6, 4)
        self.word_proj = nn.Linear(6, 4)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(2.0)))

    def forward(self, image, text, valid):
        image = self.dropout(self.bn(image.transpose(1, 2)).transpose(1, 2))
        text = self.dropout(text)
        out = dict(image_features=F.normalize(self.image_proj(image.mean(1)), dim=-1),
                   text_features=F.normalize(self.text_proj(text.mean(1)), dim=-1),
                   logit_scale=self.logit_scale.exp(), logit_bias=None)
        if self.local:
            out.update(region_features=F.normalize(self.region_proj(image), dim=-1),
                       local_text_features=F.normalize(self.word_proj(text), dim=-1),
                       local_text_valid=valid)
        return out


class TinyTask:
    def __init__(self, model, local):
        self.trainable_module = model
        self.loss = RegionClipLoss(dict(chunk_size=2, weight=0.1, attention_temperature=0.1,
                                        word_temperature=0.1, contrastive_temperature=0.07)) if local else ClipLoss()
        self.candidate_counts = []

    def compute_accum_loss(self, features, scalars, batches):
        self.candidate_counts.append(features['image_features'].shape[0])
        return self.loss(**features, **scalars, output_dict=True), scalars

    def batch_size(self, batch):
        return len(batch['image'])

    def clamp_logit_scale(self):
        with torch.no_grad():
            self.trainable_module.logit_scale.clamp_(0, math.log(100))


@pytest.mark.parametrize('local', [False, True])
@pytest.mark.parametrize('scaled', [False, True])
def test_virtual_batch_gradients_dropout_batchnorm_and_scalar_match(local, scaled):
    torch.manual_seed(4)
    model = TinyPET(local).train()
    reference = copy.deepcopy(model)
    task = TinyTask(model, local)
    reference_task = TinyTask(reference, local)
    batches = [dict(image=torch.randn(n, 3, 6), text=torch.randn(n, 4, 6),
                    valid=torch.tensor([[True, True, False, False]] * n)) for n in (2, 3)]
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    ref_opt = torch.optim.SGD(reference.parameters(), lr=0.01)
    torch.manual_seed(123)
    reference_out = [reference(**b) for b in batches]
    scalar = {'logit_scale': reference_out[-1]['logit_scale']}
    inputs = {key: torch.cat([out[key] for out in reference_out])
              for key in reference_out[0] if key not in ('logit_scale', 'logit_bias')}
    expected_losses = reference_task.loss(**inputs, **scalar, output_dict=True)
    expected = sum(expected_losses.values())
    expected.backward()
    ref_opt.step()
    expected_rng = torch.get_rng_state()

    args = SimpleNamespace(accum_freq=2, dataset_type='pet', world_size=1, grad_clip_norm=None)
    scaler = torch.amp.GradScaler('cpu', init_scale=32) if scaled else None
    state = ([], {})
    torch.manual_seed(123)
    assert _train_step_eager(task, batches[0], state, opt, scaler, nullcontext, args) is None
    assert all(p.grad is None for p in model.parameters())
    result = _train_step_eager(task, batches[1], state, opt, scaler, nullcontext, args)
    losses, _, size, remaining = result
    assert size == 5 and remaining == ([], {})
    assert task.candidate_counts == [5]  # both microbatches share all five candidates, scored once
    torch.testing.assert_close(losses['loss'], expected.detach(), atol=2e-6, rtol=2e-6)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, reference.state_dict()[key], atol=2e-6, rtol=2e-6)
    for (_, actual), (_, target) in zip(model.named_parameters(), reference.named_parameters()):
        if target.grad is not None:
            torch.testing.assert_close(actual.grad, target.grad, atol=2e-5, rtol=2e-5)
    assert model.bn.num_batches_tracked.item() == 2
    assert torch.equal(torch.get_rng_state(), expected_rng)
