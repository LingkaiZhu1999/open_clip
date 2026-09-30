"""Local alignment math, masking, chunk gradients, and native PET integration."""

import pytest
import torch
from torch.nn import functional as F

from open_clip import create_model
from open_clip.local_region import RegionClipLoss
from open_clip.model import CLIP


CONFIG = dict(dim=8, max_tokens=6, chunk_size=2, weight=0.5,
              attention_temperature=0.1, word_temperature=0.1, contrastive_temperature=0.07)


def make_model():
    return CLIP(
        embed_dim=16,
        vision_cfg=dict(timm_model_name='resnet34', timm_model_pretrained=False,
                        timm_pool='avg', timm_proj='linear', timm_model_kwargs={'in_chans': 1},
                        timm_num_views=2, timm_view_fusion='concat', image_size=64),
        text_cfg=dict(context_length=12, vocab_size=49408, width=32, heads=4, layers=1),
        region_cfg=CONFIG, output_dict=True,
    )


def test_matching_regions_score_above_mismatches_and_padding_is_ignored():
    regions = torch.eye(3)[:, None, :].repeat(1, 2, 1)
    tokens = torch.eye(3)[:, None, :].repeat(1, 4, 1)
    valid = torch.tensor([[True, True, False, False]] * 3)
    loss = RegionClipLoss(CONFIG)
    scores = loss.score_matrix(regions, tokens, valid)
    assert torch.equal(scores.argmax(1), torch.arange(3))
    assert torch.equal(scores.argmax(0), torch.arange(3))
    tokens[:, 2:] = torch.randn_like(tokens[:, 2:]) * 100
    torch.testing.assert_close(scores, loss.score_matrix(regions, tokens, valid))


def test_checkpointed_chunks_match_dense_scores_and_gradients():
    torch.manual_seed(4)
    r = F.normalize(torch.randn(3, 5, 8), dim=-1).requires_grad_()
    t = F.normalize(torch.randn(3, 6, 8), dim=-1).requires_grad_()
    mask = torch.tensor([[True] * 6, [True] * 3 + [False] * 3, [True] + [False] * 5])
    loss = RegionClipLoss(CONFIG)
    chunked = loss.score_matrix(r, t, mask)
    grad = torch.autograd.grad(chunked.square().sum(), (r, t))
    dense = loss.pair_scores(r, t, mask)
    expected = torch.autograd.grad(dense.square().sum(), (r, t))
    torch.testing.assert_close(chunked, dense)
    for actual, reference in zip(grad, expected):
        torch.testing.assert_close(actual, reference, atol=3e-5, rtol=3e-5)


def test_region_branch_preserves_global_forward_and_backpropagates():
    torch.manual_seed(5)
    model = make_model().eval()
    image = torch.randn(2, 2, 1, 64, 64)
    ids = torch.tensor([[49406, 11, 12, 13, 49407, 0, 0, 0, 0, 0, 0, 0],
                        [49406, 15, 16, 17, 18, 19, 20, 21, 22, 23, 49407, 0]])
    from open_clip.local_region import encode_image_regions
    global_image, regions = encode_image_regions(model, image)
    torch.testing.assert_close(global_image, model.encode_image(image, normalize=True))
    assert regions.shape == (2, 2, 2, 2, 8)
    tokens, mask = model.encode_local_text(ids)
    assert mask.sum(1).tolist() == [3, 6]
    assert tokens.shape == (2, 6, 8)
    model.train()
    output = model(image, ids)
    loss = RegionClipLoss(CONFIG)(**output, output_dict=True)
    assert set(loss) == {'contrastive_loss', 'region_contrastive_loss'}
    sum(loss.values()).backward()
    for parameter in (model.region_image_projection.weight, model.region_text_projection.weight,
                      model.visual.trunk.conv1.weight, model.token_embedding.weight,
                      model.visual.head.proj.weight):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    model.eval()
    assert 'region_features' not in model(image, ids)
    clone = make_model()
    clone.load_state_dict(model.state_dict(), strict=True)


def test_empty_text_rejected():
    loss = RegionClipLoss(CONFIG)
    with pytest.raises(ValueError, match='content token'):
        loss(torch.randn(2, 8), torch.randn(2, 8), torch.tensor(1.),
             torch.randn(2, 3, 8), torch.randn(2, 6, 8), torch.zeros(2, 6, dtype=torch.bool))


def test_local_config_available_through_factory():
    model = create_model('PET-ResNet34-Local', force_context_length=12,
                         text_cfg=dict(width=32, heads=4, layers=1), output_dict=True)
    assert model.region_cfg['max_tokens'] == 128
    assert model.visual.view_fusion == 'concat'


def test_invalid_region_config_rejected():
    from open_clip.local_region import configure_region_branch
    model = make_model()
    with pytest.raises(ValueError, match='chunk_size'):
        configure_region_branch(model, dict(chunk_size=0))


def test_task_selects_region_loss_and_reports_both_terms():
    from open_clip.task import CLIPTask
    task = CLIPTask(make_model())
    assert isinstance(task.loss, RegionClipLoss)
    ids = torch.tensor([[49406, 11, 12, 49407] + [0] * 8] * 2)
    losses, _ = task.training_forward({'image': torch.randn(2, 2, 1, 64, 64), 'text': ids})
    torch.testing.assert_close(losses['loss'], losses['contrastive_loss'] + losses['region_contrastive_loss'])


def test_distributed_gradient_oracle():
    import os
    import subprocess
    import sys
    from pathlib import Path
    env = dict(os.environ, OMP_NUM_THREADS='1', REGION_TEST_BACKEND='gloo')
    result = subprocess.run(
        [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2',
         str(Path(__file__).with_name('pet_region_distributed_check.py'))],
        env=env, capture_output=True, text=True, timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'gradients match global-batch oracle' in result.stdout


def test_validation_local_loss_matches_training_objective_without_collectives():
    from unittest import mock
    torch.manual_seed(12)
    regions = F.normalize(torch.randn(3, 5, 8), dim=-1)
    tokens = F.normalize(torch.randn(3, 6, 8), dim=-1)
    valid = torch.tensor([[True] * 6, [True] * 2 + [False] * 4, [True] * 4 + [False] * 2])
    single = RegionClipLoss(CONFIG)
    train_loss = single(torch.randn(3, 16), torch.randn(3, 16), torch.tensor(2.),
                        regions, tokens, valid, output_dict=True)['region_contrastive_loss']
    # Even a criterion configured for DDP must not communicate in validation_loss.
    distributed = RegionClipLoss(CONFIG, rank=0, world_size=2)
    with mock.patch('torch.distributed.all_gather', side_effect=AssertionError('Unexpected collective')):
        with torch.inference_mode():
            actual = distributed.validation_loss(regions, tokens, valid)
    torch.testing.assert_close(CONFIG['weight'] * actual, train_loss)


def test_evaluate_reports_weighted_global_local_metrics_and_keeps_model_frozen():
    import types
    from unittest import mock
    from open_clip.task import CLIPTask
    from open_clip_train.train import evaluate
    from test_eval_task import _make_args

    torch.manual_seed(15)
    model = make_model().eval()
    # No process group is initialized: rank-zero-only validation must work anyway.
    task = CLIPTask(model, rank=0, world_size=2).eval()
    batches = [{'image': torch.randn(n, 2, 1, 64, 64),
                'text': torch.tensor([[49406, 11, 12, 49407] + [0] * 8] * n)} for n in (3, 2)]
    criterion = RegionClipLoss(CONFIG)
    expected_global, expected_local = 0., 0.
    bn_before = model.visual.trunk.bn1.running_mean.clone()
    with torch.inference_mode():
        for batch in batches:
            output = task(batch)
            components = criterion(**output, output_dict=True)
            expected_global += components['contrastive_loss'].item() * len(batch['text']) / 5
            expected_local += components['region_contrastive_loss'].item() * len(batch['text']) / 5
    loader = mock.MagicMock()
    loader.__iter__.return_value = iter(batches)
    loader.num_samples = 5
    args = _make_args()
    args.world_size, args.distributed = 2, True
    with mock.patch('torch.distributed.all_gather', side_effect=AssertionError('Unexpected collective')):
        metrics = evaluate(task, {'val': types.SimpleNamespace(dataloader=loader)}, 1, args)
    assert metrics['global_val_loss'] == pytest.approx(expected_global)
    assert metrics['clip_val_loss'] == metrics['global_val_loss']
    assert metrics['weighted_local_val_loss'] == pytest.approx(expected_local)
    assert metrics['local_val_loss'] == pytest.approx(expected_local / CONFIG['weight'])
    assert metrics['total_val_loss'] == pytest.approx(expected_global + expected_local)
    assert metrics['local_val_num_samples'] == metrics['num_samples'] == 5
    assert 'image_to_text_R@10' in metrics
    torch.testing.assert_close(model.visual.trunk.bn1.running_mean, bn_before)
    assert not model.training
    assert all(p.grad is None for p in model.parameters())
