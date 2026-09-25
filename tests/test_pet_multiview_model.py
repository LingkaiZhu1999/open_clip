"""Independent-view encoding must preserve one contrastive example per scan."""
import pytest
import torch
import torch.nn.functional as F

import open_clip
from open_clip.timm_model import TimmModel


def make_tower(num_views=2, view_fusion="mean"):
    return TimmModel('resnet18', embed_dim=8, image_size=32,
                     model_kwargs={'in_chans': 1}, num_views=num_views, view_fusion=view_fusion).eval()


def test_view_pooling_matches_separate_encoder_calls_and_backpropagates():
    torch.manual_seed(0)
    tower = make_tower()
    images = torch.randn(2, 2, 1, 32, 32, requires_grad=True)
    output = tower(images)
    individual = [F.normalize(tower.head(tower.trunk(images[:, view])), dim=-1) for view in range(2)]
    expected = torch.stack(individual, dim=1).mean(dim=1)
    torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-5)
    assert output.shape == (2, 8)
    torch.testing.assert_close(tower(images.flip(1)), output, rtol=1e-4, atol=1e-5)
    output.square().sum().backward()
    assert images.grad[:, 0].abs().sum() > 0
    assert images.grad[:, 1].abs().sum() > 0
    assert tower.trunk.conv1.weight.grad is not None


def test_multiview_rejects_montage_and_wrong_view_count():
    tower = make_tower()
    for image in (torch.zeros(2, 1, 32, 64), torch.zeros(2, 3, 1, 32, 32)):
        with pytest.raises(ValueError, match='independent views'):
            tower(image)


def test_single_view_timm_path_is_unchanged():
    tower = make_tower(num_views=1)
    images = torch.randn(2, 1, 32, 32)
    torch.testing.assert_close(tower(images), tower.head(tower.trunk(images)))


def test_pet_factory_returns_one_embedding_per_scan_and_one_per_report():
    model = open_clip.create_model('PET-ResNet34', force_context_length=8, output_dict=True).eval()
    assert model.visual.num_views == 2
    assert model.visual.view_fusion == "concat"
    assert model.visual.head.proj.in_features == 1024
    assert model.visual.head.proj.out_features == 512
    assert model.visual.in_chans == 1
    assert model.visual.image_size == (310, 310)
    assert model.visual.trunk.conv1.in_channels == 1
    images = torch.randn(2, 2, 1, 32, 32)
    texts = torch.randint(0, 100, (2, 8))
    with torch.no_grad():
        result = model(images, texts)
        encoded = model.encode_image(images, normalize=True)
    assert result['image_features'].shape == result['text_features'].shape == (2, 512)
    torch.testing.assert_close(result['image_features'], encoded)
    torch.testing.assert_close(encoded.norm(dim=-1), torch.ones(2))
    assert (result['image_features'] @ result['text_features'].T).shape == (2, 2)


def test_concat_preserves_order_and_backpropagates_through_both_views():
    torch.manual_seed(0)
    tower = make_tower(view_fusion="concat")
    images = torch.randn(2, 2, 1, 32, 32, requires_grad=True)
    output = tower(images)
    features = [tower.trunk(images[:, view]) for view in range(2)]
    expected = tower.head(torch.cat(features, dim=-1))
    torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-5)
    assert output.shape == (2, 8)
    assert not torch.allclose(tower(images.flip(1)), output)
    output.square().sum().backward()
    for view in range(2):
        assert images.grad[:, view].abs().sum() > 0
        assert tower.head.proj.weight.grad[:, view * 512:(view + 1) * 512].abs().sum() > 0
    assert tower.trunk.conv1.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("kwargs", [
    {"view_fusion": "invalid"},
    {"view_fusion": "concat", "num_views": 1},
    {"view_fusion": "concat", "proj": "none"},
    {"view_fusion": "concat", "pool": "max"},
])
def test_invalid_fusion_configuration(kwargs):
    options = dict(num_views=2, view_fusion="mean")
    options.update(kwargs)
    with pytest.raises(ValueError, match="fusion"):
        TimmModel('resnet18', embed_dim=8, **options)
