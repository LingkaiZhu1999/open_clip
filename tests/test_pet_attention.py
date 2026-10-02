"""CPU/offline checks for shared-backbone PET spatial attention fusion."""
import copy

import pytest
import torch

from open_clip import create_model, get_model_config
from open_clip.multiview_attention import SpatialViewAttention
from open_clip.timm_model import TimmModel


def test_attention_connects_views_and_preserves_spatial_identity():
    torch.manual_seed(42)
    fusion = SpatialViewAttention(16, 2, width=16, heads=4, layers=2,
                                  num_queries=3, dropout=0.).eval()
    spatial = torch.randn(2, 2, 16, 3, 4, requires_grad=True)
    tokens = fusion.encode_tokens(spatial)
    assert tokens.shape == (2, 24, 16)
    # A token anchored in view 0 must depend on input features in view 1.
    gradient = torch.autograd.grad(tokens[:, 0, 0].sum(), spatial, retain_graph=True)[0]
    assert gradient[:, 1].abs().sum() > 0
    pooled = fusion.pool_tokens(tokens)
    pooled.square().sum().backward()
    assert spatial.grad[:, 0].abs().sum() > 0
    assert spatial.grad[:, 1].abs().sum() > 0
    assert fusion.view_embed.grad.abs().sum() > 0
    assert fusion.queries.grad.abs().sum() > 0
    # View embeddings prevent permutation-invariant pooling from losing view identity.
    assert not torch.allclose(pooled, fusion(spatial.flip(1)), atol=1e-6)
    # Positional encoding distinguishes spatial permutations within a view.
    assert not torch.allclose(pooled, fusion(spatial.flip(-1)), atol=1e-6)


def test_attention_checkpointing_preserves_dropout_outputs_and_gradients():
    reference = SpatialViewAttention(8, 2, width=16, heads=4, dropout=0.2).train()
    checkpointed = copy.deepcopy(reference)
    checkpointed.grad_checkpointing = True
    spatial = torch.randn(2, 2, 8, 2, 3)
    outputs = []
    for model in (reference, checkpointed):
        torch.manual_seed(9)
        output = model(spatial)
        output.square().sum().backward()
        outputs.append(output)
    torch.testing.assert_close(*outputs)
    for a, b in zip(reference.parameters(), checkpointed.parameters()):
        assert a.grad is not None and torch.isfinite(a.grad).all()
        torch.testing.assert_close(a.grad, b.grad)


def test_resnet50_shared_backbone_dynamic_resolution_and_locking():
    tower = TimmModel('resnet50', embed_dim=32, model_kwargs={'in_chans': 1},
                      num_views=2, view_fusion='attention',
                      view_fusion_cfg=dict(width=32, heads=4, dropout=0.)).eval()
    calls = []
    handle = tower.trunk.conv1.register_forward_pre_hook(lambda module, args: calls.append(args[0].shape))
    with torch.inference_mode():
        for size, grid in ((224, 7), (310, 10)):
            image = torch.randn(1, 2, 1, size, size)
            spatial = tower.trunk.forward_features(image.flatten(0, 1))
            embedding, tokens = tower.fuse_spatial_features(spatial, return_tokens=True)
            assert spatial.shape == (2, 2048, grid, grid)
            assert tokens.shape == (1, 2, grid, grid, 32)
            torch.testing.assert_close(embedding, tower(image))
            assert torch.isfinite(embedding).all()
    handle.remove()
    assert len(calls) == 4 and all(shape[:2] == (2, 1) for shape in calls)
    # No second backbone and no ungrouped fusion parameters in freeze/LR-decay logic.
    grouped = []
    for _, members in tower.layer_groups():
        for member in members:
            grouped.extend([member] if isinstance(member, torch.nn.Parameter) else member.parameters())
    assert len(grouped) == len({id(p) for p in grouped})
    assert {id(p) for p in grouped} == {id(p) for p in tower.parameters()}
    tower.lock()
    assert not any(p.requires_grad for p in tower.parameters())
    tower.lock(unlocked_groups=1)
    assert all(p.requires_grad for p in tower.head.parameters())
    assert not any(p.requires_grad for p in tower.trunk.parameters())
    tower.set_grad_checkpointing()
    assert tower.head.fusion.grad_checkpointing
    assert {'head.fusion.view_embed', 'head.fusion.queries'} <= tower.no_weight_decay()


@pytest.mark.parametrize('local', [False, True])
def test_factory_bert_contrastive_backward_and_checkpoint_roundtrip(tmp_path, local):
    from transformers import BertConfig, BertModel
    from open_clip.task import CLIPTask

    torch.manual_seed(3)
    bert_cfg = BertConfig(vocab_size=200, hidden_size=16, num_hidden_layers=1,
                          num_attention_heads=2, intermediate_size=32, max_position_embeddings=16,
                          pad_token_id=0, cls_token_id=101, sep_token_id=102)
    pretrained = BertModel(bert_cfg, add_pooling_layer=False)
    pretrained.save_pretrained(tmp_path / 'bert')
    name = 'PET-ResNet50-BioClinicalBERT-Attention' + ('-Local' if local else '')
    config = get_model_config(name)
    assert config['vision_cfg']['timm_view_fusion_cfg']['layers'] == 2
    assert config['text_cfg']['hf_model_pretrained']
    text_cfg = dict(config['text_cfg'], hf_model_name=str(tmp_path / 'bert'))
    model = create_model(name, force_image_size=64, force_context_length=12,
                         text_cfg=text_cfg,
                         output_dict=True).eval()
    torch.testing.assert_close(model.text.transformer.embeddings.word_embeddings.weight,
                               pretrained.embeddings.word_embeddings.weight)
    image = torch.randn(2, 2, 1, 64, 64)
    ids = torch.tensor([[101, 15, 17, 102] + [0] * 8, [101, 16, 18, 102] + [0] * 8])
    with torch.no_grad():
        expected = model.encode_image(image, normalize=True)
        if local:
            from open_clip.local_region import encode_image_regions
            global_features, regions = encode_image_regions(model, image)
            torch.testing.assert_close(global_features, expected)
            assert regions.shape == (2, 2, 2, 2, 128)
    task = CLIPTask(model).train()
    losses, _ = task.training_forward({'image': image, 'text': ids})
    assert torch.isfinite(losses['loss'])
    losses['loss'].backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
    params = [model.visual.trunk.conv1.weight, model.visual.head.fusion.view_embed,
              model.visual.head.fusion.blocks[0].attention.in_proj_weight,
              model.visual.head.fusion.queries, model.visual.head.proj.weight,
              model.text.proj.weight]
    if local:
        params.extend([model.region_image_projection.weight, model.region_text_projection.weight])
    for param in params:
        assert param.grad is not None and torch.isfinite(param.grad).all()
        assert param.grad.abs().sum() > 0
    # In-memory serialization avoids new large persistent checkpoints.
    import io
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = create_model(name, force_image_size=64, force_context_length=12,
                            text_cfg=text_cfg,
                            output_dict=True).eval()
    restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
    model.eval()
    with torch.no_grad():
        torch.testing.assert_close(restored.encode_image(image), model.encode_image(image))


@pytest.mark.parametrize('options', [dict(width=30), dict(layers=0), dict(heads=3),
                                     dict(num_queries=0), dict(dropout=1.)])
def test_invalid_attention_configuration(options):
    with pytest.raises(ValueError):
        SpatialViewAttention(16, 2, **options)


def test_attention_bfloat16_autocast():
    model = SpatialViewAttention(16, 2, width=16, heads=4, dropout=0.)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        output = model(torch.randn(2, 2, 16, 3, 4))
        loss = output.float().square().mean()
    loss.backward()
    assert torch.isfinite(output).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
