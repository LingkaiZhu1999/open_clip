"""Offline tiny-BERT tests for pretrained PET global/local alignment."""
import pytest
import torch
from transformers import BertConfig, BertModel

from open_clip.model import CustomTextCLIP
from open_clip.task import CLIPTask
from open_clip.local_region import RegionClipLoss
from test_pet_region_loss import CONFIG


def test_hf_pretrained_weights_masks_gradients_and_validation(tmp_path):
    torch.manual_seed(8)
    cfg = BertConfig(vocab_size=200, hidden_size=32, num_hidden_layers=1,
                     num_attention_heads=4, intermediate_size=64, max_position_embeddings=16,
                     pad_token_id=0, cls_token_id=101, sep_token_id=102)
    pretrained = BertModel(cfg, add_pooling_layer=False)
    pretrained.save_pretrained(tmp_path)
    model = CustomTextCLIP(
        embed_dim=16,
        vision_cfg=dict(timm_model_name='resnet34', timm_pool='avg', timm_proj='linear',
                        timm_model_kwargs={'in_chans':1}, timm_num_views=2, timm_view_fusion='concat'),
        text_cfg=dict(hf_model_name=str(tmp_path), hf_model_pretrained=True,
                      hf_proj_type='linear', hf_pooler_type='mean_pooler'),
        region_cfg=CONFIG, output_dict=True)
    for name, value in pretrained.state_dict().items():
        torch.testing.assert_close(model.text.transformer.state_dict()[name], value, rtol=0, atol=0)
    # Content IDs may exceed SEP: unlike native CLIP, argmax is not an EOT locator.
    ids = torch.tensor([[101, 150, 12, 102, 0, 0], [101, 155, 102, 0, 0, 0]])
    image = torch.randn(2,2,1,64,64)
    model.eval()
    with torch.no_grad():
        output = model(image,ids,return_local=True)
        assert output['local_text_valid'].sum(1).tolist() == [2,1]
        torch.testing.assert_close(output['text_features'],model.encode_text(ids,normalize=True))
        hidden = model.text.transformer(input_ids=ids, attention_mask=(ids!=0).long()).last_hidden_state
        projected = torch.nn.functional.normalize(model.region_text_projection(hidden),dim=-1)
        torch.testing.assert_close(output['local_text_features'][0,:2],projected[0,1:3])
    model.train()
    task=CLIPTask(model)
    losses,_=task.training_forward({'image':image,'text':ids})
    losses['loss'].backward()
    for param in (model.text.transformer.embeddings.word_embeddings.weight,
                  model.text.proj.weight, model.region_text_projection.weight):
        assert param.grad is not None and torch.isfinite(param.grad).all()
        assert param.grad.abs().sum()>0
    task.eval()
    with torch.inference_mode():
        output=task({'image':image,'text':ids})
        local=RegionClipLoss(CONFIG).validation_loss(output['region_features'],
                            output['local_text_features'],output['local_text_valid'])
    assert torch.isfinite(local)
    clone={k:v.clone() for k,v in model.state_dict().items()}
    model.load_state_dict(clone,strict=True)


def test_windowed_bert_preserves_tail_tokens_and_native_position_weights(tmp_path):
    from open_clip.hf_model import HFTextEncoder
    cfg = BertConfig(vocab_size=200, hidden_size=16, num_hidden_layers=1,
                     num_attention_heads=2, intermediate_size=32, max_position_embeddings=8,
                     pad_token_id=0, cls_token_id=101, sep_token_id=102,
                     hidden_dropout_prob=0., attention_probs_dropout_prob=0.)
    ref = BertModel(cfg, add_pooling_layer=False)
    ref.save_pretrained(tmp_path)
    encoder = HFTextEncoder(str(tmp_path), output_dim=8, proj_type='linear',
                            pooler_type='mean_pooler', windowed_context_length=20).eval()
    assert encoder.context_length==20 and encoder.config.max_position_embeddings==8
    torch.testing.assert_close(encoder.transformer.embeddings.position_embeddings.weight,
                               ref.embeddings.position_embeddings.weight,rtol=0,atol=0)
    ids=torch.tensor([[101]+list(range(10,25))+[102,0,0,0]])
    pooled,hidden=encoder(ids,return_tokens=True)
    assert hidden.shape==(1,20,16)
    assert torch.count_nonzero(hidden[:,16:])==0
    assert torch.count_nonzero(hidden[:,0])==0
    assert torch.count_nonzero(hidden[:,13:16])>0  # final nonempty window
    changed=ids.clone();changed[0,15]=40
    other,_=encoder(changed,return_tokens=True)
    assert not torch.allclose(pooled,other)
    expected=encoder.proj(hidden[:,1:16].mean(1))
    torch.testing.assert_close(pooled,expected)
    pooled.square().sum().backward()
    assert encoder.transformer.embeddings.word_embeddings.weight.grad[24].abs().sum()>0
    with pytest.raises(ValueError,match='context'):
        encoder(torch.zeros(1,21,dtype=torch.long))
