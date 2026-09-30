from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .image_text_task import ImageTextTask
from .base_task import unwrap_model


class CLIPTask(ImageTextTask):
    """Standard CLIP training task wrapping model + ClipLoss."""

    def __init__(
            self,
            model: nn.Module,
            *,
            loss: Optional[nn.Module] = None,
            default_loss: bool = True,
            local_loss: bool = False,
            gather_with_grad: bool = False,
            cache_labels: bool = True,
            rank: int = 0,
            world_size: int = 1,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
            verbose: bool = True,
    ):
        super().__init__(model, device=device, dtype=dtype, verbose=verbose)
        if loss is not None:
            self.loss = loss
        elif default_loss:
            from open_clip.loss import ClipLoss
            region_cfg = getattr(unwrap_model(model), 'region_cfg', None)
            if region_cfg is not None:
                from open_clip.local_region import RegionClipLoss
                loss_class = lambda **options: RegionClipLoss(region_cfg, **options)
            else:
                loss_class = ClipLoss
            self.loss = loss_class(
                local_loss=local_loss,
                gather_with_grad=gather_with_grad,
                cache_labels=cache_labels,
                rank=rank,
                world_size=world_size,
            )
        # else: eval-only construction, no self.loss attribute

    def training_forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[Dict, Dict]:
        model_out = self.trainable_module(image=batch["image"], text=batch["text"])
        losses = self.loss(**model_out, output_dict=True)
        total_loss = sum(v for k, v in losses.items() if k.endswith('_loss'))
        losses["loss"] = total_loss
        return losses, self._report(model_out)

    def eval_forward(self, batch: Dict[str, torch.Tensor]):
        inputs = {key: batch[key] for key in self.data_keys if key in batch}
        model = self.get_trainable_module(use_ema=True)
        if getattr(unwrap_model(model), 'region_cfg', None) is not None and all(k in inputs for k in ('image', 'text')):
            inputs['return_local'] = True
        return model(**inputs)
