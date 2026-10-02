"""Single-GPU PET contrastive accumulation with one full-batch loss evaluation.

Cache microbatch features, differentiate the combined loss at the feature level,
then replay each encoder microbatch with its feature gradients. RNG replay keeps
BERT dropout identical; BatchNorm running buffers update only during caching.
"""
from contextlib import contextmanager

import torch

from .utils import backward


@contextmanager
def replay_rng(state, device):
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        torch.set_rng_state(state[0])
        if devices:
            torch.cuda.set_rng_state(state[1], device)
        yield


@contextmanager
def preserve_batchnorm_buffers(model):
    buffers = [(buffer, buffer.clone()) for module in model.modules()
               if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
               for buffer in module.buffers(recurse=False) if buffer is not None]
    try:
        yield
    finally:
        with torch.no_grad():
            for buffer, saved in buffers:
                buffer.copy_(saved)


def pet_cached_accum_step(task, batch, accum_state, optimizer, scaler, autocast, args, finish_step):
    batches, cache = accum_state
    device = batch['image'].device
    rng = (torch.get_rng_state(), torch.cuda.get_rng_state(device) if device.type == 'cuda' else None)
    with torch.no_grad(), autocast():
        output = task.trainable_module(**batch)
    cache.setdefault('rng', []).append(rng)
    cache.setdefault('outputs', []).append(output)
    batches.append(batch)
    if len(batches) < args.accum_freq:
        return None

    optimizer.zero_grad()
    scalar_keys = ('logit_scale', 'logit_bias')
    outputs = cache['outputs']
    features = {key: torch.cat([out[key] for out in outputs]).detach()
                for key in outputs[0] if key not in scalar_keys}
    for value in features.values():
        if value.is_floating_point():
            value.requires_grad_(True)
    scalars = {key: outputs[-1][key].detach().requires_grad_(True)
               for key in scalar_keys if outputs[-1].get(key) is not None}
    with autocast():
        losses, report = task.compute_accum_loss(features, scalars, batches)
        total = sum(value for key, value in losses.items() if key.endswith('_loss'))
    backward(total, scaler)
    # These gradients are already scaled when GradScaler is active; do not scale
    # the replay backward again. finish_step unscales/clips/steps just once.
    feature_grads = {key: value.grad for key, value in features.items() if value.grad is not None}
    scalar_grads = {key: value.grad for key, value in scalars.items() if value.grad is not None}
    result_losses = {key: value.detach() for key, value in losses.items()}
    result_losses['loss'] = total.detach()
    report = {key: value.detach() for key, value in report.items()}
    del losses, total, features, scalars

    offset = 0
    with preserve_batchnorm_buffers(task.trainable_module):
        for j, microbatch in enumerate(batches):
            size = task.batch_size(microbatch)
            with replay_rng(cache['rng'][j], device):
                with autocast():
                    live = task.trainable_module(**microbatch)
                values, grads = [], []
                for key, grad in feature_grads.items():
                    if live[key].requires_grad:
                        values.append(live[key])
                        grads.append(grad[offset:offset + size])
                if j == len(batches) - 1:
                    for key, grad in scalar_grads.items():
                        if live[key].requires_grad:
                            values.append(live[key])
                            grads.append(grad)
                torch.autograd.backward(values, grads)
                del live, values, grads
            offset += size
    finish_step(task, optimizer, scaler, args)
    return result_losses, report, offset, ([], {})
