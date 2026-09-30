"""Run with torchrun --nproc_per_node=2; REGION_TEST_BACKEND=gloo or nccl."""
import os
import torch
import torch.distributed as dist
from torch.nn import functional as F
from open_clip.local_region import RegionClipLoss

backend = os.environ.get('REGION_TEST_BACKEND', 'gloo')
rank = int(os.environ['RANK'])
device = torch.device('cuda', int(os.environ['LOCAL_RANK'])) if backend == 'nccl' else torch.device('cpu')
if device.type == 'cuda':
    torch.cuda.set_device(device)
dist.init_process_group(backend)
torch.manual_seed(7)
config = dict(chunk_size=2, weight=0.7, attention_temperature=0.1, word_temperature=0.1, contrastive_temperature=0.07)
# Each process independently constructs the same global batch as an oracle.
global_inputs = [F.normalize(torch.randn(*shape, device=device), dim=-1).requires_grad_()
                 for shape in [(4,8), (4,8), (4,5,8), (4,6,8)]]
valid = torch.tensor([[True]*6, [True]*4+[False]*2, [True]*3+[False]*3, [True]*5+[False]],device=device)
scale = torch.tensor(2., device=device)
reference = RegionClipLoss(config)(global_inputs[0],global_inputs[1],scale,global_inputs[2],global_inputs[3],valid)
reference.backward()
local = [x.detach()[rank*2:(rank+1)*2].clone().requires_grad_() for x in global_inputs]
criterion = RegionClipLoss(config,rank=rank,world_size=2,local_loss=True,gather_with_grad=True)
actual = criterion(local[0],local[1],scale,local[2],local[3],valid[rank*2:(rank+1)*2])
actual.backward()
mean_loss=actual.detach().clone()
dist.all_reduce(mean_loss); mean_loss /= 2
torch.testing.assert_close(mean_loss,reference.detach(),rtol=1e-5,atol=1e-5)
for original, shard in zip(global_inputs,local):
    # DDP subsequently averages parameter gradients across the two ranks.
    torch.testing.assert_close(shard.grad / 2, original.grad[rank*2:(rank+1)*2],rtol=2e-4,atol=2e-5)
if rank==0: print(f'{backend}: distributed loss and all feature gradients match global-batch oracle',flush=True)
dist.destroy_process_group()
