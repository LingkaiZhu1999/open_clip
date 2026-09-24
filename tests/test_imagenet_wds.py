"""Integration coverage for the fork's ImageNet WebDataset evaluation loader."""
import io
import pickle
import tarfile
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from torchvision.transforms import ToTensor

from open_clip_train.data import get_imagenet


@pytest.mark.parametrize("workers", [0, 1])
def test_imagenet_wds_preserves_labels_and_partial_batches(tmp_path, workers):
    for shard in range(2):
        with tarfile.open(tmp_path / f"val-{shard:02d}.tar", "w") as archive:
            for index in range(3):
                image = io.BytesIO()
                Image.new("RGB", (8, 8)).save(image, format="PNG")
                for suffix, value in (
                    ("png", image.getvalue()),
                    ("cls", str(shard * 3 + index).encode()),
                ):
                    member = tarfile.TarInfo(f"{index}.{suffix}")
                    member.size = len(value)
                    archive.addfile(member, io.BytesIO(value))

    args = SimpleNamespace(
        imagenet_val=str(tmp_path / "val-{00..01}.tar"),
        batch_size=4,
        workers=workers,
    )
    loader = get_imagenet(args, (ToTensor(), ToTensor()), "val").dataloader
    # Upstream supports spawn/forkserver workers, so pipeline callables must pickle.
    pickle.dumps(loader)
    batches = list(loader)
    assert [len(labels) for _, labels in batches] == [4, 2]
    assert torch.cat([labels for _, labels in batches]).tolist() == list(range(6))
    assert all(images.shape[1:] == (3, 8, 8) for images, _ in batches)
