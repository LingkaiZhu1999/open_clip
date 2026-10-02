import math

import pytest
import torch

from open_clip_train.best_checkpoint import ValidationTracker, atomic_save_checkpoint
from open_clip_train.params import parse_args


def test_patience_resets_on_improvement_and_ties_count():
    tracker = ValidationTracker(patience=2)
    assert tracker.update(5, 1) == (True, False)
    assert tracker.update(5, 2) == (False, False)
    assert tracker.update(4, 3) == (True, False)
    assert tracker.update(4.1, 4) == (False, False)
    assert tracker.update(4.2, 5) == (False, True)
    assert (tracker.best, tracker.best_epoch) == (4, 3)


@pytest.mark.parametrize('value', [math.nan, math.inf, -math.inf])
def test_nonfinite_metric_is_rejected(value):
    with pytest.raises(ValueError, match='Non-finite'):
        ValidationTracker().update(value, 1)


def test_disabled_patience_still_tracks_best():
    tracker = ValidationTracker()
    tracker.update(1, 1)
    for epoch in range(2, 8):
        assert tracker.update(2, epoch) == (False, False)


def test_failed_write_preserves_previous_checkpoint(tmp_path, monkeypatch):
    path = tmp_path / 'best.pt'
    atomic_save_checkpoint({'epoch': 2, 'weights': torch.ones(3)}, path)
    def failed_save(obj, handle):
        handle.write(b'incomplete')
        raise OSError('Disk full')
    with monkeypatch.context() as m:
        m.setattr(torch, 'save', failed_save)
        with pytest.raises(OSError):
            atomic_save_checkpoint({'epoch': 3}, path)
    assert torch.load(path, weights_only=True)['epoch'] == 2
    assert list(tmp_path.iterdir()) == [path]
    atomic_save_checkpoint({'epoch': 4}, path)
    assert torch.load(path, weights_only=True)['epoch'] == 4
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize('extra', [['--fsdp'], ['--resume', 'latest'], ['--save-most-recent'],
                                  ['--delete-previous-checkpoint'], ['--val-frequency', '2']])
def test_best_only_rejects_unsupported_modes(extra):
    with pytest.raises(ValueError, match='fresh non-FSDP'):
        parse_args(['--save-best-only', '--val-data', 'dummy'] + extra)


def test_hpo_grid_keeps_one_winner_per_experiment(tmp_path, monkeypatch):
    """Exercise the complete scheduler/reporting loop without GPU training."""
    import importlib.util
    import json
    from pathlib import Path
    import sys
    from types import SimpleNamespace

    source = Path(__file__).resolve().parents[1] / 'scripts/run_pet_hpo.py'
    spec = importlib.util.spec_from_file_location('pet_hpo_runner_test', source)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    calls = []

    def fake_trial(cmd, **kwargs):
        calls.append(cmd)
        def arg(key):
            return cmd[cmd.index(key) + 1]
        name = arg('--name')
        key = name.split('-lr')[0][-1]
        assert arg('--batch-size') == '256'
        assert arg('--accum-freq') == '2'
        assert kwargs['env']['NPROC_PER_NODE'] == '1'
        assert kwargs['env']['CUDA_VISIBLE_DEVICES'] == '0'
        assert '--save-best-only' in cmd
        assert ('--pretrained-image' in cmd) == (key == 'C')
        assert ('--pet-region-loss-weight' in cmd) == (key == 'B')
        if key == 'B':
            assert arg('--pet-region-loss-weight') == '0.1'
        score = 5 - (float(arg('--lr')) == 5e-5) - 0.1 * (int(arg('--warmup')) == 60)
        result_dir = Path(arg('--logs')) / name / 'checkpoints'
        result_dir.mkdir(parents=True)
        (result_dir / 'results.jsonl').write_text('\n'.join(
            json.dumps({'epoch': epoch, 'clip_val_loss': loss})
            for epoch, loss in [(1, score + 0.5), (2, score)]))
        if score < float(arg('--best-checkpoint-threshold')):
            path = Path(arg('--best-checkpoint-path'))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(name)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, 'run', fake_trial)
    monkeypatch.setattr(runner.subprocess, 'check_output', lambda *a, **k: 'test revision\n')
    monkeypatch.setattr(runner.shutil, 'disk_usage', lambda p: SimpleNamespace(free=10 * 1024**3))
    monkeypatch.setattr(sys, 'argv', ['run_pet_hpo.py', '--output', str(tmp_path / 'search')])
    runner.main()
    status = json.loads((tmp_path / 'search/status.json').read_text())
    assert status['state'] == 'completed'
    assert len(calls) == 12
    assert len(list((tmp_path / 'search').rglob('*.pt'))) == 3
    for key, winner in status['winners'].items():
        assert winner['lr'] == 5e-5 and winner['warmup'] == 60
        assert winner['best_epoch'] == 2
        assert (tmp_path / 'search' / key / 'best.pt').read_text() == winner['name']
