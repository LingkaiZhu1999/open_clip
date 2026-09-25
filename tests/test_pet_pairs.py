"""Synthetic checks for PET pairing, leakage prevention, and binary decoding."""
import csv

import numpy as np
import pytest
import torch

from open_clip_train import pet_data as pet
from open_clip_train.data import get_dataset_fn
from open_clip_train.params import parse_args
from open_clip.model_traits import CLIP_TRAITS
from util_test import VariableTokenizer


def row(patient='p1', scan='s1', **kwargs):
    return dict(MRN_DID=patient, ACCESSION_DID=scan, **kwargs)


def test_pairing_uses_patient_and_accession_not_csv_order():
    scans = [row(scan='s2'), row()]
    reports = [row(Scan_Report='first'), row(scan='s2', Scan_Report='second')]
    splits = [row(split0='train'), row(scan='s2', split0='train')]
    pairs, counts = pet.pair_records(scans, reports, splits)
    assert [(r['ACCESSION_DID'], r['report']) for r in pairs] == [('s1', 'first'), ('s2', 'second')]
    assert counts['paired_scans'] == 2


def test_ambiguous_reports_are_excluded():
    pairs, counts = pet.pair_records([row()], [row(Scan_Report='one'), row(Scan_Report='two')], [row(split0='train')])
    assert not pairs
    assert counts['ambiguous_or_unmatched_keys'] == 1


def test_patient_leakage_is_rejected():
    with pytest.raises(ValueError, match='leakage'):
        pet.pair_records([], [], [row(split0='train'), row(scan='s2', split0='test')])


def test_reference_normalization_preserves_float_values():
    crop = np.array([[0., 2.13, 10., 30., 40.]], dtype=np.float32)
    image = pet.prepare_view(crop, size=5)
    expected = (np.clip(crop, 0, 30) - 2.13) / 3.39
    assert image.dtype == torch.float32
    np.testing.assert_allclose(image[0, 2].numpy(), expected[0], rtol=1e-6)
    assert image[0, 0, 0] == pytest.approx(-2.13 / 3.39)
    assert image[0, 2, -1] == image[0, 2, -2]


def test_bbox_includes_boundary_pixels_and_handles_empty():
    image = np.zeros((6, 7), dtype=np.float32)
    image[1:5, 2:6] = 3.
    assert pet.crop_nonzero(image).shape == (4, 4)
    assert pet.crop_nonzero(np.zeros((3, 3))).shape == (1, 1)
    assert pet.crop_nonzero(np.array([[2.]], dtype=np.float32)).item() == 2.


def test_training_scales_after_normalization(monkeypatch):
    monkeypatch.setattr(np.random, 'randint', lambda low, high: high - 1)
    monkeypatch.setattr(np.random, 'uniform', lambda low, high: 1.15)
    image = pet.prepare_view(np.full((2, 2), 30., dtype=np.float32), size=4, training=True)
    expected = np.zeros((4, 4), dtype=np.float32)
    expected[1:3, 1:3] = 30.
    np.testing.assert_allclose(image[0], ((expected - 2.13) / 3.39) * 1.15, rtol=1e-6)


def test_exact_fit_and_oversized_crop():
    assert pet.prepare_view(np.ones((4, 4), dtype=np.float32), size=4, training=True).shape == (1, 4, 4)
    with pytest.raises(ValueError, match='exceeds'):
        pet.prepare_view(np.ones((5, 2), dtype=np.float32), size=4)


def test_binary_path_rejects_escape(tmp_path):
    with pytest.raises(ValueError, match='directly inside'):
        pet.binary_path(tmp_path, '../other.bin')


@pytest.fixture
def source_dataset(tmp_path):
    root = tmp_path / 'source'
    (root / 'PET_binaries').mkdir(parents=True)
    for view in ('cor', 'sag'):
        np.ones((4, 2), dtype='<f4').tofile(root / 'PET_binaries' / (view + '.bin'))
    scans, reports, splits = [], [], []
    for partition, count in [('train', 3), ('val', 2), ('test', 1)]:
        for index in range(count):
            key = row(patient=partition, scan=str(index))
            scans.append(dict(key, matrixsize_1=2, matrixsize_2=2, matrixsize_3=4,
                              pixelsize_1=1, pixelsize_2=1, pixelsize_3=1,
                              filename_2d_cor='cor.bin', filename_2d_sag='sag.bin'))
            reports.append(dict(key, Scan_Report='FINDINGS: ' + str(index)))
            splits.append(dict(key, split0=partition))
    for name, rows in [('data_scan_did.csv', scans), ('data_Scan_Report.csv', reports), ('data_splits.csv', splits)]:
        with (root / name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return root


def test_direct_dataset_is_lazy_and_writes_no_files(source_dataset, monkeypatch):
    before = {p: p.stat().st_mtime_ns for p in source_dataset.rglob('*')}
    calls = []
    original = pet.read_cropped_views

    def render(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(pet, 'read_cropped_views', render)
    dataset = pet.PetReportDataset(source_dataset, VariableTokenizer(), image_size=8, canvas_size=8)
    assert len(dataset) == 3
    assert not calls  # No image reads or rendering at construction.
    sample = dataset[0]
    assert len(calls) == 1
    assert sample['image'].shape == (2, 1, 8, 8)
    assert sample['text'].shape == (16,)
    assert before == {p: p.stat().st_mtime_ns for p in source_dataset.rglob('*')}


@pytest.mark.parametrize('cache', ['none', 'ram'])
@pytest.mark.parametrize('workers', [0, 1])
@pytest.mark.parametrize('variable_text', [False, True])
def test_pet_loader_batches_and_split_selection(source_dataset, workers, variable_text, cache):
    args = parse_args(['--dataset-type', 'pet', '--train-data', str(source_dataset),
                       '--val-data', str(source_dataset), '--batch-size', '2',
                       '--workers', str(workers), '--pet-image-size', '8', '--pet-canvas-size', '8', '--pet-cache', cache])
    args.distributed = False
    args.variable_text = variable_text
    loader_fn = get_dataset_fn(str(source_dataset), args.dataset_type)
    train = loader_fn(args, None, True, tokenizer=VariableTokenizer(), model_traits=CLIP_TRAITS)
    val = loader_fn(args, None, False, tokenizer=VariableTokenizer(), model_traits=CLIP_TRAITS)
    if workers:
        train.dataloader.multiprocessing_context = 'spawn'
        val.dataloader.multiprocessing_context = 'spawn'
    assert train.dataloader.num_samples == 3
    assert val.dataloader.num_samples == 2
    assert {r['MRN_DID'] for r in train.dataloader.dataset.records} == {'train'}
    assert {r['MRN_DID'] for r in val.dataloader.dataset.records} == {'val'}
    for info in (train, val):
        if cache == 'ram':
            assert info.dataloader.dataset.image_cache.is_shared()
        batch = next(iter(info.dataloader))
        assert batch['image'].shape == (2, 2, 1, 8, 8)
        assert batch['text'].dtype == torch.long
        assert ('text_valid' in batch) == variable_text


def test_pet_smoke_limit_does_not_change_sources(source_dataset):
    dataset = pet.PetReportDataset(source_dataset, VariableTokenizer(), limit=2)
    assert len(dataset) == 2
    assert dataset.stats['paired_scans'] == 6
    assert len(pet.read_rows(source_dataset / 'data_scan_did.csv')) == 6


def test_ram_cache_matches_direct_views_and_avoids_repeated_reads(source_dataset, monkeypatch):
    before = {p: p.stat().st_mtime_ns for p in source_dataset.rglob('*')}
    dataset = pet.PetReportDataset(source_dataset, VariableTokenizer(), image_size=8, canvas_size=8, cache='ram', partition='val')
    direct = pet.PetReportDataset(source_dataset, VariableTokenizer(), image_size=8, canvas_size=8, partition='val')
    expected = direct[0]['image']

    def fail(*args, **kwargs):
        raise AssertionError('Cached sample should not reread a binary')

    monkeypatch.setattr(pet, 'read_cropped_views', fail)
    assert torch.equal(dataset[0]['image'], expected)
    assert torch.equal(dataset[0]['image'], expected)
    assert dataset.image_cache.is_shared()
    assert before == {p: p.stat().st_mtime_ns for p in source_dataset.rglob('*')}


def test_independent_view_order_and_binary_validation(source_dataset):
    np.full((4, 2), 30., dtype='<f4').tofile(source_dataset / 'PET_binaries/sag.bin')
    dataset = pet.PetReportDataset(source_dataset, VariableTokenizer(), image_size=8, canvas_size=8, partition='val')
    image = dataset[0]['image']
    assert image.shape == (2, 1, 8, 8)
    assert image[0, 0, 3, 3] == pytest.approx((1 - 2.13) / 3.39)
    assert image[1, 0, 3, 3] == pytest.approx((30 - 2.13) / 3.39)
    (source_dataset / 'PET_binaries/cor.bin').write_bytes(b'bad')
    with pytest.raises(ValueError, match='byte size'):
        dataset[0]


def test_cached_augmentation_is_not_frozen(source_dataset):
    dataset = pet.PetReportDataset(source_dataset, VariableTokenizer(), image_size=8, canvas_size=8, cache='ram')
    cache_before = dataset.image_cache.clone()
    np.random.seed(10)
    a = dataset[0]['image']
    np.random.seed(20)
    b = dataset[0]['image']
    assert not torch.equal(a, b)
    assert torch.equal(dataset.image_cache, cache_before)


def test_resize_clips_before_interpolation_and_keeps_normalization():
    crop = np.array([[60., 0.], [0., 0.]], dtype=np.float32)
    image = pet.prepare_view(crop, size=2, output_size=1)
    assert image.shape == (1, 1, 1)
    assert image.item() == pytest.approx((7.5 - pet.SUV_MEAN) / pet.SUV_STD)


@pytest.mark.parametrize('cache', ['none', 'ram'])
def test_output_smaller_than_crop_does_not_reject_or_truncate(source_dataset, cache):
    dataset = pet.PetReportDataset(source_dataset, VariableTokenizer(), partition='val',
                                   canvas_size=8, image_size=2, cache=cache)
    image = dataset[0]['image']
    crops = pet.read_cropped_views(source_dataset, dataset.records[0], size=8)
    expected = torch.stack([pet.prepare_view(c, size=8, output_size=2) for c in crops])
    torch.testing.assert_close(image, expected)
    assert image.shape == (2, 1, 2, 2)
    if cache == 'ram':
        assert dataset.image_cache.shape[-2:] == (8, 8)


def test_distinct_training_and_validation_resolutions(source_dataset):
    args = parse_args(['--dataset-type', 'pet', '--train-data', str(source_dataset),
                       '--val-data', str(source_dataset), '--batch-size', '2', '--workers', '0',
                       '--pet-canvas-size', '8', '--pet-image-size', '2', '--pet-val-image-size', '8'])
    args.distributed = False
    for training, size in [(True, 2), (False, 8)]:
        info = pet.get_pet_dataset(args, None, training, tokenizer=VariableTokenizer(), model_traits=CLIP_TRAITS)
        assert next(iter(info.dataloader))['image'].shape == (2, 2, 1, size, size)


def test_section_dataset_filters_unstructured_reports_and_keeps_full_mode(source_dataset):
    path = source_dataset / 'data_Scan_Report.csv'
    rows = pet.read_rows(path)
    rows[0]['Scan_Report'] = 'unstructured'
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    selected = pet.PetReportDataset(source_dataset, VariableTokenizer())
    full = pet.PetReportDataset(source_dataset, VariableTokenizer(), report_sections='full')
    assert len(selected) == 2 and len(full) == 3
    assert selected.section_stats['excluded_no_sections'] == 1
    assert selected.records[0]['report'] == 'FINDINGS:\n1'
    assert selected.records[0]['full_report'] == 'FINDINGS: 1'
    assert full.records[0]['report'] == 'unstructured'
