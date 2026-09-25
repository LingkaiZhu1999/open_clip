"""Read PET pairs in place, with an optional shared RAM image cache."""
import csv
import hashlib
import logging
import os
from collections import Counter, defaultdict
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .pet_text import extract_pet_sections

KEY = ('MRN_DID', 'ACCESSION_DID')


def read_rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def index_rows(rows):
    indexed = defaultdict(list)
    for row in rows:
        key = tuple(row[k].strip() for k in KEY)
        if not all(key):
            raise ValueError('Empty patient/scan key')
        indexed[key].append(row)
    return indexed


def pair_records(scans, reports, splits, split_column='split0'):
    scan_index, report_index, split_index = map(index_rows, (scans, reports, splits))
    patients = defaultdict(set)
    for row in splits:
        partition = row[split_column]
        if partition not in {'train', 'val', 'test'}:
            raise ValueError('Unknown split label')
        patients[row['MRN_DID']].add(partition)
    if any(len(partitions) != 1 for partitions in patients.values()):
        raise ValueError('Patient leakage across partitions')
    counts = Counter()
    records = []
    for key, rows in sorted(scan_index.items()):
        if len(rows) != 1 or len(report_index.get(key, [])) != 1 or len(split_index.get(key, [])) != 1:
            counts['ambiguous_or_unmatched_keys'] += 1
            continue
        report = report_index[key][0]['Scan_Report'].strip()
        if not report or report.lower() in {'nan', 'none', 'null'}:
            counts['empty_reports'] += 1
            continue
        record = dict(rows[0])
        record['report'] = report
        record['partition'] = split_index[key][0][split_column]
        # Stable opaque pair identifier; records and report text stay in memory.
        record['pair_id'] = hashlib.sha256('\0'.join(key).encode()).hexdigest()
        records.append(record)
    counts['paired_scans'] = len(records)
    counts['patients'] = len({r['MRN_DID'] for r in records})
    return records, dict(counts)


def binary_path(root, filename):
    directory = (root / 'PET_binaries').resolve()
    path = (directory / filename).resolve()
    if path.parent != directory:
        raise ValueError('Binary path must be directly inside PET_binaries')
    return path


def load_view(root, record, view):
    dim = '1' if view == 'cor' else '2'
    shape = (int(record['matrixsize_3']), int(record['matrixsize_' + dim]))
    if min(shape) <= 0:
        raise ValueError('Invalid image dimensions')
    path = binary_path(root, record['filename_2d_' + view])
    if path.stat().st_size != shape[0] * shape[1] * 4:
        raise ValueError('Binary byte size does not match metadata')
    pixels = np.fromfile(path, dtype='<f4').reshape(shape)
    if not np.isfinite(pixels).all() or (pixels < 0).any():
        raise ValueError('Nonfinite or negative PET pixels')
    spacing = (float(record['pixelsize_' + dim]), float(record['pixelsize_3']))
    if not all(np.isfinite(v) and v > 0 for v in spacing):
        raise ValueError('Invalid pixel spacing')
    return pixels, spacing


SUV_MEAN = 2.13
SUV_STD = 3.39


def crop_nonzero(pixels):
    rows = np.flatnonzero(np.any(pixels, axis=1))
    cols = np.flatnonzero(np.any(pixels, axis=0))
    if not len(rows) or not len(cols):
        return np.zeros((1, 1), dtype=np.float32)
    # Include the boundary pixels (the reference implementation drops the last
    # nonzero row/column through its exclusive slicing upper bounds).
    return pixels[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]


def read_cropped_views(root, record, size=310):
    crops = []
    for view in ('cor', 'sag'):
        pixels, (sx, sy) = load_view(root, record, view)
        if not np.isclose(sx, sy):
            raise ValueError('Reference PET padding requires isotropic in-plane spacing')
        crop = crop_nonzero(pixels)
        if max(crop.shape) > size:
            raise ValueError('PET nonzero bounding box exceeds canvas; increase --pet-canvas-size')
        crops.append(crop)
    return crops


def prepare_view(crop, size=310, intensity_max=30., training=False, output_size=None):
    """Reference float preprocessing; keep augmentations outside the RAM cache."""
    height, width = crop.shape
    if height > size or width > size:
        raise ValueError('PET crop exceeds canvas')
    if training:
        # Match the reference's offset sampling, with exact-fit cases supported.
        col = np.random.randint(0, size - width) if width < size else 0
        row = np.random.randint(0, size - height) if height < size else 0
    else:
        row, col = (size - height) // 2, (size - width) // 2
    canvas = np.zeros((size, size), dtype=np.float32)
    canvas[row:row + height, col:col + width] = crop
    canvas = torch.from_numpy(np.clip(canvas, 0., intensity_max)).unsqueeze(0)
    if output_size is not None and output_size != size:
        if output_size < 1:
            raise ValueError('PET output size must be positive')
        canvas = torch.nn.functional.interpolate(
            canvas.unsqueeze(0), size=(output_size, output_size),
            mode='bilinear', align_corners=False, antialias=True,
        ).squeeze(0)
    canvas = (canvas - SUV_MEAN) / SUV_STD
    if training:
        # Reference RandomScale runs AFTER normalization, including background.
        canvas *= np.random.uniform(0.85, 1.15)
    return canvas


def read_pairs(root, split_column='split0'):
    root = Path(root)
    return pair_records(
        read_rows(root / 'data_scan_did.csv'),
        read_rows(root / 'data_Scan_Report.csv'),
        read_rows(root / 'data_splits.csv'), split_column,
    )


class PetReportDataset(Dataset):
    """One report with independent coronal/sagittal float PET views; optional unaugmented shared RAM cache."""
    def __init__(self, root, tokenizer, partition='train', split_column='split0',
                 image_size=310, intensity_max=30., limit=0, variable_text=False,
                 output_text_mask=False, cache='none', canvas_size=310, report_sections='findings_impression'):
        from open_clip_train.data import TokenizeText

        if partition not in {'train', 'val', 'test'}:
            raise ValueError('Unknown partition')
        if limit < 0:
            raise ValueError('PET sample limit must be nonnegative')
        if image_size < 1 or canvas_size < 1 or not np.isfinite(intensity_max) or intensity_max <= 0:
            raise ValueError('Invalid PET preprocessing settings')
        if cache not in {'none', 'ram'}:
            raise ValueError('Unknown PET cache mode')
        self.root = Path(root)
        records, self.stats = read_pairs(self.root, split_column)
        if report_sections not in {'full', 'findings_impression'}:
            raise ValueError('Unknown PET report section mode')
        self.report_sections = report_sections
        self.records = []
        section_counts = Counter()
        for record in records:
            if record['partition'] != partition:
                continue
            if report_sections == 'findings_impression':
                selected, sections = extract_pet_sections(record['report'])
                if not selected:
                    section_counts['excluded_no_sections'] += 1
                    continue
                record['full_report'] = record['report']
                record['report'] = selected
                record['report_sections'] = sections
                if set(sections) == {'FINDINGS', 'IMPRESSION'}:
                    section_key = 'both'
                else:
                    section_key = '_'.join(sorted(name.lower() for name in sections))
                    if len(sections) == 1:
                        section_key += '_only'
                section_counts[section_key] += 1
            self.records.append(record)
        self.section_stats = dict(section_counts)
        logging.getLogger(__name__).info('PET %s report mode=%s: %s; retained %d pairs',
                                        partition, report_sections, self.section_stats, len(self.records))
        if limit:
            self.records = self.records[:limit]
        if not self.records:
            raise ValueError('No unambiguous PET/report pairs in selected partition')
        self.image_size = image_size
        self.canvas_size = canvas_size
        self.intensity_max = intensity_max
        self.training = partition == 'train'
        self.variable_text = variable_text
        self.tokenize_text = TokenizeText(tokenizer, variable=variable_text, output_mask=output_text_mask)
        self.image_cache = None
        self.crop_shapes = None
        if cache == 'ram':
            required_bytes = len(self.records) * 2 * canvas_size * canvas_size * 4
            if Path('/dev/shm').is_dir():
                capacity = os.statvfs('/dev/shm')
                if required_bytes > capacity.f_bavail * capacity.f_frsize:
                    raise ValueError('Not enough shared RAM for PET images; use --pet-cache none or enlarge /dev/shm')
            logger = logging.getLogger(__name__)
            logger.info('Caching %d %s PET pairs in %.2f GiB shared RAM',
                        len(self.records), partition, required_bytes / 2**30)
            self.image_cache = torch.zeros(
                (len(self.records), 2, canvas_size, canvas_size), dtype=torch.float32,
            ).share_memory_()
            self.crop_shapes = torch.empty((len(self.records), 2, 2), dtype=torch.int32).share_memory_()
            for index, record in enumerate(self.records):
                for view, crop in enumerate(read_cropped_views(self.root, record, canvas_size)):
                    h, w = crop.shape
                    self.image_cache[index, view, :h, :w].copy_(torch.from_numpy(crop))
                    self.crop_shapes[index, view] = torch.tensor((h, w), dtype=torch.int32)
                if (index + 1) % 1000 == 0:
                    logger.info('PET %s RAM cache: %d/%d pairs', partition, index + 1, len(self.records))
            logger.info('PET %s RAM cache ready', partition)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        if self.image_cache is None:
            crops = read_cropped_views(self.root, record, self.canvas_size)
        else:
            crops = [self.image_cache[index, view, :int(shape[0]), :int(shape[1])].numpy()
                     for view, shape in enumerate(self.crop_shapes[index])]
        image = torch.stack([prepare_view(crop, self.canvas_size, self.intensity_max, self.training,
                                          output_size=self.image_size)
                             for crop in crops])
        text = record['report'] if self.variable_text else [record['report']]
        return self.tokenize_text.map_sample({'image': image, 'text': text})


def get_pet_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None,
                    naflex_data_config=None, model_traits=None):
    from open_clip_train.data import (
        collate_variable_text_dicts, create_map_loader, get_text_pad_id, resolve_text_layout,
    )

    if naflex_data_config is not None:
        raise ValueError('The PET loader currently supports fixed-size image models only')
    # PET emits already-normalized float tensors. Never apply the RGB/PIL transform.
    _, variable_text = resolve_text_layout(args, model_traits)
    dataset = PetReportDataset(
        args.train_data if is_train else args.val_data, tokenizer,
        partition='train' if is_train else 'val', split_column=args.pet_split_column,
        image_size=(args.pet_image_size if is_train or args.pet_val_image_size is None
                    else args.pet_val_image_size),
        canvas_size=args.pet_canvas_size, intensity_max=args.pet_intensity_max,
        report_sections=args.pet_report_sections,
        limit=args.pet_limit_per_split, variable_text=variable_text, cache=args.pet_cache,
        output_text_mask=bool(getattr(args, 'text_attention_mask', None)) and not variable_text,
    )
    collate_fn = None
    if variable_text:
        collate_fn = partial(
            collate_variable_text_dicts, pad_id=get_text_pad_id(tokenizer),
            text_pad_multiple=getattr(args, 'text_pad_multiple', None),
            text_pad_cap=getattr(tokenizer, 'context_length', None),
        )
    if is_train and len(dataset) < args.batch_size * (args.world_size if args.distributed else 1):
        raise ValueError('PET training partition is smaller than one full global batch')
    return create_map_loader(dataset, args, is_train, collate_fn=collate_fn)
