# PET–report OpenCLIP experiment log

Updated 2026-09-25. Train both encoders **from scratch** on paired PET projections
and report text. This document summarizes current code, verified behavior, and
open decisions; historical smoke metrics are not evidence of clinical utility.

## Latest completed run analysis (2026-09-25)

The 50-epoch run `pet-resnet34-concat-2gpu-20260924-165334-526474210-1063591`
completed in about 3 hours. It shows strong overfitting: epoch-50 exact-pair R@1
was 99.4% on a 2,466-pair training subset, versus 0.36–0.45% on validation at the
same 224-pixel resolution. Validation at 310 was worse (0.16–0.20%). Best logged
validation loss was at epoch 5; best mean R@10 was at epoch 24. Only epoch 50 is
retained. Keep matched-resolution validation and best-checkpoint retention as
priorities; these recommendations have not changed launcher defaults.

See [full analysis](pet_run_analysis_20260925.md) for curves, controlled resolution
checks, metric caveats, and next experiments. The reserved test split was untouched.

## Current configuration

| Component | Setting |
| --- | --- |
| Model | `PET-ResNet34`; random image/text weights; existing CLIP tokenizer reused |
| Images | Independent single-channel coronal and sagittal MIPs, in that order |
| Vision encoder | Shared ResNet34; spatial average pooling produces 512 features per view |
| Fusion | Concatenate to 1,024 features → bias-free linear projection to 512 → L2 normalization |
| Text encoder | Native CLIP causal Transformer: 12 layers, width 512, 8 heads, vocabulary 49,408 |
| Text representation | End-of-text pooling and projection to a normalized 512-dimensional embedding |
| Text input | Extracted Findings + Impression; Summary excluded; 2,560-token fixed context |
| Objective | Symmetric scan–report contrastive loss; one positive report per scan |
| Hardware | Two RTX A6000 GPUs; DDP/NCCL, one complete model per GPU |
| Batch / workers | 256 scans per GPU, global batch 512; 16 loader workers per rank |
| Precision | `amp_bf16`: BF16 autocasting, FP32 parameters, no gradient scaler |
| Optimization | AdamW, LR 5e-4, weight decay 0.1, 50 epochs, 200 warmup updates, seed 0 |
| Memory options | Gradient checkpointing and shared RAM image caches; compilation disabled by default |
| Evaluation / outputs | Validation each epoch on rank 0; W&B project `pet-resnet34-concat-scratch`; retain latest epoch checkpoint |

No separate pretrained LLM, report summarizer, or generation objective is used.
Both views stay on the same GPU. `--local-loss --gather-with-grad` compares each
rank's local embeddings against the global batch and propagates cross-rank
feature gradients; DDP synchronizes parameter gradients.

## Data and report audit

Source: `/mnt/snotra1/ida/datasets/lymphoma_mskcc_deid`.
`PET_binaries` contains little-endian float32 **2D MIPs**, not 3D volumes.
Metadata, reports, and splits come from `data_scan_did.csv`,
`data_Scan_Report.csv`, and `data_splits.csv`.

- Join on **both `MRN_DID` and `ACCESSION_DID`**, preserving strings.
- Two scan keys have multiple differing reports; excluding those keys removes four
  rows from 16,580 and leaves **16,576 pairs from 5,073 patients**.
- Preserve `split0`: **10,785 train / 2,466 validation / 3,325 test**; no patient overlap.
- All referenced files passed existence/byte-size checks. The retained 33,152 views
  had nonempty crops no larger than 300×213. Recorded spacing is 3.27 in each axis;
  absolute intensity calibration and physical units were not independently verified.

## Text preprocessing (current, 2026-09-25)

Implemented in [pet_text.py](../../src/open_clip_train/pet_text.py) and applied by
`PetReportDataset` with `--pet-report-sections findings_impression`. This is local,
rule-based Python processing: no LLM, generated summary, or clinical rewriting.

1. **Normalize whitespace:** standardize line endings, collapse repeated spaces,
   tabs, and nonbreaking spaces, trim each line, and remove blank lines. Keep
   nonempty lines separate, including headings and numbered findings.
2. **Select Findings and Impression:** recognize case-insensitive headings with
   optional numbering, with a colon or on their own line. Preserve their source
   order, clinical wording, negation, uncertainty, numbers, and anatomy subheadings.
3. **Exclude Summary:** remove the `SUMMARY` heading and its contents up to the
   next recognized section boundary. Keep any subsequent Findings or Impression.
   Summary is no longer renamed to Impression or included as an alternative.
4. **Remove administrative content:** recognized history, technique, comparison,
   metadata, and signature headings terminate selected sections. Standalone
   `FINAL REPORT` markers and final-report communication footers also terminate
   them. Clinical mentions of “final report” within prose remain.
5. **Remove the dated introduction:** strip the specific “Integrated Imaging
   Summary for above study and CT chest abdomen pelvis performed [date]” phrase
   from retained sections, including wrapped lines and supported numeric dates.
   Preserve clinical text following it within Findings or Impression; this does
   not restore any excluded Summary content.
6. **Deduplicate sections:** remove repeated sections with the same heading and
   cleaned body; retain distinct sections. Join retained sections with single
   newlines, without blank separators.
7. **Handle missing sections:** use either Findings or Impression when only one
   is present. If neither contains usable text, exclude and count the pair. This
   includes Summary-only reports; there is no fallback to the full report.

`record['report']` contains the selected training text and `record['full_report']`
keeps the original in memory for comparison. Source CSVs remain unchanged.
`--pet-report-sections full` explicitly bypasses section filtering. The visualization
notebook displays the original and filtered reports; restart its kernel and run
all cells after changing the filter. The selected text is then tokenized with the
existing CLIP tokenizer at the configured 2,560-token context.

Latest verification: **13 filtering tests passed**, covering Summary exclusion,
later retained sections, whitespace, administrative footers, boilerplate removal,
and preservation of clinical wording. Whole-dataset coverage and token counts below
belong to earlier filter versions and have not been recomputed for Summary exclusion.
The completed training run and its retrieval analysis also predate these changes;
no new training outcome is claimed. Unusual report templates still require review.

### Historical text audit

Earlier cleanup audit (2026-09-25, before Summary exclusion): all **16,576** pairs retained, with no repeated horizontal
whitespace, excessive blank lines, or `FINAL REPORT` occurrences in extracted text.
**35 targeted tests passed**, and the visualization notebook executed successfully.
The latest experiment analysis above used the previous filter; its metrics have not
been recomputed with this cleanup.

The coverage and token counts below are historical: they precede exclusion of
`SUMMARY` sections and removal of the integrated-imaging introduction.

| Split | Both sections | Findings only | Impression only | Excluded |
| --- | ---: | ---: | ---: | ---: |
| Train | 10,510 | 254 | 21 | 0 |
| Validation | 2,394 | 70 | 2 | 0 |
| Test | 3,225 | 93 | 7 | 0 |

Extracted text lengths, including special tokens: **median 477, p95 862, max 1,923**.
All fit 2,560 tokens. A 2,048-token context would cover this audit but changes
positional-embedding/checkpoint shapes. Section extraction alone does not reduce
fixed-padding compute. The earlier full-report audit had median 895, p95 1,302,
and max 2,398 tokens; nearly all exceeded 512.

## Image preprocessing and storage

1. Crop each view to its nonzero bounding box, retaining the final row/column.
2. Place it on a **310×310 canvas**: random position in training, centered in evaluation.
3. Clip intensities to `[0, 30]`.
4. Resize to **224×224 for training** with antialiased bilinear interpolation
   (`align_corners=False`); validation keeps **310×310**.
5. Normalize with `(image - 2.13) / 3.39`, including padded background.
6. During training only, multiply each normalized view independently by a random
   factor in `[0.85, 1.15]`. Cached pixels remain unchanged.

Flags: `--pet-canvas-size 310 --pet-image-size 224 --pet-val-image-size 310`.
`--force-image-size 224` sets model training-size metadata. The base model JSON
still specifies 310; ResNet's adaptive pooling supports both resolutions.
For standalone inference, use `PetReportDataset(..., partition='test',
canvas_size=310, image_size=310)` with the same normalization.

Reference constants come from `/home/lingkai/code/lymphoma_classification_2023`:
`dataset.py` and `RandomScale`; its `lars-appendix` recipe enables normalization.
We bypass generic RGB/PIL transforms: no 8-bit conversion, inversion, or RGB replication.
The [LARS study](https://pubmed.ncbi.nlm.nih.gov/38135556/) combines independent-view
classification probabilities; our embedding concatenation is an experimental CLIP
adaptation, not a reproduction of that fusion method.

`--pet-cache ram` stores unaugmented float32 crops in shared host memory.
Workers share their rank's cache; ranks currently duplicate it: approximately
**19 GiB total** for train + validation. Smaller model inputs do not shrink this
310-canvas cache. `--pet-cache none` reads views on demand. No converted images or
report CSVs are exported. Earlier disposable exports/checkpoints were removed,
reclaiming 3.91 GiB. Saving a new checkpoint temporarily needs space for old and new files.

## Runtime and commands

Default Python: `.local/pet-cuda-venv/bin/python` (Python 3.11, PyTorch 2.10.0+cu128,
torchvision 0.25.0+cu128, timm 1.0.30), verified with driver 565.57.01.
To conserve disk, the environment reads CUDA packages through a `.pth` link to
`/home/lingkai/code/Vision-Transformer/.venv/lib/python3.11/site-packages`.
That environment must remain available and was not modified. Additional packages
are installed locally; this runtime is not fully portable.

```bash
# Recreate the runtime from the existing working CUDA environment.
bash scripts/setup_pet_cuda_env.sh /home/lingkai/code/Vision-Transformer/.venv/bin/python

# Train; additional arguments override launcher defaults.
bash scripts/train_pet_2gpu.sh /mnt/snotra1/ida/datasets/lymphoma_mskcc_deid
```

`PYTHON_BIN` can select another compatible environment. The two-GPU launcher
creates a unique timestamp/process-based name before torchrun; `--name` overrides
it. Logs/checkpoints go under `.local/pet/logs/`. Resume requires the original name
and appropriate checkpoint options; changing code does not alter an already running job.

Checkpoint-free smoke run, with external tracking disabled:

```bash
mkdir -p .local/pet
(
  cd .local/pet
  bash ../../scripts/train_pet_2gpu.sh /mnt/snotra1/ida/datasets/lymphoma_mskcc_deid \
    --pet-limit-per-split 1024 --epochs 1 --warmup 0 --logs none --report-to ''
)
```

`--logs none` still produces small `none/<run-name>/` text logs. The optional
`scripts/prepare_pet_pairs.py --data-root ... --check-pixels` audit is read-only.
The notebook is now at `tutorials/visualize_pet_report_pairs.ipynb`; select the PET
Python environment. It uses `PetReportDataset`, defaults to lazy validation reads,
and displays the same selected text that the dataset tokenizes.

## Debugging decisions and validation

| Issue | Resolution / evidence |
| --- | --- |
| Direct GPU peer transfers stalled despite advertised support | Default `NCCL_P2P_DISABLE=1`; NCCL `SHM/direct/direct` all-reduce and training passed. NVLink is inactive; underlying P2P cause remains unknown. |
| Blank CLI argument after `--torchcompile` | Replaced backslash-continued options with a Bash array; real parser checks passed. |
| Existing run name caused rank-0 exit and rank-1 NCCL failure | Unique default names; all ranks exchange collision status and fail together. Two fresh startups passed. |
| Setting canvas to 224 rejected valid crops | Separate 310 canvas from 224 output; 41 preprocessing/model/argument tests passed. |
| Compiled batch-256 backward ran out of memory | Text graph requested a 25 GiB `[256,8,2560,2560]` BF16 tensor; compilation is opt-in. Eager BF16 remains enabled. |
| Findings/Impression extraction | 37 tests passed; complete coverage/token audit and real train/validation sample checks passed. |

The mixed-resolution **eager BF16** two-GPU test passed at batch 256 and 16 workers
per rank: two optimizer updates on 1,024 training pairs, then 310-pixel validation
on 1,024 pairs. This preceded section extraction. Earlier encoder/loader regressions
passed 102 tests; notebook execution passed for both train and validation samples
at the time it was created. These counts are separate historical checks, not additive.
The completed full-run assessment above supersedes the earlier smoke-only status.

Historical capacity probes used **310-pixel views, 2,560-token context, eager mode,
gradient checkpointing, two optimizer steps, and validation**:

| Precision | Batch per GPU / global | Largest sampled memory | Result |
| --- | --- | --- | --- |
| FP16 AMP | 64 / 128 | 12.0 GiB | Passed |
| FP16 AMP | 256 / 512 | 42.2 GiB | Passed |
| FP16 AMP | 288 / 576 | 46.8 GiB | Passed |
| FP16 AMP | 320 / 640 | — | Out of memory |
| BF16 AMP | 256 / 512 | 42.2 GiB | Passed |
| BF16 AMP | 288 / 576 | 47.2 GiB | Passed; <1 GiB headroom |

Memory was sampled approximately every 0.25 s, not measured as exact allocator
peaks. Batch 256 offers more headroom; long-run stability and one-vs-two-GPU scaling
remain unmeasured. These limits do not apply automatically to compiled execution.

Evidence under `.local/pet/`: `report_sections_audit.json`, `report_sections_cleanup_audit.json`, `report_token_audit.json`,
`batch-probes/`, `bf16-batch-probes/`, `distributed-diagnostics/`, and
`none/pet-224train-310val-verified/`. Older montage/mean-fusion/CPU smokes are
superseded; their tiny-sample loss values are omitted here.

## Open concerns and next experiments

- **Data sufficiency:** 10,785 training pairs from a cohort of 5,073 total patients
  is small for two randomly initialized encoders. Repeated scans and similar
  reports reduce independence and can become false negatives. Measure retrieval
  on unseen patients; consider a smaller text encoder and a matched pretrained baseline.
- **Validation:** the contrastive loss needs no validation set to compute gradients,
  but model/schedule selection does. Preserve patient-separated splits and reserve
  test for final evaluation. Combining train + validation (13,251 pairs) is deferred.
- **Schedule:** global batch 512 gives 21 updates/epoch; 200-step warmup lasts about
  9.5 epochs. Revisit this deliberate current setting before interpreting convergence.
- **Resolution:** compare retrieval at 224 and 310. Larger inference inputs are not
  guaranteed to improve results; downsampling can smooth small uptake regions and
  train/evaluation resolution changes can affect BatchNorm behavior.
- **Report targets:** review unusual templates and references to CT-only findings.
  An Impression is a scan interpretation, not automatically a confirmed diagnosis.
- **Grounding:** deferred. Global pooled-feature concatenation alone does not retain
  spatial localization; future work needs spatial features, view/coordinate metadata,
  and appropriate phrase-to-region training/evaluation. MIPs do not uniquely locate 3D lesions.

Research context (not directly comparable sample units or initialization):

| Study | Scale and caveat |
| --- | --- |
| [CLIP](https://arxiv.org/abs/2103.00020) | 400M pairs; both encoders trained from scratch. Section 6 discusses downstream validation during development. |
| [BiomedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) | 15M figure/caption pairs; pretrained PubMedBERT. |
| [ConVIRT](https://proceedings.mlr.press/v182/zhang22a/zhang22a.pdf) | ~217k chest / ~48k musculoskeletal study-report pairs; pretrained image/text encoders and 5k validation holdouts per dataset. |
| [MedCLIP](https://aclanthology.org/2022.emnlp-main.256/) | Experiments with ~20k–570k examples; pretrained encoders, unpaired data, different matching loss. |
| [CMR-CLIP](https://www.nature.com/articles/s41467-026-73022-2) | 11,028 training studies; pretrained BioClinicalBERT and multi-frame imaging. |

These comparisons establish no minimum dataset size that guarantees success.
