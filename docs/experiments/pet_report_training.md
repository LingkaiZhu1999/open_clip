# PET–report OpenCLIP: consolidated experiment report

Updated 2026-09-25. The new Bio_ClinicalBERT experiment fine-tunes a **pretrained
text encoder** with a randomly initialized PET image encoder. Earlier experiments
trained both encoders from scratch. This report replaces
the separate training, local-learning, performance, and dated run-analysis reports.

**Current findings:** global-only training substantially overfits. GLoRIA-style
local alignment is implemented and tested, but improved generalization or grounding
is not yet established. At 128 pairs/GPU, increasing local chunk size from 4 to 32
reduced measured full training time from **22.364 to 5.439 s/step**. Use chunk 32
for the next local experiment; the launcher still defaults to 4.

## Checkpoint storage update — 2026-09-30

The latest run (`pet-bioclinicalbert-2gpu-20260930-171916-962755076-1230372`)
finished epoch-6 validation but left an incomplete 169 MB `epoch_6.pt` while the
root filesystem was full. `epoch_5.pt` passed archive CRC validation and loads
with model and optimizer state; use epoch 5 to resume, not epoch 6.

The entire run was moved to `/mnt/scapis3/lingkai/open_clip/logs/`, with every
file verified by SHA-256 before removing the source copy. The original run path
under `.local/pet/logs/` is now a symlink. The incomplete checkpoint is preserved
for diagnosis. No training was restarted.

After space was freed on the original disk (25 GB available), the
Bio_ClinicalBERT launcher was switched back to `$REPO_DIR/.local/pet/logs` for
new checkpoints, experiment logs, and W&B working files. The previously migrated
run remains on `/mnt/scapis3/lingkai/open_clip/logs`, accessible through its
original symlink; it was not moved back. Override with `PET_LOG_DIR`; an explicit `--logs`
argument overrides the training log path, and `WANDB_DIR` independently overrides
W&B's working directory. Checkpoint retention settings are unchanged.

## Bio_ClinicalBERT experiment (new)

The text encoder now supports pretrained
[`emilyalsentzer/Bio_ClinicalBERT`](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT):
12 bidirectional BERT layers, width 768, 12 heads, 28,996-token cased WordPiece
vocabulary. Encoder weights are loaded with `from_pretrained` and remain trainable.
The PET ResNet34 and global/local projection heads initialize randomly. No report
generation or LLM-based filtering is introduced.

```bash
bash scripts/train_pet_bioclinicalbert_2gpu.sh /mnt/snotra1/ida/datasets/lymphoma_mskcc_deid
```

The launcher selects `PET-ResNet34-BioClinicalBERT-Local`, 128 pairs/GPU, chunk 32,
BF16, 224/224 images, and a shared LR of **5e-5** (lower than the scratch baseline).
This is an initial fine-tuning setting, not an optimized LR. Global-only comparison
config: `PET-ResNet34-BioClinicalBERT`. Existing scratch launchers remain available.
Pretrained files are cached under `.local/pet/hf-cache` (`HF_HOME` can override it);
no patient data is sent to Hugging Face. The downloaded revision was
`d5892b39a4adaed74b92212a44081509db72f87b`.

**Long reports are preserved using windows.** Native BERT supports 512 positions;
we retain the 2,560-token report context and encode up to 510 content positions per
window with fresh CLS/SEP tokens. Pretrained positional embeddings are unchanged.
There is no attention across windows. Content-token-weighted mean pooling combines
window outputs, followed by a learned 768→512 global projection. Local alignment
samples up to 128 valid WordPiece features across the entire retained report and
projects 768→128; padding/CLS/SEP are excluded. Both validation losses are supported.
These text windows are separate from image–report loss chunks discussed below.

The cased-tokenizer audit found 7,579/10,785 training reports (**70.3%**) and
1,821/2,466 validation reports (**73.8%**) exceeded 512 tokens. All retained reports
fit 2,560: maximum 2,130 (train), 2,091 (validation), 1,924 (test), including special
tokens. Thus single-window truncation was rejected. Source:
`.local/pet/bioclinicalbert_token_audit.json`. Text filtering and image preparation
are unchanged; the notebook now previews the Bio_ClinicalBERT tokenizer by default.

Verification: every downloaded encoder tensor matched the source pretrained weights
exactly. **37 tests passed**, covering pretrained initialization, HF/native local
alignment, windowed tail-token influence/gradients, unchanged positional weights,
and global/local validation. Notebook execution passed. A two-A6000 BF16 test passed
two epochs with global/local validation and partial validation batches at 4/GPU.
The default **128 pairs/GPU** also passed two training updates and validation on
512 pairs with the 2,560-token windowed context. This verifies execution, not
convergence or generalization.
No full Bio_ClinicalBERT experiment has been started. Scratch checkpoints cannot
resume into this architecture because text weights/tokenizer/shapes differ; start
a new run, then resume only matching Bio_ClinicalBERT checkpoints.

## Scratch baseline configuration and launch

| Component | Current implementation / launcher settings |
| --- | --- |
| Models | `PET-ResNet34` (global only); `PET-ResNet34-Local` (global + local) |
| Images | Independent single-channel coronal and sagittal MIPs, in that order |
| Global vision | Shared ResNet34 → average-pool each view to 512 → concatenate to 1,024 → bias-free projection to 512 → L2 normalization |
| Global text | Native causal CLIP Transformer: 12 layers, width 512, 8 heads, vocabulary 49,408; EOT pooling → normalized 512-vector |
| Text input | Findings + Impression; Summary excluded; fixed context 2,560 tokens |
| Initialization | Random image/text weights; reuse CLIP tokenizer only; no pretrained LLM or summarizer |
| Hardware | Two RTX A6000 48-GB GPUs; DDP/NCCL, one complete model per GPU |
| Batch / workers | Global launcher: 256/GPU, 16 workers/rank; local launcher: 128/GPU, 4 workers/rank |
| Precision / memory | BF16 AMP, FP32 parameters, no gradient scaler; gradient checkpointing; RAM image cache |
| Optimization | AdamW, LR 5e-4, weight decay 0.1, 50 epochs, 200 warmup steps, seed 0; starting settings, not retuned for local learning |
| Evaluation / saving | Rank-0 validation each epoch; global retrieval and global/local losses; latest epoch retained, best checkpoint not retained automatically |
| Tracking | W&B: `pet-resnet34-concat-scratch` or `pet-resnet34-global-local-scratch` |

```bash
# Recommended local run; explicit overrides match the measured configuration.
bash scripts/train_pet_local_2gpu.sh /mnt/snotra1/ida/datasets/lymphoma_mskcc_deid \
  --batch-size 128 --pet-region-chunk-size 32

# Global-only control at the same batch size.
bash scripts/train_pet_2gpu.sh /mnt/snotra1/ida/datasets/lymphoma_mskcc_deid --batch-size 128
```

Additional arguments override defaults. Local controls are `--pet-region-loss-weight`
(default 1), `--pet-region-max-tokens` (128), and `--pet-region-chunk-size` (4).
Other local settings are in [PET-ResNet34-Local.json](../../src/open_clip/model_configs/PET-ResNet34-Local.json).
Resume with the same architecture/settings and `--resume PATH`; global-only weights
lack local projections and are not drop-in local resume checkpoints.

Runtime: `.local/pet-cuda-venv/bin/python`, Python 3.11, PyTorch 2.10.0+cu128,
torchvision 0.25.0+cu128, timm 1.0.30, driver 565.57.01. To save disk, CUDA packages
are referenced through a `.pth` link to the existing Vision-Transformer environment;
that environment must remain available. Recreate with `scripts/setup_pet_cuda_env.sh`
and its source-environment argument. `PYTHON_BIN` overrides the interpreter.
Unique run names prevent collisions; logs/checkpoints live under `.local/pet/logs/`.

## Data and preprocessing

Source: `/mnt/snotra1/ida/datasets/lymphoma_mskcc_deid`. `PET_binaries` contains
little-endian float32 **2D projections**, not volumes. Join `data_scan_did.csv`,
`data_Scan_Report.csv`, and `data_splits.csv` on both `MRN_DID` and `ACCESSION_DID`
as strings. Two ambiguous scan keys remove four of 16,580 rows, leaving
**16,576 pairs / 5,073 patients**: split0 has **10,785 train / 2,466 validation /
3,325 test**, with no patient overlap. These are pairing counts before section filtering.
All referenced binaries passed existence/size checks; crops were nonempty and at
most 300×213. Recorded spacing is 3.27 per axis; physical intensity calibration
was not independently verified.

### Text filtering

[pet_text.py](../../src/open_clip_train/pet_text.py) applies deterministic Python rules:

1. Standardize line endings, collapse repeated spaces/tabs/nonbreaking spaces, trim
   lines, and remove blank lines while keeping nonempty lines separate.
2. Retain **Findings and Impression**, recognizing case-insensitive, numbered,
   colon-delimited or standalone headings. Preserve wording, negation, uncertainty,
   numbers, anatomy subheadings, and source order.
3. Remove **Summary headings and contents**, retaining subsequent Findings/Impression.
4. End selected sections at recognized administrative headings, signatures,
   standalone `FINAL REPORT`, and final-report communication footers. Preserve
   clinical mentions of “final report” inside prose.
5. Strip the dated “Integrated Imaging Summary for above study and CT chest abdomen
   pelvis performed [date]” introduction from retained sections, preserving following
   clinical text. This does not restore excluded Summary content.
6. Deduplicate sections with identical headings and cleaned bodies. Use either
   selected section if only one exists; exclude/count reports with neither usable
   section, including Summary-only reports. There is no automatic full-report fallback.

`record['report']` is training text; `record['full_report']` preserves the original
in memory. `--pet-report-sections full` explicitly bypasses filtering. Source CSVs
are unchanged. **13 filtering tests passed**; unusual templates still need review.

Historical audit before Summary exclusion: all pairs retained, cleaned token lengths
(including special tokens) median **477**, p95 **862**, maximum **1,923**. Full reports
had median 895, p95 1,302, maximum 2,398. Final-filter token statistics have not been
recomputed. Shorter text does not reduce fixed-padding compute; changing context to
2,048 also changes checkpoint positional-embedding shapes.

### Images and storage

Crop each view to its nonzero bounding box, including the last row/column; place on
a **310×310 canvas** (random placement for training, centered for evaluation); clip
to `[0,30]`; resize with antialiased bilinear interpolation (`align_corners=False`);
normalize by **`(image - 2.13) / 3.39`**, including padding. Training then scales each
view independently by a random factor in `[0.85,1.15]`.

Both launchers now use **224×224 training and validation**. Earlier experiments used
310-pixel validation. Canvas size and model-input size are separate: shrinking the
canvas to 224 rejects valid crops. The model supports larger inference inputs through
adaptive pooling, but larger resolution has not improved the measured retrieval.
Constants follow `/home/lingkai/code/lymphoma_classification_2023`; no RGB replication,
8-bit export, or PIL preprocessing is used. The [LARS reference](https://pubmed.ncbi.nlm.nih.gov/38135556/)
combines view-level probabilities; our embedding concatenation is an adaptation.

`--pet-cache ram` stores raw float32 crops in shared host memory for each rank's
workers; ranks duplicate caches, approximately **19 GiB total** for both splits/ranks.
Output resizing does not shrink this cache. `--pet-cache none` reads on demand.
No converted images/report CSVs are exported; checkpoint replacement needs temporary
space for both old and new files.

[Visualization notebook](../../tutorials/visualize_pet_report_pairs.ipynb): actual
`PetReportDataset` views, original/filtered report panels, token counts, and normalized
pixel histograms including padding. Restart the kernel after filter edits; clear
patient-data outputs before sharing. `scripts/prepare_pet_pairs.py` provides a read-only audit.

## Local region–text objective and distributed behavior

[local_region.py](../../src/open_clip/local_region.py) adds a spatial branch before
ResNet pooling: project each 512-channel location to 128 dimensions and normalize.
At 224 pixels, two 7×7 maps give **98 regions/scan**. Tensor axes preserve view and
spatial identity; no learned view/position embeddings are added, and background
regions participate. Global concatenation remains unchanged.

The text branch projects contextual CLIP **subword** features to 128 dimensions.
It excludes SOT/EOT/padding and uniformly samples up to **128 content tokens across
the report**; shorter reports use all content tokens. Headers/punctuation remain.
The global branch still consumes the full context.

For each image–report candidate pair: compute region/token similarities, normalize
over valid tokens per region, attend over regions per token, form weighted visual
vectors, compare with token vectors by cosine similarity, then aggregate using
scaled **log-mean-exp**. Symmetric cross-entropy distinguishes paired from mismatched
examples. This adapts [GLoRIA §§3.2.3–3.2.5](https://openaccess.thecvf.com/content/ICCV2021/papers/Huang_GLoRIA_A_Multimodal_Global-Local_Representation_Learning_Framework_for_Label-Efficient_Medical_ICCV_2021_paper.pdf):
we use two PET views, causal CLIP subwords, sampling, and a mean correction instead
of the paper's log-sum-exp. It is not an exact reproduction.

Total loss = **global CLIP loss + weight × local region loss**. Logged
`region_contrastive_loss` is already weighted. Attention, token-aggregation, and
local contrastive temperatures are 0.1, 0.1, and 0.07; local temperature is fixed,
while global logit scale is learned. Local attention uses FP32, with TF32 matmul
allowed, even during BF16 encoder training.

**“Local” has two meanings:** OpenCLIP `--local-loss` distributes query rows across
GPUs; region–text alignment learns spatial associations. Both paths gather features
with gradients, compare local queries against global candidates, and offset labels
by rank. DDP averages parameter gradients. Equal per-rank shapes are required;
the training loader drops incomplete batches. Single-process eager execution also
works; the task factory rejects accumulation, FSDP, compilation, SigLIP, and distillation
for the region branch.

NCCL uses **`NCCL_P2P_DISABLE=1`**: direct peer transfers stalled on this host, while
host-memory fallback passed. NVLink is inactive. This workaround's root cause remains
unresolved. Compilation is disabled: an earlier compiled global batch-256 backward
requested a 25-GiB text-attention tensor and ran out of memory.

### Global and local validation (added 2026-09-25)

`PET-ResNet34-Local` now emits both global and spatial/token features during paired
validation, while remaining in evaluation/inference mode. The local matching function,
token mask, temperatures, and loss weight are identical to training. Rank zero scores
within-batch candidates without distributed gathers; other DDP ranks do not need to
enter the local-loss computation. No weights, BatchNorm statistics, or gradients change.

| Metric | Meaning |
| --- | --- |
| `clip_val_loss` / `global_val_loss` | Existing global symmetric contrastive validation loss; both keys have the same value |
| `local_val_loss` | Unweighted symmetric region–token contrastive validation loss |
| `weighted_local_val_loss` | Local loss multiplied by the configured region weight |
| `total_val_loss` | Global loss + weighted local loss |
| `local_val_num_samples` | Number of pairs included in the local validation loss |

Losses are sample-weighted averages of batch losses, including the last partial batch.
They appear in console output and the existing JSONL/W&B/TensorBoard evaluation sinks
when enabled. Global retrieval still ranks against the full validation gallery;
**local validation uses within-batch negatives and does not report local retrieval or
localization accuracy**. Training gathers global-batch negatives, so absolute train/val
losses can still differ because candidate counts differ. Compare trends under fixed
batch settings. The global-only model keeps its existing evaluation behavior.

Verified: **57 tests passed**, including training/validation local-objective equivalence,
partial-batch weighting, unchanged BatchNorm state/no gradients, and no validation
collectives. A real PET two-A6000 BF16 smoke completed two epochs, two updates/epoch,
and validation over 18 pairs at batch 4/GPU (last validation batch size 2). No external
tracking or checkpoint export was enabled. Existing local checkpoints remain compatible;
the new metrics appear on the next run/resume, not retroactively in old logs.

`encode_regions` returns `[B,view,H,W,D]`; `encode_local_text` returns sampled features
and masks for future visualization. Attention maps and lesion-localization evaluation
are not yet implemented; MIPs do not uniquely identify 3D lesion locations.

## Important: chunking and GPU parallelism

A chunk partitions the **image–report score matrix**, not image pixels or text.
Size `C` processes up to **C images × C reports = C² comparisons** together.
For 128 pairs total, a 128×128 matrix with C=4 has **1,024 blocks**, not 16,384/4.

Our two-GPU batch is 128/GPU, or 256 total. Each GPU compares 128 local images
against 256 reports (image→text) and 128 local reports against 256 images (text→image).
These directions select the correct report or scan; they are not PET view directions.

| Chunk C | Comparisons/block | Blocks/direction/GPU | Blocks for both directions/GPU |
| --- | ---: | ---: | ---: |
| 4 | 16 | 2,048 | 4,096 |
| 16 | 256 | 128 | 256 |
| 32 | 1,024 | 32 | 64 |

**Comparisons within a chunk run in parallel on the GPU; our loop processes chunks
sequentially. Both GPUs work in parallel.** Smaller chunks save temporary memory
but add kernel-launch/checkpoint overhead; checkpointing recomputes attention during
backward. Larger chunks provide more work per operation. More chunks are not inherently
better. Chunk boundaries do not restrict negatives: scores are assembled before
cross-entropy. Batch size, tokens, and mathematical loss remain unchanged, apart
from floating-point rounding.

Local work is quadratic in batch size: increasing 16→128 pairs/GPU increased pair
calculations 64-fold and chunks from 64→4,096 at C=4. Both directions currently
recompute pair scores; exchanging differentiable score rows could avoid duplication,
but that optimization has not been implemented.

## Measured performance on two A6000s

128 pairs/GPU, 224-pixel views, 2,560 global tokens, 128 local tokens, BF16,
checkpointing, NCCL host-memory fallback. Full-model runs used four updates on 1,024
real pairs; medians exclude the first update and include optimizer/DDP work.
Lazy reads/zero workers avoided cache setup; no validation, checkpoints, or external
tracking. Isolated loss runs used synthetic features, one warmup and three measured
updates, synchronized stages, and the slowest rank's timings.

| Configuration | Full-model median step | Isolated loss forward + backward | Isolated peak allocation |
| --- | ---: | ---: | ---: |
| Global only | 4.645 s | 1.529 ms | 24.8 MiB |
| Global + local, C=4 | 22.364 s | 10.452 s | 65.4 MiB |
| Global + local, C=16 | 5.788 s | 0.885 s | 206.7 MiB |
| Global + local, C=32 | **5.439 s** | **0.741 s** | 659.2 MiB |

C=32 gives **4.11× faster full training than C=4**, with **17.1% overhead over global
only**. Local-feature gathering/reduction took **4.218 ms including backward** in
isolation, excluding DDP parameter gradients. BF16 local-feature payload is
7.0625 MiB/rank before forward gathering; chunking does not reduce that payload.
The measurements identify chunk size 4 as a major avoidable cost, not feature exchange
as the dominant isolated-loss cost. They are not an additive full-step decomposition.

Local full-model runs peaked at **20,438 MiB/GPU**, sampled every 250 ms including
startup; the global control peaked at 21,678 MiB. These coarse device-level figures
include allocator/workspace effects and do not show that local learning needs less
memory. First steps took 31.065/14.449/14.048 s for C=4/16/32 and were excluded.
Data loading took 0.335–0.438 s/step. Short-run results are not a long-run speed or
maximum-batch guarantee. Defaults were not changed and full training was not restarted.

## Latest local-run diagnosis: snapshot through epoch 15

Run `pet-resnet34-global-local-2gpu-20260925-143758-080536858-1129810`, through
validation logged at 15:47:25 on 2026-09-25. Settings: **256 pairs/GPU**, chunk 32,
224/224 images, local weight 1, 128 local tokens, BF16, LR 5e-4, 200 warmup steps.
This is a running-job snapshot, not a completed experiment or the 128/GPU timing run.

| Epoch | Global train loss | Local train loss | Global validation loss | Mean R@1 | Mean R@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 7 (minimum validation loss) | 5.7122 | 5.8857 | 5.2618 | 0.2433% | 1.8248% |
| 11 (best mean R@1/R@10 so far) | 4.6759 | 5.2790 | 5.6883 | 0.3244% | 2.7372% |
| 15 | 3.2240 | 4.2066 | 7.7777 | 0.3041% | 2.6967% |

**Diagnosis: emerging global-objective overfitting is plausible, but a clear retrieval
collapse is not established.** From epoch 7 to 15, global training loss fell 43.6%
while validation loss rose 47.8%. However, R@10 improved over epoch 7 and is nearly
at its epoch-11 peak: 133 versus 135 top-10 successes across 4,932 directional
queries. R@1 differs by one success (15 versus 16). Those small differences alone
do not establish worsening retrieval; image→text R@10 at epoch 15 is actually
higher than at epoch 11. Absolute retrieval remains weak.

The previous resolution mismatch is absent. Logit scale increased 14.234→14.981
between epochs 7 and 15 and can amplify confident errors, but its causal contribution
has not been measured. Embedding geometry/confidence can change loss without the
same change in rankings. A peak-LR transition around epoch 10 (200 warmup updates,
21 steps/epoch) is another possible contributor, not an established explanation.

Compare the global training component—not global+local total—to global validation
trends. Training still uses 512 global candidates and sampled logging; validation
loss uses batches up to 256. All retrieval galleries contain 2,466 candidates.
At this snapshot there was **no local validation loss or localization metric**, so
these historical logs cannot diagnose local-region overfitting. Local validation loss
was subsequently implemented as described above. Unlike the earlier completed experiment,
this checkpoint has not had a matched train-subset/validation retrieval evaluation.

Next diagnostic: retain best retrieval and latest checkpoints, then evaluate equal
train/validation galleries without augmentation; report both learned- and fixed-scale
global loss, positive/negative similarities, and separately local matching metrics.
Do not select epoch 7 solely on minimum loss when the objective is retrieval.
Only epoch 15 was present at inspection, so earlier best weights are not available
for retrospective reevaluation. No job or defaults were changed; no additional GPU
workload was launched while training occupied both GPUs.

Evidence: the run's `params.txt`, `out.log`, `checkpoints/results.jsonl`; aggregate
snapshot `.local/pet/local-generalization-20260925/epoch15_snapshot.json`.

## Historical generalization results

**Completed 50-epoch run:** `pet-resnet34-concat-2gpu-20260924-165334-526474210-1063591`,
about 3 hours, 256/GPU, training 224 / validation 310, earlier report filter. Training
loss fell **6.2223→0.2892**; validation loss bottomed at **5.2917 (epoch 5)** and ended
at **12.4669**. Best mean bidirectional R@1 was 0.365% at epoch 21; best R@10 was
2.616% at epoch 24. Only epoch 50 remained at inspection.

Controlled epoch-50 evaluation used the same weights, no augmentation, and 2,466
candidates per set; the training subset was sampled with seed 0. Test data was not used.

| Set / resolution | Image→text R@1 | Text→image R@1 | Image→text R@10 | Text→image R@10 | Loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| Validation / 310 | 0.203% | 0.162% | 2.190% | 2.230% | 12.4670 |
| Same validation / 224 | 0.365% | 0.446% | 3.285% | 2.758% | 11.2618 |
| Training subset / 224 | 99.392% | 99.351% | 100% | 100% | 0.1606 |

This establishes severe overfitting plus a resolution effect; it does not identify
which model/data factor causes the remaining gap. Selected report strings were
unique within each gallery, but semantic equivalence can still affect exact-pair recall.

**Later snapshot, not a final result:** run
`pet-resnet34-concat-2gpu-20260925-110840-692375547-1103851`, through epoch 27,
also used 224/310 and 256/GPU. Minimum validation loss was **5.3044 at epoch 6**;
at epoch 27 training loss was **0.6872**, validation **11.9971**. Mean R@10 peaked
at **3.2644% (epoch 13)** versus **2.2506% at epoch 27**. Logit scale rose
14.239→18.930, potentially increasing confident-error penalties. Its contribution
was not isolated; the exact intermediate filter revision at launch was not saved.

Historical training loss used 512 global candidates and sampled logging
steps; validation loss used batches up to 256, while recall used all 2,466 candidates.
Absolute losses are not directly comparable. Random expected R@1/R@10 is
0.0406%/0.4055%; low hit counts make small recall differences noisy. Increasing
confidence can worsen loss without worsening rankings. Historical metrics predate
final Summary removal and should not be presented as evaluation of the current filter.

## Verification, evidence, and next steps

**Verified:** 39 local-loss/multi-view/task tests; dense versus chunked gradients,
masking, positive-pair ranking, encoder/projection gradients, global-forward equality,
checkpoint loading, and two-process CPU/NCCL gradient equivalence. Real PET BF16
smokes passed two updates plus validation at 2 and 16/GPU. Full-model timing runs
then passed at 128/GPU. These checks establish correctness/execution, not grounding.
Earlier separate checks included 13 final text-filter tests, 41 resizing tests,
102 encoder/loader regressions, and notebook execution; counts are not additive.

Historical **global-only 310-pixel** capacity probes: BF16 256/GPU used 42.2 GiB;
288 used 47.2 GiB with little headroom; FP16 320 failed. These are not local-model,
compiled-model, or current-resolution maximum batch sizes. Earlier startup fixes
removed a blank Bash argument and synchronized run-name collision failures.

Reproduce on idle GPUs:

```bash
PYTHONPATH=src NCCL_P2P_DISABLE=1 OMP_NUM_THREADS=2 \
  .local/pet-cuda-venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/benchmark_pet_local_loss.py --device cuda --batch-size 128 --chunks 4 16 32

.local/pet-cuda-venv/bin/python scripts/benchmark_pet_local_training.py \
  /mnt/snotra1/ida/datasets/lymphoma_mskcc_deid --batch-size 128 --chunks 4 16 32 --steps 4
# Add --global-only for the matched global control.
```

Evidence under `.local/pet/`: `local-loss-profile/a6000_batch128.json`,
`local-loss-profile/training-20260925-141339/summary.json` (local),
`local-loss-profile/training-20260925-141727/summary.json` (global control),
`latest-analysis/` (curve/metric CSV, analysis and resolution-check scripts/results),
`logs/<run>/` (params, logs, validation JSONL), `report_sections_cleanup_audit.json`,
`report_token_audit.json`, and historical `batch-probes/`, `bf16-batch-probes/`,
`distributed-diagnostics/`. CPU benchmark checks are retained in
`local-loss-profile/{cpu_check,gloo_check}.json`; GPU measurements supersede them.
The distributed gradient oracle is `tests/pet_region_distributed_check.py`.

Next priorities: use chunk 32 and matched 224/224 resolution; compare global/local
at equal batch and schedule; retain best **and** latest checkpoints and add early
stopping; consider a smaller scratch text encoder; investigate duplicate score work
and mixed-precision local matmuls; validate localization separately. Validation is
unnecessary to compute gradients but essential for model selection; reserve test data.
At 512 global batch, 200 warmup steps span ~9.5 epochs; at 256, ~4.8 epochs. Retune
schedule deliberately rather than comparing unmatched experiments.

Data volume remains a concern: ~10.8k training pairs, repeated patient scans, and
similar reports may encourage memorization/false negatives. Text cleanup alone does
not fix this. Impression is an imaging interpretation, not automatically definitive
diagnosis. The scratch baseline remains available; the new Bio_ClinicalBERT
experiment provides the requested pretrained-text comparison.

Research context (different datasets/initialization; no universal minimum size):
[CLIP](https://arxiv.org/abs/2103.00020) used 400M pairs;
[BiomedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)
15M with pretrained PubMedBERT;
[ConVIRT](https://proceedings.mlr.press/v182/zhang22a/zhang22a.pdf) ~217k chest/~48k
musculoskeletal pairs with pretrained encoders;
[MedCLIP](https://aclanthology.org/2022.emnlp-main.256/) ~20k–570k examples with
pretrained encoders and a different pairing objective;
[CMR-CLIP](https://www.nature.com/articles/s41467-026-73022-2) 11,028 studies with
pretrained BioClinicalBERT. None establishes that our scratch setup will generalize.
