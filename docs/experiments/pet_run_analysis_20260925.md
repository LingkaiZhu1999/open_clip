# PET CLIP run analysis — 2026-09-25

**Finding:** the latest run learned to match training pairs almost perfectly but
transferred poorly to unseen patients. Evaluating at the training resolution
improves retrieval, yet substantial overfitting remains. More epochs alone are
not supported by these results.

## Run and evidence

Run: `pet-resnet34-concat-2gpu-20260924-165334-526474210-1063591`.
Source: `.local/pet/logs/<run>/out.log`, `params.txt`, and
`checkpoints/results.jsonl`; the remaining checkpoint is `epoch_50.pt` (~1.04 GB).
All 50 validation records are present, finite, and use 2,466 samples. No training
process remained active at inspection. The training/evaluation log spans
2026-09-24 16:53:59–19:52:57 (2h 58m 58s; timestamps as written by the host).

Configuration verified from saved parameters: scratch ResNet34 + native CLIP text
Transformer, concatenated views, Findings/Impression targets, 2,560 tokens, training
224 / validation 310, BF16, eager execution, gradient checkpointing, two GPUs,
256 scans/GPU, 16 workers/rank, AdamW LR 5e-4, 200-step warmup, 50 epochs.

There were 21 updates/epoch, or 1,050 total. Each epoch processed 10,752 of 10,785
training pairs because incomplete batches are dropped; shuffled omissions vary by
epoch. Warmup occupied ~9.5 epochs (19% of all updates). Later logged training
throughput was approximately 57 scans/s across both GPUs. Data loading settled
near 0.026 s versus ~9 s per step; the recorded run was not primarily input-bound.

## Learning curves

![Training and validation curves](../../.local/pet/latest-analysis/learning_curves.png)

| Measurement | Early / best | Final epoch 50 |
| --- | --- | --- |
| Sampled training loss | 6.2223 at epoch 1 | 0.2892 |
| Validation loss | **5.2917 at epoch 5** | 12.4669 |
| Mean bidirectional recall@1 | **0.365% at epoch 21** | 0.182% |
| Mean bidirectional recall@10 | **2.616% at epoch 24** | 2.210% |
| Scan→report median rank | — | 711 |
| Report→scan median rank | — | 573 |

Training fitting improves throughout, while validation loss deteriorates after its
early minimum. Retrieval improves above its initial level, then fluctuates and
plateaus; the final checkpoint is not the best by any of the selection criteria
above. Validation loss and retrieval rank need not peak together: loss measures
confidence as well as ordering, and the learned logit scale grew from ~14.3 to ~19.95.
That scale growth can contribute to confidently wrong validation predictions, but
was not isolated as the cause of the loss increase.

Important metric definitions:

- Training loss uses a **512-pair global candidate batch**. Validation loss uses
  local batches of 256 (the last has 162). Their absolute levels are not directly
  comparable; trends within each series are meaningful.
- The displayed training “epoch average” averages only metric-logging steps
  0, 10, and 20, rather than all 21 updates. Treat it as a sampled estimate.
- Retrieval ranks against **all 2,466 validation candidates**, using the exact
  paired scan/report as the sole positive. Uniform-random expected recall@1 is
  **0.0406%**, and recall@10 is **0.4055%**. Above-random performance is measurable,
  but the absolute retrieval success remains low.
- Recall@1 is based on very few validation hits and is noisy. Small differences
  between epochs should not be treated as a statistically established winner.

## Controlled final-checkpoint checks

Loaded `epoch_50.pt` with strict state-dict matching, set evaluation mode, and
performed inference without modifying weights or BatchNorm statistics. Used BF16,
256-pair batches, centered placement, no intensity augmentation, identical report
processing, and FP32 retrieval scoring. The test split was not accessed.

| Evaluation set | Candidates | Scan→report R@1 | Report→scan R@1 | Scan→report R@10 | Report→scan R@10 | Batch loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation, 310 pixels | 2,466 | 0.203% (5 hits) | 0.162% (4 hits) | 2.190% | 2.230% | 12.4670 |
| Same validation pairs, 224 pixels | 2,466 | 0.365% (9 hits) | 0.446% (11 hits) | 3.285% | 2.758% | 11.2618 |
| Training subset, 224 pixels | 2,466 | **99.392%** | **99.351%** | **100%** | **100%** | **0.1606** |

The training subset was sampled without replacement with NumPy seed 0, then ordered
by dataset index. All three galleries contained 2,466 distinct selected report
strings. Exact duplicates therefore do not explain this comparison, although
semantic similarity and repeated scans can still affect retrieval interpretation.
The 310-pixel rerun reproduces logged final loss and R@1/R@10; tiny other rank
changes are consistent with BF16 numerical/tie sensitivity.

**Interpretation:** the same checkpoint performs better at 224 than 310, so the
resolution change contributes to degraded validation. Mean positive-pair cosine
similarity increases from 0.098 at 310 to 0.289 at 224. However, even matched-resolution
validation is far below the nearly perfect training-subset retrieval (positive
cosine 0.970). This is strong evidence of overfitting / poor generalization,
not merely an image-size mismatch or a failure to optimize the training objective.
These observations do not isolate whether patient-specific memorization, cohort
shift, text-encoder capacity, or other factors dominate the remaining gap.

## Recommended next experiment

1. **Use 224 for both training and validation as the baseline.** Keep 310 as a
   separate evaluation, not an assumed improvement. Compare resolutions using the
   same checkpoint and candidate set.
2. **Retain best as well as latest checkpoints**, choosing a primary validation
   metric in advance (for example mean bidirectional recall@10). Only epoch 50
   remains locally, so earlier checkpoints cannot currently be reevaluated at 224.
3. **Run a shorter controlled baseline with explicit early stopping.** Reconsider
   200 warmup steps now that an epoch has only 21 updates. Keep changes limited so
   their effects can be interpreted; the present run does not identify an optimal LR.
4. **Test capacity/initialization separately:** a smaller scratch text encoder is
   consistent with the requested scratch objective; a matched pretrained baseline
   is an optional comparison. Neither improvement is established by this run.
5. Review retrieved examples, semantically equivalent reports, and repeated-patient
   scans before interpreting exact-pair recall as clinical usefulness. Reserve the
   test split until model and preprocessing selection are fixed.

Reducing context to 2,048 would fit the audited maximum of 1,925 tokens and could
save compute, but is an efficiency change—not an established fix for overfitting.
It also changes positional-embedding/checkpoint shapes.

## Reproducibility and limits

Artifacts under `.local/pet/latest-analysis/`:
`analyze.py`, `epoch_metrics.csv`, `summary.json`, `learning_curves.png`,
`check_resolution.py`, and `resolution_check.json`. The two Python scripts reproduce
the curve analysis and final-checkpoint diagnostics using the local PET environment.
No patient text/images or embedding arrays were exported; saved artifacts contain
aggregate metrics. No training configuration or model weights were changed.

This is one seed and one train/validation split. The findings support a generalization
problem and a resolution effect for this checkpoint, not a clinical-performance
claim or a causal ranking of all possible remedies.
