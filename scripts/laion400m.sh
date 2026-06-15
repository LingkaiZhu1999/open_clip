#!/bin/bash
#SBATCH --time=5-0:0:00
#SBATCH --mem=800G
#SBATCH --output=pi-gpu.out
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=5
#SBATCH --gpus-per-node=5
#SBATCH --partition=gpu-h200-141g-ellis
#SBATCH --cpus-per-task=20
#SBATCH --wait-all-nodes=1

module load scicomp-python-env

cd /scratch/work/zhul2/code/open_clip
export PYTHONPATH="$PWD/src:$PYTHONPATH"
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_PORT="${MASTER_PORT:-12802}"
export RUN_NAME="${RUN_NAME:-ViT-B-32-Vanilla-${SLURM_JOB_ID:-manual}}"

srun --cpu_bind=v --accel-bind=gn python -u src/open_clip_train/main.py \
  --train-data '/scratch/shareddata/dldata/laion400M/img2dataset/laion400m-data/{00000..41407}.tar' \
  --train-num-samples 268836185 \
  --dataset-type webdataset \
  --report-to wandb \
  --model ViT-B-32 \
  --torchcompile \
  --torchcompile-strategy step \
  --torchcompile-mode reduce-overhead \
  --ddp-static-graph \
  --batch-size 3840 \
  --epochs 48 \
  --workers 10 \
  --precision amp_bfloat16 \
  --name "$RUN_NAME" \
  --seed 0 \
  --save-frequency 1 \
  --local-loss \
  --gather-with-grad \
  --imagenet-val '/scratch/shareddata/dldata/imagenet-1k-wds/imagenet-1k-wds/imagenet1k-validation-{00..63}.tar'
  # --resume /scratch/work/zhul2/code/open_clip/logs/ViT-B-32-Vanilla-resume/checkpoints/epoch_32.pt
