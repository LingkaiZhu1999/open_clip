#!/bin/bash
#SBATCH --time=96:10:00
#SBATCH --mem=500G
#SBATCH --output=pi-gpu.out
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --partition=gpu-h200-141g-ellis
#SBATCH --cpus-per-task=16
#SBATCH --wait-all-nodes=1

module load scicomp-python-env

cd /scratch/work/zhul2/code/open_clip
export PYTHONPATH="$PWD/src:$PYTHONPATH"
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_PORT="${MASTER_PORT:-12802}"

srun --cpu_bind=v --accel-bind=gn python -u src/open_clip_train/main.py \
  --train-data '/scratch/shareddata/dldata/laion400M/img2dataset/laion400m-data/{00000..41407}.tar' \
  --train-num-samples 268836185 \
  --dataset-type webdataset \
  --report-to wandb \
  --model ViT-B-32 \
  --torchcompile \
  --batch-size 4096 \
  --epochs 32 \
  --workers 8 \
  --precision amp_bfloat16 \
  --warmup 2000 \
  --name "ViT-B-32-Vanilla" \
  --seed 0 \
  --save-frequency 1 \
  --local-loss \
  --gather-with-grad
