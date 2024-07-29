#!/bin/bash

#SBATCH --job-name=evac_transformer
#SBATCH --output=evac_transformer_log.txt
#SBATCH --ntasks=1
#SBATCH --time=600:00
#SBATCH --mem-per-cpu=100
#SBATCH --cpus-per-task=4
#SBATCH --partition=htc

export DEVICE="cpu"
export WANDB_PROJECT="evacuation_june"
export WANDB_ENTITY="albinakl"
export WANDB_API_KEY="5e7ca36a9a2b12b4cda477f8047ff713694b350e"

EXPERIMENT_NAME="transformer-emb"

#nodes to choose from: gpu, gpu_devel, htc, mem
# srun -p htc -n 4 pwd

python3 src/main.py \
--env.experiment-name $EXPERIMENT_NAME \
--env.number-of-pedestrians 60 \
--env.noise-coef 0.2 \
--env.enslaving-degree 1. \
--env.is-new-exiting-reward \
--env.no-is-new-followers-reward \
--env.intrinsic-reward-coef 0. \
--env.giff-freq 500 \
--env.wandb-enabled \
--wrap.positions rel \
--wrap.statuses ohe \
--wrap.type Box \
model:clean-rl-config \
--model.agent.learning_rate 0.0005 \
--model.agent.num_envs 3 \
model.network:rpo-transformer-embedding-config \
--model.network.num-blocks 2 \
--model.network.num-heads 3 \
--model.network.no-use-resid