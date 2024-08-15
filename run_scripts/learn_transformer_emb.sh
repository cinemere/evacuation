#!/bin/sh
export DEVICE="cpu"
export WANDB_PROJECT="evacuation_june"
export WANDB_ENTITY="albinakl"

EXPERIMENT_NAME="transformer-emb"


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
--model.agent.learning_rate 0.001 \
--model.agent.num_envs 3 \
model.network:rpo-transformer-embedding-config \
--model.network.num-blocks 2 \
--model.network.num-heads 3 \
--model.network.no-use-resid
