export DEVICE="cpu"
export WANDB_PROJECT="evacuation"

EXPERIMENT_NAME="grav-emb"

python3 src/main.py \
--env.experiment-name $EXPERIMENT_NAME \
--env.number-of-pedestrians 60 \
--env.noise-coef 0.2 \
--env.enslaving-degree 1. \
--env.is-new-exiting-reward \
--env.is-new-followers-reward \
--env.intrinsic-reward-coef 0. \
--env.giff-freq 500 \
--env.wandb-enabled \
--wrap.positions grav \
model:clean-rl-config \
--model.agent.learning_rate 0.0003 \
--model.agent.num_envs 3 \
model.network:rpo-linear-network-config