# evacuation

RL environment to study the evacuation of pedestrians for dummly rooms.

## Installation

```
git clone https://github.com/cinemere/evacuation
cd evacuation
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run experiments

**Before running experints** don't forget to:

```
export PYTHONPATH=$PWD
```
**To enable [wandb](https://wandb.ai/site) logging** you need to create your wandb profile and run the following once:
```
wandb init
```

To run experiment:
```
python main.py --exp-name "my-first-experiment"
```

## Documentation

Most valuable parametes can be set throw argparse module. However some parameters are still in files:\
- `src/params.py` $\rightarrow$ parameters of the model (+ logging params)
- `src/env/constants.py` $\rightarrow$ parameters of the environment

**Note.** Currently only stable baselines model is available.

```
usage: main.py [-h] [--origin {ppo,a2c}] [--learn-timesteps LEARN_TIMESTEPS] [--learning-rate LEARNING_RATE] [--gamma GAMMA] [--device {cpu,cuda}]
               [--exp-name EXP_NAME] [-v] [--draw] [-n NUMBER_OF_PEDESTRIANS] [--width WIDTH] [--height HEIGHT] [--step-size STEP_SIZE]
               [--noise-coef NOISE_COEF] [--intrinsic-reward-coef INTRINSIC_REWARD_COEF] [--max-timesteps MAX_TIMESTEPS] [--n-episodes N_EPISODES]
               [--n-timesteps N_TIMESTEPS] [-e ENABLED_GRAVITY_EMBEDDING] [--alpha ALPHA]

options:
  -h, --help            show this help message and exit

model:
  --origin {ppo,a2c}    which model to use (default: ppo)
  --learn-timesteps LEARN_TIMESTEPS
                        number of timesteps to learn the model (default: 5000000)
  --learning-rate LEARNING_RATE
                        learning rate for stable baselines ppo model (default: 0.0003)
  --gamma GAMMA         gammma for stable baselines ppo model (default: 0.99)
  --device {cpu,cuda}   device for the model (default: cpu)

experiment:
  --exp-name EXP_NAME   prefix of the experiment name for logging results (default: test)
  -v, --verbose         debug mode of logging (default: False)
  --draw                save animation at each step (default: False)

env:
  -n NUMBER_OF_PEDESTRIANS, --number-of-pedestrians NUMBER_OF_PEDESTRIANS
                        number of pedestrians in the simulation (default: 10)
  --width WIDTH         geometry of environment space: width (default: 1.0)
  --height HEIGHT       geometry of environment space: height (default: 1.0)
  --step-size STEP_SIZE
                        length of pedestrian's and agent's step (default: 0.01)
  --noise-coef NOISE_COEF
                        noise coefficient of randomization in viscek model (default: 0.2)
  --intrinsic-reward-coef INTRINSIC_REWARD_COEF
                        coefficient in front of intrinsic reward (default: 1.0)

time:
  --max-timesteps MAX_TIMESTEPS
                        max timesteps before truncation (default: 2000)
  --n-episodes N_EPISODES
                        number of episodes already done (for pretrained models) (default: 0)
  --n-timesteps N_TIMESTEPS
                        number of timesteps already done (for pretrained models) (default: 0)

gravity embedding params:
  -e ENABLED_GRAVITY_EMBEDDING, --enabled-gravity-embedding ENABLED_GRAVITY_EMBEDDING
                        if True use gravity embedding (default: True)
  --alpha ALPHA         alpha parameter of gravity gradient embedding (default: 3)
```