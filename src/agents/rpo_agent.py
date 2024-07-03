from dataclasses import dataclass

from . import BaseAgent
from .networks.rpo_linear_agent_network import RPOLinearNetwork, RPOLinearNetworkConfig
from .networks.rpo_transformer_agent_network import RPOTransformerEmbedding, RPOTransformerEmbeddingConfig
from .networks.rpo_deep_sets_agent_network import RPODeepSetsEmbedding, RPODeepSetsEmbeddingConfig
        
from dataclasses import dataclass
import gymnasium as gym
import random
import numpy as np
import torch

from env import EvacuationEnv, EnvConfig, EnvWrappersConfig, setup_env

import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter


TBLOGS_DIR = os.getenv("TBLOGS_DIR", "saved_data/tb_logs")


def wrapping(env, gamma):    
    env = gym.wrappers.FlattenObservation(env)  # deal with Dict observation space
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)
    env = gym.wrappers.NormalizeObservation(env)
    env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -1, 1))
    env = gym.wrappers.NormalizeReward(env, gamma=gamma)
    env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -100, 100)) ## TODO: Check without this
    return env

def make_env(env_config, env_wrappers_config, gamma):
    def thunk():
        env = setup_env(env_config, env_wrappers_config)        
        return wrapping(env, gamma)
    return thunk

@dataclass
class RPOAgentTrainingConfig(BaseAgent):
    exp_name: str = "rpo-agent"
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    
    total_timesteps: int = 80000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 3
    """the number of parallel game environments"""
    num_steps: int = 2048
    """the number of steps to run in each environment per policy rollout"""

    # agent training params
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float | None = None
    """the target KL divergence threshold"""

    @property
    def batch_size(self):
        return int(self.num_envs * self.num_steps)

    @property
    def minibatch_size(self):
        return int(self.batch_size // self.num_minibatches)

    @property
    def num_iterations(self):
        return self.total_timesteps // self.batch_size

    @property
    def num_updates(self):
        return self.total_timesteps // self.batch_size

    
class RPOAgent:
    def __init__(
            self, 
            env_config: EnvConfig,
            env_wrappers_config: EnvWrappersConfig, 
            training_config: RPOAgentTrainingConfig,
            network_config: RPOLinearNetworkConfig | RPOTransformerEmbeddingConfig,
        ):
        """
        Learnable RPO (Robust Policy Optimization) agent.
        
        Original paper: https://arxiv.org/abs/2212.07536 
        Code based on: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/rpo_continuous_action.py
        """
               
        self.cfg = training_config

        self.envs = gym.vector.SyncVectorEnv(
            [make_env(env_config, env_wrappers_config, self.cfg.gamma)\
                for _ in range(self.cfg.num_envs)]
        )
        self.action_space = self.envs.single_action_space
        self.obs_space = self.envs.single_observation_space
        
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() and self.cfg.cuda else 'cpu')

        self.wandb_enabled = env_config.wandb_enabled

        # seeding        
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)
        torch.backends.cudnn.deterministic = self.cfg.torch_deterministic

        if isinstance(network_config, RPOLinearNetworkConfig):
            self.net = RPOLinearNetwork(self.envs, network_config, self.device)
        elif isinstance(network_config, RPOTransformerEmbeddingConfig):
            self.net = RPOTransformerEmbedding(self.envs, env_config.number_of_pedestrians,
                                               config=network_config, device=self.device)
        elif isinstance(network_config, RPODeepSetsEmbeddingConfig):
            self.net = RPODeepSetsEmbedding(self.envs, env_config.number_of_pedestrians, 
                                            config=network_config, device=self.device)
        else:
            raise NotImplementedError
        self.optimizer = optim.Adam(self.net.parameters(), 
                                    lr=self.cfg.learning_rate, eps=1e-5)        
        self.writer = SummaryWriter(os.path.join(TBLOGS_DIR, env_config.experiment_name))

    def learn(self):

        # prepare a storage for one update        
        obs = torch.zeros((self.cfg.num_steps, self.cfg.num_envs) + self.obs_space.shape).to(self.device)
        actions = torch.zeros((self.cfg.num_steps, self.cfg.num_envs) + self.action_space.shape).to(self.device)
        logprobs = torch.zeros((self.cfg.num_steps, self.cfg.num_envs)).to(self.device)
        rewards = torch.zeros((self.cfg.num_steps, self.cfg.num_envs)).to(self.device)
        dones = torch.zeros((self.cfg.num_steps, self.cfg.num_envs)).to(self.device)
        values = torch.zeros((self.cfg.num_steps, self.cfg.num_envs)).to(self.device)
        
        # TRY NOT TO MODIFY: start the game
        global_step = 0
        start_time = time.time()
        next_obs, _ = self.envs.reset(seed=self.cfg.seed)
        next_obs = torch.Tensor(next_obs).to(self.device)
        next_done = torch.zeros(self.cfg.num_envs).to(self.device)
        
        for update in range(1, self.cfg.num_updates + 1):   
            # Annealing the rate if instructed to do so.
            if self.cfg.anneal_lr:
                frac = 1.0 - (update - 1.0) / self.cfg.num_updates
                lrnow = frac * self.cfg.learning_rate
                self.optimizer.param_groups[0]["lr"] = lrnow

            # Collect statistics (fill the storage)
            for step in range(0, self.cfg.num_steps):
                global_step += 1 * self.cfg.num_envs                
                obs[step] = next_obs                                                # mem
                dones[step] = next_done                                             # mem
                
                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, logprob, _, value = self.net.get_action_and_value(next_obs)
                    values[step] = value.flatten()                                  # mem
                actions[step] = action                                              # mem
                logprobs[step] = logprob                                            # mem

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, terminations, truncations, infos = self.envs.step(action.cpu().numpy())
                done = np.logical_or(terminations, truncations)
                rewards[step] = torch.tensor(reward).to(self.device).view(-1)       # mem
                next_obs, next_done = torch.Tensor(next_obs).to(self.device), torch.Tensor(done).to(self.device)
                
                if "final_info" in infos:
                    for info in infos["final_info"]:
                        if info and "episode" in info:
                            print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                            self.writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                            self.writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

            # Bootstrap value if not done (estimate advantages and returns)
            with torch.no_grad():
                next_value = self.net.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(self.device)
                lastgaelam = 0
                for t in reversed(range(self.cfg.num_steps)):
                    if t == self.cfg.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + self.cfg.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = \
                        delta + self.cfg.gamma * self.cfg.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values

            # Flatten the batch
            b_obs = obs.reshape((-1,) + self.envs.single_observation_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + self.envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)
    
            # Optimizing the policy and value network
            b_inds = np.arange(self.cfg.batch_size)
            clipfracs = []
            for epoch in range(self.cfg.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, self.cfg.batch_size, self.cfg.minibatch_size):
                    end = start + self.cfg.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = self.net.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > self.cfg.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if self.cfg.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.cfg.clip_coef, 1 + self.cfg.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if self.cfg.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -self.cfg.clip_coef,
                            self.cfg.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - self.cfg.ent_coef * entropy_loss + v_loss * self.cfg.vf_coef

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.max_grad_norm)
                    self.optimizer.step()

                if self.cfg.target_kl is not None:
                    if approx_kl > self.cfg.target_kl:
                        break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            self.writer.add_scalar("charts/learning_rate", self.optimizer.param_groups[0]["lr"], global_step)
            self.writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            self.writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            self.writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            self.writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            self.writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            self.writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            self.writer.add_scalar("losses/explained_variance", explained_var, global_step)
            print("SPS:", int(global_step / (time.time() - start_time)))
            self.writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
    
        self.writer.close()