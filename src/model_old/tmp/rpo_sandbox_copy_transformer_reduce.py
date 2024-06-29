# %%
# This implementations is taken from https://github.com/vwxyzjn/cleanrl
# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/rpo/#rpo_continuous_actionpy
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


@dataclass
class Args:
    exp_name: str = "rpo-tr3-lr3e5"##os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True  #False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "evacuation+cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v4"
    """the id of the environment"""
    total_timesteps: int = 8000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-5
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 2048
    """the number of steps to run in each environment per policy rollout"""
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
    target_kl: float = None
    """the target KL divergence threshold"""
    rpo_alpha: float = 0.5
    """the alpha parameter for RPO"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""
# %%
from src.env import EvacuationEnv, RelativePosition, constants, PedestriansStatuses
from src import params_old
from src.utils import get_experiment_name, parse_args

def setup_env(args, experiment_name):
    env = EvacuationEnv(
        experiment_name=experiment_name,
        number_of_pedestrians=args.number_of_pedestrians,
        enslaving_degree=args.enslaving_degree, 
        width=args.width,
        height=args.height,
        step_size=args.step_size,
        noise_coef=args.noise_coef,
        is_termination_agent_wall_collision=args.is_termination_agent_wall_collision,
        is_new_exiting_reward=args.is_new_exiting_reward,
        is_new_followers_reward=args.is_new_followers_reward,
        intrinsic_reward_coef=args.intrinsic_reward_coef,
        max_timesteps=args.max_timesteps,
        n_episodes=args.n_episodes,
        n_timesteps=args.n_timesteps,
        enabled_gravity_embedding=args.enabled_gravity_embedding,
        alpha=args.alpha,
        verbose=args.verbose,
        render_mode=None,
        draw=args.draw
    ) 
    return env

# %%
def setup_evacuation_env():
    env_args = parse_args(True, [
        "--exp-name", "rpo-debug-tr3-randinit-noresidual-reduced-stat-lr3e5",
        # "--init-reward-each-step", "-1." set in constants
        # "-e", "true",
        "-e", "false",
        "--intrinsic-reward-coef", "0",
        ])
    experiment_name = get_experiment_name(env_args)
    env = setup_env(env_args, experiment_name)
    env = RelativePosition(env)
    env = PedestriansStatuses(env)
    return env

# %%
def wrap_env(env, gamma):    
    env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)
    env = gym.wrappers.NormalizeObservation(env)
    env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -1, 1)) # FIXME check this clipping
    env = gym.wrappers.NormalizeReward(env, gamma=gamma)
    env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -100, 100)) # FIXME check this clipping
    return env

def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():  
        env = setup_evacuation_env()
        env = wrap_env(env, gamma)
        return env
    return thunk

env = make_env(None, None, None, None, 0.99)
# def make_env(env_id, idx, capture_video, run_name, gamma):
#     def thunk():
#         if capture_video and idx == 0:
#             env = gym.make(env_id, render_mode="rgb_array")
#             env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
#         else:
#             env = gym.make(env_id)
#         env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
#         env = gym.wrappers.RecordEpisodeStatistics(env)
#         env = gym.wrappers.ClipAction(env)
#         env = gym.wrappers.NormalizeObservation(env)
#         env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
#         env = gym.wrappers.NormalizeReward(env, gamma=gamma)
#         env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
#         return env
#     return thunk
# %%

import torch
import torch.nn as nn
import torch.optim as optim

class DeepSets(nn.Module):
    def __init__(self, input_dim, embedding_dim, output_dim, n_sets: int = 3):
        super(DeepSets, self).__init__()
        self.n_sets = n_sets
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, embedding_dim * self.n_sets),
            nn.ReLU(),
            nn.Linear(embedding_dim * self.n_sets, embedding_dim * self.n_sets)
        )
        self.aggregation = nn.Sequential(
            nn.Linear(embedding_dim, output_dim)
        )

    def forward(self, x):                           # (batch_size, set_size, input_dim)
        # print('====')
        # print(f"{x.shape=}")
        # x = x.unsqueeze(1)
        # x = x.view(1, 2, -1)
        # print(f"{x.shape=}")
        embedded = self.embedding(x)                # (batch_size, set_size, embedding_dim)
        dim = x.dim() - 1  # 1
        # print(f'{embedded.shape=} {dim=}')
        embedded = embedded.view(x.shape[0], self.n_sets, -1)
        # print(f'{embedded.shape=} {dim=}')
        aggregated = embedded.sum(dim=dim)          # (batch_size, embedding_dim)
        # print(f'{aggregated.shape=}')
        output = self.aggregation(aggregated)       # (batch_size, output_dim)
        # print(f'{output.shape}')
        # print('-----')
        return output
    
#%%
x = torch.rand(1, 24)
ds = DeepSets(24, 64, 24)
ds(x)
#%%
ds.embedding
#%%
# class DeepSelector(nn.Module):
#     def __init__(self, input_dim, embedding_dim, output_dim) -> None:
#         self.k = layer_init(
#             nn.Linear(np.array(envs.single_observation_space.shape).prod(), embedding_dim))
#         self.q = layer_init(
#             nn.Linear(np.array(envs.single_observation_space.shape).prod(), embedding_dim))
#         self.v = layer_init(
#             nn.Linear(np.array(envs.single_observation_space.shape).prod(), embedding_dim))
import torch
import torch.nn as nn
import torch.nn.functional as F
# %%
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads: int = 3):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads
        
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.dense = nn.Linear(d_model, d_model)
        
    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        # x = x.view(batch_size, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)
        # return x.permute(0, 1, 2)
    
    # def forward(self, q, k, v, mask=None):
    #     batch_size = q.size(0)
        
    #     q = self.split_heads(self.Wq(q), batch_size)
    #     k = self.split_heads(self.Wk(k), batch_size)
    #     v = self.split_heads(self.Wv(v), batch_size)
    #     # print(f"{q.shape=} {k.shape=} {v.shape=}")
        
    #     scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.depth, dtype=torch.float32))
    #     if mask is not None:
    #         scores = scores.masked_fill(mask == 0, -1e9)
        
    #     attention_weights = F.softmax(scores, dim=-1)
    #     output = torch.matmul(attention_weights, v)
        
    #     output = output.permute(0, 2, 1, 3).contiguous()
    #     output = output.view(batch_size, -1, self.d_model)
    #     output = self.dense(output)
    #     return output
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        q = self.split_heads(self.Wq(q), batch_size)
        k = self.split_heads(self.Wk(k), batch_size)
        v = self.split_heads(self.Wv(v), batch_size)
        # print(f"{q.shape=} {k.shape=} {v.shape=}")
        
        # scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.depth, dtype=torch.float32))
        scores = torch.matmul(q.transpose(-2, -1), k) / torch.sqrt(torch.tensor(self.depth, dtype=torch.float32))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        # output = torch.matmul(attention_weights, v)
        output = torch.matmul(attention_weights, v.transpose(-2, -1))
        
        output = output.permute(0, 2, 1, 3).contiguous()
        output = output.view(batch_size, -1, self.d_model)
        output = self.dense(output)
        return output



class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1, d_output=4):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_output)
        
    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, d_model, 
                 num_heads, 
                 d_ff: int = 64, 
                 dropout: float = 0.1
        ) -> None:
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        # self.norm2 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(4)
        self.ff = FeedForward(d_model, d_ff, dropout, d_output=4)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        attn_output = self.attention(x, x, x, mask)
        # print(f"{attn_output.shape=}")
        # out1 = self.norm1(x + self.dropout(attn_output))
        out1 = self.norm1(self.dropout(attn_output))
        ff_output = self.ff(out1)
        # out2 = self.norm2(out1 + self.dropout(ff_output))
        out2 = self.norm2(self.dropout(ff_output))
        if out2.dim() == 3:
            out2 = out2.squeeze(1)
        return out2
#%%

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs, rpo_alpha, input_shape):
        super().__init__()
        self.rpo_alpha = rpo_alpha
        self.critic = nn.Sequential(
            layer_init(nn.Linear(input_shape, 64)),
            # layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(input_shape, 64)),
            # layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        else:  # new to RPO
            # sample again to add stochasticity to the policy
            z = torch.FloatTensor(action_mean.shape).uniform_(-self.rpo_alpha, self.rpo_alpha).to(device)
            action_mean = action_mean + z
            probs = Normal(action_mean, action_std)

        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

# %%
class DeepSetsAgent(Agent):
    def __init__(self, envs, rpo_alpha):
        super().__init__(envs, rpo_alpha, 34)
        self.deep_sets = DeepSets(
            input_dim=envs.observation_space.shape[-1],
            embedding_dim=34,
            output_dim=envs.observation_space.shape[-1]
        )
        
    def get_value(self, x):
        x = self.deep_sets(x)
        return super().get_value(x)
    
    def get_action_and_value(self, x, action=None):
        x = self.deep_sets(x)
        return super().get_action_and_value(x, action)
# %%
class TransformerAgent(Agent):
    def __init__(self, envs, rpo_alpha):
        super().__init__(envs, rpo_alpha, input_shape=4)
        # self.deep_sets = DeepSets(
        #     input_dim=envs.observation_space.shape[-1],
        #     embedding_dim=24,
        #     output_dim=envs.observation_space.shape[-1]
        # )
        self.transformer = TransformerBlock(
            d_model=envs.observation_space.shape[-1],
            num_heads=2,  # 3
            d_ff=34,  # 24
        )
        
    def get_value(self, x):
        x = self.transformer(x)
        return super().get_value(x)
    
    def get_action_and_value(self, x, action=None):
        x = self.transformer(x)
        return super().get_action_and_value(x, action)
# %%

# %% SETUP ARGS

# if __name__ == "__main__":

# args = tyro.cli(Args)
args = tyro.cli(Args, args=[]) 
# %%
args.batch_size = int(args.num_envs * args.num_steps)
args.minibatch_size = int(args.batch_size // args.num_minibatches)
args.num_iterations = args.total_timesteps // args.batch_size
run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
if args.track:
    import wandb

    wandb.init(
        project=args.wandb_project_name,
        entity=args.wandb_entity,
        sync_tensorboard=True,
        config=vars(args),
        name=run_name,
        monitor_gym=True,
        save_code=True,
    )
    
writer = SummaryWriter(f"runs/{run_name}")
writer.add_text(
    "hyperparameters",
    "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
)

# TRY NOT TO MODIFY: seeding
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic

# device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
device = torch.device("cpu")

# env setup
envs = gym.vector.SyncVectorEnv(
    [make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
)
assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
# %% INITIALIZE
# agent = Agent(envs, args.rpo_alpha).to(device)
# agent = DeepSetsAgent(envs, args.rpo_alpha).to(device)
agent = TransformerAgent(envs, args.rpo_alpha).to(device)
optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

# ALGO Logic: Storage setup
obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
values = torch.zeros((args.num_steps, args.num_envs)).to(device)

# %%

# TRY NOT TO MODIFY: start the game
global_step = 0
start_time = time.time()
next_obs, _ = envs.reset(seed=args.seed)
next_obs = torch.Tensor(next_obs).to(device)
next_done = torch.zeros(args.num_envs).to(device)
num_updates = args.total_timesteps // args.batch_size

for update in range(1, num_updates + 1):
    # Annealing the rate if instructed to do so.
    if args.anneal_lr:
        frac = 1.0 - (update - 1.0) / num_updates
        lrnow = frac * args.learning_rate
        optimizer.param_groups[0]["lr"] = lrnow

    for step in range(0, args.num_steps):
        global_step += 1 * args.num_envs
        obs[step] = next_obs
        dones[step] = next_done

        # ALGO LOGIC: action logic
        with torch.no_grad():
            action, logprob, _, value = agent.get_action_and_value(next_obs)
            values[step] = value.flatten()
        actions[step] = action
        logprobs[step] = logprob

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
        done = np.logical_or(terminations, truncations)
        rewards[step] = torch.tensor(reward).to(device).view(-1)
        next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

    # bootstrap value if not done
    with torch.no_grad():
        next_value = agent.get_value(next_obs).reshape(1, -1)
        advantages = torch.zeros_like(rewards).to(device)
        lastgaelam = 0
        for t in reversed(range(args.num_steps)):
            if t == args.num_steps - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]
            delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
        returns = advantages + values

    # flatten the batch
    b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
    b_logprobs = logprobs.reshape(-1)
    b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
    b_advantages = advantages.reshape(-1)
    b_returns = returns.reshape(-1)
    b_values = values.reshape(-1)

    # Optimizing the policy and value network
    b_inds = np.arange(args.batch_size)
    clipfracs = []
    for epoch in range(args.update_epochs):
        np.random.shuffle(b_inds)
        for start in range(0, args.batch_size, args.minibatch_size):
            end = start + args.minibatch_size
            mb_inds = b_inds[start:end]

            _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
            logratio = newlogprob - b_logprobs[mb_inds]
            ratio = logratio.exp()

            with torch.no_grad():
                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

            mb_advantages = b_advantages[mb_inds]
            if args.norm_adv:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            # Policy loss
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # Value loss
            newvalue = newvalue.view(-1)
            if args.clip_vloss:
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    newvalue - b_values[mb_inds],
                    -args.clip_coef,
                    args.clip_coef,
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
            else:
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

            entropy_loss = entropy.mean()
            loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            optimizer.step()

        if args.target_kl is not None:
            if approx_kl > args.target_kl:
                break

    y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
    var_y = np.var(y_true)
    explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

    # TRY NOT TO MODIFY: record rewards for plotting purposes
    writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
    writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
    writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
    writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
    writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
    writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
    writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
    writer.add_scalar("losses/explained_variance", explained_var, global_step)
    print("SPS:", int(global_step / (time.time() - start_time)))
    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

envs.close()
writer.close()

# %%
