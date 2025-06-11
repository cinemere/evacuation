# %%%
import wandb
import pandas as pd
import os
import tqdm

# Initialize the W&B API
api = wandb.Api()

project_name = "evacuation_dec"

# Fetch all runs from the project
runs = api.runs(project_name)

# Create a directory to store downloaded data
output_dir = "bulk_downloads"
os.makedirs(output_dir, exist_ok=True)

# Iterate over all runs and download their history
for run in tqdm.tqdm(runs):
    history = run.history(samples=None)
    history_file = os.path.join(output_dir, f"history_{run.id}.csv")
    history.to_csv(history_file, index=False)
    print(f"Saved history for run {run.id} to {history_file}")
# %%

filters = {
    "$and": [
        {"config.env.number_of_pedestrians": 15},  # Filter by config parameter
        {"config.model.agent.total_timesteps": 3000000},  # Filter by config parameter
        {"created_at": {"$lte": "2025-02-14T00:00:00"}}  # Filter by creation timestamp
    ]
}

# Fetch runs matching the filters
runs = api.runs(project_name, filters=filters)

# Print the number of runs fetched
print(f"Found {len(runs)} runs matching the filters.")

import pandas as pd
from tqdm import tqdm

# Initialize an empty list to store all rows
data = []

# Define the smoothing factor for EMA (commonly between 0 and 1)
alpha_ema = 0.1  # Adjust this value based on your needs

# Iterate over all runs
for run in tqdm(runs, desc="Processing runs"):
    try:
        # Extract alpha from config, with a default value if missing
        alpha = run.config.get("wrap", {}).get("alpha", None)
        
        # Fetch the history for specific keys
        history = run.history(keys=["episode_length", "overall_timesteps", "episode_reward", "_step"])
        
        # Compute EMA for each metric
        history["episode_length_ema"] = history["episode_length"].ewm(alpha=alpha_ema, adjust=False).mean()
        history["episode_reward_ema"] = history["episode_reward"].ewm(alpha=alpha_ema, adjust=False).mean()
        
        # Add metadata to each row in the history
        for _, row in history.iterrows():
            data.append({
                "run_id": run.id,
                "name": run.name,
                "state": run.state,
                "alpha": alpha,
                "episode_length": row.get("episode_length", None),
                "overall_timesteps": row.get("overall_timesteps", None),
                "episode_reward": row.get("episode_reward", None),
                "episode_length_ema": row.get("episode_length_ema", None),
                "episode_reward_ema": row.get("episode_reward_ema", None),
                "step": row.get("_step", None)
            })
    except Exception as e:
        print(f"Error processing run {run.id}: {e}")

# Convert to a Pandas DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv("filtered_data_with_ema.csv", index=False)
print("All training data with EMA saved to 'all_training_data_with_ema.csv'.")
# %%
df
# %%
df[df["alpha"] == 1].run_id.unique()
# %%
df[df["run_id"] == df[df["alpha"] == 1].run_id.unique()[1]]["step"]
# %%
import matplotlib.pyplot as plt
# grouped = df.groupby("alpha")
# df["smooth_overall_timesteps"] = df["overall_timesteps"] // 1000 * 1000
df["smooth_overall_timesteps"] = df["overall_timesteps"] // 2000 * 2000
# df["smooth_overall_timesteps"] = df["overall_timesteps"] // 3000 * 3000
# %%
plot_df = df.groupby(["alpha", "smooth_overall_timesteps"]).agg({
# plot_df = df.groupby(["alpha", "step"]).agg({
    # "overall_timesteps" : ["mean"],
    "episode_length": ["mean", "std"],
    "episode_reward": ["mean", "std"],
    "episode_length_ema": ["mean", "std"],
    "episode_reward_ema": ["mean", "std"],
}).reset_index()
# %%
from matplotlib import rcParams
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = ['Computer Modern']

rcParams['font.size'] = 14  # Default font size
rcParams['axes.titlesize'] = 20  # Title font size
rcParams['axes.titleweight'] = 'bold'  # Title font weight
rcParams['axes.labelsize'] = 15  # X/Y axis label font size
rcParams['legend.fontsize'] = 12  # Legend font size
rcParams["legend.handleheight"] = 1
rcParams['xtick.labelsize'] = 12  # X-axis tick label font size
rcParams['ytick.labelsize'] = 12  # Y-axis tick label font size

SAVEDIR="/Users/Klepach/work/repo/tmp/evacuation/final_plots"
# 
# colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
# colors = ["#635147", "#E74C3C", "#EFC473", "#61B6B6", "#3498DB"]
# colors = ["#1F77B4", "#2CA02C", "#E9A73D", "#b563d1", "#E74C3C"]
# colors = ["#E74C3C", "#F39C12", "#F1C40F", "#EC7063", "#D35400"]
# colors = ["#4B77BE", "#68C3A3", "#6C7A89", "#9B59B6", "#3498DB"]
# colors = ["#FF6F61", "#FFD700", "#87CEEB", "#FF4500", "#7FFF00"]
# colors = ["#000000", "#B91DC2", "#FF0000", "#00FF00", "#0000FF"]
colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442"]
# %%

# Show the plot

# Create the plot
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
for alpha, color in zip(df.alpha.unique(), colors):
    filter_alpha = plot_df["alpha"] == alpha
    ax.plot(
        plot_df[filter_alpha]["smooth_overall_timesteps"],
        plot_df[filter_alpha][("episode_length_ema", "mean")],
        lw=1,
        alpha=0.8,
        color=color
    )
    ax.fill_between(
        x=plot_df[filter_alpha]["smooth_overall_timesteps"],
        y1=plot_df[filter_alpha][("episode_length_ema", "mean")]+plot_df[filter_alpha][("episode_length_ema", "std")],
        y2=plot_df[filter_alpha][("episode_length_ema", "mean")]-plot_df[filter_alpha][("episode_length_ema", "std")],
        alpha=0.2,
        color=color
    )
for alpha, color in zip(df.alpha.unique(), colors):
    label=f"{alpha}"
    ax.plot([], [], linestyle="-", color=color, label=label, lw=2)

# Add labels and legend
# ax.set_xscale("log")
ax.set_xlabel("Overall Timesteps")
ax.set_ylabel("Episode length")  # Replace "Metric" with the actual metric name
num=filters["$and"][0]["config.env.number_of_pedestrians"]
ax.set_title(f"Episode length for {num} particles\n\n")
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, 
           ncol=5, bbox_to_anchor=(.9, .9), frameon=False,
           title=r"for varying $\alpha$:")
plt.grid(True, color="grey", alpha=0.2)
plt.tight_layout()
plt.savefig(os.path.join(SAVEDIR,f"length-alpha-{num}.pdf"))
plt.show()
# %%
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
for alpha, color in zip(df.alpha.unique(), colors):
    filter_alpha = plot_df["alpha"] == alpha
    ax.plot(
        plot_df[filter_alpha]["smooth_overall_timesteps"],
        # plot_df[filter_alpha][("overall_timesteps", "mean")],
        plot_df[filter_alpha][("episode_reward_ema", "mean")],
        # label=f"alpha={alpha}",
        lw=1,
        alpha=0.8,
        color=color,
    )
    skip = 1
    ax.fill_between(
        x=plot_df[filter_alpha]["smooth_overall_timesteps"][::skip],
        # x=plot_df[filter_alpha][("overall_timesteps", "mean")][::skip],
        y1=plot_df[filter_alpha][("episode_reward_ema", "mean")][::skip]+\
            abs(plot_df[filter_alpha][("episode_reward_ema", "std")][::skip]),
        y2=plot_df[filter_alpha][("episode_reward_ema", "mean")][::skip]-\
            abs(plot_df[filter_alpha][("episode_reward_ema", "std")][::skip]),
        step="pre",
        alpha=0.2,
        color=color,
    )
    # break

for alpha, color in zip(df.alpha.unique(), colors):
    label=f"{alpha}" 
    ax.plot([], [], linestyle="-", color=color, label=label, lw=2)

# Add labels and legend|
# ax.set_xscale("log")
ax.set_xlabel("Overall Timesteps")
ax.set_ylabel("Episode reward")  # Replace "Metric" with the actual metric name
num=filters["$and"][0]["config.env.number_of_pedestrians"]
ax.set_title(f"Episode reward for {num} particles\n\n")
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, 
           ncol=5, bbox_to_anchor=(.9, .9), frameon=False,
           title=r"for varying $\alpha$:")
plt.grid(True, color="grey", alpha=0.2)
plt.tight_layout()
plt.savefig(os.path.join(SAVEDIR,f"reward-alpha-{num}.pdf"))
plt.show()
# %%
# %%
