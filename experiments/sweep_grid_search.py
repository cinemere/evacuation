
import wandb
import yaml

config_path = "experiments/sweep_grid_search_config.yaml"
wandb.login()



with open(config_path, 'r') as file:
    sweep_configuration = yaml.safe_load(file)

sweep_id = wandb.sweep(sweep=sweep_configuration, project="my-first-sweep")


bash_script_content = f"""#!/bin/bash

# Bash script to run wandb agent with the provided sweep_id
echo "Running wandb agent with sweep_id: {sweep_id}"
wandb agent {sweep_id}
"""

# Save bash-script
with open("experiments/run_sweep.sh", "w") as file:
    file.write(bash_script_content)

# Make it executable
import os
os.chmod("experiments/run_sweep.sh", 0o755)

print("Bash script 'run_sweep.sh' has been generated and is ready to use.")
print("Run it on all devices with:")
print("./experiments/run_sweep.sh")

