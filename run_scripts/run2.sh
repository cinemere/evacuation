#!/bin/bash

#SBATCH --job-name=parallel_job               # Job name
#SBATCH --partition=cpu_small                 # Queue name
#SBATCH --mail-type=END,FAIL                  # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=klepach.aa@phystech.edu   # Where to send mail
#SBATCH --nodes=1                             # Run all processes on a single node
#SBATCH --ntasks=1                            # Run a single task
#SBATCH --cpus-per-task=1                     # Number of CPU cores per task
#SBATCH --mem=1gb                             # Job memory request
#SBATCH --time=24:00:00                       # Time limit hrs:min:sec or dd-hrs:min:sec
#SBATCH --output=/gpfs/gpfs0/n.lastname/parallel_%j.log     # Standard output and error log

#MODULE LOAD PART

module load python/anaconda3

#COMMAND PART

conda activate aklepach_env
cd /home/rice/aklepach/code/evacuation
export PYTHONPATH=.

n_ped=60
learn_timesteps=2000000

emb_states=(True False)
emb_names=("grademb" "noemb")

reward_states_isexitingreward=(True False)
reward_names_isexitingreward=("exitrew" "noexitrew")

reward_states_isfollowreward=(True False)
reward_names_isfollowreward=("followrew" "nofollowrew")

reward_states_intrinsicrewardcoef=("0." "1.")
# reward_initrew=("0." "-1.")
reward_initrew=("-1.")
alpha_states=("2" "3" "4" "5")
noisecoef_states=(".2" ".5" ".8")
ensldegree_states=("1." ".5" ".1")

for i_emb in ${!emb_states[@]}; do
    emb_state=${emb_states[$i_emb]}
    emb_name=${emb_names[$i_emb]}

    for i_exitrew in ${!reward_states_isexitingreward[@]}; do
        exitrew_state=${reward_states_isexitingreward[$i_exitrew]}
        exitrew_name=${reward_names_isexitingreward[$i_exitrew]}
        
        for i_follrew in ${!reward_states_isfollowreward[@]}; do
            follrew_state=${reward_states_isfollowreward[$i_follrew]}
            follrew_name=${reward_names_isfollowreward[$i_follrew]}
        
            for intrrew in ${reward_states_intrinsicrewardcoef[@]}; do

                for initrew in ${reward_initrew[@]}; do
                    for alpha in ${alpha_states[@]}; do

                        for noisecoef in ${noisecoef_states[@]}; do
                        
                            for ensldegree in ${ensldegree_states[@]}; do

                                echo ""; echo "----------------------------------------------------------------"; echo "";

                                exp_name="n${n_ped}_${emb_name}_${exitrew_name}_${follrew_name}_intrrew-${intrrew}_initrew-${initrew}_alpha-${alpha}_noise-${noisecoef}_ensldegree-${ensldegree}"
                                echo "Starting: $exp_name"

                                env_setup="-n ${n_ped} --noise-coef ${noisecoef} --enslaving-degree ${ensldegree}"
                                echo "Env params: ${env_setup}"

                                reward_setup="--is-new-exiting-reward ${exitrew_state} --is-new-followers-reward ${follrew_state} --intrinsic-reward-coef ${intrrew} --init-reward-each-step=${initrew}"
                                echo "Reward params: ${reward_setup}"
                                
                                learn_setup="--learn-timesteps ${learn_timesteps} -e ${emb_state} --alpha ${alpha}"
                                echo "Learning experiment params: ${learn_setup}"

                                query="--exp-name ${exp_name} ${reward_setup} ${env_setup} ${learn_setup}"
                                echo ""; echo "All params:"; echo ""; echo "---->    ${query}"; echo ""
                            
                                python3 main.py $query &

                            done
                            
                            wait
                        done
                    done
                done
            done
        done
    done
done


