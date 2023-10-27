source venv/bin/activate
export PYTHONPATH=.

n_ped=60
learn_timesteps=2000000

emb_states=("True")
emb_names=("grademb")

reward_states_isexitingreward=("True")
reward_names_isexitingreward=("exitrew")

reward_states_isfollowreward=("False")
reward_names_isfollowreward=("nofollowrew")

reward_states_intrinsicrewardcoef=("1." "0.")
alpha_states=("2")
noisecoef_states=(".05" ".2" ".5")
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

                for alpha in ${alpha_states[@]}; do

                    for noisecoef in ${noisecoef_states[@]}; do
                    
                        for ensldegree in ${ensldegree_states[@]}; do


                            exp_name="n${n_ped}_${emb_name}_${exitrew_name}_${follrew_name}_intrrew-${intrrew}_alpha-${alpha}_noise-${noisecoef}_ensldegree-${ensldegree}"
                            echo "Starting: $exp_name"
                            echo "python main.py --exp-name $exp_name -n $n_ped --learn-timesteps $learn_timesteps -e $emb_state --is-new-exiting-reward $exitrew_state --is-new-followers-reward $follrew_state --intrinsic-reward-coef $intrrew --alpha $alpha --noise-coef $noisecoef --enslaving-degree $ensldegree"
                            python main.py --exp-name $exp_name -n $n_ped --learn-timesteps $learn_timesteps -e $emb_state --is-new-exiting-reward $exitrew_state --is-new-followers-reward $follrew_state --intrinsic-reward-coef $intrrew --alpha $alpha --noise-coef $noisecoef --enslaving-degree $ensldegree &
                        done
                    
                    wait
                    done
                done
            done
        done
    done
done


