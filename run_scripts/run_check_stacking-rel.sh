common_args="-rel true -e false --init-reward-each-step=-1. --intrinsic-reward-coef 0 --enslaving-degree 1"

query1="--exp-name test-stacking-2-rel-ppo --num-obs-stacks 2 --origin ppo ${common_args}"
query2="--exp-name test-stacking-1-rel-ppo --num-obs-stacks 1 --origin ppo ${common_args}"
query3="--exp-name test-stacking-2-rel-sac --num-obs-stacks 2 --origin sac ${common_args}"
query4="--exp-name test-stacking-1-rel-sac --num-obs-stacks 1 --origin sac ${common_args}"

echo "python3 main.py ${query1}"
echo "python3 main.py ${query2}"
echo "python3 main.py ${query3}"
echo "python3 main.py ${query4}"

python3 main.py $query1 &
python3 main.py $query2 &
python3 main.py $query3 &
python3 main.py $query4 &
wait