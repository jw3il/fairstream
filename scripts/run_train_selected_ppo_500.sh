base_params="--agents ppo ppo ppo ppo --learning-rate=1e-5 --iterations=500 --eval-interval=10 --num-gpus=1 --rollout-workers=6 --eval-workers=14 --test-export-traces=10 --frame-stacking=8"
EXEC=./scripts/train_full.sh

for run in 0 1 2
do
# main agent
$EXEC ${base_params} --trace-seed "${run}" --no-parameter-sharing --quality-fairness-coeff=0.25 --comment="ppo-500-lr1e-5-nosharing-fs8-${run}"

# ablations with different qf-coefficient
$EXEC ${base_params} --trace-seed "${run}" --no-parameter-sharing --quality-fairness-coeff=0.5 --comment="ppo-500-lr1e-5-nosharing-fs8-qf0.5-${run}"
$EXEC ${base_params} --trace-seed "${run}" --no-parameter-sharing --quality-fairness-coeff=0.75 --comment="ppo-500-lr1e-5-nosharing-fs8-qf0.75-${run}"

# other ablations
$EXEC ${base_params} --trace-seed "${run}" --parameter-sharing --quality-fairness-coeff=0.25 --comment="ppo-500-lr1e-5-fs8-${run}"
$EXEC ${base_params} --trace-seed "${run}" --no-parameter-sharing --lstm --quality-fairness-coeff=0.25 --comment="ppo-500-lr1e-5-nosharing-lstm-${run}"
$EXEC ${base_params} --trace-seed "${run}" --no-parameter-sharing --bw-sharing=minerva --quality-fairness-coeff=0.25 --comment="ppo-500-lr1e-5-nosharing-minerva-${run}"
done
