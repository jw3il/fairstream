test_export="--test-export-traces=10"
EXEC=./scripts/train_full.sh

for minerva in 0 1
do

if [ $minerva -eq 1 ]
then
minerva_comment="-minerva"
minerva_arg="--bw-sharing=minerva"
else
minerva_comment=""
minerva_arg=""
fi

$EXEC --eval-only --agents min min min min --comment="eval-only-min${minerva_comment}" --eval-workers=16 ${minerva_arg} ${test_export}
$EXEC --eval-only --agents max max max max --comment="eval-only-max${minerva_comment}" --eval-workers=16 ${minerva_arg} ${test_export}
$EXEC --eval-only --agents random random random random --comment="eval-only-random${minerva_comment}" --eval-workers=16 ${minerva_arg} ${test_export}
$EXEC --eval-only --agents greedy greedy greedy greedy --comment="eval-only-greedy-8${minerva_comment}" --eval-workers=16 ${minerva_arg} ${test_export}

for i in 1 2 4 8 16 32
do
   $EXEC --eval-only --agents greedy greedy greedy greedy --greedy-k="$i" --comment="eval-val-only-greedy-${i}${minerva_comment}" --eval-workers=16 ${minerva_arg} --eval-validation
done
done