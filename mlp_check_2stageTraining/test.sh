layer=5
hidden=32
noises="0.2 0.4 0.6 0.8"
worst_among_ns="1 5 10 20 50"
opt="SGD"
log_dir="logs"
model_dir="saved_models"
epoch=400
decay_epoch=100
decay_ratio=0.1
lr=0.01
samples=10000

for worst_among_n in $worst_among_ns
do
  for noise in $noises
  do
    model_id="l"$layer"h"$hidden"noise"$noise"n"$worst_among_n"_lr"$lr"ep"$epoch"decay"$decay_epoch"rate"$decay_ratio
    load=$model_dir"/"$model_id".ckpt"
    noise_test=$noise
    results_dir="logs/"$model_id"/noise${noise_test}"
    mkdir $results_dir
    echo "start noise ${noise_test} testing for saved model "${load}
    python3 ../mlp/testModel.py --layer $layer --hidden $hidden --batch-size 1000 --samples $samples \
                          --noise $noise_test --logdir $results_dir\
                          --load $load &
  done
  noise=1.0
  model_id="l"$layer"h"$hidden"noise"$noise"n"$worst_among_n"_lr"$lr"ep"$epoch"decay"$decay_epoch"rate"$decay_ratio
  load=$model_dir"/"$model_id".ckpt"
  noise_test=$noise
  results_dir="logs/"$model_id"/noise${noise_test}"
  mkdir $results_dir
  echo "start noise ${noise_test} testing for saved model "${load}
  python3 ../mlp/testModel.py --layer $layer --hidden $hidden --batch-size 1000 --samples $samples \
                        --noise $noise_test --logdir $results_dir\
                        --load $load
done
