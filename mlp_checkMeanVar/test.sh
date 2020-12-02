layer=5
hidden=32
lambs="0 1 5 10"
noises="0.2 0.4 0.6 0.8"
opt="SGD"
log_dir="logs"
model_dir="saved_models"
epoch=1000
decay_epoch=250
decay_ratio=0.2
lr=0.01
samples=10000

for lamb in $lambs
do
  for noise in $noises
  do
    model_id="l"$layer"h"$hidden"noise"$noise"lamb"$lamb"_lr"$lr"ep"$epoch"decay"$decay_epoch"rate"$decay_ratio
    load=$model_dir"/"$model_id".ckpt__epoch"${epoch}
    noise_test=$noise
    results_dir="logs/"$model_id"/noise${noise_test}"
    mkdir $results_dir
    echo "start noise ${noise_test} testing for saved model "${load}
    python3 testModel.py --layer $layer --hidden $hidden --batch-size 1000 --samples $samples \
                          --noise $noise_test --logdir $results_dir\
                          --load $load &
  done
  noise=1.0
  model_id="l"$layer"h"$hidden"noise"$noise"lamb"$lamb"_lr"$lr"ep"$epoch"decay"$decay_epoch"rate"$decay_ratio
  load=$model_dir"/"$model_id".ckpt__epoch"${epoch}
  noise_test=$noise
  results_dir="logs/"$model_id"/noise${noise_test}"
  mkdir $results_dir
  echo "start noise ${noise_test} testing for saved model "${load}
  python3 testModel.py --layer $layer --hidden $hidden --batch-size 1000 --samples $samples \
                        --noise $noise_test --logdir $results_dir\
                        --load $load
done
