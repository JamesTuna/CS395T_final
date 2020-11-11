layer=2
hidden=32
noises="0 0.2 0.4 0.8 1.0 1.2 1.4 1.6"
worst_among_n=10
opt="SGD"
log_dir="logs"
model_dir="saved_models"
epoch=400
decay_epoch=80
decay_ratio=0.1
lr=0.01

for noise in $noises
do
  model_id="l"$layer"h"$hidden"noise"$noise"n"$worst_among_n"_lr"$lr"ep"$epoch"decay"$decay_epoch"rate"$decay_ratio
  load=$model_dir"/"$model_id".ckpt"

  noise_test=$noise

  samples=1000
  results_dir="logs/"$model_id"/noise${noise_test}"
  mkdir $results_dir
  echo "start noise ${noise_test} testing for saved model"
  echo $load
  python3 testModel.py --layer $layer --hidden $hidden --batch-size 1000 --samples $samples \
                        --noise $noise_test --logdir $results_dir\
                        --load $load &
done
