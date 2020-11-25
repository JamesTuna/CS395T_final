noises="0.2 0.4 0.6 0.8 1.0"
worst_among_n=50
opt="SGD"
log_dir="logs"
model_dir="saved_models"

epoch=400
decay_epoch=80
decay_ratio=0.1
lr=0.01
batch_size=128
print_step=1000

cuda=0



if [ ! -d $log_dir ]; then
  mkdir $log_dir
fi

if [ ! -d $model_dir ]; then
  mkdir $model_dir
fi

# train model
for noise in $noises
do
  layer=5
  hidden=32
  model_id="l"$layer"h"$hidden"noise"$noise"n"$worst_among_n"_lr"$lr"ep"$epoch"decay"$decay_epoch"rate"$decay_ratio
  log_dir_specific=$log_dir"/"$model_id
  if [ ! -d $log_dir_specific ]; then
    mkdir $log_dir_specific
  fi

  save_as=$model_dir"/"$model_id".ckpt"

  echo "start training model "$model_id"..."
  python3 trainDASO.py --layer $layer --hidden $hidden \
                        --noise $noise --n $worst_among_n \
                        --opt $opt --lr $lr --batch_size $batch_size \
                        --epoch $epoch --lr_decay_epoch $decay_epoch --lr_decay_rate $decay_ratio \
                        --ps $print_step --logdir $log_dir_specific --save_as $save_as --cuda $cuda &

done
