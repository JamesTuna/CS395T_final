#!/bin/bash
layer=5
hidden=32
noises="0.2 0.4 0.6 0.8 1.0"
lambs='0 1 5 10'
opt="SGD"
log_dir="logs"
model_dir="saved_models"

epoch=1000
decay_epoch=250
decay_ratio=0.2
lr=0.01
batch_size=128
print_step=1000
cuda=0

mkdir $log_dir
mkdir $model_dir

# train model
lamb=0
cuda=0
for noise in $noises; do
  model_id="l"$layer"h"$hidden"noise"$noise"lamb"$lamb"_lr"$lr"ep"$epoch"decay"$decay_epoch"rate"$decay_ratio
  log_dir_specific=$log_dir"/"$model_id
  if [ ! -d "$log_dir_specific" ]; then
    mkdir "$log_dir_specific"
  fi
  save_as="$model_dir/$model_id.ckpt"
  echo "start training model $model_id..."
  python3 trainModel.py --layer $layer --hidden $hidden \
                        --noise $noise --lamb $lamb \
                        --opt $opt --lr $lr --batch_size $batch_size \
                        --epoch $epoch --lr_decay_epoch $decay_epoch --lr_decay_rate $decay_ratio \
                        --ps $print_step --logdir "$log_dir_specific" --save_as "$save_as" \
                        --save-per-epochs 100 --cuda $cuda &
done

lamb=1
cuda=1
for noise in $noises; do
  model_id="l"$layer"h"$hidden"noise"$noise"lamb"$lamb"_lr"$lr"ep"$epoch"decay"$decay_epoch"rate"$decay_ratio
  log_dir_specific=$log_dir"/"$model_id
  if [ ! -d "$log_dir_specific" ]; then
    mkdir "$log_dir_specific"
  fi
  save_as="$model_dir/$model_id.ckpt"
  echo "start training model $model_id..."
  python3 trainModel.py --layer $layer --hidden $hidden \
                        --noise $noise --lamb $lamb \
                        --opt $opt --lr $lr --batch_size $batch_size \
                        --epoch $epoch --lr_decay_epoch $decay_epoch --lr_decay_rate $decay_ratio \
                        --ps $print_step --logdir "$log_dir_specific" --save_as "$save_as" \
                        --save-per-epochs 100 --cuda $cuda &
done

lamb=5
cuda=2
for noise in $noises; do
  model_id="l"$layer"h"$hidden"noise"$noise"lamb"$lamb"_lr"$lr"ep"$epoch"decay"$decay_epoch"rate"$decay_ratio
  log_dir_specific=$log_dir"/"$model_id
  if [ ! -d "$log_dir_specific" ]; then
    mkdir "$log_dir_specific"
  fi
  save_as="$model_dir/$model_id.ckpt"
  echo "start training model $model_id..."
  python3 trainModel.py --layer $layer --hidden $hidden \
                        --noise $noise --lamb $lamb \
                        --opt $opt --lr $lr --batch_size $batch_size \
                        --epoch $epoch --lr_decay_epoch $decay_epoch --lr_decay_rate $decay_ratio \
                        --ps $print_step --logdir "$log_dir_specific" --save_as "$save_as" \
                        --save-per-epochs 100 --cuda $cuda &
done

lamb=10
cuda=3
for noise in $noises; do
  model_id="l"$layer"h"$hidden"noise"$noise"lamb"$lamb"_lr"$lr"ep"$epoch"decay"$decay_epoch"rate"$decay_ratio
  log_dir_specific=$log_dir"/"$model_id
  if [ ! -d "$log_dir_specific" ]; then
    mkdir "$log_dir_specific"
  fi
  save_as="$model_dir/$model_id.ckpt"
  echo "start training model $model_id..."
  python3 trainModel.py --layer $layer --hidden $hidden \
                        --noise $noise --lamb $lamb \
                        --opt $opt --lr $lr --batch_size $batch_size \
                        --epoch $epoch --lr_decay_epoch $decay_epoch --lr_decay_rate $decay_ratio \
                        --ps $print_step --logdir "$log_dir_specific" --save_as "$save_as" \
                        --save-per-epochs 100 --cuda $cuda &
done
