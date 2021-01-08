opt="SGD"
log_dir="logs"
model_dir="saved_models"
epoch=500
decay_epoch=100
decay_ratio=0.1
lr=0.01
batch_size=128
print_step=1000
cuda=0
layer=5
hidden=8
mkdir $log_dir
mkdir $model_dir


############################################### post train model #########################################
daso_n=50
noise=0.4
lrs="0.001 0.0001 0.00001 0.0000001"

for lr in $lrs
do
  load=$model_dir"/pretrain_l"$layer"h"$hidden"noise"$noise"n1_lr0.01ep500decay100rate0.1.ckpt"
  model_id="l"$layer"h"$hidden"noise"$noise"n"$daso_n"_lr"$lr"ep"$epoch"decay"$decay_epoch"rate"$decay_ratio
  log_dir_specific=$log_dir"/"$model_id
  mkdir $log_dir_specific
  save_as=$model_dir"/"$model_id".ckpt"

  echo "start post-training model "$model_id"..."
  python3 ../mlp/trainDASO.py --layer $layer --hidden $hidden \
                        --noise $noise --n $daso_n \
                        --opt $opt --lr $lr --batch_size $batch_size \
                        --epoch $epoch --lr_decay_epoch $decay_epoch --lr_decay_rate $decay_ratio \
                        --ps $print_step --logdir $log_dir_specific --save_as $save_as \
                        --load $load --cuda $cuda &
done
