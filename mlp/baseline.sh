noise=0
worst_among_n=0
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

layer=5
hidden=32
model_id="l"$layer"h"$hidden"noise"$noise"n"$worst_among_n"_lr"$lr"ep"$epoch"decay"$decay_epoch"rate"$decay_ratio
log_dir_specific=$log_dir"/"$model_id

mkdir $log_dir_specific
save_as=$model_dir"/"$model_id".ckpt"

echo "start training model "$model_id"..."
  python3 trainDASO.py --layer $layer --hidden $hidden \
                        --noise $noise --n $worst_among_n \
                        --opt $opt --lr $lr --batch_size $batch_size \
                        --epoch $epoch --lr_decay_epoch $decay_epoch --lr_decay_rate $decay_ratio \
                        --ps $print_step --logdir $log_dir_specific --save_as $save_as --cuda $cuda



noises="0.2 0.4 0.6 0.8 1.0"
for noise in $noises
do
    model_id="l"$layer"h"$hidden"noise"$noise"n"$worst_among_n"_lr"$lr"ep"$epoch"decay"$decay_epoch"rate"$decay_ratio
    load=$model_dir"/"$model_id".ckpt"
    noise_test=$noise
    results_dir="logs/"$model_id"/noise${noise_test}"
    mkdir $results_dir
    echo "start noise ${noise_test} testing for saved model "${load}
    python3 testModel.py --layer $layer --hidden $hidden --batch-size 1000 --samples $samples \
                          --noise $noise_test --logdir $results_dir\
                          --load $load &
done
