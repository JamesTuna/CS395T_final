noises="0.2 0.4 0.6 0.8 1.0"
daso_n="1 5"
model_dir="saved_models/lr_ep1000_decay250_rate0.2"
samples=10000
cuda=0
mkdir test_results
for noise in $noises
do

  logdir=test_results/noise${noise}_daso1
  mkdir $logdir
  model_file=$model_dir/noise_${noise}_daso1/saved.ckpt__epoch1000
  python3 test.py --load $model_file --noise $noise --samples $samples --cuda $cuda --logdir $logdir

  logdir=test_results/noise${noise}_daso5
  mkdir $logdir
  model_file=$model_dir/noise_${noise}_daso5/saved.ckpt__epoch1000
  python3 test.py --load $model_file --noise $noise --samples $samples --cuda $cuda --logdir $logdir
done
