noises="0.5"
daso_ns="1 50"
model_dir="saved_models/lr0.0001_ep1000_decay400_rate0.1"
samples=10000
cuda=0
mkdir test_results
for noise in $noises
do

  logdir=test_results/noise${noise}_daso1
  mkdir $logdir
  model_file=$model_dir/noise_${noise}_daso1__epoch1000
  python3 test.py --load $model_file --noise $noise --samples $samples --cuda $cuda --logdir $logdir &

  logdir=test_results/noise${noise}_daso50
  mkdir $logdir
  model_file=$model_dir/noise_${noise}_daso50__epoch1000
  python3 test.py --load $model_file --noise $noise --samples $samples --cuda $cuda --logdir $logdir
  sleep 10m
done
