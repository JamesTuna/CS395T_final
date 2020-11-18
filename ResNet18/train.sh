noise="0.2 0.4 0.6 0.8 1.0"
daso_n=50
epochs=1000
decay_rate=0.2
decay_epochs=250
cuda=0
mkdir logs
mkdir saved_models
mkdir logs/lr${lr}_ep${epochs}_decay${decay_epochs}_rate$decay_rate
mkdir saved_models/lr${lr}_ep${epochs}_decay${decay_epochs}_rate$decay_rate

for n in $noise
do
  A=logs/lr${lr}_ep${epochs}_decay${decay_epochs}_rate${decay_rate}/noise_${n}_daso${daso_n}
  B=saved_models/lr${lr}_ep${epochs}_decay${decay_epochs}_rate${decay_rate}/noise_${n}_daso${daso_n}
  mkdir $A
  mkdir $B
  B="$B/saved.ckpt"
  echo "run python3 train.py --n $daso_n --opt SGD --lr 0.01 --batch-size 128 --epoch $epochs --decay-epochs $decay_epochs --decay-rate $decay_rate --noise $n --ps 1000 --logdir $A --saveas $B"

  python3 train.py --n $daso_n --opt SGD --lr 0.01 --batch-size 128 --epoch $epochs \
                    --decay-epochs $decay_epochs --decay-rate $decay_rate \
                      --noise $n --ps 1000 --logdir $A --saveas $B --cuda $cuda
done
