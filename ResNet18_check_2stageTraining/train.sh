# noise 0.1 0.2 0.3 0.4
# daso_n 1 5 20 50

########################################### pretrain #################################################
daso_n=1
epochs=1000
decay_rate=0.1
decay_epochs=250
lr=0.1
mkdir logs
mkdir saved_models
mkdir logs/pretrain_lr${lr}_ep${epochs}_decay${decay_epochs}_rate$decay_rate
mkdir saved_models/pretrain_lr${lr}_ep${epochs}_decay${decay_epochs}_rate$decay_rate

n=0.1
cuda=0
A=logs/pretrain_lr${lr}_ep${epochs}_decay${decay_epochs}_rate${decay_rate}/noise_${n}_daso${daso_n}
B=saved_models/pretrain_lr${lr}_ep${epochs}_decay${decay_epochs}_rate${decay_rate}/noise_${n}_daso${daso_n}
mkdir $A
mkdir $B
B="$B/saved.ckpt"
echo "run python3 train.py --n $daso_n --opt SGD --lr 0.01 --batch-size 128 --epoch $epochs --decay-epochs $decay_epochs --decay-rate $decay_rate --noise $n --ps 1000 --logdir $A --saveas $B"

python3 train.py --n $daso_n --opt SGD --lr $lr --batch-size 128 --epoch $epochs \
                  --decay-epochs $decay_epochs --decay-rate $decay_rate \
                    --noise $n --ps 1000 --logdir $A --saveas $B --cuda $cuda &
n=0.2
cuda=1
A=logs/pretrain_lr${lr}_ep${epochs}_decay${decay_epochs}_rate${decay_rate}/noise_${n}_daso${daso_n}
B=saved_models/pretrain_lr${lr}_ep${epochs}_decay${decay_epochs}_rate${decay_rate}/noise_${n}_daso${daso_n}
mkdir $A
mkdir $B
B="$B/saved.ckpt"
echo "run python3 train.py --n $daso_n --opt SGD --lr 0.01 --batch-size 128 --epoch $epochs --decay-epochs $decay_epochs --decay-rate $decay_rate --noise $n --ps 1000 --logdir $A --saveas $B"

python3 train.py --n $daso_n --opt SGD --lr $lr --batch-size 128 --epoch $epochs \
                  --decay-epochs $decay_epochs --decay-rate $decay_rate \
                    --noise $n --ps 1000 --logdir $A --saveas $B --cuda $cuda &

n=0.3
cuda=2
A=logs/pretrain_lr${lr}_ep${epochs}_decay${decay_epochs}_rate${decay_rate}/noise_${n}_daso${daso_n}
B=saved_models/pretrain_lr${lr}_ep${epochs}_decay${decay_epochs}_rate${decay_rate}/noise_${n}_daso${daso_n}
mkdir $A
mkdir $B
B="$B/saved.ckpt"
echo "run python3 train.py --n $daso_n --opt SGD --lr 0.01 --batch-size 128 --epoch $epochs --decay-epochs $decay_epochs --decay-rate $decay_rate --noise $n --ps 1000 --logdir $A --saveas $B"

python3 train.py --n $daso_n --opt SGD --lr $lr --batch-size 128 --epoch $epochs \
                  --decay-epochs $decay_epochs --decay-rate $decay_rate \
                    --noise $n --ps 1000 --logdir $A --saveas $B --cuda $cuda &

n=0.4
cuda=3
A=logs/pretrain_lr${lr}_ep${epochs}_decay${decay_epochs}_rate${decay_rate}/noise_${n}_daso${daso_n}
B=saved_models/pretrain_lr${lr}_ep${epochs}_decay${decay_epochs}_rate${decay_rate}/noise_${n}_daso${daso_n}
mkdir $A
mkdir $B
B="$B/saved.ckpt"
echo "run python3 train.py --n $daso_n --opt SGD --lr 0.01 --batch-size 128 --epoch $epochs --decay-epochs $decay_epochs --decay-rate $decay_rate --noise $n --ps 1000 --logdir $A --saveas $B"

python3 train.py --n $daso_n --opt SGD --lr $lr --batch-size 128 --epoch $epochs \
                  --decay-epochs $decay_epochs --decay-rate $decay_rate \
                    --noise $n --ps 1000 --logdir $A --saveas $B --cuda $cuda &


########################################### posttrain #################################################
daso_ns="1 5 20 50"
epochs=1000
decay_rate=0.1
decay_epochs=400
cuda=0
lr=0.0001
mkdir logs
mkdir saved_models
mkdir logs/lr${lr}_ep${epochs}_decay${decay_epochs}_rate$decay_rate
mkdir saved_models/lr${lr}_ep${epochs}_decay${decay_epochs}_rate$decay_rate



for daso_n in daso_ns
do
  n=0.4
  cuda=3
  A=logs/lr${lr}_ep${epochs}_decay${decay_epochs}_rate${decay_rate}/noise_${n}_daso${daso_n}
  B=saved_models/lr${lr}_ep${epochs}_decay${decay_epochs}_rate${decay_rate}/noise_${n}_daso${daso_n}
  load=saved_models/pretrain_lr0.1_ep1000_decay250_rate0.1/noise_${n}_daso${daso_n}/saved.ckpt__epoch1000
  python3 train.py --n $daso_n --opt SGD --lr 0.01 --batch-size 128 --epoch $epochs \
                    --decay-epochs $decay_epochs --decay-rate $decay_rate \
                      --noise $n --ps 1000 --logdir $A --saveas $B --cuda $cuda \
                      --load $load &

  n=0.3
  cuda=2
  A=logs/lr${lr}_ep${epochs}_decay${decay_epochs}_rate${decay_rate}/noise_${n}_daso${daso_n}
  B=saved_models/lr${lr}_ep${epochs}_decay${decay_epochs}_rate${decay_rate}/noise_${n}_daso${daso_n}
  load=saved_models/pretrain_lr0.1_ep1000_decay250_rate0.1/noise_${n}_daso${daso_n}/saved.ckpt__epoch1000
  python3 train.py --n $daso_n --opt SGD --lr 0.01 --batch-size 128 --epoch $epochs \
                    --decay-epochs $decay_epochs --decay-rate $decay_rate \
                      --noise $n --ps 1000 --logdir $A --saveas $B --cuda $cuda \
                      --load $load &
  n=0.1
  cuda=0
  A=logs/lr${lr}_ep${epochs}_decay${decay_epochs}_rate${decay_rate}/noise_${n}_daso${daso_n}
  B=saved_models/lr${lr}_ep${epochs}_decay${decay_epochs}_rate${decay_rate}/noise_${n}_daso${daso_n}
  load=saved_models/pretrain_lr0.1_ep1000_decay250_rate0.1/noise_${n}_daso${daso_n}/saved.ckpt__epoch1000
  python3 train.py --n $daso_n --opt SGD --lr 0.01 --batch-size 128 --epoch $epochs \
                    --decay-epochs $decay_epochs --decay-rate $decay_rate \
                      --noise $n --ps 1000 --logdir $A --saveas $B --cuda $cuda \
                      --load $load &

  n=0.2
  cuda=1
  A=logs/lr${lr}_ep${epochs}_decay${decay_epochs}_rate${decay_rate}/noise_${n}_daso${daso_n}
  B=saved_models/lr${lr}_ep${epochs}_decay${decay_epochs}_rate${decay_rate}/noise_${n}_daso${daso_n}
  load=saved_models/pretrain_lr0.1_ep1000_decay250_rate0.1/noise_${n}_daso${daso_n}/saved.ckpt__epoch1000
  python3 train.py --n $daso_n --opt SGD --lr 0.01 --batch-size 128 --epoch $epochs \
                    --decay-epochs $decay_epochs --decay-rate $decay_rate \
                      --noise $n --ps 1000 --logdir $A --saveas $B --cuda $cuda \
                      --load $load
done
