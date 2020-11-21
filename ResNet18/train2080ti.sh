noise="1.0 0.8 0.6 0.4 0.2"
daso_n=10
epochs=300
decay_rate=0.1
decay_epochs=100
save_per_epochs=50
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


                      # handling cuda error and keep training
                      while [ ! -f $B__epoch$epochs ] # when saved models with $epochs training doesn't exist
                      do
                        echo "training unfinished, try to locate the latest model..."
                        latest=0
                        for file in $(ls $B"__epoch"*)
                        do
                          for word in $(echo $file | tr "__epoch" "\n")
                          do
                            epoch=$word
                          done
                          if [ $epoch -gt $latest ]; then latest=$epoch; fi
                        done
                        echo "keep training from epoch ${latest}"

                        if [ $latest -eq 0 ]
                        then
                          echo "train from scratch"
                          python3 train.py --n $daso_n --opt SGD --lr 0.01 --batch-size 128 --epoch $epochs \
                                          --decay-epochs $decay_epochs --decay-rate $decay_rate \
                                            --noise $n --ps 1000 --logdir $A --saveas $B --cuda $cuda \
                                            --save-per-epochs $save_per_epochs
                        else
                          echo "continue training from epoch ${latest}"
                          python3 train.py --n $daso_n --opt SGD --lr 0.01 --batch-size 128 --epoch $epochs \
                                          --decay-epochs $decay_epochs --decay-rate $decay_rate \
                                            --noise $n --ps 1000 --logdir $A --saveas $B --cuda $cuda \
                                            --save-per-epochs $save_per_epochs \
                                            --continueEp $latest --load "${B}__epoch${latest}"
                        fi
                      done
done
