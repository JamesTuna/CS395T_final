# Vortex
#noise="0.2 0.4 0.6 0.8 1.0"
#gamma="0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0 2.2 2.4 2.6 2.8 3.0 3.2 3.4 3.6 3.8 4.0 4.2 4.4 4.6"
#samples=10000
#mkdir test_results

#for g in $gamma
#do
#  for n in $noise
#  do
#    echo "test W_${g} under noise $n"
#    mkdir test_results/w${g}
#    mkdir test_results/w${g}/noise${n}
#    python3 testVortex.py --load saved_models/W_$g.npy --noise $n --logdir test_results/w${g}/noise${n} --samples $samples
#  done
#done

# noise injection
samples=10000
mkdir test_results
noise="0.2 0.4 0.6 0.8 1.0"
for n in $noise
do
  echo "test noise injection model under noise $n"
  mkdir test_results/NI${n}
  mkdir test_results/NI${n}/noise${n}
  python3 testVortex.py --load saved_models/NI_$n.npy --noise $n --logdir test_results/NI${n}/noise${n} --samples $samples
done
