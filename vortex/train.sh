# vortex
#ro=1
#gamma="0 0.1 0.2 0.4 0.8 1.6 3.2 6.4 12.8"
#gamma="0.6 1.0 1.2 1.4 1.8 2.0 2.2 2.4 2.6"
#gamma="2.8 3.0 3.4 3.6 3.8 4.0 4.2 4.4 4.6"
#for g in $gamma
#do
#  python3 trainVortex.py --cuda --ro $ro --gamma $g --saveas "saved_models/W_${g}.npy" &
#done

# noise injection
noise="0.2 0.4 0.6 0.8 1.0"
for n in $noise
do
  python3 trainNI.py --noise $n --saveas "saved_models/NI_${n}.npy"
done
