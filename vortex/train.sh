ro=1
gamma="0 0.1 0.2 0.4 0.8 1.6 3.2 6.4"
for g in $gamma
do
  python3 trainVortex.py --ro $ro --gamma $g --saveas 'W_${gamma}.npy' &
done
