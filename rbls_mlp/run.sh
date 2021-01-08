noise="0.6 0.8"
for n in $noise
do
  python3 mlpRegressor.py --noise $n
done
