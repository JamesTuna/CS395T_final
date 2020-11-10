# Train DNN with Parameter Variation
### Robust Least Square on MNIST
- Under directory rbls
- run ```preprocessData.py``` first to generares npy files
- then try ```python3 rbls.py --noise-train x --noise-test y``` to see rbls performance on MNIST
- ```--noise-train``` and ```--noise-test``` defines standard deviation of multiplicative Gaussian noise used in training and testing 
