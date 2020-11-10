# Train DNN with Parameter Variation
### Robust Least Square on MNIST
- Under directory rbls
- run ```preprocessData.py``` first to generate npy files
- try ```python3 rbls.py --noise-train x --noise-test y``` to see rbls performance on MNIST
- ```x``` and ```y``` defines standard deviation of multiplicative Gaussian noise used in training and testing 
