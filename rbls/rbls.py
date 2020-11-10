#! /usr/bin/env python3
import numpy as np
import time

# robust least square for multiplicative noise
# solve Ax = y
# minimize E [||(A+dA)(x+dx)-y||^2 ]
# dA = (1+e1)A, Var(e1) = varA
# dx = (1+e2)A, Var(e2) = varx
# returns tuple of (solution, mean square error)

def rbSolve(A,y,varA,varx):
	assert A.shape[0] == y.shape[0], 'shape mismatch: A %s and y %s'%(A.shape[0],y.shape[0])
	y = np.concatenate([y,np.zeros((A.shape[1],1))],axis=0)
	CA = np.sqrt((A**2).sum(axis=0))
	CA = np.diag(CA)
	CA *= np.sqrt(varA+varx+varA*varx)
	A = np.concatenate([A,CA],axis=0)
	sol = np.linalg.pinv(A).dot(y)
	err = ((A.dot(sol) - y)**2).sum()/A.shape[0]
	return sol,err

if __name__ == '__main__':
	import argparse
	# parser
	parser = argparse.ArgumentParser()
	parser.add_argument('--noise-train', type=float, default=0, help='variation std assumed in solving RBLS')
	parser.add_argument('--noise-test', type=float, default=2, help='variation std during test')
	args = parser.parse_args()


	stdx = args.noise_train# variance of parameter variation
	A_train = np.load('data/A.npy') # [60000,784]
	A_test = np.load('data/A_test.npy') # [10000,784]
	y_train_onehot = np.load('data/y_onehot.npy') # [60000,10]
	y_test_onehot = np.load('data/y_test_onehot.npy') # [10000,10]
	y_test = np.load('data/y_test.npy') #[10000,1]

	#y_train_onehot[y_train_onehot==0]=-1
	#y_test_onehot[y_test_onehot==0]=-1

	# robust ls regression 10 times (output is 10 dimensional vector on MNIST)
	W = np.zeros((784,10)) # parameters [784,10]
	print("robust training with variation std=%.4f"%(stdx))
	start = time.time()
	for i in range(10):
		sol,error = rbSolve(A_train,y_train_onehot[:,i].reshape(-1,1),0,stdx**2)
		print('Robust LS %s: err %.4f'%(i,error))
		W[:,i] = sol.reshape(784,)
	end = time.time()
	#W = np.array(W)
	print("training finished: %s seconds"%(end-start))
	#W = W.reshape(784,10)
	np.save('W_%s.npy'%(stdx),W)

	# test
	test_noise_std = args.noise_test
	noise_sample = 1000
	square_err = []
	accuracy = []
	print("test under variation std %s"%test_noise_std)
	for test in range(noise_sample):
		W_ = W * np.random.normal(1,test_noise_std,W.shape)
		y_hat = A_test.dot(W_)
		err = ((y_hat - y_test_onehot)**2).sum()/y_test_onehot.shape[0]
		pred = y_hat.argmax(axis=1)
		acc = np.count_nonzero(pred == y_test.reshape(-1))/y_test.shape[0]
		if test % 100 == 99:
			print("test %s/%s err %.4f acc %.4f"%(test+1,noise_sample,err,acc))
			#print('pred',pred[:10],'true',y_test[:10])
		square_err.append(err)
		accuracy.append(acc)
	square_err = np.array(square_err)
	accuracy = np.array(accuracy)
	print("avg square err %.4f"%(square_err.mean()))
	print("avg acc %.4f"%(accuracy.mean()))
