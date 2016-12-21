import numpy as np 
np.random.seed(0) # for debugging

def nonlin(x,deriv=False):
	if(deriv==True):
		return x*(1-x)
	return 1/(1+np.exp(-x))

# input data 

X = np.array([[0,0],
			  [0,1],
			  [1,0],
			  [1,1]])

# output data

y = np.array([[0,1,1,0]]).T

# initializing weights in the synapses 

syn0 = 2*np.random.random((2,3)) - 1
syn1 = 2*np.random.random((3,1)) - 1

for cur in range(100000):

	#forward propogation
	l0 = X
	l1 = nonlin(np.dot(l0,syn0))
	l2 = nonlin(np.dot(l1,syn1))

	#calculating error at the output layer & hidden layer
	l2_error = y - l2
	l2_delta = l2_error * nonlin(l2,deriv=True)
	l1_delta = l2_delta.dot(syn1.T) * nonlin(l1,deriv=True)

	# updating weights 

	syn0 += np.dot(l0.T,l1_delta)
	syn1 += np.dot(l1.T,l2_delta)

	if cur%10000==0:
	    print("---------------")
	    print("Output : ")
	    print(str(np.round(l2)))
