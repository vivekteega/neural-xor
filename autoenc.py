import numpy as np 
np.random.seed(0) # for debugging

def nonlin(x,deriv=False):
	if(deriv==True):
		return x*(1-x)
	return 1/(1+np.exp(-x))

# input data 

inp = 4

X = np.array([[0,0,1,1],
			  [0,1,1,1],
			  [1,1,1,1],
			  [0,0,0,1],
			  [1,0,0,1],
			  [1,1,0,1],
			  [0,0,1,0],
			  [0,1,1,0],
			  [1,1,1,0],
			  [0,0,0,0],
			  [1,0,0,0],
			  [1,1,0,0]])


# initializing weights in the synapses 

syn0 = 2*np.random.random((inp,3)) - 1
syn1 = 2*np.random.random((3,inp)) - 1


for cur in range(100000):

	#forward propogation
	l0 = X
	l1 = nonlin(np.dot(l0,syn0))
	l2 = nonlin(np.dot(l1,syn1))

	#calculating error at the output layer & hidden layer
	l2_error = X - l2

	l2_delta = l2_error * nonlin(l2,deriv=True)
	l1_delta = l2_delta.dot(syn1.T) * nonlin(l1,deriv=True)


	# updating weights 

	syn0 += np.dot(l0.T,l1_delta)
	syn1 += np.dot(l1.T,l2_delta)

	if cur%10000==0:
	    print("---------------")
	    print("Output : Layer 1 ")
	    print(str(np.round(l1)))
	    print("Output : Layer 2 ")
	    print(str(np.round(l2)))

print("\n\nTraining is over")

while True:
	a = float(input("\n\nEnter a: "))
	b = float(input("Enter b: "))
	c = float(input("Enter c: "))
	d = float(input("Enter d: "))

		   
	temp = np.array([[a,b,c,d]])
	layer1 = nonlin(np.dot(temp,syn0))
	layer2 = nonlin(np.dot(layer1,syn1))

	print("---------------")
	print("Output : Layer 1 ")
	print(str(np.round(layer1)))
	print("Output : Layer 2 ")
	print(str(np.round(layer2)))


