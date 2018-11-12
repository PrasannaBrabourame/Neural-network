import numpy as np
import time

#Network Definition
n_hidden = 10
n_in = 10
n_out = 10
n_samples = 300

#hyperParameters
learning_rate = 0.01
momentum = 0.9

np.random.seed(0)

#Activation Function for Output Layer(Binary Classification)
def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

#Activation function for Hidden Units
def tanh_prime(x):
    return  1 - np.tanh(x)**2

#Network Train
def train(x, t, V, W, bv, bw):

    #Feed Forward
    A = np.dot(x, V) + bv
    Z = np.tanh(A)

    B = np.dot(Z, W) + bw
    Y = sigmoid(B)

    #BackProb
    Ew = Y - t
    Ev = tanh_prime(A) * np.dot(W, Ew)

    dW = np.outer(Z, Ew)
    dV = np.outer(x, Ev)

    loss = -np.mean ( t * np.log(Y) + (1 - t) * np.log(1 - Y) )

    return  loss, (dV, dW, Ev, Ew)

def predict(x, V, W, bv, bw):
    A = np.dot(x, V) + bv
    B = np.dot(np.tanh(A), W) + bw
    return (sigmoid(B) > 0.5).astype(int)

#Parms for Network

V = np.random.normal(scale=0.1, size=(n_in, n_hidden))
W = np.random.normal(scale=0.1, size=(n_hidden, n_out))

bv = np.zeros(n_hidden)
bw = np.zeros(n_out)

params = [V,W,bv,bw]

#Generate some data

X = np.random.binomial(1, 0.5, (n_samples, n_in))
T = X ^ 1

#Train out Network

for epoch in range(100):
    err = []
    upd = [0]*len(params)

    t0 = time.clock()
    for i in range(X.shape[0]):
        loss, grad = train(X[i], T[i], *params)

        for j in range(len(params)):
            params[j] -= upd[j]

        for j in range(len(params)):
            upd[j] = learning_rate * grad[j] + momentum * upd[j]

        err.append( loss )

    print("Epoch: %d, Loss: %.8f, Time: %.4fs" % (
                epoch, np.mean( err ), time.clock()-t0 ))

#Predict

x = np.random.binomial(1, 0.5, n_in)
print("XOR prediction:")
print(x)
print(predict(x, *params))