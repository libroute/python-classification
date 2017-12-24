# A neural network classifier in python using scikit-neuralnetwork
# Number of variables read from params.txt input file

params = {}
with open("params.txt") as f:
    for line in f:
        name, var = line.partition("=")[::2]
        params[name.strip()] = int(var)

from numpy import genfromtxt
data = genfromtxt('train.csv', delimiter=',')

from sknn.mlp import Classifier, Layer
nn = Classifier(layers=[Layer("Tanh",units=5),Layer("Sigmoid",units=5),Layer("Softmax")],learning_rate = 0.01, n_iter=500, verbose=True)
nn.fit(data[:,params['n_truth']:], data[:,0:params['n_truth']])

data_test = genfromtxt('test.csv', delimiter=',')
outp=nn.predict(data_test)
print(data_test.shape)
print(outp.shape)
from numpy import concatenate

data_out = concatenate((outp,data_test), axis=1)

from numpy import savetxt
savetxt("test_output.csv", data_out, delimiter=",")
