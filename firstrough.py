import numpy as np
from matplotlib import pyplot as plt
def sigmoid(x):
    return 1/(1 + np.exp(-x))

def neural_net(x1,x2,w1,w2,b):
    z = w1*x1 + w2*x2 + b
    return sigmoid(z)

#np.random.randn() function to generate random numbers
w1 = np.random.randn()
w2 = np.random.randn()
b = np.random.randn()
learning_rate = 0.01
training_set = [[0,0,0],
                [0,1,0],
                [1,0,0],
                [1,1,1]]
for i in range(len(training_set)):
    point = training_set[i]
    plt.scatter(point[0],point[1])
#training loop
for i in range(100000):
    ind = np.random.randint(len(training_set))
    predict = neural_net(training_set[ind][0],training_set[ind][1],w1,w2,b)
    #squared error
    cost = (training_set[ind][2] - predict)**2
    #derivatives of terms
    dcost_dpredict = -2*(training_set[ind][2] - predict)
    #dpredict_dz = np.exp(-(training_set[ind][0]*w1 + training_set[ind][1]*w2 + b))*predict**2
    temp = training_set[ind][0]*w1 + training_set[ind][1]*w2 + b;
    dpredict_dz = sigmoid(temp)*(1 - sigmoid(temp))
    dz_dw1 = training_set[ind][0]
    dz_dw2 = training_set[ind][1]
    dz_db = 1
    dcost_dw1 = dcost_dpredict * dpredict_dz * dz_dw1
    dcost_dw2 = dcost_dpredict * dpredict_dz * dz_dw2
    dcost_db = dcost_dpredict * dpredict_dz * dz_db
    w1 = w1 - learning_rate*dcost_dw1
    w2 = w2 - learning_rate*dcost_dw2
    b = b - learning_rate*dcost_db


def retreive(x1,x2):
    temp = neural_net(x1,x2,w1,w2,b)
    if temp >= .5:
        print 1
    else:
        print 0
retreive(0,0)
retreive(0,1)
retreive(1,0)
retreive(1,1)