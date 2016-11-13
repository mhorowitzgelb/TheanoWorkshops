import theano
from matplotlib import pyplot as plt
import numpy as np

from theano import tensor as T

def plot__function(x, f,lb, ub):
    function = theano.function(inputs=[x], outputs=f)
    xlist =  np.arange(lb,ub,(ub-lb)/1000.0, np.float32)
    ylist =  function(xlist)
    plt.plot(xlist,ylist.transpose())
    plt.show()


def show_predictions(x,pred):
    for i in range(x.shape[0]):
        a = x[i].reshape((28,28))
        plt.imshow(a,cmap='Greys_r')
        plt.title(str(pred[i]))
        plt.figure(i+1)
    plt.show()