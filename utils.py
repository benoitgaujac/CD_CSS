import os
import pdb
import numpy as np
import theano
import theano.tensor as T
from theano.gradient import zero_grad
import lasagne as lg
from functools import partial
import datetime
from six.moves import cPickle

import pdb

eps=1e-6

######################################## Energy ########################################
def init_BM_params(archi):
    """
    Initialize BN parameters
    """
    W = np.random.randn(archi["nhidden_0"], archi["nhidden_0"]) * 1e-8
    W = 0.5 * (W + W.T)
    W = theano.shared(W.astype(dtype=theano.config.floatX), name="W")
    return W

def build_net(architecture, energy_type='FC_net'):
    """
    Takes in a list of layer widths and returns the last layer
    of a feed-forward net that has those dimensions with an extra
    linear layer of width 1 at the end.
    """
    if energy_type=='FC_net':
        l = lg.layers.InputLayer(shape=[None, architecture["nhidden_0"]],dtype=theano.config.floatX)
        for i in range(architecture["hidden"]):
            l = lg.layers.DenseLayer(l, num_units=architecture["nhidden_"+str(i+1)],
                                        W=lg.init.GlorotUniform(),
                                        b=lg.init.Constant(0.),
                                        nonlinearity=lg.nonlinearities.elu)
        # Output layer
        l_out = lg.layers.DenseLayer(l, num_units=architecture["noutput"],
                                        W=lg.init.GlorotUniform(),
                                        b=lg.init.Constant(0.),
                                        nonlinearity=lg.nonlinearities.linear)
    elif energy_type=='CONV_net':
        l = lg.layers.InputLayer(shape=[None, 1, architecture["nhidden_0"], architecture["nhidden_0"]],dtype=theano.config.floatX)
        ## Convolutional layers
        for i in range(architecture["conv"]):
            l = lg.layers.Conv2DLayer(l, num_filters=architecture["num_filters_" + str(i)],
                                        filter_size=architecture["filter_size_" + str(i)],
                                        W=lg.init.GlorotUniform(),
                                        b=lg.init.Constant(0.),
                                        nonlinearity=lg.nonlinearities.elu)
            if i==0:
                l = lg.layers.MaxPool2DLayer(l, pool_size=(2, 2))
        ## Pooling
        l = lg.layers.MaxPool2DLayer(l, pool_size=(2, 2))
        ## Dense layer
        l = lg.layers.DenseLayer(l, num_units=architecture["FC_units"],
                                    W=lg.init.GlorotUniform(),
                                    b=lg.init.Constant(0.),
                                    nonlinearity=lg.nonlinearities.tanh)
        ## output
        l_out = lg.layers.DenseLayer(l, num_units=architecture["noutput"],
                                        W=lg.init.GlorotUniform(),
                                        b=lg.init.Constant(0.),
                                        nonlinearity=lg.nonlinearities.linear)
    return l_out

######################################## Sampling ########################################
def build_taylor_q(X, E_data, srng):
    """
    Build the taylor expansion of the energy.
    """
    pvals = T.nnet.softmax(E_data.reshape((1, -1)))
    pvals = T.repeat(pvals, X.shape[0], axis=0)
    pi = T.argmax(srng.multinomial(pvals=pvals,
                                   dtype=theano.config.floatX), axis=1)
    q = T.nnet.sigmoid(T.grad(T.sum(E_data), X)[pi])
    return q
