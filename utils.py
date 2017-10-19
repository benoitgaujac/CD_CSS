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


######################################## Save/load params ########################################
def save_params(params, filename, date_time=True):
    if date_time:
        filename = filename + datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
    with open(filename + ".pmz", 'wb') as f:
        for param in params:
            cPickle.dump(param, f, protocol=cPickle.HIGHEST_PROTOCOL)

######################################## Energy ########################################
def build_energy(x,energy_type='boltzman',archi=None):
    """
    Build the ennergy function of our model. Return the output of the energy, the params and the enrgy function.
    -x:             Input
    -energy_type:   Energy function used
    -archi:         net architecture (None for boltzman)
    """
    D = x.shape[1]
    # Initialize params
    if energy_type=='boltzman':
        W = init_BM_params(archi)
        params = [W]
        l_out = botlmzan_energy(x,W)
        energy = partial(botlmzan_energy,W=W)
    elif energy_type=='FC_net' or energy_type=='CONV_net':
        l_out = build_net(archi, energy_type)
        params = lg.layers.get_all_params(l_out)
        energy = partial(net_energy,l_out=l_out,energy_type=energy_type,im_resize=archi["nhidden_0"])
    else:
        raise ValueError("Incorrect Energy. Not FC_net nor CONV_net.")
    return l_out, params, energy

def init_BM_params(archi):
    """
    Initialize BN parameters
    """
    W = np.random.randn(archi["nhidden_0"], archi["nhidden_0"]) * 1e-8
    W = 0.5 * (W + W.T)
    W = theano.shared(W.astype(dtype='float64'), name="W")
    return W

def botlmzan_energy(x, W):
    """
    The energy function for the Boltzman machine
    """
    return T.sum(T.dot(x, W) * x, axis=1)

def build_net(architecture, energy_type='FC_net'):
    """
    Takes in a list of layer widths and returns the last layer
    of a feed-forward net that has those dimensions with an extra
    linear layer of width 1 at the end.
    """
    if energy_type=='FC_net':
        l = lg.layers.InputLayer(shape=[None, architecture["nhidden_0"]])
        for i in range(architecture["hidden"]):
            l = lg.layers.DenseLayer(l, num_units=architecture["nhidden_"+str(i+1)],
                                        W=lg.init.GlorotUniform(),
                                        b=lg.init.Constant(0.),
                                        nonlinearity=lg.nonlinearities.elu)
        l_out = lg.layers.DenseLayer(l, num_units=architecture["noutput"],
                                        W=lg.init.GlorotUniform(),
                                        b=lg.init.Constant(0.),
                                        nonlinearity=lg.nonlinearities.linear)
    elif energy_type=='CONV_net':
        l = lg.layers.InputLayer(shape=[None, 1, architecture["nhidden_0"], architecture["nhidden_0"]],dtype='float64')
        ## Convolutional layers
        for i in range(architecture["conv"]):
            l = lg.layers.Conv2DLayer(l, num_filters=2^i*architecture["num_filters"],
                                        filter_size=architecture["filter_size"],
                                        W=lg.init.GlorotUniform(),
                                        b=lg.init.Constant(0.),
                                        nonlinearity=lg.nonlinearities.elu)
            l = lg.layers.MaxPool2DLayer(l, pool_size=(2, 2))
        ## Dense layer
        l = lg.layers.DenseLayer(l, num_units=architecture["FC_units"],
                                    W=lg.init.GlorotUniform(),
                                    b=lg.init.Constant(0.),
                                    nonlinearity=lg.nonlinearities.elu)
        """
        ## Dropout
        l = lg.layers.dropout(l, p=.5)
        """
        ## output
        l_out = lg.layers.DenseLayer(l, num_units=architecture["noutput"],
                                        W=lg.init.GlorotUniform(),
                                        b=lg.init.Constant(0.),
                                        nonlinearity=lg.nonlinearities.linear)


    return l_out

def net_energy(x, l_out, energy_type, im_resize=None):
    """
        The energy function for the NNET
        l_out - lasagne layer
        x - theano.tensor.matrix
    """
    if energy_type=='CONV_net':
        Xin = T.reshape(x, (-1,1,im_resize,im_resize),ndim=4)
    else:
        Xin = x
    #pdb.set_trace()
    return lg.layers.get_output(l_out, Xin)

"""
def init_params(architecture,energy_type="boltzman"):
    #Initialize models parameters
    if energy_type == "boltzman":
        W = np.random.randn(architecture["nhidden_0"], architecture["nhidden_0"]) * 1e-8
        W = 0.5 * (W + W.T)
        W = theano.shared(W.astype(theano.config.floatX), name="W")
        params =  {"W":W}
    elif energy_type == "FC_net":
        params = {}
        for i in range(architecture["hidden"]):
            W = lg.init.GlorotNormal()(shape=(architecture["nhidden_"+str(i)], architecture["nhidden_"+str(i+1)]))
            b = np.zeros(architecture["nhidden_"+str(i+1)])
            params["W"+str(i)] = theano.shared(W.astype(theano.config.floatX))
            params["b"+str(i)] = theano.shared(b.astype(theano.config.floatX))
        W = lg.init.GlorotNormal()(shape=(architecture["nhidden_"+str(i+1)], architecture["noutput"]))
        b = np.zeros(architecture["noutput"])
        params["W"+str(i+1)] = theano.shared(W.astype(theano.config.floatX))
        params["b"+str(i+1)] = theano.shared(b.astype(theano.config.floatX))
    elif energy_type == "CONV_net":
        params = {}
        W = lg.init.GlorotNormal()(shape=(architecture["num_filter"], 1, architecture["size_filter"],architecture["size_filter"]))
        b = lg.init.GlorotNormal()(shape=(architecture["num_filter"]))
        params["W0"] = theano.shared(W.astype(theano.config.floatX))
        params["b0"] = theano.shared(b.astype(theano.config.floatX))
        for i in range(architecture["conv"]-1):
            W = lg.init.GlorotNormal()(shape=(2^(i+1)*architecture["num_filter"], 2^i*architecture["num_filter"], architecture["size_filter"],architecture["size_filter"]))
            b = lg.init.GlorotNormal()(shape=(2^(i+1)*architecture["num_filter"]))
            params["W"+str(i+1)] = theano.shared(W.astype(theano.config.floatX))
            params["b"+str(i+1)] = theano.shared(b.astype(theano.config.floatX))
        W = lg.init.GlorotNormal()(shape=(architecture["nhidden_"+str(i+1)], architecture["noutput"]))
        b = np.zeros(architecture["noutput"])
        params["W"+str(i+1)] = theano.shared(W.astype(theano.config.floatX))
        params["b"+str(i+1)] = theano.shared(b.astype(theano.config.floatX))
    else:
        raise ValueError("Incorrect net type. Not FC_net nor CONV_net.")

    return params
"""

######################################## Sampling ########################################
def sampler(x, energy, E_data, num_steps, params, sampling_method, srng):
    """
    Sampler for MC approximation of the energy. Return samples.
    -x:                 Input data
    -energy:            Energy function
    -E_data:            Energy of the training data
    -num_steps:         Number of steps in the MCMC
    -params:            Params of the model
    -sampling_method:   Sampling method used for sampling (gibbs, taylor)
    """
    if sampling_method=="gibbs":
        samples, updates = gibbs_sample(x, energy, num_steps, params, srng)
    elif sampling_method=="naive_taylor":
        samples, updates = taylor_sample(x, E_data, srng)
    else:
        raise ValueError("Incorrect sampling method. Not gibbs nor naive_taylor.")

    return samples, updates

def gibbs_sample(X, energy, num_steps, params, srng):
    """
    Gibbs sampling
    """
    def gibbs_step(i, x, *args):
        "perform one step of gibbs sampling from the energy model for all N chains"
        x_i = x[T.arange(x.shape[0]), i]
        x_zero = T.set_subtensor(x_i, 0.0)
        x_one = T.set_subtensor(x_i, 1.0)
        merged = T.concatenate([x_one, x_zero], axis=0)
        eng = energy(merged).flatten()
        eng_one = eng[:x.shape[0]]
        eng_zero = eng[x.shape[0]:]
        q = T.nnet.sigmoid(eng_one - eng_zero)
        samps = binary_sample(q.shape, q, srng=srng)
        return T.set_subtensor(x_i, samps)

    for i in range(num_steps):
        shuffle = srng.uniform(size=X.shape)
        shuffled = T.argsort(shuffle, axis=1)
        result, updates = theano.scan(fn=gibbs_step,
                                sequences=shuffled.T,
                                outputs_info=X,
                                non_sequences=params)
    return result[-1], updates

def taylor_sample(X, E_data, srng):
    """
    Sample from taylor expansion of the energy
    """
    q = build_taylor_q(X, E_data, srng)
    return binary_sample(q.shape, q, srng=srng), {}

def build_taylor_q(X, E_data, srng):
    """
    Build the taylor expansion of the energy
    """
    pvals = T.nnet.softmax(E_data.reshape((1, -1)))
    pvals = T.repeat(pvals, X.shape[0], axis=0)
    pi = T.argmax(srng.multinomial(pvals=pvals,
                                   dtype='float64'), axis=1)
    #pdb.set_trace()
    q = T.nnet.sigmoid(T.grad(T.sum(E_data), X)[pi])
    return q

def binary_sample(size, p=0.5, dtype='float64', srng=None):
    """
    Samples binary data
    """
    x = srng.uniform(size=size)
    x = zero_grad(x)
    if not isinstance(p, float):
        p = zero_grad(p)
    return T.cast(x < p, dtype)
