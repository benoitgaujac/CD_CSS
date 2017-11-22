import os
import pdb
import numpy as np
import theano
import theano.tensor as T
from theano.gradient import zero_grad

from utils import logsumexp

eps=1e-6

def sampler(x, energy, E_data, num_samples, params, p_flip, sampling_method, num_samples, srng, uniform_taylor=True):
    """
    Sampler for MC approximation of the energy. Return samples.
    -x:                 Input data
    -energy:            Energy function
    -E_data:            Energy of the training data
    -num_steps:         Number of steps in the MCMC
    -params:            Params of the model
    -sampling_method:   Sampling method used for sampling (gibbs, taylor)
    -num_samples:       Number of samples for importance sampling/MC
    -uniform_taylor:    Weather or not to use uniform mixture weights for taylor distribution
    """
    if sampling_method=="gibbs":
        # TODO
        pass
        #samples, logq, updates = gibbs_sample(x, energy, num_steps, num_samples, params, srng)
    elif sampling_method=="naive_taylor":
        samples, logq, updates = taylor_sample(x, E_data, num_samples, uniform_taylor, srng)
    elif sampling_method=="stupid_q":
        samples, logq, updates = stupidq(x,p_flip,srng)
    else:
        raise ValueError("Incorrect sampling method. Not gibbs nor naive_taylor.")

    return samples, logq, updates

def taylor_sample(X, E_data, num_samples, uniform_taylor, srng):
    """
    Sample from taylor expansion of the energy.
    -X:                 batch x D
    -E_data:            batch x 1
    -uniform_taylor:    Weather or not to use uniform mixture weights
    """
    # Build density
    means, pvals = build_taylor_q(X, E_data,uniform_taylor) #shape: (batch,D), (1,batch)

    # Sampling component of the mixture.
    pi = T.argmax(srng.multinomial(pvals=T.repeat(pvals, num_samples, axis=0),
                                   dtype=theano.config.floatX), axis=1) #shape: (num_samples,)
    q = T.repeat(T.nnet.sigmoid(means), num_samples, axis=0)[pi] #shape: (num_samples,D)
    q_sample = binary_sample(q.shape, q, srng=srng) #shape: (num_samples,D)

    # Calculate log[q(xs)]
    #log[q(xs|n)]
    means = T.repeat(means.dimshuffle(["x", 0, 1]),num_samples,axis=0) #shape: (num_samples, batch, D)
    q_sample_ext = T.repeat(q_sample.dimshuffle([0, "x", 1]),X.shape[0],axis=1)  #shape: (num_samples, batch, D)
    log_qx = -T.sum(T.nnet.nnet.binary_crossentropy(means,q_sample_ext),axis=-1,keepdims=False)  #shape: (num_samples, batch)
    """
    means = means.dimshuffle([0, "x", 1]) #shape: (batch, 1, D)
    q_sample_ext = q_sample.dimshuffle(["x", 0, 1])  #shape: (1, num_samples, D)
    Xentr = T.switch(T.eq(q_sample_ext, 0), 1 - means, means)  #shape: (batch, num_samples, D)
    log_qx = T.sum(Xentr,axis=-1).T  #shape: (num_samples,batch)
    """
    #log[q(n)]
    log_qn = T.log(pvals) #shape: (1,batch)
    log_q = logsumexp(log_qx + log_qn)  #shape: (num_samples,1)

    return q_sample, log_q, dict()

def build_taylor_q(X, E_data, uniform):
    """
    Build the taylor expansion of the energy for bernoulli mixtures of batch mixtures.
    -X:         batch x D
    -E_data:    batch x 1
    """
    # Responsability of each of the batch mixtures q(n).
    if uniform:
        # uniform mixture weights
        pvals = T.ones_like(E_data.reshape((1, -1)))/T.cast(X.shape[0],theano.config.floatX) #shape: (1,batch)
    else:
        # mixture components distributed as softmax(E(xn))
        pvals = T.nnet.softmax(E_data.reshape((1, -1))) #shape: (1,batch)

    # Mean of bernoulli phi_n. We have batch mixtures, so batch means of dimension D
    means = T.grad(T.sum(E_data), X) #shape: (batch,D)
    return means, pvals

def gibbs_sample(X, energy, num_steps, num_samples, params, srng):
    """
    Gibbs sampling.
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
        return T.set_subtensor(x_i, samps), q

    for i in range(num_samples):
        shuffle = srng.uniform(size=X.shape)
        shuffled = T.argsort(shuffle, axis=1)
        result, updates = theano.scan(fn=gibbs_step,
                                sequences=shuffled.T,
                                outputs_info=[X,None],
                                non_sequences=params)
        q_samples = result[0][-1]
        q = result[1].T
        q = q * (1.0 - 2*eps) + eps
        logq = - T.sum(T.nnet.binary_crossentropy(q, X[shuffled]), axis=1,keepdims=True)

    return q_samples, logq, updates

def stupidq(X,p_flip,srng):
    size = X.shape
    Xflipped = 1.0 - X
    binomial = binary_sample(size=size, p=p_flip, srng=srng)
    q_sample = T.switch(binomial, X, Xflipped)
    log_q = - T.sum(T.nnet.binary_crossentropy(p_flip*T.ones_like(X), q_sample), axis=1,keepdims=True)

    return q_sample, log_q, dict()

def binary_sample(size, p=0.5, dtype=theano.config.floatX, srng=None):
    """
    Samples binary data.
    """
    x = srng.uniform(size=size)
    x = zero_grad(x)
    if not isinstance(p, float):
        p = zero_grad(p)
    return T.cast(x < p, dtype)
