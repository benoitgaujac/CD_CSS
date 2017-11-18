import os
import pdb
import numpy as np
import theano
import theano.tensor as T
from theano.gradient import zero_grad

from utils import build_taylor_q, logsumexp

eps=1e-6

def sampler(x, energy, E_data, num_steps, params, p_flip, sampling_method, srng):
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
        samples, logq, updates = gibbs_sample(x, energy, num_steps, params, srng)
    elif sampling_method=="naive_taylor":
        samples, logq, updates = taylor_sample(x, E_data, num_steps, srng)
    elif sampling_method=="stupid_q":
        samples, logq, updates = stupidq(x,p_flip,srng)
    else:
        raise ValueError("Incorrect sampling method. Not gibbs nor naive_taylor.")

    return samples, logq, updates

def gibbs_sample(X, energy, num_steps, params, srng):
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

    for i in range(num_steps):
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

def taylor_sample(X, E_data, num_steps, srng):
    """
    Sample from taylor expansion of the energy.
    -X:         batch x D
    -E_data:    batch x 1
    """
    # Build density
    means, pvals = build_taylor_q(X, E_data, srng) #shape: #shape: (batch,D), (batch,batch)

    # Sampling component of the mixture. We need to expand dim for num of samples.
    pi = T.argmax(srng.multinomial(pvals=pvals,
                                   dtype=theano.config.floatX), axis=1) #shape: (batch,)
    q = T.nnet.sigmoid(means)[pi] #shape: (batch,D)
    q_ext = T.repeat(q, num_steps, axis=0) #shape: (num_steps*batch,D)
    q_sample = binary_sample(q_ext.shape, q_ext, srng=srng) #shape: (num_steps*batch,D)

    # Calculate log[q(q_sample)]
    means = T.repeat(T.nnet.sigmoid(means), num_steps, axis=0) #shape: (num_steps*batch,D)
    log_qs = - (T.nnet.binary_crossentropy(means, q_sample)) #shape: (num_steps*batch,D)
    log_qn = T.repeat(T.log(pvals[0]).T.dimshuffle([0,"x"]),num_steps, axis=0) #shape: (num_steps*batch,1)
    log_q = logsumexp(log_qs + log_qn)  #shape: (batch,1)

    return q_sample, log_q, dict()

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
