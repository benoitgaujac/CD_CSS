import os
import pdb
import numpy as np
import theano
import theano.tensor as T
from theano.gradient import zero_grad

from utils import build_taylor_q

eps=1e-6

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
        samples, logq, updates = gibbs_sample(x, energy, num_steps, params, srng)
    elif sampling_method=="naive_taylor":
        samples, logq, updates = taylor_sample(x, E_data, srng)
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

def taylor_sample(X, E_data, srng):
    """
    Sample from taylor expansion of the energy.
    """
    q = build_taylor_q(X, E_data, srng)
    q_sample = binary_sample(q.shape, q, srng=srng)
    # Calculate log[q(q_sample)]
    q = q * (1.0 - 2*eps) + eps
    log_q = - T.sum(T.nnet.binary_crossentropy(q, q_sample), axis=1,keepdims=True)
    # Return the objective
    return q_sample, log_q, {}

def binary_sample(size, p=0.5, dtype='float64', srng=None):
    """
    Samples binary data.
    """
    x = srng.uniform(size=size)
    x = zero_grad(x)
    if not isinstance(p, float):
        p = zero_grad(p)
    return T.cast(x < p, dtype)
