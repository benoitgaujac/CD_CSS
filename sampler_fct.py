import os
import pdb
import numpy as np
import theano
import theano.tensor as T
from theano.gradient import zero_grad

from utils import logsumexp

eps=1e-6

def sampler(x, energy, E_data, num_steps, params, p_flip, sampling_method, num_samples, srng):
    """
    Sampler for MC approximation of the energy. Return samples.
    -x:                 Input data
    -energy:            Energy function
    -E_data:            Energy of the training data
    -num_steps:         Number of steps in the MCMC
    -params:            Params of the model
    -sampling_method:   Sampling method used for sampling (gibbs, taylor, uniform)
    -num_samples:       Number of samples for importance sampling/MC
    """
    if sampling_method=="gibbs":
        samples, logq, updates = gibbs_sample(x,energy,num_steps,params,srng)
    elif sampling_method=="taylor_uniform":
        samples, logq, updates = taylor_sample(x,E_data,num_samples,True,srng)
    elif sampling_method=="taylor_softmax":
        samples, logq, updates = taylor_sample(x,E_data,num_samples,False,srng)
    elif sampling_method=="uniform":
        samples, logq, updates = uniform(x,num_samples,srng)
    elif sampling_method=="stupid_q":
        samples, logq, updates = stupidq(x,num_samples,p_flip,srng)
    elif sampling_method=="mixtures":
        samples, logq, updates = mix_q(x,num_samples,srng)
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
    q = T.repeat(means, num_samples, axis=0)[pi] #shape: (num_samples,D)
    q_sample = binary_sample(q.shape, q, srng=srng) #shape: (num_samples,D)

    # Calculate log[q(xs)]
    #log[q(xs|n)]
    means = T.repeat(means.dimshuffle(["x", 0, 1]),num_samples,axis=0) #shape: (num_samples, batch, D)
    q_sample_ext = T.repeat(q_sample.dimshuffle([0, "x", 1]),X.shape[0],axis=1)  #shape: (num_samples, batch, D)
    q_sample_ext = q_sample_ext * (1.0 - 2*eps) + eps
    log_qx = -T.sum(T.nnet.binary_crossentropy(means,q_sample_ext),axis=-1,keepdims=False)  #shape: (num_samples, batch)
    #log[q(n)]
    log_qn = T.log(pvals) #shape: (1,batch)
    log_q = logsumexp(log_qx + log_qn)  #shape: (num_samples,1)

    return q_sample, log_q, dict()

def build_taylor_q(X, E_data, uniform):
    """
    Build the taylor expansion of the energy for bernoulli mixtures of batch mixtures.
    -X:         batch x D
    -E_data:    batch x 1
    -uniform:    Weather or not to use uniform mixture weights
    """
    # Responsability of each of the batch mixtures q(n).
    if uniform:
        # uniform mixture weights
        pvals = T.ones_like(E_data.reshape((1, -1)))/T.cast(X.shape[0],theano.config.floatX) #shape: (1,batch)
    else:
        # mixture components distributed as softmax(E(xn))
        pvals = T.nnet.softmax(E_data.reshape((1, -1))) #shape: (1,batch)

    # Mean of bernoulli phi_n. We have batch mixtures, so batch means of dimension D
    means = T.nnet.sigmoid(T.grad(T.sum(E_data), X)) #shape: (batch,D)
    return means, pvals

def mix_q(x,num_samples,srng):
    samples_u, logq_u, _ = uniform(x, int(num_samples/4), srng)
    samples_bu, logq_bu, _ = blobs_uniform(x, int(num_samples/4), srng)
    samples_s1, logq_s1, _ = stupidq(x, int(num_samples/4), 0.1, srng)
    samples_s2, logq_s2, _ = stupidq(x, int(num_samples/4), 0.4, srng)

    q_sample = T.concatenate((samples_u,samples_bu,samples_s1,samples_s2))
    log_q = T.concatenate((logq_u,logq_bu,logq_s1,logq_s2))

    return q_sample, log_q, dict()

def stupidq(X,num_samples,p_flip,srng):
    N = X.shape[0]
    # Samling training point from batch
    pvals = T.ones((1,N), dtype=theano.config.floatX)/T.cast(N,theano.config.floatX) #shape: (1,batch)
    pi = T.argmax(srng.multinomial(pvals=T.repeat(pvals, num_samples, axis=0),
                                        dtype=theano.config.floatX), axis=1) #shape: (num_samples,)
    x = T.repeat(X, num_samples, axis=0)[pi] #shape: (num_samples,D)
    xflipped = 1.0 - x
    switching = binary_sample(size=x.shape, p=1.0-p_flip, srng=srng)
    q_sample = T.switch(switching, x, xflipped)

    #log[q(n)]
    means = T.exp(-T.nnet.binary_crossentropy(p_flip*T.ones_like(X),1-X)) #shape: (batch,D)
    means = T.repeat(means.dimshuffle(["x", 0, 1]),num_samples,axis=0) #shape: (num_samples, batch, D)
    q_sample_ext = T.repeat(q_sample.dimshuffle([0, "x", 1]),N,axis=1)  #shape: (num_samples, batch, D)
    log_qx = -T.sum(T.nnet.binary_crossentropy(means,q_sample_ext),axis=-1,keepdims=False)  #shape: (num_samples, batch)
    #log[q(n)]
    log_qn = T.log(pvals) #shape: (1,batch)
    log_q = logsumexp(log_qx + log_qn)  #shape: (num_samples,1)

    return q_sample, log_q, dict()

def uniform(X, num_samples, srng):
    """
    Sample from uniform q.
    -X: batch x D
    """
    # Sampling from uniform
    q = T.cast(0.5,theano.config.floatX)*T.ones((num_samples,X.shape[-1]),dtype=theano.config.floatX) #shape: (num_samples,D)
    q_sample = binary_sample(q.shape, q, srng=srng) #shape: (num_samples,D)

    # Calculate log[q(xs)]
    log_q = -T.log(2)*T.cast(X.shape[-1],theano.config.floatX)*T.ones((num_samples,1),dtype=theano.config.floatX) #shape: (num_samples,1)

    return q_sample, log_q, dict()

def blobs_uniform(X, num_samples, srng):
    """
    Sample from blobs uniform q.
    -X: batch x D
    """
    #!!!! value hard coded !!!!
    im_w = 28
    im_h = 28
    blobs_size = 7
    blobs_area = blobs_size*blobs_size
    blobs_number = T.cast(im_h/blobs_size,dtype='int32')

    # Sampling from uniform
    q_blobs = T.cast(0.5,theano.config.floatX)*T.ones((num_samples,blobs_number*blobs_number),dtype=theano.config.floatX) #shape: (num_samples,blobs_number)
    q_sample_blobs = binary_sample(q_blobs.shape, q_blobs, srng=srng).reshape((-1,blobs_number,blobs_number)) #shape: (num_samples,blobs_number)
    q_sample = T.repeat(T.repeat(q_sample_blobs,blobs_size,axis=1),blobs_size,axis=2).reshape((-1,X.shape[-1])) #shape: (num_samples,D)

    # Calculate log[q(xs)] (wrong q but we do not use for new CSS)
    log_q = -T.log(2)*T.cast(blobs_size,theano.config.floatX)*T.ones((num_samples,1),dtype=theano.config.floatX) #shape: (num_samples,1)

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

def gibbs_sample(X, energy, num_steps, params, srng):
    """
    Gibbs sampling
    """
    def gibbs_step(i, x, *args):
        """perform one step of gibbs sampling """
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
    logq = T.zeros((result[-1].shape[0],1)) #we dont need that as we do CD

    return result[-1], logq, updates
