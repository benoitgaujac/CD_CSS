import os
import pdb
import numpy as np
import theano
import theano.tensor as T
from theano.gradient import zero_grad
from utils import logsumexp

eps=1e-6

def objectives(E_data,E_samples,log_q,obj_fct,datasize,approx_grad=True):
    if obj_fct=='CD':
        l, logz, sig = cd_objective(E_data, log_q, E_samples)
    elif obj_fct=='IMP':
        l, logz, sig = imp_objective(E_data, E_samples, log_q, approx_grad)
    elif obj_fct=='CSS':
        l, logz, sig = css_objective(E_data, E_samples, log_q, datasize, approx_grad)
    else:
        raise ValueError("Incorrect objective function.")

    return l, logz, sig

def cd_objective(E_data, logq, E_samples):
    """
    An objective whose gradient is equal to the CD gradient.
    """
    z1 = T.mean(E_data)
    z2 = T.mean(E_samples)
    return z1 - z2, 0.0, 0.0

def imp_objective(E_data, logq, E_samples, approx_grad=True):
    """
    Pseudo CD with importance sampling for Z.
    -log_q:         log[q(q_sample)] Sx1
    -E_data:        Energy of the true data Nx1
    -E_samples:     Energy of the samples Sx1
    -approx_grad:   Whether to take gradients with respect to log_q (True means we don't take)
    """
    if approx_grad:
        logq = zero_grad(logq)

    N = T.cast(E_samples.shape[0],theano.config.floatX)

    # Expand the energy for the Q samples
    e_q = E_samples - logq - T.log(N) #shape: (nsamples,1)

    # Calculate the objective
    z_1 = T.mean(E_data)
    logZ = T.squeeze(logsumexp(e_q.T))

    # Compute variance variance estimator
    m = zero_grad(T.max(e_q, axis=0))
    e_q = e_q - m
    sqr_diff = T.sqr(N*T.exp(e_q)-T.exp(logZ-m))
    logsig = T.log(T.sum(sqr_diff)) + 2*m - 0.5*T.log(N-T.cast(1.0,theano.config.floatX))

    return z_1 - logZ, logZ, logsig[0]


def css_objective(E_data, E_samples, logq, datasize, approx_grad=True):
    """
    CSS objective.
    -log_q:         log[q(q_sample)] Sx1
    -E_data:        Energy of the true data Nx1
    -E_samples:     Energy of the samples Sx1
    -datasize:      Training size for importance sampling
    -approx_grad:   Whether to take gradients with respect to log_q (True means we don't take)
    """
    if approx_grad:
        logq = zero_grad(logq)

    N = T.cast(E_data.shape[0] + E_samples.shape[0],theano.config.floatX)

    # Expand the energy for the Q samples
    e_q = E_samples - logq - T.log(T.cast(E_samples.shape[0],theano.config.floatX)) #shape: (nsamples,1)
    e_x = E_data + T.log(datasize/T.cast(E_data.shape[0],theano.config.floatX))

    # Concatenate energies
    e_p = T.concatenate((e_x, e_q), axis=0)

    # Calculate the objective
    z_1 = T.mean(e_x)
    logZ = T.squeeze(logsumexp(e_p.T))

    # Compute variance variance estimator
    m = zero_grad(T.max(e_p, axis=0))
    e_p = e_p - m
    sqr_diff = T.sqr(N*T.exp(e_p)-T.exp(logZ-m))
    logsig = T.log(T.sum(sqr_diff)) + 2*m - 0.5*T.log(N-T.cast(1.0,theano.config.floatX))

    return z_1 - logZ, logZ, logsig[0]

def variance_estimator(E_data,E_samples,logq,logZ,datasize):
    """
    Empirical variance estimator.
    -E_data:        Energy of the true data Nbatchx1
    -E_samples:     E(X_samples) Energy of the samples Nsamplesx1
    -logq:         log[q(xs)] Nsamplesx1
    -logZ:          log[Z_est]
    """
    N = T.cast(E_data.shape[0] + E_samples.shape[0],theano.config.floatX)
    en = E_data + T.log(datasize/T.cast(E_data.shape[0],theano.config.floatX))
    es = E_samples - logq - T.log(T.cast(E_samples.shape[0],theano.config.floatX))
    e = T.concatenate((en,es),axis=0)
    sqr_diff = T.sqr(T.exp(e)+N-T.exp(logZ))

    return T.sum(sqr_diff)/(N-T.cast(1.0,theano.config.floatX))
