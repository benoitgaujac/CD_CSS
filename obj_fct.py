import os
import pdb
import numpy as np
import theano
import theano.tensor as T
from theano.gradient import zero_grad

eps=1e-6

def objectives(E_data,E_samples,log_q,obj_fct,approx_grad=True):
    if obj_fct=='CD':
        l, z1, z2 = cd_objective(E_data, E_samples)
    elif obj_fct=='CSS':
        l, z1, z2 = css_objective(E_data, E_samples, log_q, approx_grad)
    else:
        raise ValueError("Incorrect objective function. Not CD nor CSS.")

    return l, z1, z2

def cd_objective(E_data, E_samples):
    """
    An objective whose gradient is equal to the CD gradient.
    """
    z1 = T.mean(E_data)
    z2 = T.mean(E_samples)
    return z1 - z2, z1, z2


def css_objective(E_data, E_samples, log_q, approx_grad=True):
    """
    CSS objective.
    -log_q:         log[q(q_sample)] (NxS)x1
    -E_data:        Energy of the true data Nx1
    -E_samples:     Energy of the samples (NxS)x1
    -approx_grad:   Whether to take gradients with respect to log_q (True means we don't take)
    """
    if approx_grad:
        log_q = zero_grad(log_q)

    # Expand the energy for the Q samples
    e_q = E_samples - log_q - T.log(T.cast(log_q.shape[0], theano.config.floatX)) #shape: (nsamples*batch,1)
    e_x = E_data #shape: (batch,1)

    # Concatenate energies
    e_p = T.concatenate((e_x, e_q), axis=0)

    # Calculate the objective
    m = zero_grad(T.max(e_p, axis=0))
    e_p = e_p - m
    z_1 = T.mean(e_x)
    z_2 = T.mean(e_q)
    """
    z_1 = T.log(T.sum(T.exp(e_p[:e_x.shape[0]]), axis=0)) + m
    z_2 = T.log(T.sum(T.exp(e_p[e_x.shape[0]:]), axis=0)) + m
    """
    logsumexp = T.log(T.sum(T.exp(e_p), axis=0)) + m
    return z_1 - logsumexp[0], z_1, z_2
