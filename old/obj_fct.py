import os
import pdb
import numpy as np
import theano
import theano.tensor as T
from theano.gradient import zero_grad

eps=1e-6

def objectives(true_x,q_sample,log_q,energy,E_data,obj_fct,approx_grad=True):
    if obj_fct=='CD':
        l, z1, z2 = cd_objective(true_x, q_sample, energy, E_data)
    elif obj_fct=='CSS':
        l, z1, z2 = css_objective(true_x, q_sample, log_q, energy, E_data, approx_grad)
    else:
        raise ValueError("Incorrect objective function. Not CD nor CSS.")

    return l, z1, z2

def cd_objective(true_x, q_sample, energy, E_data):
    """
    An objective whose gradient is equal to the CD gradient.
    """
    z1 = T.mean(E_data)
    z2 = T.mean(energy(q_sample))
    return z1 - z2, z1, z2


def css_objective(true_x, q_sample, log_q, energy, E_data, approx_grad=True):
    """
    CSS objective.
    -true_x:        The data points samples
    -q_sample:      Samples from the q distribution
    -log_q:         log[q(q_sample)]
    -approx_grad:   Whether to take gradients with respect to log_q (True means we don't take)
    """
    if approx_grad:
        log_q = zero_grad(log_q)

    # Expand the energy for the Q samples
    e_q = energy(q_sample) - log_q - T.log(T.cast(log_q.shape[0], theano.config.floatX))
    e_x = E_data

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