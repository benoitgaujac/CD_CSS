import theano
import theano.tensor as T
from theano.gradient import zero_grad
from .utils import rnd, binary_sample, gibbs_step, EPS


def pseudo_lik(true_x, params, optimal_func):
    """ Approximates the loglikihood as the sum over the
    conditional logliklihoods"""
    _, logq = gibbs_step(true_x, params, optimal_func, outputq=True, eps=EPS, sample=False)
    return T.sum(logq)


def cd_objective(true_x, q_sample, energy):
    """
    An objective whose gradient is equal to the CD gradient
    """
    e_x = energy(true_x)
    e_q = energy(q_sample)
    m_x = zero_grad(T.max(e_x, axis=0))
    m_q = zero_grad(T.max(e_q, axis=0))

    z1 = T.log(T.sum(T.exp(e_x - m_x), axis=0)) + m_x
    z2 = T.log(T.sum(T.exp(e_q - m_q), axis=0)) + m_q
    return T.mean(e_x) - T.mean(e_q), z1, z2


def css_objective(true_x, q_sample, log_q, energy, approx_grad=True, css=True):
    """
    CSS objective

    true_x - The data points samples
    q_sample - Samples from the q distribution
    log_q - log[q(q_sample)]
    approx_grad - Whether to take gradients with respect to log_q (True means we don't take)
    """
    if approx_grad:
        log_q = zero_grad(log_q)

    # Expand the energy for the Q samples
    e_q = energy(q_sample) - log_q - T.log(T.cast(log_q.shape[0], theano.config.floatX))
    e_x = energy(true_x) - T.log(T.cast(true_x.shape[0], theano.config.floatX)) + T.log(T.cast(50000, theano.config.floatX))

    # Concatenate energies
    e_p = T.concatenate((e_x, e_q), axis=0)

    # Calculate the objective
    m_x = zero_grad(T.max(e_x, axis=0))
    m_q = zero_grad(T.max(e_q, axis=0))
    m = zero_grad(T.max(e_p, axis=0))

    z1 = T.log(T.sum(T.exp(e_x - m_x), axis=0)) + m_x
    z2 = T.log(T.sum(T.exp(e_q - m_q), axis=0)) + m_q
    if css:
        logsumexp = T.log(T.sum(T.exp(e_p - m), axis=0)) + m
    else:
        logsumexp = T.log(T.sum(T.exp(e_q - m_q), axis=0)) + m_q
    return T.mean(e_x) - logsumexp, z1, z2


def css_mf_alex(true_x, q, energy, eps=EPS, **kwargs):
    """
    Mean Field CSS where we take a single sample per mixture.
    """
    # Sample q
    q_sample = binary_sample(q.shape, q)
    # Calculate log[q(q_sample)]
    q = q * (1.0 - 2 * eps) + eps
    log_q = - T.sum(T.nnet.binary_crossentropy(q, q_sample), axis=1)
    # Return the objective
    return css_objective(true_x, q_sample, log_q, energy, **kwargs)


def css_mf(true_x, q, energy, adaptive=False, **kwargs):
    """
    Mean Field CSS where we sample mixture components uniformly.
    """
    if adaptive:
        pvals = T.nnet.softmax(energy(true_x).reshape((1, -1)))
        pvals = T.repeat(pvals, q.shape[0], axis=0)
    else:
        pvals = T.ones((q.shape[0],)) / T.cast(q.shape[0], theano.config.floatX)
        pvals = T.repeat(pvals.reshape((1, -1)), q.shape[0], axis=0)
    pi = T.argmax(rnd.multinomial(pvals=pvals), axis=1)
    return css_mf_alex(true_x, q[pi], energy, **kwargs)


def css_taylor(true_x, energy, adaptive=False, **kwargs):
    if adaptive:
        pvals = T.nnet.softmax(energy(true_x).reshape(1, -1))
        pvals = T.repeat(pvals, true_x.shape[0], axis=0)
    else:
        pvals = T.ones((true_x.shape[0],)) / T.cast(true_x.shape[0], theano.config.floatX)
        pvals = T.repeat(pvals.reshape((1, -1)), true_x.shape[0], axis=0)
    pi = T.argmax(rnd.multinomial(pvals=pvals), axis=1)
    e = energy(true_x)
    g = T.grad(T.sum(e), true_x)
    return css_mf_alex(true_x, T.nnet.sigmoid(g[pi]), energy, **kwargs)
