import os
import pdb
import numpy as np
import theano
import theano.tensor as T
import lasagne as lg
from functools import partial

from utils import init_BM_params, build_net

eps=1e-6

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
        train_energy = partial(botlmzan_energy,W=W)
        test_energy  = train_energy
    elif energy_type=='FC_net' or energy_type=='CONV_net':
        l_out = build_net(archi, energy_type)
        params = lg.layers.get_all_params(l_out)
        train_energy = partial(net_energy,l_out=l_out,energy_type=energy_type,im_resize=archi["nhidden_0"],deterministic=False)
        test_energy = partial(net_energy,l_out=l_out,energy_type=energy_type,im_resize=archi["nhidden_0"],deterministic=True)
    else:
        raise ValueError("Incorrect Energy. Not FC_net nor CONV_net.")
    return l_out, params, train_energy, test_energy

def botlmzan_energy(x, W):
    """
    The energy function for the Boltzman machine
    """
    return T.sum(T.dot(x, W) * x, axis=1,keepdims=True)

def net_energy(x, l_out, energy_type, im_resize=None, deterministic=False):
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
    return lg.layers.get_output(l_out, Xin, deterministic=deterministic)
