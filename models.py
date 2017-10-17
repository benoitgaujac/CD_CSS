import pdb
import lasagne.updates as upd
import theano
from theano.tensor.shared_randomstreams import RandomStreams
import numpy as np

from functools import partial
import utils as u
from eval_reconstruct import reconstruct_images

srng = RandomStreams(100)
np.random.seed(42)

def build_model(X, obj_fct, sampling_method, alpha,
                            num_steps_MC=1,
                            num_steps_reconstruct=10,
                            energy_type="boltzman",
                            archi=None):
    """
    Build model and return train and test function, as well as output
    -X:                 Input
    -obj_fct:           Name of the training objective
    -sampling_method:   Sampling method used
    -alpha:             Learning rate
    -num_steps_MC:      Number of steps for sampling (either Gibbs or Taylor)
    -reconstruct_steps: Number of steps for reconstructing the images
    -energy_type:       Energy function type (either boltzman or nnet)
    -archi:             Architecture of the energy network
    """

    # Build energy
    l_out, params, energy = u.build_energy(X,energy_type,archi)
    E_data = energy(X)
    # Sampling from Q
    samples, updts = u.sampler(X, energy, E_data, num_steps_MC, params, sampling_method, srng)

    # Build loss function & updates dictionary
    E_q = energy(samples)
    loss = E_data.mean() - E_q.mean()### build method to deal with CSS objectives, obj_fct arg
    updates = upd.adam(-loss, params, learning_rate=alpha)
    updates.update(updts) #we need to ad the update dictionary

    # Evaluation
    recon, acc = reconstruct_images(X, num_steps=num_steps_reconstruct,
                                                        params=params,
                                                        energy=energy,
                                                        srng=srng,
                                                        fraction=0.7,
                                                        D=784)

    # Build theano function
    train = theano.function(inputs=[X], outputs=loss, updates=updates)
    test = theano.function(inputs=[X], outputs=(acc,recon))

    return train, test, l_out, params
