import pdb
import lasagne.updates as upd
import lasagne.layers
from lasagne.regularization import regularize_layer_params, regularize_layer_params_weighted, l2
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import numpy as np

from functools import partial
import utils as u
from energy_fct import build_energy
from sampler_fct import sampler
from obj_fct import objectives
from eval_fct import reconstruct_images

srng = RandomStreams(100)
np.random.seed(42)
coef_regu = 0.0
regularization=False

def build_model(X, obj_fct, alpha, sampling_method, p_flip,
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
    l_out, params, energy = build_energy(X,energy_type,archi)
    E_data = energy(X)

    # Sampling from Q
    samples, log_q, updts = sampler(X, energy, E_data, num_steps_MC, params, p_flip, sampling_method, srng)
    E_samples =energy(samples)

    # Build loss function, regularization & updates dictionary
    loss, z1, z2 = objectives(samples,log_q,E_data,E_samples,obj_fct,approx_grad=True)
    if regularization and energy_type!='boltzman':
        all_layers = lasagne.layers.get_all_layers(l_out)
        layers={}
        for i, layer in enumerate(all_layers):
            layers[layer]=coef_regu
        regu = regularize_layer_params_weighted(layers,l2)
        #regu = coef_regu*regularize_layer_params(l_out,l2)
    else:
        regu = T.zeros_like(loss)
    updates = upd.adam(-loss+regu, params, learning_rate=alpha)
    updates.update(updts) #we need to ad the update dictionary

    # Logilike evaluation with 10N samples
    samples_10, logq_10, _ = sampler(X, energy, E_data, 50*num_steps_MC, params, p_flip, sampling_method, srng)
    E_samples_10 = energy(samples_10)
    loss_10, z1_10, z2_10 = objectives(samples_10,logq_10,E_data,E_samples_10,obj_fct,approx_grad=True)

    # Evaluation (you lazy)
    recon_01, acc_01 = reconstruct_images(X, num_steps=num_steps_reconstruct,
                                                        params=params,
                                                        energy=energy,
                                                        srng=srng,
                                                        fraction=0.1,
                                                        D=784)

    recon_03, acc_03 = reconstruct_images(X, num_steps=num_steps_reconstruct,
                                                        params=params,
                                                        energy=energy,
                                                        srng=srng,
                                                        fraction=0.3,
                                                        D=784)
    recon_05, acc_05 = reconstruct_images(X, num_steps=num_steps_reconstruct,
                                                        params=params,
                                                        energy=energy,
                                                        srng=srng,
                                                        fraction=0.5,
                                                        D=784)
    recon_07, acc_07 = reconstruct_images(X, num_steps=num_steps_reconstruct,
                                                        params=params,
                                                        energy=energy,
                                                        srng=srng,
                                                        fraction=0.7,
                                                        D=784)

    # Build theano learning function
    trainloss_function = theano.function(inputs=[X,p_flip], outputs=(loss,z1,z2), updates=updates,on_unused_input='ignore')
    testloss_function = theano.function(inputs=[X,p_flip], outputs=(loss,z1,z2,loss_10,z1_10,z2_10),on_unused_input='ignore')
    #eval_function = theano.function(inputs=[X], outputs=(acc_01,acc_03,acc_05,acc_07,recon_01,recon_03,recon_05,recon_07))
    eval_function = theano.function(inputs=[X], outputs=(acc_01,acc_05,acc_07,recon_01,recon_05,recon_07))

    #loglike_eval = theano.function(inputs=[X], outputs=(loss_10,z1_10,z2_10), on_unused_input='ignore')

    # Debug Function
    debugf = theano.function(inputs=[X,p_flip], outputs=(samples_10, logq_10, E_samples),on_unused_input='ignore')

    return debugf, trainloss_function, testloss_function, eval_function, l_out, params
