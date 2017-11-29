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
from obj_fct import objectives, variance_estimator
from eval_fct import reconstruct_images

srng = RandomStreams(100)
np.random.seed(42)
coef_regu = 0.0
regularization=False

def build_model(X, obj_fct, alpha, datasize, sampling_method, p_flip,
                                                        num_samples,
                                                        num_steps_MC=1,
                                                        num_steps_reconstruct=10,
                                                        energy_type="boltzman",
                                                        archi=None):
    """
    Build model and return train and test function, as well as output
    -X:                 Input
    -obj_fct:           Name of the training objective
    -sampling_method:   Sampling method used
    -num_samples:       Number of samples for importance sampling/MC
    -alpha:             Learning rate
    -num_steps_MC:      Number of steps for MCMC with gibbs sampling
    -reconstruct_steps: Number of steps for reconstructing the images
    -energy_type:       Energy function type (either boltzman or nnet)
    -archi:             Architecture of the energy network
    """

    # Build energy
    l_out, params, energy = build_energy(X,energy_type,archi)
    E_data = energy(X)

    # Sampling from Q
    samples, logq, updts = sampler(X, energy, E_data, num_steps_MC, params, p_flip, sampling_method, num_samples, srng)
    #samples, logq, updts = sampler(X, energy, E_data, num_steps_MC, params, p_flip, sampling_method, 5, srng)
    E_samples =energy(samples)

    # Build loss function, variance estimator, regularization & updates dictionary
    loss, logZ, _, _ = objectives(E_data,E_samples,logq,obj_fct,datasize,approx_grad=True)
    sig = variance_estimator(E_data,E_samples,logq,logZ,datasize)
    if regularization and energy_type!='boltzman':
        all_layers = lasagne.layers.get_all_layers(l_out)
        layers={}
        for i, layer in enumerate(all_layers):
            layers[layer]=coef_regu
        loss = loss - regularize_layer_params_weighted(layers,l2)
        #loss = loss coef_regu*regularize_layer_params(l_out,l2)
    updates = upd.adam(-loss, params, learning_rate=alpha)
    updates.update(updts) #we need to ad the update dictionary

    # Logilike & variance evaluation with 100,500,1000N samples
    samples100, logq100, _ = sampler(X, energy, E_data, num_steps_MC, params, p_flip, sampling_method, 100*num_samples, srng)
    E_samples100 = energy(samples100)
    loss100, logZ100, _, _ = objectives(E_data,E_samples100,logq100,obj_fct,datasize,approx_grad=True)
    sig100 = variance_estimator(E_data,E_samples100,logq100,logZ100,datasize)
    samples500, logq500, _ = sampler(X, energy, E_data, num_steps_MC, params, p_flip, sampling_method, 500*num_samples, srng)
    E_samples500 = energy(samples500)
    loss500, logZ500, _, _ = objectives(E_data,E_samples500,logq500,obj_fct,datasize,approx_grad=True)
    sig500 = variance_estimator(E_data,E_samples500,logq500,logZ500,datasize)
    samples1000, logq1000, _ = sampler(X, energy, E_data, num_steps_MC, params, p_flip, sampling_method, 1000*num_samples, srng)
    E_samples1000 = energy(samples1000)
    loss1000, logZ1000, _, _ = objectives(E_data,E_samples1000,logq1000,obj_fct,datasize,approx_grad=True)
    sig1000 = variance_estimator(E_data,E_samples1000,logq1000,logZ1000,datasize)

    # Evaluation (you lazy)
    recon_01, acc_01 = reconstruct_images(X, num_steps=num_steps_reconstruct,params=params,energy=energy,srng=srng,fraction=0.1,D=784)
    recon_05, acc_05 = reconstruct_images(X, num_steps=num_steps_reconstruct,params=params,energy=energy,srng=srng,fraction=0.5,D=784)
    recon_07, acc_07 = reconstruct_images(X, num_steps=num_steps_reconstruct,params=params,energy=energy,srng=srng,fraction=0.7,D=784)

    # Build theano learning function
    trainloss_function = theano.function(inputs=[X,p_flip], outputs=(E_data,E_samples,logq,loss,logZ,sig), updates=updates,on_unused_input='ignore')
    testloss_function = theano.function(inputs=[X,p_flip],
                                        outputs=(E_data,E_samples,logq,loss,logZ,sig,
                                                loss100,sig100,
                                                loss500,sig500,
                                                E_samples1000,logq1000,loss1000,logZ1000,sig1000),
                                        on_unused_input='ignore')
    #eval_function = theano.function(inputs=[X], outputs=(acc_01,acc_03,acc_05,acc_07,recon_01,recon_03,recon_05,recon_07))
    eval_function = theano.function(inputs=[X], outputs=(acc_01,acc_05,acc_07,recon_01,recon_05,recon_07))

    """
    # Debug Function
    debugf = theano.function(inputs=[X,p_flip], outputs=(samples_10, logq_10, E_samples),on_unused_input='ignore')
    """

    return trainloss_function, testloss_function, eval_function, params
