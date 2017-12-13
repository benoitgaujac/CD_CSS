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

def build_model(X, obj_fct, alpha, datasize, sampling_method, annealed_logq,
                                                                num_samples,
                                                                num_steps_MC=1,
                                                                num_steps_reconstruct=15,
                                                                energy_type="boltzman",
                                                                archi=None):
    """
    Build model and return train and test function, as well as output
    -X:                 Input
    -obj_fct:           Name of the training objective
    -sampling_method:   Sampling method used
    -annealed_logq:     Annealed logq for CSS with annealed logq
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
    #samples, logq, updts = sampler(X, energy, E_data, num_steps_MC, params, p_flip, sampling_method, 1, srng)
    E_samples =energy(samples)

    # Build loss function, variance estimator, regularization & updates dictionary
    if obj_fct=='CSSann':
        obj_fct = 'CSS'
        logq = annealed_logq
    loss, logZ, sig = objectives(E_data,E_samples,logq,obj_fct,datasize,approx_grad=True)
    updates = upd.adam(-loss, params, learning_rate=alpha)
    updates.update(updts) #we need to add the update dictionary (for gibbs sampling)

    """
    Regularization
    if regularization and energy_type!='boltzman':
        all_layers = lasagne.layers.get_all_layers(l_out)
        layers={}
        for i, layer in enumerate(all_layers):
            layers[layer]=coef_regu
        loss = loss - regularize_layer_params_weighted(layers,l2)
        #loss = loss coef_regu*regularize_layer_params(l_out,l2)

    # Alternative sampling
    altsamples, altlogq, _ = sampler(X, energy, E_data, num_steps_MC, params, p_flip, alt_sampling, num_samples, srng)
    altE_samples =energy(altsamples)
    altloss, altlogZ, _ = objectives(E_data,altE_samples,altlogq,obj_fct,datasize,approx_grad=True)

    # Logilike & variance evaluation with 100,500,1000N samples
    samples1000, logq1000, _ = sampler(X, energy, E_data, num_steps_MC, params, p_flip, sampling_method, 1000*num_samples, srng)
    E_samples1000 = energy(samples1000)
    loss1000, logZ1000, sig1000 = objectives(E_data,E_samples1000,logq1000,obj_fct,datasize,approx_grad=True)
    """

    # Evaluation
    recon_01, acc_01 = reconstruct_images(X, num_steps=num_steps_reconstruct,params=params,energy=energy,srng=srng,fraction=0.1,D=784)
    recon_05, acc_05 = reconstruct_images(X, num_steps=num_steps_reconstruct,params=params,energy=energy,srng=srng,fraction=0.5,D=784)
    recon_07, acc_07 = reconstruct_images(X, num_steps=num_steps_reconstruct,params=params,energy=energy,srng=srng,fraction=0.7,D=784)

    # Build theano learning function
    trainloss_function = theano.function(inputs=[X,annealed_logq], outputs=(E_data,E_samples,loss,logZ,sig), updates=updates,on_unused_input='ignore')
    testloss_function = theano.function(inputs=[X,annealed_logq],
                                        outputs=(E_data,E_samples,loss,logZ,sig),
                                        on_unused_input='ignore')
    eval_function = theano.function(inputs=[X], outputs=(acc_01,acc_05,acc_07,recon_01,recon_05,recon_07))


    return trainloss_function, testloss_function, eval_function, params
