import os
import pdb
import time
from argparse import ArgumentParser
from collections import OrderedDict
from six.moves import cPickle

import sys
DATA_PATH = "../data-sets"
if not DATA_PATH in sys.path:
    sys.path.append(DATA_PATH)
from bb_datasets import get_dataset

import numpy as np
from math import log,exp
import pandas as pd
from functools import partial
import theano
import theano.tensor as T
import lasagne as lg

from models import build_model
from energy_fct import net_energy, botlmzan_energy
from utils import build_net

objectives = ['CSS','CD',]
#objectives = ['CSS',]
ene = ['FC_net',]
#ene = ['CONV_net','boltzman']
samp = ['taylor_uniform','taylor_softmax','uniform']
#samp = ['taylor_uniform','taylor_softmax',]
#fractions = [0.1,0.3,0.5,0.7]
fractions = [0.1,0.5,0.7]

#NUM_SAMPLES = [1,5,10] # Nb of sampling steps
NUM_SAMPLES = [1,] # Nb of sampling steps
RECONSTRUCT_STEPS = 10 # Nb of Gibbs steps for reconstruction
IM_SIZE = 28 # MNIST images size
D = IM_SIZE*IM_SIZE # Dimension
BATCH_SIZE = 32 # batch size
NUM_EPOCH = 10
LOG_FREQ = 32
NUM_RECON = 10
IND_RECON = 2000
LR = 0.00005

######################################## Models architectures ########################################
FC_net = {"hidden":3,"nhidden_0":D,"nhidden_1":1024,"nhidden_2":2048,"nhidden_3":2048,"noutput":1}
CONV_net = {"conv":3,"nhidden_0":IM_SIZE,
            "filter_size_0":5,"num_filters_0":32,#conv1
            "filter_size_1":3,"num_filters_1":64,#conv2
            "filter_size_2":3,"num_filters_2":64,#conv3
            "FC_units":256,#FC1
            "noutput":1}
"""
CONV_net = {"conv":2,"nhidden_0":IM_SIZE,
            "filter_size_0":3,"num_filters_0":16,#conv1
            "filter_size_1":3,"num_filters_1":16,#conv2
            "FC_units":128,#FC1
            "noutput":1}
"""
arch = {"FC_net":FC_net, "CONV_net":CONV_net, "boltzman":FC_net}
######################################## helper ########################################
def save_np(array,name,path,column=False):
    file_name = path + '_' + name + '.csv'
    if column:
        shape = np.shape(array)
        if shape[0]!=9:
            raise ValueError("Wrong energies array shape. Should have dim0 = 9(1+4+4)")
        columns = ['true_x', 'fin. recon w. 0.1', 'fin. recon w. 0.3', 'fin. recon w. 0.5', 'fin. recon w. 0.7', 'bes. recon w. 0.1', 'bes. recon w. 0.3', 'bes. recon w. 0.5', 'bes. recon w. 0.7']
        df = pd.DataFrame(np.transpose(array),columns=columns)
    else:
        df = pd.DataFrame(array)
    df.to_csv(file_name)

def save_params(params, filename, date_time=True):
    if date_time:
        filename = filename + datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
    with open(filename + ".pmz", 'wb') as f:
        cPickle.dump(params, f, protocol=cPickle.HIGHEST_PROTOCOL)

def load_data(FILE_PATH):
    with open(FILE_PATH,'rb') as f:
        x = np.asarray(cPickle.load(f))
    return x

def create_directory(directory,samples,experiment):
    if not os.path.exists(directory):
        os.makedirs(directory)
    samples_dir,_ = create_subdirectory(directory,str(samples) + 'samples')
    _,checkpoint_file = create_subdirectory(samples_dir,"weights",experiment)
    _,result_file = create_subdirectory(samples_dir,"log",experiment)

    return checkpoint_file, result_file

def create_subdirectory(DIR,SUBDIR,experiment=None):
    sub = os.path.join(DIR,SUBDIR) # Path to sub
    if not os.path.exists(sub):
        os.makedirs(sub)
    if experiment!=None:
        result_file = os.path.join(sub,experiment)
    else:
        result_file = '_'
    return sub,result_file


######################################## Main ########################################
def main(dataset, batch_size=BATCH_SIZE, num_epochs=NUM_EPOCH, energy_type='boltzman', archi=None, sampling_method='gibbs', num_samples=2, obj_fct="CD", mode="train",directory="results"):
    # Create directories
    experiment = obj_fct + "_" +energy_type + "_" + sampling_method
    checkpoint_file, result_file = create_directory(directory,num_samples,experiment)

    if mode=="train":
        # Image to visualize reconstruction
        true_x = dataset.data["test"][0][IND_RECON:IND_RECON+NUM_RECON]
        # Flipping prob for stupidq
        p_flip = T.scalar(dtype=theano.config.floatX)
        nm_steps_tot = NUM_EPOCH*dataset.data['train'][0].shape[0]//batch_size
        prob_init = 0.4
        decay_rate = exp((1/float(nm_steps_tot))*log(0.05/prob_init))
        # Input tensor
        X = T.matrix()
        # Build Model
        print("\ncompiling " + energy_type + " with " + sampling_method + " " + str(num_samples) + "samples for " + obj_fct + " objective...")
        trainloss_f, testloss_f, eval_f, l_out, params = build_model(X, obj_fct=obj_fct,
                                                                        alpha=LR,
                                                                        sampling_method=sampling_method,
                                                                        p_flip = p_flip,
                                                                        num_samples=BATCH_SIZE*num_samples,
                                                                        num_steps_MC=1,
                                                                        num_steps_reconstruct=RECONSTRUCT_STEPS,
                                                                        energy_type=energy_type,
                                                                        archi=archi)
        # Training loop
        print("\nstarting training...")
        shape = (num_epochs*dataset.data['train'][0].shape[0]//(LOG_FREQ*batch_size)+1,len(fractions))
        train_accuracy  = np.zeros(shape)
        train_energy    = np.zeros((shape[0],2))
        train_loss      = np.zeros(shape[0])
        train_samples   = np.zeros((shape[0],BATCH_SIZE*num_samples,2))
        #train_samples   = np.zeros((shape[0],5,2))
        test_accuracy   = np.zeros(shape)
        test_energy     = np.zeros((shape[0],2))
        test_loss       = np.zeros(shape[0])
        test_samples    = np.zeros((shape[0],BATCH_SIZE*num_samples,2))
        #test_samples    = np.zeros((shape[0],5,2))
        eval_loglike    = np.zeros(shape)
        eval_energy     = np.zeros((shape[0],1))
        eval_samples    = np.zeros((shape[0],1000*BATCH_SIZE*num_samples,2))
        #eval_samples    = np.zeros((shape[0],100*BATCH_SIZE*num_samples,2))
        time_ite        = np.zeros(shape[0])
        norm_params     = np.zeros((shape[0],len(params)))
        i, s = 0, time.time() #counter for iteration, time
        best_acc, best_loss = 0.0, -100.0
        for epoch in range(num_epochs):
            for x, y in dataset.iter("train", batch_size):
                train_l, Z1, LogZ, Esamples, Logq = trainloss_f(x,prob_init*exp(i*log(decay_rate)))
                if train_l>best_loss:
                    best_loss = train_l
                if i%LOG_FREQ==0:
                    # Compute params params norm
                    if energy_type=='boltzman':
                        norm = [np.sum(W.get_value()**2)/float(W.get_value().size) for W in params]
                    elif energy_type[-3:]=='net':
                        norm = [np.sum(W**2)/float(W.size) for W in lg.layers.get_all_param_values(l_out)]
                    # Eval train
                    """
                    train_a1,train_a3,train_a5,train_a7,_,_,_,_ = eval_f(x)
                    train_a = np.array([train_a1,train_a3,train_a5,train_a7])
                    """
                    train_a1,train_a5,train_a7,_,_,_ = eval_f(x)
                    train_a = np.array([train_a1,train_a5,train_a7])
                    # Test
                    test_l, n = 0.0, 0
                    loglikelihood = np.zeros((len(fractions)))
                    test_a = np.zeros((len(fractions)))
                    for x_test, y_test in dataset.iter("test", batch_size):
                        l, z1, logZ, esamples, logq, ll100, ll500, ll1000, logZ1000, esamples1000, logq1000 = testloss_f(x_test,prob_init*exp(i*log(decay_rate)))
                        test_l += l
                        loglikelihood += np.array([ll100, ll500, ll1000])
                        """
                        acc1,acc3,acc5,acc7,_,_,_,_ = eval_f(x_test)
                        test_a += np.array([acc1,acc3,acc5,acc7])
                        """
                        acc1,acc5,acc7,_,_,_ = eval_f(x_test)
                        test_a += np.array([acc1,acc5,acc7])
                        n += 1
                        if n==1:
                            break
                    test_l = test_l/float(n)
                    loglikelihood = loglikelihood/float(n)
                    test_a = test_a/float(n)
                    if test_a[-1]>best_acc:
                        best_acc = test_a[-1]
                        """
                        _,_,_,_,recon1,recon3,recon5,recon7 = eval_f(true_x)
                        save_params([recon1,recon3,recon5,recon7], result_file + '_best_recons', date_time=False)
                        """
                        _,_,_,recon1,recon5,recon7 = eval_f(true_x)
                        save_params([recon1,recon5,recon7], result_file + '_best_recons', date_time=False)
                    # Store info
                    train_accuracy[(i)//LOG_FREQ] = train_a
                    train_energy[(i)//LOG_FREQ] = np.asarray([Z1,LogZ])
                    train_loss[(i)//LOG_FREQ] = train_l
                    train_samples[(i)//LOG_FREQ] = np.concatenate((Esamples,Logq), axis=-1)
                    test_accuracy[(i)//LOG_FREQ] = test_a
                    test_energy[(i)//LOG_FREQ] = np.asarray([z1,logZ])
                    test_loss[(i)//LOG_FREQ] = test_l
                    test_samples[(i)//LOG_FREQ] = np.concatenate((esamples,logq), axis=-1)
                    eval_loglike[(i)//LOG_FREQ] = loglikelihood
                    eval_energy[(i)//LOG_FREQ] = np.asarray([logZ1000])
                    eval_samples[(i)//LOG_FREQ] = np.concatenate((esamples1000,logq1000), axis=-1)
                    ti = time.time() - s
                    time_ite[(i)//LOG_FREQ] = ti
                    norm_params[(i)//LOG_FREQ] = norm
                    # Save info
                    save_np(train_accuracy,'train_acc',result_file)
                    save_np(train_energy,'train_energy',result_file)
                    save_np(train_loss,'train_loss',result_file)
                    save_params([train_samples],result_file + '_train_samples', date_time=False)
                    save_np(test_accuracy,'test_acc',result_file)
                    save_np(test_energy,'test_energy',result_file)
                    save_np(test_loss,'test_loss',result_file)
                    save_params([test_samples],result_file + '_test_samples', date_time=False)
                    save_np(eval_loglike,'eval_loglike',result_file)
                    save_np(eval_energy,'eval_energy',result_file)
                    save_params([eval_samples],result_file + '_eval_samples', date_time=False)
                    save_np(time_ite,'time',result_file)
                    save_np(norm_params,'norm_params',result_file)
                    # log info
                    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                    print("[{:.3f}s]iteration {}".format(ti, i+1))
                    print("train loss: {:.3e}, test loss: {:.3f}".format(float(train_l),float(test_l)))
                    print("train acc: {:.3f}%, test acc: {:.3f}%\n".format(100.0*train_a[-1],100.0*test_a[-1]))
                    s = time.time()
                i += 1
        # Reconstructing images after training ends
        """
        _,_,_,_,recon1,recon3,recon5,recon7 = eval_f(true_x)
        save_params([recon1,recon3,recon5,recon7], result_file + '_final_recons', date_time=False)
        """
        _,_,_,recon1,recon5,recon7 = eval_f(true_x)
        save_params([recon1,recon5,recon7], result_file + '_final_recons', date_time=False)
        save_params([true_x], result_file + '_truex', date_time=False)
        # Save final params
        print("Saving weights..")
        if energy_type=='boltzman':
            save_params([W.get_value() for W in params], checkpoint_file,False)
        elif energy_type[-3:]=='net':
            save_params(lg.layers.get_all_param_values(l_out), checkpoint_file,False)
        else:
            raise ValueError("Incorrect Energy. Not net or boltzman.")

    elif mode=="energy":
        # Load data
        true_x_PATH = result_file + '_truex.pmz'
        true_x = load_data(true_x_PATH)
        true_x = np.reshape(true_x,(-1,D))
        recons_PATH = result_file + '_final_recons.pmz'
        final_recons = load_data(recons_PATH)
        final_recons = np.reshape(final_recons,(-1,D))
        recons_PATH = result_file + '_best_recons.pmz'
        best_recons = load_data(recons_PATH)
        shape = np.shape(best_recons)[:-1]
        best_recons = np.reshape(best_recons,(-1,D))
        # Load weights
        params = load_data(checkpoint_file+'.pmz')
        # Input tensor
        X = T.matrix()
        # Build Model
        print("\ncompiling energy " + energy_type + "...")
        if energy_type=='boltzman':
            energy = botlmzan_energy(X,params[0])
        elif energy_type=='FC_net' or energy_type=='CONV_net':
            l_out = build_net(archi, energy_type)
            lg.layers.set_all_param_values(l_out, params)
            energy = net_energy(X, l_out, energy_type=energy_type, im_resize=archi["nhidden_0"])
        else:
            raise ValueError("Incorrect Energy. Not FC_net nor CONV_net.")
        energy_function = theano.function(inputs=[X], outputs=energy)
        # Compute energy
        print("\nComputing energies...")
        Edata = np.expand_dims(np.squeeze(energy_function(true_x)),axis=0)
        Efrecons = np.reshape(energy_function(final_recons),shape)
        Ebrecons = np.reshape(energy_function(best_recons),shape)
        print("Data energies:")
        print(Edata[0,:])
        print("Reconstruction energies:")
        print(Efrecons[-1,:])
        # Save energies
        save_np(np.concatenate((Edata,Efrecons,Ebrecons),axis=0),'energy',result_file,column=True)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--batch_size", "-b", action='store', dest="BATCH_SIZE",type=int, default=100)
    parser.add_argument("--num_epochs", "-e", action='store', dest="NUM_EPOCH", type=int, default=100)
    parser.add_argument("--num_data", action='store', dest="num_data", type=int, default=-1)
    parser.add_argument("--energy", action='store', dest="energy", type=str, default='boltzman')
    parser.add_argument("--sampling", action='store', dest="sampling", type=str, default='gibbs')
    parser.add_argument("--nsamples", "-s", action='store', dest="nsamples", type=int, default=1)
    parser.add_argument("--objective","-o", action='store', dest="obj", type=str, default='CD')
    parser.add_argument("--mode","-m", action='store', dest="mode", type=str, default='train')
    parser.add_argument("--results_directory","-d", action='store', dest="dir", type=str)
    options = parser.parse_args()

    # Get Data
    dataset = get_dataset("mnist")
    dataset.load()

    for k in ("train", "valid", "test"):
        dataset.data[k] = ((0.5 < dataset.data[k][0][:-1]).astype(theano.config.floatX),dataset.data[k][1][:-1])
    dataset.data["train"] = (dataset.data[k][0][:options.num_data],dataset.data[k][1][:options.num_data])

    """

    main(dataset,batch_size=options.BATCH_SIZE,
                num_epochs=options.NUM_EPOCH,
                energy_type=options.energy,
                archi=arch[options.energy],
                sampling_method=options.sampling,
                num_samples=options.nsamples,
                obj_fct=options.obj,
                mode=options.mode,
                directory=options.dir)
    """

    for nsampl in NUM_SAMPLES[::-1]:
        for sampl in samp:
            for energ in ene:
                for ob in objectives:
                    main(dataset,batch_size=options.BATCH_SIZE,
                                    num_epochs=options.NUM_EPOCH,
                                    energy_type=energ,
                                    archi=arch[energ],
                                    sampling_method=sampl,
                                    num_samples=nsampl,
                                    obj_fct=ob,
                                    mode=options.mode,
                                    directory=options.dir)
