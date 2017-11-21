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

CD_STEPS = 1 # Nb of sampling steps
RECONSTRUCT_STEPS = 10 # Nb of Gibbs steps for reconstruction
IM_SIZE = 28 # MNIST images size
D = IM_SIZE*IM_SIZE # Dimension
BATCH_SIZE = 50 # batch size
NUM_EPOCH = 10
LOG_FREQ = 32
NUM_RECON = 10
IND_RECON = 2000
LR = 0.00005
RESULTS_DIR = "./results26" # Path to results
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)
PARAMS_SUBDIR = os.path.join(RESULTS_DIR,"weights") # Path to parameters
LOG_SUBDIR = os.path.join(RESULTS_DIR,"log") # Path to log

fractions = [0.1,0.3,0.5,0.7]

######################################## Models architectures ########################################
FC_net = {"hidden":3,"nhidden_0":D,"nhidden_1":1024,"nhidden_2":2048,"nhidden_3":2048,"noutput":1}
CONV_net = {"conv":3,"nhidden_0":IM_SIZE,
            "filter_size_0":5,"num_filters_0":32,#conv1
            "filter_size_1":3,"num_filters_1":64,#conv2
            "filter_size_2":3,"num_filters_2":64,#conv3
            "FC_units":256,#FC1
            "noutput":1}
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

######################################## Main ########################################
def main(dataset, batch_size=BATCH_SIZE, num_epochs=NUM_EPOCH, energy_type='boltzman', archi=None, sampling_method='gibbs', obj_fct="CD", mode="train"):
    # Create directories
    experiment = obj_fct + "_" +energy_type + "_" + sampling_method
    if not os.path.exists(PARAMS_SUBDIR):
        os.makedirs(PARAMS_SUBDIR)
    checkpoint_file = os.path.join(PARAMS_SUBDIR,experiment)
    if not os.path.exists(LOG_SUBDIR):
        os.makedirs(LOG_SUBDIR)
    result_file = os.path.join(LOG_SUBDIR,experiment)

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
        print("\ncompiling model " + energy_type + " with " + sampling_method + " sampling for " + obj_fct + " objective...")
        loss_function, eval_function, l_out, params = build_model(X, obj_fct=obj_fct,
                                                                            alpha=LR,
                                                                            sampling_method=sampling_method,
                                                                            p_flip = p_flip,
                                                                            num_steps_MC=CD_STEPS,
                                                                            num_steps_reconstruct=RECONSTRUCT_STEPS,
                                                                            energy_type=energy_type,
                                                                            archi=archi)
        # Training loop
        print("starting training...")
        shape = (num_epochs*dataset.data['train'][0].shape[0]//(LOG_FREQ*batch_size)+1,len(fractions),NUM_RECON,D)
        train_accuracy  = np.zeros(shape[:2])
        test_accuracy   = np.zeros(shape[:2])
        train_loss      = np.zeros(shape[0])
        test_loss       = np.zeros(shape[0])
        train_energy    = np.zeros((shape[0],2))
        test_energy     = np.zeros((shape[0],2))
        time_ite        = np.zeros(shape[0])
        i, s = 0, time.time() #counter for iteration, time
        best_acc, best_loss = 0.0, -100.0
        for epoch in range(num_epochs):
            for x, y in dataset.iter("train", batch_size):
                train_l, Z1, Z2 = loss_function(x,prob_init*exp(i*log(decay_rate)))
                if train_l>best_loss:
                    best_loss = train_l
                if i%LOG_FREQ==0:
                    # Eval train
                    train_a1,train_a3,train_a5,train_a7,_,_,_,_ = eval_function(x)
                    train_a = np.array([train_a1,train_a3,train_a5,train_a7])
                    # Test
                    test_l, n = 0.0, 0
                    test_a = np.zeros((len(fractions)))
                    for x_test, y_test in dataset.iter("test", batch_size):
                        l, z1, z2 = loss_function(x_test,prob_init*exp(i*log(decay_rate)))
                        test_l += l
                        acc1,acc3,acc5,acc7,_,_,_,_ = eval_function(x_test)
                        test_a += np.array([acc1,acc3,acc5,acc7])
                        n += 1
                        if n==2:
                            break
                    test_a = test_a/float(n)
                    test_l = test_l/float(n)
                    if test_a[-1]>best_acc:
                        best_acc = test_a[-1]
                        _,_,_,_,recon1,recon3,recon5,recon7 = eval_function(true_x)
                        #save_params(np.split(recons,np.shape(recons)[0]), result_file + '_best_recons', date_time=False)
                        save_params([recon1,recon3,recon5,recon7], result_file + '_best_recons', date_time=False)
                    # Store info
                    train_accuracy[(i)//LOG_FREQ] = train_a
                    test_accuracy[(i)//LOG_FREQ] = test_a
                    train_loss[(i)//LOG_FREQ] = train_l
                    test_loss[(i)//LOG_FREQ] = test_l
                    train_energy[(i)//LOG_FREQ] = np.asarray([Z1,Z2])
                    test_energy[(i)//LOG_FREQ] = np.asarray([z1,z2])
                    ti = time.time() - s
                    time_ite[(i)//LOG_FREQ] = ti
                    # Save info
                    save_np(train_accuracy,'train_acc',result_file)
                    save_np(test_accuracy,'test_acc',result_file)
                    save_np(train_loss,'train_loss',result_file)
                    save_np(test_loss,'test_loss',result_file)
                    save_np(train_energy,'train_energy',result_file)
                    save_np(test_energy,'test_energy',result_file)
                    save_np(time_ite,'time',result_file)
                    # log info
                    print("")
                    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                    print("[{:.3f}s]iteration {}".format(ti, i+1))
                    print("train loss: {:.3e}, test loss: {:.3f}".format(float(train_l),float(test_l)))
                    print("train acc: {:.3f}%, test acc: {:.3f}%".format(100.0*train_a[-1],100.0*test_a[-1]))
                    s = time.time()
                i += 1
        # Reconstructing images after training ends
        _,_,_,_,recon1,recon3,recon5,recon7 = eval_function(true_x)
        save_params([recon1,recon3,recon5,recon7], result_file + '_final_recons', date_time=False)
        save_params([true_x], result_file + '_truex', date_time=False)
        # Save final params
        print("\nSaving weights..")
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
    parser.add_argument("--objective","-o", action='store', dest="obj", type=str, default='CD')
    parser.add_argument("--mode","-m", action='store', dest="mode", type=str, default='train')
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
                obj_fct=options.obj,
                mode=options.mode)
    """

    objectives = ['CD','CSS',]
    ene = ['CONV_net','boltzman','FC_net']
    #samp = ['naive_taylor','stupid_q']
    samp = ['naive_taylor',]
    for sampl in samp:
        for energ in ene:
            for ob in objectives:
                main(dataset,batch_size=options.BATCH_SIZE,
                                num_epochs=options.NUM_EPOCH,
                                energy_type=energ,
                                archi=arch[energ],
                                sampling_method=sampl,
                                obj_fct=ob,
                                mode=options.mode)