import os
import pdb
import time
from argparse import ArgumentParser
from collections import OrderedDict
#from matplotlib import pyplot as plt

import sys
DATA_PATH = "../data-sets"
if not DATA_PATH in sys.path:
    sys.path.append(DATA_PATH)
from bb_datasets import get_dataset

import numpy as np
from functools import partial
import theano
import theano.tensor as T
import lasagne as lg

from models import build_model
from utils import save_params

CD_STEPS = 1 # Nb of sampling steps
RECONSTRUCT_STEPS = 10 # Nb of Gibbs steps for reconstruction
IM_SIZE = 28 # MNIST images size
D = IM_SIZE*IM_SIZE # Dimension
BATCH_SIZE = 50 # batch size
NUM_EPOCH = 10
LOG_FREQ = 4
LR = 0.0002
PARAMS_DIR = "./trained_models" # Path to parameters
RESULTS_DIR = "./results4" # Path to results

######################################## Models architectures ########################################
FC_net = {"hidden":3,"nhidden_0":D,"nhidden_1":1024,"nhidden_2":2048,"nhidden_3":2048,"noutput":1}
CONV_net = {"conv":3,"nhidden_0":IM_SIZE,
            "filter_size_0":5,"num_filters_0":32,#conv1
            "filter_size_1":3,"num_filters_1":64,#conv2
            "filter_size_2":3,"num_filters_2":64,#conv3
            "FC_units":256,#FC1
            "noutput":1}
arch = {"FC_net":FC_net, "CONV_net":CONV_net, "boltzman":FC_net}
######################################## Main ########################################
def main(batch_size=BATCH_SIZE, size_data=100, num_epochs=NUM_EPOCH, energy_type='boltzman', archi=None, sampling_method='gibbs', obj_fct="CD"):
    # Create directories
    if not os.path.exists(PARAMS_DIR):
        os.makedirs(PARAMS_DIR)
    checkpoint_file = os.path.join(PARAMS_DIR,"ckpt")
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    experiment = obj_fct + "_" +energy_type + "_" + sampling_method
    result_file = os.path.join(RESULTS_DIR,experiment)

    # Get Data
    dataset = get_dataset("mnist")
    dataset.load()
    for k in ("train", "valid", "test"):
        dataset.data[k] = ((0.5 < dataset.data[k][0][:size_data]).astype(theano.config.floatX),
                                                    dataset.data[k][1][:size_data])

    # Input tensor
    X = T.matrix()

    # Build Model
    print("compiling model " + energy_type + " with " + sampling_method + " sampling for " + obj_fct + " objective...")
    # Train function
    train_func, test_func, l_out, params = build_model(X, obj_fct=obj_fct,
                                                                alpha=LR,
                                                                sampling_method=sampling_method,
                                                                num_steps_MC=CD_STEPS,
                                                                num_steps_reconstruct=RECONSTRUCT_STEPS,
                                                                energy_type=energy_type,
                                                                archi=archi)

    # Training loop
    print("starting training...")
    train_accuracy  = np.zeros(num_epochs*dataset.data['train'][0].shape[0]//(LOG_FREQ*batch_size)+1)
    test_accuracy   = np.zeros(num_epochs*dataset.data['train'][0].shape[0]//(LOG_FREQ*batch_size)+1)
    train_loss      = np.zeros(num_epochs*dataset.data['train'][0].shape[0]//(LOG_FREQ*batch_size)+1)
    test_loss       = np.zeros(num_epochs*dataset.data['train'][0].shape[0]//(LOG_FREQ*batch_size)+1)
    train_z1        = np.zeros(num_epochs*dataset.data['train'][0].shape[0]//(LOG_FREQ*batch_size)+1)
    train_z2        = np.zeros(num_epochs*dataset.data['train'][0].shape[0]//(LOG_FREQ*batch_size)+1)
    time_ite        = np.zeros(num_epochs*dataset.data['train'][0].shape[0]//(LOG_FREQ*batch_size)+1)
    recons          = np.zeros((num_epochs*dataset.data['train'][0].shape[0]//(LOG_FREQ*batch_size)+1, D))
    i, s = 0, time.time() #counter for iteration, time
    best_acc, best_loss = 0.0, -100.0
    for epoch in range(num_epochs):
        for x, y in dataset.iter("train", batch_size):
            train_l, z1, z2, train_a = train_func(x)
            if train_l>best_loss:
                best_loss = train_l
            if i%LOG_FREQ==0:
                test_l, test_a, n = 0.0, 0.0, 0 #test_acc, counter for test
                for x_test, y_test in dataset.iter("test", batch_size):
                    l, acc, recon = test_func(x_test)
                    test_a += acc
                    test_l += l
                    n += 1
                    if n==3:
                        break
                test_a = test_a/float(n)
                test_l = test_l/float(n)
                if test_a>best_acc:
                    best_acc = test_a
                    """
                    Save params
                    if energy_type=='boltzman':
                        save_params([W.get_value() for W in params], checkpoint_file+"_"+str(epoch)+"_")
                    elif energy_type=='net':
                        save_params(lg.layers.get_all_param_values(l_out), checkpoint_file+"_"+str(epoch)+"_")
                    else:
                        raise ValueError("Incorrect Energy. Not net or boltzman.")
                    """
                # Store info
                train_loss[(i)//LOG_FREQ] = train_l
                test_loss[(i)//LOG_FREQ] = test_l
                train_accuracy[(i)//LOG_FREQ] = train_a
                test_accuracy[(i)//LOG_FREQ] = test_a
                recons[(i)//LOG_FREQ] = recon[0]
                train_z1[(i)//LOG_FREQ] = z1
                train_z2[(i)//LOG_FREQ] = z2
                ti = time.time() - s
                time_ite[(i)//LOG_FREQ] = ti
                np.savetxt(result_file + '_train_acc.txt', train_accuracy)
                np.savetxt(result_file + '_test_acc.txt', test_accuracy)
                np.savetxt(result_file + '_train_loss.txt', train_loss)
                np.savetxt(result_file + '_test_loss.txt', test_loss)
                np.savetxt(result_file + '_recon.txt', recons)
                np.savetxt(result_file + '_z1.txt', train_z1)
                np.savetxt(result_file + '_z2.txt', train_z2)
                np.savetxt(result_file + '_time.txt', time_ite)
                # log info
                print("")
                print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                print("[{:.3f}s]iteration {}".format(ti, i+1))
                print("train loss: {:.3e}, test loss: {:.3f}".format(float(train_l),float(test_l)))
                print("train acc: {:.3f}%, test acc: {:.3f}%".format(100.0*train_a,100.0*test_a))
                s = time.time()
            i += 1


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--batch_size", "-b", action='store', dest="BATCH_SIZE",type=int, default=100)
    parser.add_argument("--num_epochs", "-e", action='store', dest="NUM_EPOCH", type=int, default=100)
    parser.add_argument("--num_data", action='store', dest="num_data", type=int, default=-1)
    parser.add_argument("--energy", action='store', dest="energy", type=str, default='boltzman')
    parser.add_argument("--sampling", action='store', dest="sampling", type=str, default='gibbs')
    parser.add_argument("--objective","-o", action='store', dest="obj", type=str, default='CD')
    options = parser.parse_args()

    """
    main(batch_size=options.BATCH_SIZE,
            size_data=options.num_data,
            num_epochs=options.NUM_EPOCH,
            energy_type=options.energy,
            archi=arch[options.energy],
            sampling_method=options.sampling,
            obj_fct=options.obj)
    """
    objectives = ['CD','CSS']
    ene = ['boltzman','FC_net','CONV_net']
    #ene = ['boltzman','FC_net','CONV_net']
    samp = ['naive_taylor',]
    for ob in objectives:
        for energ in ene:
            for sampl in samp:
                    main(batch_size=options.BATCH_SIZE,
                            size_data=options.num_data,
                            num_epochs=options.NUM_EPOCH,
                            energy_type=energ,
                            archi=arch[energ],
                            sampling_method=sampl,
                            obj_fct=ob)
