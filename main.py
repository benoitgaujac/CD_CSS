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

objectives = ['CSSann','CSSnewM','CSS','CSSnew','IMP']
#ene = ['FC_net','CONV_net','boltzman']
ene = ['CONV_net','boltzman']
#samp = ['taylor_uniform','taylor_softmax','uniform']
samp = ['uniform',]
fractions = [0.1,0.5,0.7]

NUM_SAMPLES = [1,] # Nb of sampling steps
RECONSTRUCT_STEPS = 10 # Nb of Gibbs steps for reconstruction
IM_SIZE = 28 # MNIST images size
D = IM_SIZE*IM_SIZE # Dimension
BATCH_SIZE = 32 # batch size
NUM_EPOCH = 10
LOG_FREQ = 128
NUM_RECON = 10
IND_RECON = 2000
LR = 0.0001
ANNEALED_THRESHOLD = 1.0

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
    # Sanity check gibbs/neww CSS
    if sampling_method=='gibbs' and obj_fct!="CD":
        raise ValueError("Gibbs only with CD")
    if obj_fct=='CSSnewM':
        sampling_method='mixtures'
        obj_fct='CSSnew'

    # Create directories
    experiment = obj_fct + "_" +energy_type + "_" + sampling_method
    checkpoint_file, result_file = create_directory(directory,num_samples,experiment)

    if mode=="train":
        # Image to visualize reconstruction
        true_x = dataset.data["test"][0][IND_RECON:IND_RECON+NUM_RECON]
        # Annealing logq
        logq = T.scalar(dtype=theano.config.floatX) #logq = -n*log(2) for uniform
        init_n = 1.0
        nm_steps_tot = NUM_EPOCH*dataset.data['train'][0].shape[0]//batch_size
        annealing_rate = nm_steps_tot//D # we bound the volume of the distribution
        # Input tensor
        X = T.matrix()
        # Build Model
        print("\ncompiling " + energy_type + " with " + sampling_method + " " + str(num_samples) + "samples for " + obj_fct + " objective...")
        trainloss_f, testloss_f, eval_f, params = build_model(X, obj_fct=obj_fct,
                                                                alpha=LR,
                                                                datasize = T.cast(dataset.data["train"][0].shape[0],theano.config.floatX),
                                                                alt_sampling='stupid_q',
                                                                annealed_logq = logq,
                                                                num_samples=batch_size*num_samples,
                                                                num_steps_MC=1,
                                                                num_steps_reconstruct=RECONSTRUCT_STEPS,
                                                                energy_type=energy_type,
                                                                archi=archi)
        # Training loop
        print("\nstarting training...")
        shape = (num_epochs*dataset.data['train'][0].shape[0]//(LOG_FREQ*batch_size)+1,len(fractions))
        train_accuracy  = np.zeros(shape) # accuracy
        train_loss      = np.zeros((shape[0])) # loss
        train_energy    = np.zeros((shape[0],batch_size,1)) # Edata
        train_samples   = np.zeros((shape[0],batch_size*num_samples,1)) # Esamples
        train_sig       = np.zeros((shape[0])) # sigma
        train_z         = np.zeros((shape[0])) # logz
        test_accuracy   = np.zeros(shape) # accuracy
        #test_loss       = np.zeros((shape[0],3)) # l,l1000, altloss
        test_loss       = np.zeros((shape[0],1)) # l
        test_energy     = np.zeros((shape[0],batch_size,1)) # Edata
        #test_samples    = np.zeros((shape[0],batch_size*num_samples,2)) # Esamples,alternative Esamples
        test_samples    = np.zeros((shape[0],batch_size*num_samples,1)) # Esamples
        #test_sig        = np.zeros((shape[0],2)) # sigma,sigma1000
        test_sig        = np.zeros((shape[0],1)) # sigma
        #test_z          = np.zeros((shape[0],3)) # logz,logz1000,alternative logz
        test_z          = np.zeros((shape[0],1)) # logz
        #eval_samples    = np.zeros((shape[0],1000*batch_size*num_samples,1)) #1000*Esamples
        time_ite        = np.zeros(shape[0])
        iteration       = np.zeros((shape[0])) # iteration
        norm_params     = np.zeros((shape[0],len(params)))
        i, s = 0, time.time() #counter for iteration, time
        best_acc = 0.0
        n = init_n
        for epoch in range(num_epochs):
            for x, y in dataset.iter("train", batch_size):
                Edata,Esamples,Loss,logZ,Sig = trainloss_f(x,-n*log(2.0))
                if i%LOG_FREQ==0:
                    # Compute params params norm
                    norm = [np.sum(W.get_value()**2)/float(W.get_value().size) for W in params]
                    # Eval train
                    train_a1,train_a5,train_a7,_,_,_ = eval_f(x)
                    train_a = np.array([train_a1,train_a5,train_a7])
                    # Test
                    batch_count = 0
                    #loss = np.zeros((3))
                    loss = np.zeros((1))
                    test_a = np.zeros((len(fractions)))
                    for x_test, y_test in dataset.iter("test", batch_size):
                        #edata,esamples,logq,l,logz,sig,esamples1000,logq1000,l1000,logz1000,sig1000,alte_samples,altloss,altlogz = testloss_f(x_test,prob_init*exp(i*log(decay_rate)))
                        #loss += np.array([l,l1000,altloss])
                        edata,esamples,l,logz,sig = testloss_f(x_test,-n*log(2.0))
                        loss += np.array([l])
                        acc1,acc5,acc7,_,_,_ = eval_f(x_test)
                        test_a += np.array([acc1,acc5,acc7])
                        batch_count += 1
                        if batch_count==2:
                            break
                    loss = loss/float(batch_count)
                    test_a = test_a/float(batch_count)
                    if test_a[-1]>best_acc:
                        best_acc = test_a[-1]
                        _,_,_,recon1,recon5,recon7 = eval_f(true_x)
                        save_params([recon1,recon5,recon7], result_file + '_best_recons', date_time=False)
                    # Store info
                    train_accuracy[(i)//LOG_FREQ] = train_a
                    train_loss[(i)//LOG_FREQ] = Loss
                    train_energy[(i)//LOG_FREQ] = Edata
                    #train_samples[(i)//LOG_FREQ] = np.concatenate((Esamples,Logq), axis=-1)
                    train_samples[(i)//LOG_FREQ] = Esamples
                    train_sig[(i)//LOG_FREQ] = Sig
                    train_z[(i)//LOG_FREQ] = logZ
                    test_accuracy[(i)//LOG_FREQ] = test_a
                    test_loss[(i)//LOG_FREQ] = loss
                    test_energy[(i)//LOG_FREQ] = edata
                    #test_samples[(i)//LOG_FREQ] = np.concatenate((esamples,logq,alte_samples), axis=-1)
                    test_samples[(i)//LOG_FREQ] = esamples
                    #test_sig[(i)//LOG_FREQ] = np.asarray([sig,sig1000])
                    test_sig[(i)//LOG_FREQ] = np.asarray([sig])
                    #test_z[(i)//LOG_FREQ] = np.asarray([logz,logz1000,altlogz])
                    test_z[(i)//LOG_FREQ] = np.asarray([logz])
                    #eval_samples[(i)//LOG_FREQ] = np.concatenate((esamples1000,logq1000), axis=-1)
                    ti = time.time() - s
                    time_ite[(i)//LOG_FREQ] = ti
                    iteration[(i)//LOG_FREQ] = i
                    norm_params[(i)//LOG_FREQ] = norm
                    # Save info
                    save_np(train_accuracy,'train_acc',result_file)
                    save_np(train_loss,'train_loss',result_file)
                    save_params([train_energy],result_file + '_train_energy', date_time=False)
                    save_params([train_samples],result_file + '_train_samples', date_time=False)
                    save_np(train_sig,'train_sig',result_file)
                    save_np(train_z,'train_z',result_file)
                    save_np(test_accuracy,'test_acc',result_file)
                    save_params([test_loss],result_file + '_test_loss', date_time=False)
                    save_params([test_energy],result_file + '_test_energy', date_time=False)
                    save_params([test_samples],result_file + '_test_samples', date_time=False)
                    save_params([test_sig],result_file + '_test_sig', date_time=False)
                    save_params([test_z],result_file + '_test_z', date_time=False)
                    #save_params([eval_samples],result_file + '_eval_samples', date_time=False)
                    save_np(time_ite,'time',result_file)
                    save_np(iteration,'iteration',result_file)
                    save_np(norm_params,'norm_params',result_file)
                    # log info
                    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                    print("[{:.3f}s]iteration {}".format(ti, i+1))
                    print("train loss: {:.3e}, test loss: {:.3f}".format(float(Loss),float(loss[0])))
                    print("train acc: {:.3f}%, test acc: {:.3f}%".format(100.0*train_a[-1],100.0*test_a[-1]))
                    print("train sig: {:.3f}, test sig: {:.3f}".format(float(Sig),float(sig)))
                    print("annealed logq: -{}xlog(2)\n".format(int(n)))
                    s = time.time()
                i += 1
                #updating annealed logq
                if obj_fct=='CSSann' and np.average(Esamples)+n*log(2.0)<ANNEALED_THRESHOLD:
                    n += annealing_rate

        # Reconstructing images after training ends
        _,_,_,recon1,recon5,recon7 = eval_f(true_x)
        save_params([recon1,recon5,recon7], result_file + '_final_recons', date_time=False)
        save_params([true_x], result_file + '_truex', date_time=False)
        # Save final params
        print("Saving weights..")
        save_params([W.get_value() for W in params], checkpoint_file,False)

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
    dataset.data["train"] = (dataset.data["train"][0][:options.num_data],dataset.data["train"][1][:options.num_data])

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
