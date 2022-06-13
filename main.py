import torch
from torch.nn.functional import one_hot,softmax
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
import itertools
import scipy.stats as st
import math
from more_itertools import chunked
import time
import random
import sys

print("Using pytorch version: " + torch.__version__)

### local imports
print("Importing local files: ", end = '')
from args import args
from utils import *
from sampling_utils import *


def load_dataset(dataset_filename, train_c, device):

    """
    loads dataset into memory and performs centering and unit sphere projection of features

    Parameters
    ----------
    dataset_filename : name of file contianing extracted features of dataset (performs ensembling if list contains multiple filenames)
    train_c : number or training classes in dataset (used for centering and projection)
    device : cpu or cuda
    

    Returns
    -------
    dataset
    """

    dataset = torch.cat([torch.load(fn, map_location=torch.device(device)).to(device) for fn in dataset_filename], dim = 2)
    shape = dataset.shape
    average = dataset[:train_c].reshape(-1, dataset.shape[-1]).mean(dim = 0)

    dataset = dataset.reshape(-1, dataset.shape[-1]) - average
    dataset = dataset / torch.norm(dataset, dim = 1, keepdim = True)
    dataset = dataset.reshape(shape)
 
    return dataset



def generate_runs(n_runs, dataset, n_ways, n_samples, alpha_dir, gs, beta, kappa, m, train_c, test_c, val_c, device):
    
    """
    generates runs for active few-shot tasks with dirichlet class imbalance and and performs graph smoothing

    Parameters
    ----------
    n_runs : number of runs
    dataset : dataset used for tasks (after feature extraction)
    n_ways : number of classes
    n_samples : number of samples
    alpha_dir : dirichlet imbalance parameter
    gs : False to turn off graph smoothing
    beta, kappa, m : graph smoothing parameters 
    train_c : number of train classes in dataset
    val_c : number of validation classes in dataset
    test_c : number of test classes in dataset
    device : cpu or cuda
    

    Returns
    -------
    runs_x
        runs inputs
    runs_y
        runs targets
    """

    runs_x = torch.zeros(n_runs,n_samples,dataset.shape[-1]).to(device)
    runs_y = torch.zeros(n_runs,n_samples).to(device)
    dirs = get_dirichlet_query_dist(alpha_dir, n_runs,  n_ways, n_samples)
    
    for i in range(n_runs):
        classes = torch.randperm(test_c)[:n_ways] + train_c +val_c
        
        dq = 0 #dirichlet number of samples per class
        for c,j in enumerate(dirs[i]):
            runs_x[i,dq:dq+j]= dataset[classes[c]][torch.randperm(dataset.shape[1])[:j]]
            runs_y[i,dq:dq+j] += c 
            dq += j

    if gs==True:
        runs_x = graph_smoothing(runs_x, device, beta, kappa, m)
    
    return runs_x,runs_y


def main(n_runs, n_ways, n_samples, n_labels, comb_batch_size, run_batch_size, device, gs, beta, kappa, m, dataset_filename, test_c, train_c, val_c, n_init, temp, n_iter,switch,crit,alpha_dir):
    
    dataset = load_dataset(dataset_filename, train_c, device)

    print('number of classes: ',n_ways,'.number of labels: ',n_labels,'.number of samples: ', n_samples, ".number of runs: ",n_runs)
    print('Runs generation start..')
    start_time = time.time()
    
    runs_x, runs_y = generate_runs(n_runs, dataset, n_ways, n_samples, alpha_dir, gs, beta, kappa, m, train_c, test_c, val_c, device)
    print("generation end : --- %s seconds ---" % (time.time() - start_time))

    start_time_alg = time.time()
    if crit == 'oracle':
        
        #using NCM
        scores,_ = oracle(runs_x, runs_y, n_ways, n_samples, n_labels, device, comb_batch_size, run_batch_size)
        
    elif crit == 'random':
        
        scores=[]
        labels_idx_rand = torch.zeros(n_runs,n_labels)
        for i in range(n_runs):
            labels_idx_rand[i] = torch.randperm(n_samples)[:n_labels]
        labels_idx_rand = labels_idx_rand.reshape(n_runs,-1).type(torch.long)

        pred_y_rand,_,_ = softkmeans(runs_x,runs_y,labels_idx_rand,n_ways,n_samples,n_labels,run_batch_size,temp)
        scores  = wacc_acc_f1(pred_y_rand.cpu(),runs_y,n_ways)[0]

    else:
        scores = []
        for i in tqdm(range(int(n_runs/run_batch_size))):
            
            runs_x_batch = runs_x[i*run_batch_size:(i+1)*run_batch_size] 
            runs_y_batch = runs_y[i*run_batch_size:(i+1)*run_batch_size] 
            labels_idx_crit = softkm_sample(runs_x_batch, runs_y_batch, n_ways,n_labels, device, n_init, temp, n_iter, crit, switch)

            pred_y_crit,_,_ = softkmeans(runs_x_batch,runs_y_batch,labels_idx_crit,n_ways,n_samples,n_labels,temp, n_iter)
            scores  += wacc_acc_f1(pred_y_crit.cpu(),runs_y_batch,n_ways)[0]

    print("total runtime : --- %s seconds ---" % (time.time() - start_time_alg))
    print("best: {:.2f}% (± {:.2f}%)".format(stats(scores, "")[0]*100,stats(scores, "")[1]*100))


if __name__ == "__main__":
   main(args.n_runs, args.n_ways, args.n_samples, args.n_labels, args.comb_batch_size, args.run_batch_size, args.device, args.gs, args.beta, args.kappa, args.m, 
   args.dataset_fn, args.test_c, args.train_c, args.val_c, args.n_init, args.temp, args.n_iter, args.switch,args.crit,args.alpha_dir)