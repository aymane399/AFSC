import torch
from torch.nn.functional import one_hot,softmax
import numpy as np
from tqdm import tqdm

import itertools
from more_itertools import chunked
import time


from inference_methods import softkmeans
from utils import cluster_inertia

def label_oh_mat(labels_y, n_ways,avg=True):

    """
    generates one hot encoding of labels. used to create class means when multiplied with labels X

    Parameters
    ----------
    labels : categorical labels values
    n_ways : number of ways
    avg : return only class mean when true, return also class sum and shots per class count otherwise

    Returns
    -------
    oh_labels_avg
        class means from labels
    oh_labels
        class sum from labels
    optional ~ shots_per_class 
        classes count in labels
    """
    
    oh_labels = one_hot(labels_y.type(torch.int64),n_ways).permute(0,1,2).type(torch.float)
    
    labels_class_count = oh_labels.sum(axis=1)  #(q,c)
    oh_labels_avg = oh_labels/(labels_class_count.unsqueeze(1))
    oh_labels_avg = oh_labels_avg.nan_to_num(0)

    if avg==True:
        return oh_labels_avg
    else:
        return oh_labels_avg, oh_labels, labels_class_count


def means_from_labels(runs_x,runs_y,shots_idx, n_ways):

    """
    generates class means from labels

    Parameters
    ----------
    runs_x : runs inputs
    runs_y : runs targets
    n_ways : number of ways
    

    Returns
    -------
    class_means
        class means in labels
    """

    n_runs = runs_x.shape[0]
    labels_x = runs_x[np.arange(n_runs).reshape(-1,1),shots_idx].permute(0,2,1)
    labels_y = runs_y[np.arange(n_runs).reshape(-1,1),shots_idx]
    oh_labels_avg =  label_oh_mat(labels_y, n_ways)
    class_means = torch.matmul(labels_x,oh_labels_avg)                          
    #shape (r*c,f,l)
    class_means = torch.permute(class_means, (0, 2, 1))
    #shape (r*c,l,f)
    return class_means


def select_with_criterion(runs_x, labels_idx, num_classes, centroids, soft_allocations, clust_labels, crit, rd, ns):

    """
    selects new samples to turn into labels according to criterion on clustered data while making sure previous labels are not reselected

    Parameters
    ----------
    runs_x : runs inputs
    labels_idx : labels_idx
    num_classes : number of ways/ classes
    centroids : clustering centroids
    soft_allocations : clustering soft allocations (used for margin criterion)
    clust_labels : clustering labels
    crit : criterion used (LSS- K-medoid - margin)
    rd : round number
    ns : multiplicative factor for switching between high and low confidence samples selection
    

    Returns
    -------
    labels_idx
        updated labels_idx list
    """

    n_runs = runs_x.shape[0]
    for r in range(n_runs):
            if False:
                
                    if crit == 'vr':
                        probs = -torch.max(soft_allocations[r],dim=1)[0]
                    elif crit =='entropy':
                        probs = torch.sum(-torch.log(soft_allocations[r])*soft_allocations[r],dim=1)
                    probs=ns*probs
                    for step in range(5):
                        probs[labels_idx[r,:rd*num_classes+step].type(torch.long)] = np.NINF
                        labels_idx[r,rd*num_classes+step] = torch.argmax(probs)
                        

            else:
                for i in range(num_classes):
                    if crit=='K-medoid':
                        probs = -torch.norm(runs_x[r]-centroids[r,i],dim=-1)
                    elif crit=='LSS':
                        probs = (torch.norm(runs_x[r].unsqueeze(0).repeat(num_classes,1,1) - centroids[r].unsqueeze(1).repeat(1,runs_x.shape[1],1),dim=2)**2)
                        probs/=probs.clone()[i,:]
                        probs = probs.sum(dim=0)
                    elif crit=='margin':
                        probs = soft_allocations[r,:,i] - torch.argmax(soft_allocations[r,:,np.delete(np.arange(num_classes),i)],dim=-1)
                    elif crit == 'vr':
                        probs = torch.max(soft_allocations[r],dim=1)[0]
                    elif crit =='entropy':
                        probs = -torch.sum(-torch.log(soft_allocations[r])*soft_allocations[r],dim=1)
                    probs=ns*probs
                    for step in range(5):
                        probs[labels_idx[r,:rd*num_classes+step].type(torch.long)] = np.NINF
                        labels_idx[r,rd*num_classes+step] = torch.argmax(probs)

                    probs=ns*probs
                    probs[clust_labels[r]!=i] = -10e6
                    probs[labels_idx[r,:rd*num_classes+i].type(torch.long)] = np.NINF
                    labels_idx[r,rd*num_classes+i] = torch.argmax(probs)

    return labels_idx



def softkm_sample(runs_x, runs_y, num_classes, num_labels, device, n_init=10, temp=5, num_iter=30, crit='LSS',switch=1):
    
    """
    selects new samples to turn into labels according to criterion on clustered data while making sure previous labels are not reselected

    Parameters
    ----------
    runs_x : runs inputs
    runs_y : runs targets
    num_classes : number of ways/ classes
    num_labels : number of labels
    device : cpu or cuda
    n_init, temp, num_iter : soft k-means parameters
    crit : criterion used for active selection
    switch : number of switching round from high confidence to low confidence sampling
    

    Returns
    -------
    labels_idx
        updated labels_idx list
    """

    rounds = num_labels//num_classes
    ns = 1
    if switch==0:
        ns=-1

    n_runs = runs_x.shape[0]
    num_samples = runs_x.shape[1]
    labels_idx = torch.zeros(n_runs,num_labels).to(device).long()
    centroids = torch.zeros(n_runs,n_init,num_classes,runs_x.shape[-1]).to(device)
    run_rep = runs_x.unsqueeze(1).repeat((1,n_init,1,1)).reshape(-1,runs_x.shape[1],runs_x.shape[2])

    for r in range(n_runs):
        for i in range(n_init):
            centroids[r,i,:] = runs_x[r,torch.randperm(runs_x.shape[1])[:num_classes]]
    
    
    centroids = centroids.reshape(-1,num_classes,runs_x.shape[-1])  
    for i in range(num_iter):

        distances = torch.norm(run_rep.unsqueeze(2) - centroids.unsqueeze(1),dim=-1)
        soft_allocations = softmax(-distances.pow(2)*temp, dim=2)  
        centroids = torch.einsum("rqc,rqd->rcd", soft_allocations, run_rep)
        centroids = centroids/((soft_allocations.sum(dim = 1)).unsqueeze(-1))

    clust_labels = torch.min(distances, dim = 2)[1]
    inertias = cluster_inertia(run_rep,clust_labels,device,num_classes).reshape(n_runs,n_init)
    best_clust = torch.min(inertias,dim=1)[1]

    clust_labels = clust_labels.reshape(n_runs,n_init,-1)[np.arange(n_runs),best_clust]
    centroids = centroids.reshape(n_runs,n_init,num_classes,-1)[np.arange(n_runs),best_clust,:,:]
    labels_idx = select_with_criterion(runs_x, labels_idx, num_classes, centroids, distances, clust_labels, crit, 0, ns)

    if rounds>1:
        for rd in range(1,rounds):
            if rd>=switch:
                ns=-1
            clust_labels,soft_allocations,mask = softkmeans(runs_x, runs_y, labels_idx, num_classes, num_samples, num_classes*rd, temp, num_iter) 
            if crit in ['margin','vr','entropy']:
                allocs = torch.ones(n_runs,num_samples,num_classes).to(device)
                allocs[mask] = soft_allocations.reshape(-1,num_classes)
            else:
                allocs = None
            labels_idx = select_with_criterion(runs_x, labels_idx, num_classes, centroids, allocs, clust_labels, crit, rd, ns)

    
    return labels_idx


def oracle(runs_x, runs_y, num_classes, num_samples, num_labels, device, comb_batch_size, run_batch_size):
    
    """
    tries all combinations of labels from samples and selects the one that yields the best performance

    Parameters
    ----------
    runs_x : runs inputs
    runs_y : runs targets
    num_classes : number of ways/ classes
    num_samples : number of samples
    num_labels : number of labels
    device : cpu or cuda
    comb_batch_size : number of combinations to process in a single batch
    run_batch_size : number of runs to process in a single batch
    

    Returns
    -------
    scores
        oracle scores list
    labels_best
        oracle-selected labels list           
    """

    n_runs = runs_y.shape[0]
    
    scores = torch.zeros(n_runs,1).to(device)
    scores_comb = torch.zeros(n_runs,1).to(device)

    labels_best = torch.zeros(n_runs,1,num_labels).to(device)
    labels_best_comb = torch.zeros(n_runs,1,num_labels).to(device)
    
    print('combination batch creation')
    start_time = time.time()
    combis = itertools.combinations(range(num_samples),num_labels)
    comb_batches = chunked(combis,comb_batch_size)
    
    print("comb batches created : --- %s seconds ---" % (time.time() - start_time))
    print('run started')
    
    for batch in tqdm(comb_batches):
        #batching combinations
        for i in range(n_runs//run_batch_size):    

            comb_ids = torch.tensor(batch).to(device)

            #batching runs
            batch_x = runs_x[i*run_batch_size:(i+1)*run_batch_size]
            batch_y = runs_y[i*run_batch_size:(i+1)*run_batch_size]
            batch_x_comb = batch_x.unsqueeze(1).repeat((1,comb_ids.shape[0],1,1)).reshape(-1,batch_x.shape[-2],batch_x.shape[-1])                #(r*c,q,f) 
            batch_y_comb = batch_y.unsqueeze(1).repeat((1,comb_ids.shape[0],1))                                                         #(r,c,q)
            batch_x_comb_unflat = batch_x.unsqueeze(1).repeat((1,comb_ids.shape[0],1,1))                                                           #(r,c,q,f)

            batch_labels_y = batch_y_comb[:,np.arange(comb_ids.shape[0]).reshape(-1,1),comb_ids]   
                                               #(r,c,s)

            oh_labels_avg =  label_oh_mat(batch_labels_y.reshape(run_batch_size*comb_ids.shape[0],-1), num_classes)                 #(r*c,n_labels,s)

            batch_labels_x = batch_x_comb_unflat[:,np.arange(comb_ids.shape[0]).reshape(-1,1),comb_ids,:]                                        #(r,c,s,f)
            batch_labels_x = batch_labels_x.reshape(-1,batch_labels_x.shape[-2],batch_labels_x.shape[-1])                                 #(r*c,s,f)
            batch_labels_x = torch.permute(batch_labels_x, (0, 2, 1))                                                                   #(r*c,f,s)


            #adding sampled samples to centroid corresponding to their class
            ncm_centroids = torch.matmul(batch_labels_x,oh_labels_avg)                                                             #(r*c,f,n_labels)
            ncm_centroids = torch.permute(ncm_centroids, (0, 2, 1))                                                                   #(r*c,n_labels,f)
            
            #ncm an score
            
            pred_y = torch.argmin(torch.norm(batch_x_comb.unsqueeze(2) - ncm_centroids.unsqueeze(1),dim=-1),dim=-1)                #(r*c,q)
            pred_y[np.arange(pred_y.shape[0]).reshape(-1,1),comb_ids.unsqueeze(0).repeat((run_batch_size,1,1)).reshape(-1,num_labels)]=batch_labels_y.reshape(-1,num_labels).type(torch.long)

            #weighted accuracy (average of classes recalls)
            avg_recall = torch.zeros((pred_y.shape[0],1))
            
            for i_c in range(num_classes):
                class_pred = (pred_y==i_c).float().cpu()
                class_true = (batch_y_comb.reshape(-1,num_samples)==i_c).float().cpu()
                avg_recall += torch.nan_to_num(((class_true*class_pred).sum(axis=1))/class_true.sum(axis=1),nan=1).reshape(-1,1)/num_classes


            comb_batch_scores, indices = avg_recall.reshape(run_batch_size,-1).max(dim=1)
            comb_batch_scores = comb_batch_scores.reshape(-1,1)
            #add score to runs for combs
            scores_comb[i*run_batch_size:(i+1)*run_batch_size] = comb_batch_scores
            labels_best_comb[i*run_batch_size:(i+1)*run_batch_size,0] = comb_ids.unsqueeze(0).repeat(run_batch_size,1,1)[np.arange(run_batch_size),indices,:] 
        #select best combination score per run<<<

        scores,indices = torch.cat((scores,scores_comb),dim=-1).max(dim=-1)
        labels_best = torch.cat((labels_best,labels_best_comb),dim=-2)[np.arange(n_runs),indices,:]
        labels_best = labels_best.reshape((n_runs,1,num_labels))
        scores = scores.reshape(-1,1)
        
    #torch.save(labels_best, 'best_labels_'+str(num_classes)+'_'+str(num_samples)+'_'+str(num_labels)+'.pt')

    scores = scores.reshape(-1).tolist()
    labels_best = labels_best.reshape(n_runs,-1).type(torch.long)

    return scores,labels_best
