import torch
import numpy as np
from torch.nn.functional import softmax

import sampling_utils 

def ncm(runs_x,runs_y,labels_idx,class_means):
    
    """
    classifies runs inputs according to nearest class mean and forces ground truth value on 
    selected labels

    Parameters
    ----------
    runs_x : runs inputs
    runs_y : runs targets
    labels_idx : labels indices in runs
    class_means : class means used for class assignment

    Returns
    -------
    preds_y
        target predictions
    """

    n_runs = runs_x.shape[0]
    preds_y = torch.argmin(torch.norm(runs_x.unsqueeze(2) - class_means.unsqueeze(1),dim=-1),dim=-1) 
    preds_y[np.arange(n_runs).reshape(-1,1),labels_idx]= runs_y[np.arange(n_runs).reshape(-1,1),labels_idx].type(torch.long)

    return preds_y


def softkmeans(runs_x,runs_y,labels_idx,n_ways,n_samples,n_labels,temp=5, n_iter=30):

    
    """
    classifies runs inputs according to a soft k-means classifier: a soft-kmeans clustering is initialized using the labels
    we force the labels to stay within the clusters they initialized, and classify each sample as belonging to the class corresponding
    the label that initialized its cluster.

    Parameters
    ----------
    runs_x : runs inputs
    runs_y : runs targets
    labels_idx : labels indices in runs
    n_ways : number of classes
    n_samples : number of samples
    n_labels : number of labels
    temp : soft k-means hyperparameter for the soft allocations
    n_iter : number of iterations for the soft k-means

    Returns
    -------
    preds_y
        target predictions

    soft_allocations
        algorithm final soft allocations
    mask 
        mask to ignore the labels
    """

    runs_y_clone = runs_y.clone().detach().type(torch.long)
    n_runs = runs_x.shape[0]

    labels_x = runs_x[np.arange(n_runs).reshape(-1,1),labels_idx].permute(0,2,1)
    labels_y = runs_y[np.arange(n_runs).reshape(-1,1),labels_idx]

    oh_labels_avg, oh_labels, labels_class_count =  sampling_utils.label_oh_mat(labels_y, n_ways, avg=False)
    centroids = torch.matmul(labels_x,oh_labels_avg).permute(0,2,1)
    cents_non_avg = torch.matmul(labels_x,oh_labels).permute(0,2,1)
    
    #using a mask on the labels in order to not account for them in the soft k-means iterations as we force
    #them to stay withing the cluster they initialize
    mask = torch.arange(n_samples).reshape(1,-1).repeat(n_runs,1)
    for i in range(n_labels):
        mask[np.arange(n_runs).reshape(-1,1),labels_idx[:,i].reshape(n_runs,1)]= -1
    mask = (mask!=-1)
    runs_x = runs_x[mask].reshape(n_runs,n_samples-n_labels,-1)


    for i in range(n_iter):

        distances = torch.norm(runs_x.unsqueeze(2) - centroids.unsqueeze(1),dim=-1)
        soft_allocations = softmax(-distances.pow(2)*temp, dim=2)  #(r,q,n_c)
        centroids = cents_non_avg + torch.einsum("rsw,rsd->rwd", soft_allocations, runs_x)
        centroids = centroids/((labels_class_count+soft_allocations.sum(dim = 1)).unsqueeze(-1))
    
    preds_non_labels = torch.min(distances, dim = 2)[1]
    runs_y_clone[mask] = preds_non_labels.reshape(-1)
    pred_labels = runs_y_clone.reshape(runs_y_clone.shape)
    pred_labels[np.arange(n_runs).reshape(-1,1),labels_idx]= runs_y[np.arange(n_runs).reshape(-1,1),labels_idx].type(torch.long)

    return pred_labels,soft_allocations,mask