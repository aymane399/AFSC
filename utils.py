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
from sklearn.metrics import f1_score,accuracy_score


def wacc_acc_f1(runs_preds,runs_y,n_ways):

    """
    computes weighted accuracy, unweighted accuracy and weighted F1-score for predictions

    Parameters
    ----------
    runs_x : runs inputs
    runs_y : runs targets
    n_ways : number of ways
    

    Returns
    -------
    mat_avg
        average class means from label
    optional ~ shots_per_class 
        classes count in labels
    """

    acc = []
    f1 = []
    n_runs = runs_preds.shape[0]
    wacc = torch.zeros((n_runs,1))        
    for i in range(n_runs):
        acc.append(accuracy_score(runs_preds[i].cpu(),runs_y[i].cpu()))
        f1.append(f1_score(runs_preds[i].cpu(),runs_y[i].cpu(),average='weighted'))        
    
    for i in range(n_ways):
        class_pred = (runs_preds==i).float().cpu()
        class_true = (runs_y==i).float().cpu()
        wacc += torch.nan_to_num(((class_true*class_pred).sum(axis=1))/class_true.sum(axis=1),nan=1).reshape(-1,1)/n_ways
    wacc = wacc.reshape(-1).tolist()
    return wacc,acc,f1  


def stats(scores, name):

    """
    computes mean and 95% confidence interval for runs
    function reused from EASY : https://github.com/ybendou/easy
    
    """

    if len(scores) == 1:
        low, up = 0., 1.
    elif len(scores) < 30:
        low, up = st.t.interval(0.95, df = len(scores) - 1, loc = np.mean(scores), scale = st.sem(scores))
    else:
        low, up = st.norm.interval(0.95, loc = np.mean(scores), scale = st.sem(scores))
    if name == "":
        return np.mean(scores), up - np.mean(scores)
    else:
        print("{:s} {:.2f} (± {:.2f}) (conf: [{:.2f}, {:.2f}]) (worst: {:.2f}, best: {:.2f})".format(name, 100 * np.mean(scores), 100 * np.std(scores), 100 * low, 100 * up, 100 * np.min(scores), 100 * np.max(scores)))




def get_dirichlet_query_dist(alpha, n_runs, n_ways, q_shots, CUB=False):
    
    """
    generate dirichlet probabilities and
    converts them numbers of samples per class
    reused : https://github.com/oveilleux/Realistic_Transductive_Few_Shot
    
    """    
    alpha = np.full(n_ways, alpha)
    prob_dist = np.random.dirichlet(alpha, n_runs)
    if CUB ==True:
        while len(prob_dist[prob_dist>=60/80])!=0:
            # hard constraint : maximum class sample = 60 on a 80 samples setting. For settings with more than 
            # 80 samples, the CUB dataset is not considered
            prob_dist = np.random.dirichlet(alpha, n_runs)
    return np.array(convert_prob_to_samples(prob=prob_dist, q_shot=q_shots))

def convert_prob_to_samples(prob, q_shot):
    """
    convert class probabilities to numbers of samples per class
    reused : https://github.com/oveilleux/Realistic_Transductive_Few_Shot
    
    """
    prob = prob * q_shot
    for i in range(len(prob)):
        if sum(np.round(prob[i])) > q_shot:
            while sum(np.round(prob[i])) != q_shot:
                idx = 0
                for j in range(len(prob[i])):
                    frac, whole = math.modf(prob[i, j])
                    if j == 0:
                        frac_clos = abs(frac - 0.5)
                    else:
                        if abs(frac - 0.5) < frac_clos:
                            idx = j
                            frac_clos = abs(frac - 0.5)
                prob[i, idx] = np.floor(prob[i, idx])
            prob[i] = np.round(prob[i])
        elif sum(np.round(prob[i])) < q_shot:
            while sum(np.round(prob[i])) != q_shot:
                idx = 0
                for j in range(len(prob[i])):
                    frac, whole = math.modf(prob[i, j])
                    if j == 0:
                        frac_clos = abs(frac - 0.5)
                    else:
                        if abs(frac - 0.5) < frac_clos:
                            idx = j
                            frac_clos = abs(frac - 0.5)
                prob[i, idx] = np.ceil(prob[i, idx])
            prob[i] = np.round(prob[i])
        else:
            prob[i] = np.round(prob[i])
    return prob.astype(int)


def graph_smoothing(run, device, beta = 2, kappa = 3, m = 15):

    """
    computes weighted accuracy, unweighted accuracy and weighted F1-score for predictions

    Parameters
    ----------
    runs : runs inputs
    beta : identity matrix multiplication factor
    kappa : order of the smoothing
    m : number of neighbors for adjacency matrix 
    

    Returns
    -------
    smoothed_run
        runs after graph smoothing
    """

    features = run.reshape(-1, run.shape[-1])
    normalized_features = features / torch.norm(features, dim = 1, keepdim = True)
    cos_graph = torch.matmul(normalized_features, normalized_features.transpose(0,1))
    cos_graph = (1 - torch.eye(features.shape[0]).to(device)) * cos_graph
    
    for i in range(cos_graph.shape[0]):
        indices = torch.argsort(cos_graph[i,:])
        cos_graph[i,indices[:cos_graph.shape[0] - m]] = 0.
    
    cos_graph = ((cos_graph + cos_graph.transpose(0,1)) > 0.).int().float()
    scaled_degree_matrix = torch.diag(torch.pow(cos_graph.sum(dim = 1), -0.5))
    E = torch.matmul(scaled_degree_matrix, torch.matmul(cos_graph, scaled_degree_matrix))
    IE = beta * torch.eye(E.shape[0]).to(device) + E
    result = IE.clone()
    for i in range(kappa - 1):
        result = torch.matmul(result, IE)
    smoothed_run = torch.matmul(result, features)
    smoothed_run = smoothed_run.reshape(run.shape)  
    return smoothed_run  




def cluster_inertia(run_x,clusters,device,num_clusters):

    """
    Takes several clusterings as input and returns their inertias i.e sum of squared distances of each element to its centroid

    Parameters
    ----------
    run_x : runs inputs
    clusters : multiple cluster assignments 
    device : cuda or cpu
    num_clusters : number of clusterings
    centroids : number of neighbors for adjacency matrix 
    

    Returns
    -------
    inertias
        list of inertias
    """

    clusters_oh = one_hot(clusters.long(),num_clusters)
    clusters_count = clusters_oh.sum(axis=1)
    centroids = torch.einsum("rqf,rqc->rcf",run_x,one_hot(clusters.long(),5).float().to(device)/(clusters_count.unsqueeze(1)).to(device))
    
    distances_cents = torch.norm(run_x.unsqueeze(2) - centroids.unsqueeze(1).to(device),dim=-1)
    inertias = (distances_cents.reshape(-1,num_clusters)[np.arange(run_x.shape[0]*run_x.shape[1]),clusters.reshape(-1).long()]).reshape(run_x.shape[0],run_x.shape[1]).sum(axis=1)

    return inertias


