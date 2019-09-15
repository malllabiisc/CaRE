import numpy as np
import random
import copy
import time

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.cuda as cuda
import torch.optim as optim
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore")

def seq_batch(phrase_id, args, phrase2word):
    phrase_batch = np.ones((len(phrase_id),11),dtype = int)*args.pad_id
    phrase_len = torch.LongTensor(len(phrase_id))
    
    for i,ID in enumerate(phrase_id):
        phrase_batch[i,0:len(phrase2word[ID])] = np.array(phrase2word[ID])
        phrase_len[i] = len(phrase2word[ID])
        
    phrase_batch = torch.from_numpy(phrase_batch)
    phrase_batch = Variable(torch.LongTensor(phrase_batch))
    phrase_len = Variable(phrase_len)
    
    if args.use_cuda:
        phrase_batch = phrase_batch.cuda()
        phrase_len = phrase_len.cuda()
    
    return phrase_batch, phrase_len


def get_next_batch(id_list, data, args, train):
    entTotal = args.num_nodes
    samples = []
    labels = np.zeros((len(id_list),entTotal))
    for i in range(len(id_list)):
        trip = train[id_list[i]]
        samples.append([trip[0],trip[1]])
        pos_ids = list(data.label_graph[(trip[0],trip[1])])
        labels[i][pos_ids] = 1    
    return np.array(samples),labels


def get_rank(scores,clust,Hits,entid2clustid,filter_clustID):
    hits = np.ones((len(Hits)))
    scores = np.argsort(scores)
    rank = 1
    high_rank_clust = set()
    for i in range(scores.shape[0]):
        if scores[i] in clust: break
        else:
            if entid2clustid[scores[i]] not in high_rank_clust and entid2clustid[scores[i]] not in filter_clustID:
                rank+=1
                high_rank_clust.add(entid2clustid[scores[i]])
    for i,r in enumerate(Hits):
        if rank>r: hits[i]=0
        else: break
    return rank,hits


def evaluate(model, entTotal, test_trips, args, data):
    ents = torch.arange(0, entTotal, dtype=torch.long)
    H_Rank = []
    H_inv_Rank = []
    H_Hits = np.zeros((len(args.Hits)))
    T_Rank = []
    T_inv_Rank = []
    T_Hits = np.zeros((len(args.Hits)))
    head = test_trips[:,0]
    rel = test_trips[:,1]
    tail = test_trips[:,2]
    id2ent = data.id2ent
    id2rel = data.id2rel
    true_clusts = data.true_clusts
    entid2clustid = data.entid2clustid
    ent_filter = data.label_filter
    bs = args.batch_size
    
    edges = torch.tensor(data.edges,dtype=torch.long)
    if args.use_cuda: 
        ents = ents.cuda()
        edges = edges.cuda()
    
    r_embed,ent_embed = model.get_embed(edges,ents,rel)
    
    test_scores = np.zeros((test_trips.shape[0],entTotal))
    n_batches = int(test_trips.shape[0]/bs) + 1
    for i in range(n_batches):
        ent = head[i*bs:min((i+1)*bs,test_trips.shape[0])]
        ent = ent_embed[ent,:]
        r = r_embed[i*bs:min((i+1)*bs,test_trips.shape[0]),:]
        scores = model.get_scores(ent,r,ent_embed,ent.shape[0]).cpu().data.numpy()
        test_scores[i*bs:min((i+1)*bs,test_trips.shape[0]),:] = scores
        
    for j in range(test_trips.shape[0]):
        print("Evaluation Phase: sample {}/{} total samples".format(j + 1,test_trips.shape[0]),end="\r")
        
        sample_scores = -test_scores[j,:]
        
        t_clust = set(true_clusts[tail[j]])
        
        _filter = []
        if (head[j],rel[j]) in ent_filter: _filter = ent_filter[(head[j],rel[j])]
        
        if j%2==1:
            H_r,H_h = get_rank(sample_scores,t_clust,args.Hits,entid2clustid,_filter)
            H_Rank.append(H_r)
            H_inv_Rank.append(1/H_r)
            H_Hits += H_h            
        else:
            T_r,T_h = get_rank(sample_scores,t_clust,args.Hits,entid2clustid,_filter)
            T_Rank.append(T_r)
            T_inv_Rank.append(1/T_r) 
            T_Hits += T_h
    print("Mean Rank: Head = {}  Tail = {}  Avg = {}"
          .format(np.mean(np.array(H_Rank)),np.mean(np.array(T_Rank)),(np.mean(np.array(H_Rank)) + np.mean(np.array(T_Rank)))/2))
    print("MRR: Head = {}  Tail = {}  Avg = {}"
          .format(np.mean(np.array(H_inv_Rank)),np.mean(np.array(T_inv_Rank)),(np.mean(np.array(H_inv_Rank)) + np.mean(np.array(T_inv_Rank)))/2))
    
    for i,hits in enumerate(args.Hits):
        print("Hits@{}: Head = {}  Tail={}  Avg = {}"
              .format(hits,H_Hits[i]/len(H_Rank),T_Hits[i]/len(H_Rank),(H_Hits[i] + T_Hits[i])/(2*len(H_Rank))))
    return (np.mean(np.array(H_Rank)) + np.mean(np.array(T_Rank)))/2,(np.mean(np.array(H_inv_Rank)) + np.mean(np.array(T_inv_Rank)))/2


