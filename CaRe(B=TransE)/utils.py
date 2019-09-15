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


def get_neg_samples(pos_samples,unq_ent, args):
    size_of_batch = pos_samples.shape[0]
    num_to_generate = size_of_batch * args.neg_samples
    neg_samples = np.tile(pos_samples, (args.neg_samples, 1))
    
    values = np.random.choice(unq_ent, size=num_to_generate)
    choices = np.random.uniform(size=num_to_generate)
    subj = choices > 0.5
    obj = choices <= 0.5
    neg_samples[subj, 0] = values[subj]
    neg_samples[obj, 2] = values[obj]

    return np.concatenate((pos_samples, neg_samples))



def get_next_batch(id_list, data, args):
    pos_samples = data.train_trips[id_list]
    unq_ents = set()
    for i in range(pos_samples.shape[0]):
        unq_ents.add(pos_samples[i][0])
        unq_ents.add(pos_samples[i][2])
    unq_ents = list(unq_ents)
    samples = get_neg_samples(pos_samples, unq_ents, args)
    
    return samples


def get_rank(h,r,t,clust,args,entid2clustid,filter_clustID):
    hits = np.ones((len(args.Hits)))
    scores = torch.norm(h + r - t, args.p_norm, -1)
    scores = scores.cpu().data.numpy()
    scores = np.argsort(scores)
    rank = 1
    high_rank_clust = set()
    for i in range(scores.shape[0]):
        if scores[i] in clust: break
        else:
            if entid2clustid[scores[i]] not in high_rank_clust and entid2clustid[scores[i]] not in filter_clustID:
                rank+=1
                high_rank_clust.add(entid2clustid[scores[i]])
    for i,r in enumerate(args.Hits):
        if rank>r: hits[i]=0
        else: break
    return rank,hits


def evaluate(model, test_trips, entTotal, args, data):
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
    H_filter = data.H_filter
    T_filter = data.T_filter
    
    ents = torch.arange(0, entTotal, dtype=torch.long)
    edges = torch.tensor(data.edges,dtype=torch.long)
    if args.use_cuda: 
        ents = ents.cuda()
        edges = edges.cuda()
    
    r_embed,ent_embed = model.get_embed(edges,ents,rel)
        
    for j in range(test_trips.shape[0]):
        print("Evaluation Phase: sample {}/{} total samples".format(j + 1,test_trips.shape[0]),end="\r")
        h = ent_embed[head[j],:]
        r = r_embed[j,:]
        t = ent_embed[tail[j],:]
        h_clust = set(true_clusts[head[j]])
        t_clust = set(true_clusts[tail[j]])
        h = h.repeat(entTotal,1)
        r = r.repeat(entTotal,1)
        t = t.repeat(entTotal,1)
        if (rel[j],tail[j]) in H_filter: _filter = H_filter[(rel[j],tail[j])]
        else: _filter = []
        H_r,H_h = get_rank(ent_embed,r,t,h_clust,args,entid2clustid,_filter)
        if (rel[j],head[j]) in T_filter: _filter = T_filter[(rel[j],head[j])]
        else: _filter = []
        T_r,T_h = get_rank(h,r,ent_embed,t_clust,args,entid2clustid,_filter)
        H_Rank.append(H_r)
        H_inv_Rank.append(1/H_r)
        T_Rank.append(T_r)
        T_inv_Rank.append(1/T_r)
        H_Hits += H_h
        T_Hits += T_h
    print("Mean Rank: Head = {}  Tail = {}  Avg = {}"
          .format(np.mean(np.array(H_Rank)),np.mean(np.array(T_Rank)),(np.mean(np.array(H_Rank)) + np.mean(np.array(T_Rank)))/2))
    print("MRR: Head = {}  Tail = {}  Avg = {}"
          .format(np.mean(np.array(H_inv_Rank)),np.mean(np.array(T_inv_Rank)),(np.mean(np.array(H_inv_Rank)) + np.mean(np.array(T_inv_Rank)))/2))
    
    for i,hits in enumerate(args.Hits):
        print("Hits@{}: Head = {}  Tail={}  Avg = {}"
              .format(hits,H_Hits[i]/len(H_Rank),T_Hits[i]/len(H_Rank),(H_Hits[i] + T_Hits[i])/(2*len(H_Rank))))
    return (np.mean(np.array(H_Rank)) + np.mean(np.array(T_Rank)))/2,(np.mean(np.array(H_inv_Rank)) + np.mean(np.array(T_inv_Rank)))/2

