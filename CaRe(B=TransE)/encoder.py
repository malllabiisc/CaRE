import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.cuda as cuda
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
import torch.nn.functional as F


class GRUEncoder(nn.Module):
    def __init__(self, embed_matrix, args):        
        super(GRUEncoder, self).__init__()
        self.args = args
        self.bi = self.args.bidirectional
        self.hidden_size = self.args.nfeats//2 if self.bi else self.args.nfeats
        self.num_layers = self.args.num_layers
        self.embed_matrix = embed_matrix
        self.pad_id = self.args.pad_id
        self.poolType = self.args.poolType
        self.dropout = self.args.dropout
        
        self.embed = nn.Embedding(num_embeddings = self.embed_matrix.shape[0], 
                                  embedding_dim = self.embed_matrix.shape[1],
                                  padding_idx = self.pad_id)
        
        
        self.encoder = nn.GRU(self.embed_matrix.shape[1], self.hidden_size, 
                              self.num_layers, dropout=self.dropout, batch_first = True, bidirectional = self.bi)
        
        self.embed.weight.data.copy_(torch.from_numpy(self.embed_matrix))
        
        
    def _encode(self, batch, doc_len):
        size, sort = torch.sort(doc_len, dim=0, descending=True)
        _, unsort = torch.sort(sort, dim=0)
        batch = torch.index_select(batch, dim=0, index=sort)
        embedded = self.embed(batch)
        packed = pack(embedded, size.data.tolist(), batch_first=True)
        encoded, h = self.encoder(packed)
        unpacked, _ = unpack(encoded, batch_first=True)
        unpacked = torch.index_select(unpacked, dim=0, index=unsort)
        h = torch.index_select(h, dim=1, index=unsort)
        return unpacked, h

    def _pool(self, unpacked, h):
        batchSize, Seqlength, _ = unpacked.size()
        if self.poolType == 'last':
            idx = 2 if self.bi else 1
            pooled = h[-idx:].transpose(0, 1).contiguous().view(batchSize, -1)
        elif self.poolType == 'mean':
            pooled = torch.mean(unpacked, dim=1)
        elif self.poolType == 'max':
            pooled, _ = torch.max(unpacked, dim=1)
        return pooled
    
    def _getFeatures(self, batch,doc_len):
        encoded, hidden = self._encode(batch, doc_len)
        hidden = self._pool(encoded, hidden)
        return hidden
    
    def forward(self,batch,doc_len):
        phrase_encode = self._getFeatures(batch,doc_len)
        return phrase_encode