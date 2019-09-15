#### Import all the supporting classes

import argparse

from utils import *
from encoder import GRUEncoder
from cn_variants import LAN, GCN, GAT
from data import load_data

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class ConvEParam(nn.Module):
    def __init__(self, args, embed_matrix,rel2words):
        super(ConvEParam, self).__init__()
        self.args = args

        self.rel2words = rel2words
        self.phrase_embed_model = GRUEncoder(embed_matrix, self.args)

        if self.args.CN=='LAN':
        	self.cn = CaRe(self.args.nfeats, self.args.nfeats)
        elif self.args.CN=='GCN':
        	self.cn = CaReGCN(self.args.nfeats, self.args.nfeats)
        else:
        	self.cn = CaReGAT(self.args.nfeats, self.args.nfeats//self.args.nheads, heads=self.args.nheads, dropout=self.args.dropout)

        
        self.np_embeddings = nn.Embedding(self.args.num_nodes, self.args.nfeats)        
        nn.init.xavier_normal_(self.np_embeddings.weight.data)
        
        self.inp_drop = torch.nn.Dropout(self.args.dropout)
        self.hidden_drop = torch.nn.Dropout(self.args.dropout)
        self.feature_map_drop = torch.nn.Dropout2d(self.args.dropout)
        self.loss = torch.nn.BCELoss()

        self.conv1 = torch.nn.Conv2d(1, 32, (3, 3))
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.bn2 = torch.nn.BatchNorm1d(self.args.nfeats)
        self.register_parameter('b', nn.Parameter(torch.zeros(self.args.num_nodes)))
        self.fc = torch.nn.Linear(16128,self.args.nfeats)        
        
    def forward(self, x, edges):
        return self.cn(x, edges)

    def get_scores(self,ent,rel,ent_embed,batch_size):
        
        ent = ent.view(-1, 1, 15, 20)
        rel = rel.view(-1, 1, 15, 20)

        stacked_inputs = torch.cat([ent, rel], 2)

        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, ent_embed.transpose(1,0))
        x += self.b.expand_as(x)
        return x
        

    def get_embed(self, edges, node_id, r):
        
        np_embed = self.np_embeddings(node_id)
        if self.args.CN != 'Phi':
        	np_embed = self.forward(np_embed, edges)


        r,r_len = seq_batch(r,self.args,self.rel2words)
        r_embed = self.phrase_embed_model(r,r_len)
        
        return r_embed, np_embed
    
    
    def get_loss(self,samples,labels,edges,node_id):

        np_embed = self.np_embeddings(node_id)
        if self.args.cn != 'Phi':
        	np_embed = self.forward(np_embed, edges)
        
        sub_embed = np_embed[samples[:,0]]
        r = samples[:,1]
        

        r_batch,r_len = seq_batch(r.cpu().numpy(), self.args, self.rel2words)
        rel_embed = self.phrase_embed_model(r_batch,r_len)
        
        scores = self.get_scores(sub_embed, rel_embed, np_embed, self.args.batch_size)
        pred = F.sigmoid(scores)
        
        predict_loss = self.loss(pred, labels)
        
        return predict_loss




def main(args):
	data = load_data(args)
	args.pad_id = data.word2id['<PAD>']
	args.num_nodes = len(data.ent2id)
	if torch.cuda.is_available(): args.use_cuda = True
	else: args.use_cuda = False

	model = ConvEParam(args,data.embed_matrix,data.rel2word)
	

	if args.use_cuda:
	    model.cuda()

	model_state_file = args.model_path

	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'max',factor = 0.5, patience = 2)

	
	train_pairs = list(data.label_graph.keys())

	train_id = np.arange(len(train_pairs))

	node_id = torch.arange(0, args.num_nodes, dtype=torch.long)
	edges = torch.tensor(data.edges,dtype=torch.long)
	if args.use_cuda: 
	    edges = edges.cuda()
	    node_id = node_id.cuda()

	best_MR = 20000
	best_MRR = 0
	best_epoch = 0
	count = 0
	for epoch in range(args.n_epochs):
	    model.train()
	    if count >= args.early_stop: break
	    epoch_loss = 0
	    permute = np.random.permutation(train_id)
	    train_id = train_id[permute]
	    n_batches = train_id.shape[0]//args.batch_size
	    
	    t1 = time.time()
	    for i in range(n_batches):
	        id_list = train_id[i*args.batch_size:(i+1)*args.batch_size]
	        samples,labels = get_next_batch(id_list, data, args, train_pairs)
	        
	        samples = Variable(torch.from_numpy(samples))
	        labels = Variable(torch.from_numpy(labels).float())
	        if args.use_cuda:
	            samples = samples.cuda()
	            labels = labels.cuda()
	        
	        optimizer.zero_grad()
	        loss = model.get_loss(samples,labels,edges,node_id)
	        loss.backward()
	        print("batch {}/{} batches, batch_loss: {}".format(i,n_batches,(loss.data).cpu().numpy()),end='\r')
	        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
	        optimizer.step()
	        epoch_loss += (loss.data).cpu().numpy()
	    print("epoch {}/{} total epochs, epoch_loss: {}".format(epoch+1,args.n_epochs,epoch_loss/n_batches))
	    
	    if (epoch + 1)%args.eval_epoch==0:
	        model.eval()
	        MR,MRR = evaluate(model, args.num_nodes, data.valid_trips, args, data)
	        if MRR>best_MRR or MR<best_MR:
	            count = 0
	            if MRR>best_MRR: best_MRR = MRR
	            if MR<best_MR: best_MR = MR
	            best_epoch = epoch + 1
	            torch.save({'state_dict': model.state_dict(), 'epoch': epoch},model_state_file)
	        else: count+=1
	        print("Best Valid MRR: {}, Best Valid MR: {}, Best Epoch: {}".format(best_MRR,best_MR,best_epoch))
	        scheduler.step(best_epoch)


	### Get Embeddings
	print("Test Set Evaluation ---")
	checkpoint = torch.load(model_state_file)
	model.eval()
	model.load_state_dict(checkpoint['state_dict'])
	_,_ = evaluate(model, args.num_nodes, data.test_trips, args, data)



if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='CaRe: Canonicalization Infused Representations for Open KGs')

	### Model and Dataset choice
	parser.add_argument('-CN',   dest='CN', default='LAN', choices=['LAN','GCN','GAT','Phi'], help='Choice of Canonical Cluster Encoder Network')
	parser.add_argument('-dataset', 	    dest='dataset', 	    default='ReVerb45K',choices=['ReVerb45K','ReVerb20K'],		            help='Dataset Choice')

	### Data Paths
	parser.add_argument('-data_path',       dest='data_path',       default='../Data', 			help='Data folder')

	#### Hyper-parameters
	parser.add_argument('-nfeats',      dest='nfeats',       default=300,   type=int,       help='Embedding Dimensions')
	parser.add_argument('-nheads',      dest='nheads',       default=3,     type=int,       help='multi-head attantion in GAT')
	parser.add_argument('-num_layers',  dest='num_layers',   default=1,     type=int,       help='No. of layers in encoder network')
	parser.add_argument('-bidirectional',  dest='bidirectional',   default=True,     type=bool,       help='type of encoder network')
	parser.add_argument('-poolType',    dest='poolType',     default='last',choices=['last','max','mean'], help='pooling operation for encoder network')
	parser.add_argument('-dropout',     dest='dropout',      default=0.5,   type=float,     help='Dropout')
	parser.add_argument('-reg_param',   dest='reg_param',    default=0.0,   type=float,     help='regularization parameter')
	parser.add_argument('-lr',          dest='lr',           default=0.001, type=float,     help='learning rate')
	parser.add_argument('-p_norm',      dest='p_norm',       default=1,     type=int,       help='TransE scoring function')
	parser.add_argument('-batch_size',  dest='batch_size',   default=128,   type=int,       help='batch size for training')
	parser.add_argument('-neg_samples', dest='neg_samples',  default=10,    type=int,       help='No of Negative Samples for TransE')
	parser.add_argument('-n_epochs',    dest='n_epochs',     default=500,   type=int,       help='maximum no. of epochs')
	parser.add_argument('-grad_norm',   dest='grad_norm',    default=1.0,   type=float,     help='gradient clipping')
	parser.add_argument('-eval_epoch',  dest='eval_epoch',   default=5,     type=int,       help='Interval for evaluating on validation dataset')
	parser.add_argument('-Hits',        dest='Hits',         default= [10,30,50],           help='Choice of n in Hits@n')
	parser.add_argument('-early_stop',  dest='early_stop',   default=10,    type=int,       help='Stopping training after validation performance stops improving')
	

	args = parser.parse_args()


	args.data_files = {
	'ent2id_path'       : args.data_path + '/' + args.dataset + '/ent2id.txt',
	'rel2id_path'       : args.data_path + '/' + args.dataset + '/rel2id.txt',
	'train_trip_path'   : args.data_path + '/' + args.dataset + '/train_trip.txt',
	'test_trip_path'    : args.data_path + '/' + args.dataset + '/test_trip.txt',
	'valid_trip_path'   : args.data_path + '/' + args.dataset + '/valid_trip.txt',
	'gold_npclust_path' : args.data_path + '/' + args.dataset + '/gold_npclust.txt',
	'cesi_npclust_path' : args.data_path + '/' + args.dataset + '/cesi_npclust.txt',
	'glove_path'        : 'glove/glove.6B.300d.txt'
	}

	args.model_path = "ConvE" + "-" + args.CN + "_modelpath.pth"


	main(args)
