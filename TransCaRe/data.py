import pathlib
import numpy as np

class load_data():
    def __init__(self, args):
        self.args = args
        self.data_files = self.args.data_files
        
        
        self.fetch_data()
        
    def get_phrases(self,file_path):
        f = open(file_path,"r").readlines()
        phrase2id = {}
        id2phrase = {}
        word2id = set()
        for line in f[1:]:
            phrase,ID = line.strip().split("\t")
            phrase2id[phrase] = int(ID)
            id2phrase[int(ID)] = phrase
            for word in phrase.split():
                word2id.add(word)
        return phrase2id,id2phrase,word2id
    
    def get_phrase2word(self,phrase2id,word2id):
        phrase2word = {}
        for phrase in phrase2id:
            words = []
            for word in phrase.split(): words.append(word2id[word])
            phrase2word[phrase2id[phrase]] = words
        return phrase2word

    
    def get_word_embed(self,word2id,GLOVE_PATH):
        word_embed = {}
        if pathlib.Path(GLOVE_PATH).is_file():
            print("utilizing pre-trained word embeddings")
            with open(GLOVE_PATH, encoding="utf8") as f:
                for line in f:
                    word, vec = line.split(' ', 1)
                    if word in word2id:
                        word_embed[word2id[word]] = np.fromstring(vec, sep=' ')
        else: print("word embeddings are randomly initialized")

        wordkeys = word2id.values()
        a = [words for words in wordkeys if words not in word_embed]
        for word in a:
            word_embed[word] = np.random.normal(size = 300)

        self.embed_matrix  = np.zeros((len(word_embed),300))
        for word in word_embed:
            self.embed_matrix[word] = word_embed[word]
            
    def get_train_triples(self,triples_path,entid2clustid):
        trip_list = []
        self.H_filter = {}
        self.T_filter = {}
        f = open(triples_path,"r").readlines()
        for trip in f[1:]:
            trip = trip.strip().split()
            e1,r,e2 = int(trip[0]),int(trip[1]),int(trip[2])
            
            if (r,e2) not in self.H_filter:
                self.H_filter[(r,e2)] = set()
            self.H_filter[(r,e2)].add(entid2clustid[e1])
            
            if (r,e1) not in self.T_filter:
                self.T_filter[(r,e1)] = set()
            self.T_filter[(r,e1)].add(entid2clustid[e2])

            trip_list.append([e1,r,e2])
        return np.array(trip_list)
    
    def get_test_triples(self, triples_path):
        trip_list = []
        f = open(triples_path,"r").readlines()
        for trip in f[1:]:
            trip = trip.strip().split()
            e1,r,e2 = int(trip[0]),int(trip[1]),int(trip[2])
            trip_list.append([e1,r,e2])
        return np.array(trip_list)
    
    def get_clusters(self,clust_path):
        ent_clusts = {}
        entid2clustid = {}
        ent_list = []
        unique_clusts = []
        ID = 0
        f = open(clust_path,"r").readlines()
        for line in f:
            line = line.strip().split()
            clust = [int(ent) for ent in line[2:]]
            ent_clusts[int(line[0])] = clust
            if line[0] not in ent_list: 
                ID+=1
                unique_clusts.append(clust)
                ent_list.extend(line[2:])
                for ent in clust: entid2clustid[ent] = ID
        return ent_clusts,entid2clustid,unique_clusts
    
    def get_edges(self,ent_clusts):
        head_list = []
        tail_list = []
        for ent in ent_clusts:
            if self.args.model_variant=='CaRe' and len(ent_clusts[ent])==1:
                head_list.append(ent)
                tail_list.append(ent)
            for neigh in ent_clusts[ent]:
                if neigh!=ent:
                    head_list.append(neigh)
                    tail_list.append(ent)
                    
        head_list = np.array(head_list).reshape(1,-1)
        tail_list = np.array(tail_list).reshape(1,-1)
        
        self.edges = np.concatenate((np.array(head_list),np.array(tail_list)),axis = 0)
    
    
    def fetch_data(self):
        self.rel2id,self.id2rel,self.word2id = self.get_phrases(self.data_files["rel2id_path"])
        self.ent2id,self.id2ent,_ = self.get_phrases(self.data_files["ent2id_path"])
        
        self.canon_clusts,_,self.unique_clusts = self.get_clusters(self.data_files["cesi_npclust_path"])
        self.true_clusts, self.entid2clustid,_ = self.get_clusters(self.data_files["gold_npclust_path"])
        
        self.get_edges(self.canon_clusts)
        
        self.train_trips,self.rel2id,self.label_graph = self.get_train_triples(self.data_files["train_trip_path"],
                                                                               self.entid2clustid)

        self.test_trips = self.get_test_triples(self.data_files["test_trip_path"])
        self.valid_trips = self.get_test_triples(self.data_files["valid_trip_path"])
        
        
        self.word2id = {word:index for index,word in enumerate(list(self.word2id))}
        self.word2id['<PAD>'] = len(self.word2id)

        self.rel2word = self.get_phrase2word(self.rel2id,self.word2id)

        self.get_word_embed(self.word2id,self.data_files["glove_path"])


