# CaRe
## CaRe: Open Knowledge Graph Embeddings

Source code and dataset for [EMNLP 2019](https://www.emnlp-ijcnlp2019.org/) paper: [CaRe: Open Knowledge Graph Embeddings](http://talukdar.net/papers/CaRe_EMNLP2019.pdf).

![](https://github.com/malllabiisc/CaRE/blob/master/CaRe_model.png)
*Overview of CaRe. CaRe learns KG embeddings from the augmented OpenKG. Base model can be any existing KG embedding model (e.g., TransE, ConvE). RP embeddings are parameterized by encoding vector representations of the word sequence composing them. This enables CaRe to capture semantic similarity of RPs. Embeddings of NPs are made more context rich by updating them with the represenations of canonical NPs (connected with dotted lines). A generic nomenclature for CaRe framework is defined as CaRe(B,PN,CN). We define Bi-GRU and LAN as default val- ues for the PN and CN arguments respectively. Please refer to the paper for more details.* 
### Dependencies:

* Compatible with Pytorch 1.1 and Python 3.x.
* Dependencies can be installed using `requirements.txt`.

### Dataset:
* Pre-processed datasets ReVerb45k, ReVerb20K are included with the repository present in the `Data` directory.
* The datasets are originally taken from [CESI](https://github.com/malllabiisc/cesi).
* Both the datasets contain the following files:

  ent2id.txt: all noun phrases and corresponding ids, one per line. The first line is the number of noun phrases.

  rel2id.txt: all relation phrases and corresponding ids, one per line. The first line is the number of relations.

  train_trip.txt: training file, the first line is the number of triples for training. Then the following lines are all in the format ***(s,r,o)*** which indicates there is a relation ***rel*** between ***s*** and ***o*** .
  **Note that train_trip.txt contains ids from ent2id.txt and rel2id.txt instead of the actual noun and relation phrases.**
  
  test_trip.txt: testing file, the first line is the number of triples for testing. Then the following lines are all in the format ***(s,r,o)*** .

  valid_trip.txt: validating file, the first line is the number of triples for validating. Then the following lines are all in the format ***(s,r,o)*** .
  
  cesi_npclust.txt: The noun phrase canonicalization output of [CESI](https://github.com/malllabiisc/cesi). Each line corresponds to the canonicalization information of a noun phrase in the following format ***(NP_id, no. of canonical NPs, list ids of canonical NPs)*** . 
  
  gold_npclust.txt: The ground truth noun phrase canonicalization information. This information is used during evaluations. Each line corresponds to the canonicalization information of a noun phrase in the following format ***(NP_id, no. of canonical NPs, list ids of canonical NPs)*** .
  
  
### Usage:
Any existing KG embedding model can used in the CaRe framework. Codes for the following Base Models (B) is provided:
* ConvE in the directory `CaRe(B=ConvE)`.
* TransE in the directory `CaRe(B=TransE)`.
* Some of the important available options include:
  ```shell
  '-CN',   dest='CN', default='LAN', choices=['LAN','GCN','GAT','Phi'], help='Choice of Canonical Cluster Encoder Network'
  '-dataset', dest='dataset', default='ReVerb45K',choices=['ReVerb45K','ReVerb20K'], help='Dataset Choice'
  '-nfeats', dest='nfeats', default=300, type=int, help='Embedding Dimensions'
  '-bidirectional',  dest='bidirectional', default=True, type=bool, help='type of encoder network'
  '-poolType', dest='poolType', default='last', choices=['last','max','mean'], help='pooling operation for encoder network'
  ```

##### Run the main code:
* After installing python dependencies, execute `sh setup.sh` to download pre-trained glove embeddings. (Experiments can be run without this step as well. In that case word vectors would get randomly initialized.)
* Move to the directory of corresponding to the choice of Base model and execute: `python CaRe_main.py -CN LAN -dataset ReVerb45K`.


