# CaRE
## CaRe: Open Knowledge Graph Embeddings

![](https://github.com/malllabiisc/CaRE/blob/master/CaRe_model.png)
*Overview of CaRe. CaRe learns KG embeddings from the augmented OpenKG. Base model can be any existing KG embedding model (e.g., TransE, ConvE). RP embeddings are parameterized by encoding vector representations of the word sequence composing them. This enables CaRe to capture semantic similarity of RPs. Embeddings of NPs are made more context rich by updating them with the represenations of canonical NPs (connected with dotted lines).*
### Dependencies

* Compatible with Pytorch 1.1 and Python 3.x.
* Dependencies can be installed using `requirements.txt`.
