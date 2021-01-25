# Copyright

<ins>That software is currently unlicensed and, therefore, cannot be used in any way</ins>. We will license it under the GPL 3.0 license as soon as our thesis is reviewed and we obtain the proper permissions of the Dean of Faculty of Mathematics and Information Science, WUT.


# ReLeGy - Representation Learning of Graphs in Python
The ReLeGy package offers multiple methods of embedding graph vertices into vector spaces. Currently supported methods are:
* Laplacian Eigenmaps [[1]](#1)
* Graph Factorization [[2]](#2)
* GraRep [[3]](#3)
* HOPE [[4]](#4)
* DeepWalk [[5]](#5)
* Node2Vec [[6]](#6)
* LINE [[7]](#7)
* HARP for DeepWalk and Node2Vec [[8]](#8)
* SDNE [[9]](#9)
* DNGR [[10]](#10)
* Struc2Vec [[11]](#11)
* GraphWave [[12]](#12)
* GNN [[13]](#13)
* GCN [[14]](#14)

## Installation

Installing relegy is simple, use 

```
pip3 install git+https://github.com/lukaszbrzozowski/ReLeGy.git
```

## Examples

Let's start with generating an example graph we wish to embed. To to this, we use one of the methods implemented in the **graphs** module:
```python
import relegy.graphs as rlg

G = rlg.generate_clusters_graph(n=100, k=4, out_density=0.05, in_density=0.6)
```

We generated a graph consisting of 4 clusters having 100 degrees in total. The cluster density is 0.6 and the between-cluster density is 0.05.

We may now proceed to the embedding process. Each method offers two interfaces. Let's assume that we wish to embed the graph G with HOPE in 4-dimensional space. We may generate the embedding using a static function:
```python
import relegy.embeddings as rle

Z = rle.HOPE.fast_embed(G, d=4) # other parameters may be passed here
```
We may divide the process into stages:
```python
hope = rle.HOPE(G)

hope.initialize()
hope.fit()
Z = hope.embed(d=4)

```
The output of both the above chunks will be the same. However, using stages means that you do not lose the progress achieved in the previous steps. That is, if we wished to change the embedding dimension to 5 instead of 4, we could just run:
```python
Z = hope.embed(d=5)
```
As the __d__ parameter for HOPE method is passed in the **embed** stage, that means that we do not have to rerun the previous steps, just **embed**. In general, sometimes you may find that the same parameter for different methods is passed in different stages - this is because we aimed to minimize the loss of embedding progress when changing the parameter's value.

## References
<a id="1">[1]</a> 
Mikhail Belkin and Partha Niyogi. Laplacian eigenmaps and spectral techniques for em-bedding and clustering. In Advances in Neural Information Processing Systems 14 \[Neural Information Processing Systems: Natural and Synthetic\], NIPS 2001, December 3-8, 2001, Vancouver, British Columbia, Canada], pages 585–591. MIT Press, 2001.

<a id="2">[2]</a> 
Amr Ahmed, Nino Shervashidze, Shravan M. Narayanamurthy, Vanja Josifovski, and Alexander J. Smola. Distributed large-scale natural graph factorization. In 22nd International World Wide Web Conference, WWW ’13, Rio de Janeiro, Brazil, May 13-17, 2013, pages 37–48. International World Wide Web Conferences Steering Committee / ACM, 2013.

<a id="3">[3]</a> 
Shaosheng Cao, Wei Lu, and Qiongkai Xu. GraRep: Learning graph representations with global structural information. In Proceedings of the 24th ACM International Conference on Information and Knowledge Management, CIKM 2015, Melbourne, VIC, Australia, October 19 - 23, 2015, pages 891–900. ACM, 2015.

<a id="4">[4]</a> 
Mingdong Ou, Peng Cui, Jian Pei, Ziwei Zhang, and Wenwu Zhu. Asymmetric transitivity preserving graph embedding. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, KDD ’16, pages 1105–1114, 2016.

<a id="5">[5]</a> 
Bryan Perozzi, Rami Al-Rfou, and Steven Skiena. DeepWalk: Online learning of social representations. CoRR, abs/1403.6652, 2014.

<a id="6">[6]</a> 
Aditya Grover and Jure Leskovec. Node2Vec: Scalable feature learning for networks. CoRR, abs/1607.00653, 2016.

<a id="7">[7]</a> 
Jian Tang, Meng Qu, Mingzhe Wang, Ming Zhang, Jun Yan, and Qiaozhu Mei.  LINE: Large-scale information network embedding. CoRR, abs/1503.03578, 2015.

<a id="8">[8]</a> 
Haochen Chen, Bryan Perozzi, Yifan Hu, and Steven Skiena. HARP: Hierarchical representation learning for networks. CoRR, abs/1706.07845, 2017

<a id="9">[9]</a> 
Daixin Wang, Peng Cui, and Wenwu Zhu. Structural deep network embedding. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, San Francisco, CA, USA, August 13-17, 2016, pages 1225–1234. ACM, 2016.

<a id="10">[10]</a> 
Shaosheng Cao, Wei Lu, and Qiongkai Xu. Deep neural networks for learning graph representations. In Proceedings of the Thirtieth AAAI Conference on Artificial Intelligence, February 12-17, 2016, Phoenix, Arizona, USA, pages 1145–1152. AAAI Press, 2016.

<a id="11">[11]</a> 
Daniel R. Figueiredo, Leonardo Filipe Rodrigues Ribeiro, and Pedro H. P. Saverese. Struc2Vec: Learning node representations from structural identity. CoRR, abs/1704.03165,2017.

<a id="12">[12]</a> 
Claire Donnat, Marinka Zitnik, David Hallac, and Jure Leskovec. Learning structural node embeddings via diffusion wavelets. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, KDD 2018, London, UK, August 19-23, 2018, pages 1320–1329. ACM, 2018.

<a id="13">[13]</a> 
Franco Scarselli, Marco Gori, Ah Chung Tsoi, Markus Hagenbuchner, and Gabriele Monfardini. The graph neural network model. IEEE Trans. Neural Networks, 20(1):61–80, 2009.

<a id="14">[14]</a> 
Thomas N. Kipf and Max Welling. Semi-supervised classification with graph convolutional networks. CoRR, abs/1609.02907, 2016.
