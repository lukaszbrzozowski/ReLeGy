from relegy.embeddings.node.laplacianembeddings import LaplacianEmbeddings
from relegy.embeddings.node.graphfactorization import GraphFactorization
from relegy.embeddings.node.grarep import GraRep
from relegy.embeddings.node.hope import HOPE
import networkx as nx

#Data preparation
G1 = nx.barbell_graph(50, 0)
G2 = nx.complete_graph(100)
G3 = nx.random_graphs.erdos_renyi_graph(100, 0.2)
print("Graphs generated")
#LE
le1 = LaplacianEmbeddings(G1, d=5)
le2 = LaplacianEmbeddings(G2, d=5)
le3 = LaplacianEmbeddings(G3, d=5)

le1.embed()
print("LE1 embedded successfully")
le2.embed()
print("LE2 embedded successfully")
le3.embed()
print("LE3 embedded successfully")
#GF
gf1 = GraphFactorization(G1, d=5)
gf2 = GraphFactorization(G2, d=5)
gf3 = GraphFactorization(G3, d=5)

gf1.embed()
print("GF1 embedded successfully")
gf2.embed()
print("GF2 embedded successfully")
gf3.embed()
print("GF3 embedded successfully")

#GraRep
gp1 = GraRep(G1, d=5, K=3)
gp2 = GraRep(G2, d=5, K=3)
gp3 = GraRep(G3, d=5, K=3)

gp1.embed()
print("GraRep1 embedded successfully")
gp2.embed()
print("GraRep2 embedded successfully")
gp3.embed()
print("GraRep3 embedded successfully")

#HOPE
hope1_1 = HOPE(G1, d=5, proximity="Katz")
hope1_2 = HOPE(G1, d=5, proximity="RPR")
hope1_3 = HOPE(G1, d=5, proximity="AA")
hope1_4 = HOPE(G1, d=5, proximity="CN")

hope2_1 = HOPE(G2, d=5, proximity="Katz")
hope2_2 = HOPE(G2, d=5, proximity="RPR")
hope2_3 = HOPE(G2, d=5, proximity="AA")
hope2_4 = HOPE(G2, d=5, proximity="CN")

hope3_1 = HOPE(G3, d=5, proximity="Katz")
hope3_2 = HOPE(G3, d=5, proximity="RPR")
hope3_3 = HOPE(G3, d=5, proximity="AA")
hope3_4 = HOPE(G3, d=5, proximity="CN")

hope1_1.embed()
print("HOPE1 Katz embedded successfully")
hope1_2.embed()
print("HOPE1 RPR embedded successfully")
hope1_3.embed()
print("HOPE1 AA embedded successfully")
hope1_4.embed()
print("HOPE1 CN embedded successfully")

hope2_1.embed()
print("HOPE2 Katz embedded successfully")
hope2_2.embed()
print("HOPE2 RPR embedded successfully")
hope2_3.embed()
print("HOPE2 AA embedded successfully")
hope2_4.embed()
print("HOPE2 CN embedded successfully")

hope3_1.embed()
print("HOPE3 Katz embedded successfully")
hope3_2.embed()
print("HOPE3 RPR embedded successfully")
hope3_3.embed()
print("HOPE3 AA embedded successfully")
hope3_4.embed()
print("HOPE3 CN embedded successfully")

