import unittest

from engthesis.embeddings.node.node2vec import Node2Vec
from engthesis.graphs.examples import examplesDict

class Node2VecTestCase(unittest.TestCase):
    def testEmbedsWithoutError(self):
        for graph in examplesDict.values():
            model = Node2Vec(graph)
            model.embed()
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
