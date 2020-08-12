import unittest

from engthesis.embeddings.node.laplacianembeddings import LaplacianEmbeddings
from engthesis.graphs.examples import examplesDict

class MyTestCase(unittest.TestCase):
    def testEmbedsWithoutError(self):
        for graph in examplesDict.values():
            model = LaplacianEmbeddings(graph)
            model.embed()
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
