import unittest

from engthesis.embeddings.node.grarep import GraRep
from engthesis.graphs.examples import examplesDict

class GraRepTestCase(unittest.TestCase):
    def testEmbedsWithoutError(self):
        for graph in examplesDict.values():
            model = GraRep(graph)
            model.embed()
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
