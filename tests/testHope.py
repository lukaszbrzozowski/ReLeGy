import unittest

from engthesis.embeddings.node.hope import HOPE
from engthesis.graphs.examples import examplesDict

class HopeTestCase(unittest.TestCase):
    def testEmbedsWithoutError(self):
        for graph in examplesDict.values():
            model = HOPE(graph)
            model.embed()
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
