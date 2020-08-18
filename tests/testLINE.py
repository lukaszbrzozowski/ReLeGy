import unittest

from engthesis.embeddings.node.line import LINE
from engthesis.graphs.examples import examplesDict

class LINETestCase(unittest.TestCase):
    def testEmbedsWithoutError(self):
        for graph in examplesDict.values():
            model = LINE(graph)
            model.embed()
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
