import unittest

from engthesis.embeddings.node.deepwalk import DeepWalk
from engthesis.graphs.examples import examplesDict

class GraphFactorizationTestCase(unittest.TestCase):
    def testEmbedsWithoutError(self):
        for graph in examplesDict.values():
            model = DeepWalk(graph)
            model.embed()
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
