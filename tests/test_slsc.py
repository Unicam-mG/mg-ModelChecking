import os
import tensorflow as tf
from sources.slsc.SLSC import build_model, to_mG
from sources.slsc.datasets.simplicial_complex_dataset import SimplicialComplexDataset, AdjacencyRelation, \
    random_maximal_simplicial_complex
from libmg import MultipleGraphLoader


class SLSCToMgTest(tf.test.TestCase):
    def test_parse(self):
        self.assertEqual('try', to_mG('try'))
        self.assertEqual('true', to_mG('true'))
        self.assertEqual('false', to_mG('false'))
        self.assertEqual('(try);not', to_mG('not try'))
        self.assertEqual('((try) || (succ));and', to_mG('try and succ'))
        self.assertEqual('((try) || (succ));or', to_mG('try or succ'))
        self.assertEqual('(try);|p3>or', to_mG('N try'))
        self.assertEqual('mu X,b . (((((try) || (X;|p3>or));and) || (succ));or)', to_mG('try R succ'))


class SimplicialComplexTest(tf.test.TestCase):

    def setUp(self):
        super().setUp()
        self.formulae = ['true', 'false', 'a', 'not a', 'a and b', 'a or b', 'N a', 'a R b']
        self.atomic_propositions = ['a', 'b', 'c']
        self.datasets = [SimplicialComplexDataset('test_dataset_lower', self.atomic_propositions,
                                                  random_maximal_simplicial_complex(10, self.atomic_propositions),
                                                  AdjacencyRelation.LOWER_ADJACENCY),
                         SimplicialComplexDataset('test_dataset_upper', self.atomic_propositions,
                                                  random_maximal_simplicial_complex(10, self.atomic_propositions),
                                                  AdjacencyRelation.UPPER_ADJACENCY),
                         SimplicialComplexDataset('test_dataset_spatial', self.atomic_propositions,
                                                  random_maximal_simplicial_complex(10, self.atomic_propositions),
                                                  AdjacencyRelation.SPATIAL_ADJACENCY),
                         ]
        self.model = build_model(self.datasets[0], self.formulae)

    def test_run(self):
        for dataset in self.datasets:
            d_loader = MultipleGraphLoader(dataset, node_level=True, batch_size=1, shuffle=False, epochs=1)
            for inputs, y in d_loader.load():
                self.model.call([inputs], training=False)


if __name__ == '__main__':
    tf.test.main()
