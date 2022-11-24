import os
import tensorflow as tf
from sources.logics.slsc.SLSC import build_model
from sources.datasets.simplicial_complex_dataset import SimplicialComplexDataset, AdjacencyRelation, \
    random_maximal_simplicial_complex
from libmg import MultipleGraphLoader

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "0"


class CudaTest(tf.test.TestCase):
    def setUp(self):
        super(CudaTest, self).setUp()

    def test_cuda(self):
        self.assertEqual(tf.test.is_built_with_cuda(), True)


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
