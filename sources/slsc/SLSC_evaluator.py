import string

from libmg import PredictPerformance, CallPerformance
from sources.slsc.SLSC import build_model
from sources.datasets.simplicial_complex_dataset import AdjacencyRelation, random_maximal_simplicial_complex, \
    SimplicialComplexDataset
from libmg import MultipleGraphLoader

if __name__ == '__main__':
    dataset = SimplicialComplexDataset('testing_dataset9', list(string.ascii_lowercase),
                                       random_maximal_simplicial_complex(10, list(string.ascii_lowercase)),
                                       AdjacencyRelation.LOWER_ADJACENCY)
    PredictPerformance(lambda dataset: build_model(dataset, ['N a']),
                       lambda dataset: MultipleGraphLoader(dataset, node_level=True, batch_size=1, shuffle=False,
                                                           epochs=1))(dataset)
    CallPerformance(lambda dataset: build_model(dataset, ['N a']),
                    lambda dataset: MultipleGraphLoader(dataset, node_level=True, batch_size=1, shuffle=False,
                                                        epochs=1))(dataset)
