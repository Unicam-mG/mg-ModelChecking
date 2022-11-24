import os
import itertools
from typing import Dict, Set, Tuple, List, Any

import numpy as np
import random
from enum import auto, Enum
from gudhi.simplex_tree import SimplexTree
from scipy.sparse import coo_matrix
from spektral.data import Graph
from spektral.utils import reorder
from libmg import Dataset

from sources.datasets.dataset_utils import get_np_data_type


class AdjacencyRelation(Enum):
    LOWER_ADJACENCY = auto()
    UPPER_ADJACENCY = auto()
    SPATIAL_ADJACENCY = auto()


def lower_adjacency(s1, s2, _):
    return len(s1 & s2) == (len(s1) - 1) == (len(s2) - 1)


def upper_adjacency(s1, s2, st):
    s3 = s1 | s2
    return (len(s3) == len(s1) + 1 == len(s2) + 1) and st.find(s3)


def spatial_adjacency(s1, s2, _):
    return len(s1 & s2) != 0


adjacency_relations = {
    AdjacencyRelation.LOWER_ADJACENCY: lower_adjacency,
    AdjacencyRelation.UPPER_ADJACENCY: upper_adjacency,
    AdjacencyRelation.SPATIAL_ADJACENCY: spatial_adjacency
}


def random_maximal_simplicial_complex(max_dimension, atomic_propositions_set):
    def powerset(iterable):
        """powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3) without empty set"""
        s = list(iterable)
        return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(1, len(s) + 1))

    vertices = {i for i in range(max_dimension + 1)}
    simplices = sorted(powerset(vertices), key=lambda x: len(x), reverse=True)
    labelled_complex = {(i, frozenset(random.sample(atomic_propositions_set,
                                                    k=random.randint(0, len(atomic_propositions_set))))):
                        powerset(simplex) for i, simplex in enumerate(simplices)}
    output: Dict[Tuple[int], Set[str]] = {simplex: set() for simplex in simplices}
    for i, label in labelled_complex:
        for simplex in labelled_complex[(i, label)]:
            output[simplex] |= label
    return output


class SimplicialComplexDataset(Dataset):

    def __init__(self, name, atomic_propositions_set, simplices, adjacency_relation: AdjacencyRelation, **kwargs):
        self._atomic_propositions_set = sorted(atomic_propositions_set)
        self.simplices = simplices
        self.adjacency_relation = adjacency_relation
        super().__init__(name, **kwargs)

    def download(self):
        os.makedirs(self.path)
        st = SimplexTree()
        for simplex in self.simplices:
            st.insert(simplex)
        simplices_with_id_by_rank: Dict[int, List[Tuple[Set[str], int]]] = {i: [] for i in range(st.dimension() + 1)}

        print("Generating x feature vector")
        # Node features (x) are a multi-hot representation of atomic actions
        if len(self.atomic_proposition_set) > 64:
            x = np.zeros((st.num_simplices(), len(self.atomic_proposition_set)), dtype=np.uint8)
            for i, (simplex, _) in enumerate(st.get_simplices()):
                simplices_with_id_by_rank[len(simplex) - 1].append((set(simplex), i))
                labels = self.simplices.get(tuple(simplex), [])
                for label in labels:
                    x[i][self.atomic_proposition_set.index(label)] = 1
        else:
            tmp_x = []
            for i, (simplex, _) in enumerate(st.get_simplices()):
                simplices_with_id_by_rank[len(simplex) - 1].append((set(simplex), i))
                labels = self.simplices.get(tuple(simplex), [])
                indices = [self.atomic_proposition_set.index(label) for label in labels]
                tmp_x.append([sum(2 ** idx for idx in indices)])
            x = np.array(tmp_x, dtype=get_np_data_type(self.atomic_proposition_set))

        print("Generating adjacency matrix")
        adj = adjacency_relations[self.adjacency_relation]
        data = []
        idx = []

        if self.adjacency_relation is AdjacencyRelation.UPPER_ADJACENCY:
            ranks = range(st.dimension())
        else:
            ranks = range(1, st.dimension() + 1)
        for i in ranks:
            for (s1, id1), (s2, id2) in itertools.combinations(simplices_with_id_by_rank[i], 2):
                if adj(s1, s2, st):
                    idx.append([id1, id2])
                    idx.append([id2, id1])
                    data.append(1)
                    data.append(1)
        new_idx = np.swapaxes(reorder(np.array(idx))[0], 0, 1)
        a = coo_matrix((data, (new_idx[0], new_idx[1])), shape=(st.num_simplices(), st.num_simplices()), dtype=np.uint8)

        y: List[List[Any]] = [[] for _ in range(st.num_simplices())]

        filename = os.path.join(self.path, 'Graph_' + str(self.adjacency_relation.name))
        np.savez(filename, x=x, a=a, y=y)

    def read(self):
        examples = []
        for file in os.scandir(self.path):
            loaded_graph = np.load(file.path, allow_pickle=True)
            examples.append(Graph(x=loaded_graph['x'],
                                  a=loaded_graph['a'].item(),
                                  y=loaded_graph['y']))
        return examples

    @property
    def path(self):
        return os.path.join(super().path, self.name)

    @property
    def atomic_proposition_set(self):
        return self._atomic_propositions_set
