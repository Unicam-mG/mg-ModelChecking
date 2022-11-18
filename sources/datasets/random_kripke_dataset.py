import random
import os
import networkx as nx
import numpy as np
import pickle
import itertools
from spektral.data import Dataset
from pyModelChecking import Kripke

from sources.datasets.dataset_utils import ctl_to_mu_formulae, get_graphs, create_graphs


def generate_random_graph(max_possible_nodes, max_edges_per_node, force_max):
    """
    Generates a random weakly-connected digraph with or without deadlock that can be considered an LTS for some system
    :param max_possible_nodes: Sets an upper bound to the number of nodes to generate.  Must be >= 2
    :param max_edges_per_node: Sets an upper bound to the number of outgoing edges for any given node. Must be >= 1
    :param force_max: Generate max nodes
    :return: A random weakly-connected directed networkx Graph
    """
    if max_possible_nodes <= 1:
        raise ValueError("Argument max_possible_nodes must be >= 2")
    if max_edges_per_node <= 0:
        raise ValueError("Argument max_possible_out_edges_per_node must be >= 1")
    print("Generating networkx graph")

    # Select the number of nodes in the graph
    n_nodes = max_possible_nodes if force_max else random.randint(2, max_possible_nodes)
    # Generate the in and out degrees of the nodes in the graph
    degrees = [random.randint(1, max_edges_per_node) for _ in range(n_nodes)]
    # The sum of these two lists must match
    nodes_in_degrees = []
    nodes_out_degrees = []
    # Set random in and out degrees
    nodes_in_degrees.extend(random.sample(degrees, len(degrees)))
    nodes_out_degrees.extend(random.sample(degrees, len(degrees)))

    # Generate the directed graph
    g = nx.generators.degree_seq.directed_configuration_model(nodes_in_degrees, nodes_out_degrees, nx.DiGraph)
    return g


def networkx_to_kripke(g, initial_state, graph_labels):
    """
    Convert a networkx directed Graph to Aldebaran format
    :param g: A networkx directed Graph
    :param initial_state: Initial state of the graph, numbered from 0 to n_states - 1
    :return: A kripke structure object
    """
    graph_labels = iter(graph_labels)
    print("Generating adjacency list 1")
    adjacency_list = [row.split() for row in nx.generate_adjlist(g)]
    print("Generating adjacency list 2")
    adjacency_list = [[(int(row[0]), int(i)) for i in row[1:]] for row in adjacency_list]
    print("Generating adjacency list 3")
    adjacency_list = list(filter(None, adjacency_list))
    print("Generating transitions")
    transitions = list(itertools.chain.from_iterable(adjacency_list))

    print("Generating Kripke Structure")
    K = Kripke(S=list(range(g.order())),
               S0=[initial_state],
               R=transitions,
               L={i: [next(graph_labels) for _ in range(g.out_degree(i))]
                  for i in range(g.order())})
    return K


def networkx_to_lts(g, initial_state, graph_labels):
    """
    Convert a networkx directed Graph to LTS in Aldebaran format
    :param g: A networkx directed Graph
    :param initial_state: Initial state of the graph, numbered from 0 to n_states - 1
    :return: A string representing the input graph in Aldebaran (.aut) format
    """
    graph_labels = iter(graph_labels)
    aut_header = 'des (' + str(initial_state) + ',' + str(g.size()) + ',' + str(g.order()) + ')\n'
    adj_list = list(nx.generate_adjlist(g))
    edges_list = [row.split() for row in adj_list]
    edges_list = ['(' + row[0] + ',\"' + next(graph_labels) + '\",' + i + ')\n' for row in edges_list for i in row[1:]]
    aut_edges = ''.join(edges_list)
    return aut_header + aut_edges


class RandomKripkeDataset(Dataset):
    def __init__(self, examples, max_possible_nodes, max_edges_per_node, atomic_propositions_set, formulae=None,
                 name=None, skip_model_checking=False, probabilistic=False, **kwargs):
        self.examples = examples
        self.max_possible_nodes = max_possible_nodes
        self.max_edges_per_node = max_edges_per_node
        self._atomic_propositions_set = sorted(atomic_propositions_set)
        self._formulae = formulae or []
        self._mu_formulae = ctl_to_mu_formulae(self._formulae)
        self._name = name
        self.skip_model_checking = skip_model_checking or not formulae or probabilistic
        self.probabilistic = probabilistic
        self.string_params = '_'.join([str(self.name), str(self.examples), str(self.max_possible_nodes),
                                       str(self.max_edges_per_node), str(self.probabilistic),
                                       str(self.skip_model_checking)])
        super().__init__(**kwargs)

    def download(self):
        os.makedirs(self.path)
        os.mkdir(self.kripke_path)
        os.mkdir(self.lts_path)

        generated_graphs = 0
        while generated_graphs < self.examples:
            print("Generated graphs: " + str(generated_graphs))
            # Generate a random networkx graph
            g = generate_random_graph(self.max_possible_nodes, self.max_edges_per_node, True)
            # Generate sequence of atomic propositions / action labels equal to the number of edges
            graph_labels = random.choices(self.atomic_proposition_set, k=g.size())

            # Convert the graph to a LTS, with initial state 0 and the given sequence of action labels
            lts = networkx_to_lts(g, 0, graph_labels)
            # Save the LTS to file
            with open(os.path.join(self.lts_path, 'LTS_' + str(generated_graphs)), 'wb') as f:
                pickle.dump(lts, f)

            del lts

            # Convert the graph to a Kripke structure, with initial state 0 and given sequence of atomic props.
            kripke_structure = networkx_to_kripke(g, 0, graph_labels)
            # Save the Kripke structure to file
            with open(os.path.join(self.kripke_path, 'Kripke_' + str(generated_graphs)), 'wb') as f:
                pickle.dump(kripke_structure, f)

            x, a, y = create_graphs(kripke_structure, self._atomic_propositions_set, self._formulae,
                                    self.skip_model_checking, self.probabilistic,)

            # We can now save the graph to file as npz
            filename = os.path.join(self.path, 'Graph_' + str(generated_graphs))
            if self.skip_model_checking:
                np.savez(filename, x=x, a=a)
            else:
                np.savez(filename, x=x, a=a, y=y)
            generated_graphs += 1

    def read(self):
        return get_graphs(self.path)

    @property
    def path(self):
        return os.path.join(super().path, self.string_params)

    @property
    def kripke_path(self):
        return os.path.join(self.path, 'kripke_structures')

    @property
    def lts_path(self):
        return os.path.join(self.path, 'lts')

    @property
    def name(self):
        return self._name

    @property
    def kripke_structures(self):
        for file in os.scandir(self.kripke_path):
            with open(file.path, 'rb') as f:
                yield pickle.load(f)

    @property
    def labelled_transition_systems(self):
        for file in os.scandir(self.lts_path):
            with open(file.path, 'rb') as f:
                yield pickle.load(f)

    @property
    def mu_calculus_formulae(self):
        return self._mu_formulae

    @property
    def formulae(self):
        return self._formulae

    @property
    def atomic_proposition_set(self):
        return self._atomic_propositions_set


if __name__ == '__main__':
    dataset = RandomKripkeDataset(1, 10, 16, ['a', 'b', 'c'], ['a', 'b', 'c'], None, False, False)
    print(list(dataset.kripke_structures))
    print(list(dataset.labelled_transition_systems))
    print(dataset.formulae)
    print(dataset.mu_calculus_formulae)
    print(dataset[0].x)
    a = dataset[0].a
    print(a)
