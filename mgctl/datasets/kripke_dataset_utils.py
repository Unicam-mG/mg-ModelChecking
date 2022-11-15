import os
import numpy as np
from pyModelChecking import CTL
from scipy.sparse import coo_matrix
from spektral.data import Graph
from mgctl.utils import get_np_data_type


def ctl_to_mu_formulae(CTL_formulae):
    def to_mu_calculus(phi):
        if isinstance(phi, CTL.Bool):
            return str(phi)
        elif isinstance(phi, CTL.AtomicProposition):
            return '<' + str(phi) + '>true'
        elif isinstance(phi, CTL.Not):
            return '!(' + to_mu_calculus(phi.subformula(0)) + ')'
        elif isinstance(phi, CTL.Or):
            return '(' + ' || '.join([to_mu_calculus(sub_formula) for sub_formula in phi.subformulas()]) + ')'
        elif isinstance(phi, CTL.And):
            return '(' + ' && '.join([to_mu_calculus(sub_formula) for sub_formula in phi.subformulas()]) + ')'
        elif isinstance(phi, CTL.Imply):
            return '(' + to_mu_calculus(phi.subformula(0)) + ' => ' + to_mu_calculus(phi.subformula(1)) + ')'
        elif isinstance(phi, CTL.E):
            sub_phi = phi.subformula(0)
            if isinstance(sub_phi, CTL.X):
                return '<true>' + to_mu_calculus(sub_phi.subformula(0))
            elif isinstance(sub_phi, CTL.G):
                var = vars.pop()
                return 'nu ' + var + '.(' + to_mu_calculus(sub_phi.subformula(0)) + ' && <true>' + var + ')'
            elif isinstance(sub_phi, CTL.F):
                var = vars.pop()
                return 'mu ' + var + '.(' + to_mu_calculus(sub_phi.subformula(0)) + ' || <true>' + var + ')'
            elif isinstance(sub_phi, CTL.U):
                var = vars.pop()
                return 'mu ' + var + '.(' + to_mu_calculus(sub_phi.subformula(1)) + ' || (' + to_mu_calculus(
                    sub_phi.subformula(0)) + ' && <true>' + var + '))'
        elif isinstance(phi, CTL.A):
            sub_phi = phi.subformula(0)
            if isinstance(sub_phi, CTL.X):
                return '[true]' + to_mu_calculus(sub_phi.subformula(0))
            elif isinstance(sub_phi, CTL.G):
                var = vars.pop()
                return 'nu ' + var + '.(' + to_mu_calculus(sub_phi.subformula(0)) + ' && [true]' + var + ')'
            elif isinstance(sub_phi, CTL.F):
                var = vars.pop()
                return 'mu ' + var + '.(' + to_mu_calculus(sub_phi.subformula(0)) + ' || [true]' + var + ')'
            elif isinstance(sub_phi, CTL.U):
                var = vars.pop()
                return 'mu ' + var + '.(' + to_mu_calculus(sub_phi.subformula(1)) + ' || (' + to_mu_calculus(
                    sub_phi.subformula(0)) + ' && [true]' + var + '))'
        else:
            raise ValueError("Error parsing formula ", phi)

    parser = CTL.Parser()
    mu_formulae = []
    for formula in CTL_formulae:
        parsed_formula = parser(formula)
        vars = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
                'V', 'W', 'X', 'Y', 'Z']
        mu_formulae.append(to_mu_calculus(parsed_formula))
    return mu_formulae


def create_graphs(kripke_structure, atomic_propositions_set, formulae, skip_model_checking=False, probabilistic=False):
    # We generate the x, a, y values we need to create a spektral graph
    n_nodes = len(kripke_structure.states())
    n_edges = len(kripke_structure.transitions())
    n_ap = len(atomic_propositions_set)

    print("Generating x feature vector")
    # Node features (x) are a multi-hot representation of atomic actions
    if n_ap > 64:
        x = np.zeros((n_nodes, n_ap), dtype=np.uint8)
        for i in range(n_nodes):
            labels = kripke_structure.labels(i)
            for label in labels:
                x[i][atomic_propositions_set.index(label)] = 1
    else:
        x = []
        for i in range(n_nodes):
            labels = kripke_structure.labels(i)
            indices = [atomic_propositions_set.index(label) for label in labels]
            x.append([sum(2 ** idx for idx in indices)])
        x = np.array(x, dtype=get_np_data_type(atomic_propositions_set))

    print("Generating adjacency matrix")
    # Initialize empty sparse adjacency matrix
    data = np.random.random(n_edges) if probabilistic else np.ones(n_edges)
    rows = []
    cols = []

    # Populate it in row-major order
    for edge in sorted(kripke_structure.transitions()):
        (source, target) = edge
        rows.append(source)
        cols.append(target)

    # The sum of out-going edges probability must be 1.0 for each node
    if probabilistic:
        for i in range(n_nodes):
            idx = [j for j, x in enumerate(rows) if x == i]
            data[idx] /= sum(data[idx])

    # Create the coo sparse adjacency matrix (a)
    a = coo_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes), dtype=np.float16 if probabilistic else np.uint8)

    print("Generating y feature vector")
    # Lastly, we model check all nodes of the graph on all the formulae to get our labels (y)
    mc_results = []
    parser = CTL.Parser()
    if not skip_model_checking:
        for formula in formulae:
            mc_results.append(CTL.modelcheck(kripke_structure, formula, parser))
        # Labels are in multi-hot notation using binary values
        y = np.zeros((n_nodes, len(formulae)), dtype=np.bool_)
        for j, mc_result in enumerate(mc_results):
            y[list(mc_result), j] = True
    else:
        y = None

    return x, a, y


def get_graphs(path):
    examples = []
    for file in filter(lambda f: not f.is_dir(), os.scandir(path)):
        loaded_graph = np.load(file.path, allow_pickle=True)
        y = loaded_graph['y'] if 'y' in loaded_graph.files else None
        examples.append(Graph(x=loaded_graph['x'],
                              a=loaded_graph['a'].item(),
                              y=y))
    return examples
