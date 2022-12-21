import stormpy
import numpy as np
from libmg import Dataset
from scipy.sparse import coo_matrix
from spektral.data import Graph

from sources.dataset_utils import get_np_data_type


# TODO: remove 0 edges
def ctmc_to_graph(ctmc, formula, qualitative):
    n_nodes = ctmc.nr_states
    atomic_propositions_set = sorted(list(ctmc.labeling.get_labels()))
    n_ap = len(atomic_propositions_set)

    print("Generating x feature vector")
    # Node features (x) are a multi-hot representation of atomic actions
    if n_ap > 64:
        x = np.zeros((n_nodes, n_ap), dtype=np.uint8)
        for i in range(n_nodes):
            labels = ctmc.labels_state(i)
            for label in labels:
                x[i][atomic_propositions_set.index(label)] = 1
    else:
        tmp_x = []
        for i in range(n_nodes):
            labels = ctmc.labels_state(i)
            indices = [atomic_propositions_set.index(label) for label in labels]
            tmp_x.append([sum(2 ** idx for idx in indices)])
        x = np.array(tmp_x, dtype=get_np_data_type(atomic_propositions_set))

    print("Generating adjacency matrix and edge labels")
    # Initialize empty sparse adjacency matrix
    data = []
    rows = []
    cols = []
    tmp_e = []
    maxq = max(ctmc.exit_rates)
    for state in ctmc.states:
        has_self_loop = False
        edges = []
        iidx = []
        jidx = []
        qii = 0
        for action in state.actions:
            for transition in action.transitions:
                if transition.column == state.id:  # we are going to add a self transition, set flag and skip ahead
                    has_self_loop = True
                elif transition.column > state.id and not has_self_loop:  #we dont have a self transition, then we add self loop with rate 0 before adding the current transition
                    data.append(1)
                    iidx.append(state.id)
                    jidx.append(state.id)
                    edges.append(0.0)
                    has_self_loop = True
                data.append(1)
                iidx.append(state.id)
                jidx.append(transition.column)
                edges.append(transition.value())
                if state.id != transition.column:
                    qii = qii - transition.value()
        if not has_self_loop: # we didn't add a self transition yet because all transitions were added before transition.column > state.id
            data.append(1)
            iidx.append(state.id)
            jidx.append(state.id)
            edges.append(0.0)
        tmp_e.extend([[edges[i], edges[i]/ctmc.exit_rates[state.id], 1 + qii/maxq if iidx[i] == jidx[i] else edges[i]/maxq] for i in range(len(edges))])
        rows.extend(iidx)
        cols.extend(jidx)

    e = np.array(tmp_e, dtype=np.float32)
    a = coo_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes), dtype=np.uint8)

    print("Generating y feature vector")
    properties = stormpy.parse_properties(formula)
    # f = properties[0].raw_formula
    result = stormpy.model_checking(ctmc, properties[0])
    assert result.result_for_all_states
    tmp_y = [[result.at(i)] for i in range(n_nodes)]
    y = np.array(tmp_y, dtype=np.bool_) if qualitative else np.array(tmp_y, dtype=np.float32)

    return [Graph(x=x, a=a, e=e, y=y)]


class DebugCTMCDataset(Dataset):
    def __init__(self, formula, qualitative):
        self.formula = formula
        self.qualitative = qualitative
        super().__init__('debug')

    def read(self):
        builder = stormpy.SparseMatrixBuilder(rows=4, columns=4, entries=6, force_dimensions=False,
                                              has_custom_row_grouping=False)
        builder.add_next_value(row=0, column=1, value=1.5)
        builder.add_next_value(row=1, column=0, value=3)
        builder.add_next_value(row=1, column=2, value=1.5)
        builder.add_next_value(row=2, column=1, value=3)
        builder.add_next_value(row=2, column=3, value=1.5)
        builder.add_next_value(row=3, column=2, value=3)
        transition_matrix = builder.build()

        state_labeling = stormpy.storage.StateLabeling(4)
        labels = {'empty', 'full'}
        for label in labels:
            state_labeling.add_label(label)

        state_labeling.add_label_to_state('empty', 0)
        state_labeling.add_label_to_state('full', 3)

        exit_rates = [1.5, 4.5, 4.5, 3.0]

        components = stormpy.SparseModelComponents(transition_matrix=transition_matrix, state_labeling=state_labeling,
                                                   rate_transitions=True)
        components.exit_rates = exit_rates
        ctmc = stormpy.storage.SparseCtmc(components)
        return ctmc_to_graph(ctmc, self.formula, self.qualitative)

    @property
    def formulae(self):
        return [self.formula]

    @property
    def atomic_proposition_set(self):
        return sorted(['empty', 'full'])


if __name__ == '__main__':
    formula = "P>=0.65 [ true U[0, 7.5] \"full\" ]"
    dataset = DebugCTMCDataset(formula, True)
    print(dataset[0].y)
