import stormpy
import numpy as np
from libmg import Dataset
from scipy.sparse import coo_matrix
from spektral.data import Graph

from sources.dataset_utils import get_np_data_type


def dtmc_to_graph(dtmc, formula, qualitative):
    n_nodes = dtmc.nr_states
    atomic_propositions_set = sorted(list(dtmc.labeling.get_labels()))
    n_ap = len(atomic_propositions_set)

    print("Generating x feature vector")
    # Node features (x) are a multi-hot representation of atomic actions
    if n_ap > 64:
        x = np.zeros((n_nodes, n_ap), dtype=np.uint8)
        for i in range(n_nodes):
            labels = dtmc.labels_state(i)
            for label in labels:
                x[i][atomic_propositions_set.index(label)] = 1
    else:
        tmp_x = []
        for i in range(n_nodes):
            labels = dtmc.labels_state(i)
            indices = [atomic_propositions_set.index(label) for label in labels]
            tmp_x.append([sum(2 ** idx for idx in indices)])
        x = np.array(tmp_x, dtype=get_np_data_type(atomic_propositions_set))

    print("Generating adjacency matrix and edge labels")
    # Initialize empty sparse adjacency matrix
    data = []
    rows = []
    cols = []
    tmp_e = []
    for state in dtmc.states:
        for action in state.actions:
            for transition in action.transitions:
                data.append(1)
                rows.append(state.id)
                cols.append(transition.column)
                tmp_e.append([transition.value()])
    e = np.array(tmp_e, dtype=np.float32)
    a = coo_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes), dtype=np.uint8)

    print("Generating y feature vector")
    properties = stormpy.parse_properties(formula)
    # f = properties[0].raw_formula
    result = stormpy.model_checking(dtmc, properties[0])
    assert result.result_for_all_states
    tmp_y = [[result.at(i)] for i in range(n_nodes)]
    y = np.array(tmp_y, dtype=np.bool_) if qualitative else np.array(tmp_y, dtype=np.float32)

    return [Graph(x=x, a=a, e=e, y=y)]


class DebugDTMCDataset(Dataset):
    def __init__(self, formula, qualitative, ex=1):
        self.formula = formula
        self.qualitative = qualitative
        self.ex = ex
        super().__init__('debug')

    def read(self):
        if self.ex == 1:
            builder = stormpy.SparseMatrixBuilder(rows=4, columns=4, entries=6, force_dimensions=False,
                                                  has_custom_row_grouping=False)
            builder.add_next_value(row=0, column=1, value=1)
            builder.add_next_value(row=1, column=1, value=0.01)
            builder.add_next_value(row=1, column=2, value=0.01)
            builder.add_next_value(row=1, column=3, value=0.98)
            builder.add_next_value(row=2, column=0, value=1)
            builder.add_next_value(row=3, column=3, value=1)
            transition_matrix = builder.build()

            state_labeling = stormpy.storage.StateLabeling(4)
            labels = {'try', 'succ', 'fail'}
            for label in labels:
                state_labeling.add_label(label)

            state_labeling.add_label_to_state('try', 1)
            state_labeling.add_label_to_state('fail', 2)
            state_labeling.add_label_to_state('succ', 3)

            components = stormpy.SparseModelComponents(transition_matrix=transition_matrix, state_labeling=state_labeling)
            dtmc = stormpy.storage.SparseDtmc(components)
            return dtmc_to_graph(dtmc, self.formula, self.qualitative)
        else:
            builder = stormpy.SparseMatrixBuilder(rows=4, columns=4, entries=6, force_dimensions=False,
                                                  has_custom_row_grouping=False)
            builder.add_next_value(row=0, column=1, value=1)
            builder.add_next_value(row=1, column=0, value=0.5)
            builder.add_next_value(row=1, column=2, value=0.5)
            builder.add_next_value(row=2, column=1, value=0.5)
            builder.add_next_value(row=2, column=3, value=0.5)
            builder.add_next_value(row=3, column=2, value=1)
            transition_matrix = builder.build()

            state_labeling = stormpy.storage.StateLabeling(4)
            labels = {'empty', 'full'}
            for label in labels:
                state_labeling.add_label(label)

            state_labeling.add_label_to_state('empty', 0)
            state_labeling.add_label_to_state('full', 3)


            components = stormpy.SparseModelComponents(transition_matrix=transition_matrix,
                                                       state_labeling=state_labeling)
            dtmc = stormpy.storage.SparseDtmc(components)
            return dtmc_to_graph(dtmc, self.formula, self.qualitative)

    @property
    def formulae(self):
        return [self.formula]

    @property
    def atomic_proposition_set(self):
        return sorted(['try', 'succ', 'fail'])


if __name__ == '__main__':
    formula = "P>=0.9 [ X (!\"try\" | \"succ\" ) ]"
    dataset = DebugDTMCDataset(formula)
    print(dataset[0])
