import os
import pathlib
import pickle
from typing import Dict, List

import numpy as np
import regex
from enum import Enum, auto
from pyModelChecking import Kripke
from snakes.pnml import loads
from snakes.nets import StateGraph
from libmg import Dataset

from sources.ctl.datasets.kripke_dataset_utils import ctl_to_mu_formulae, get_graphs, create_graphs


class MCCTypes(Enum):
    CARDINALITY = auto()
    FIREABILITY = auto()


def pn_to_kripke(ts, stuttering, atomic_propositions_type: MCCTypes):
    L: Dict[str, List[str]] = {}
    R = []
    if atomic_propositions_type is MCCTypes.FIREABILITY:
        for i in ts:
            L[i] = []
            for out_state, transition, _ in ts.successors(i):
                L[i].append(transition.name)
                R.append((i, out_state))
            if len(L[i]) == 0 and stuttering:
                R.append((i, i))  # add a self loop
    else:
        raise ValueError("Type not implemented!")

    K = Kripke(S=list(range(len(ts))),
               S0=[0],
               R=R,
               L=L)
    return K


def pn_to_lts(ts, atomic_propositions_type: MCCTypes):
    """
    :return: A string representing the input graph in Aldebaran (.aut) format
    """
    if atomic_propositions_type is MCCTypes.FIREABILITY:
        aut_body = ''
        n_edges = 0
        for i in ts:
            for out_state, transition, _ in ts.successors(i):
                aut_body += '(' + str(i) + ',\"' + transition.name + '\",' + str(
                    out_state) + ')\n'
                n_edges += 1
    else:
        raise ValueError("Type not implemented!")
    aut_header = 'des (0,' + str(n_edges) + ',' + str(len(ts)) + ')\n'
    return aut_header + aut_body


def parse_petri_net_formulae(formulae_folder, formula_type: MCCTypes):
    def fireability_parser(match):
        sub_clauses = regex.findall(r'"(.*?)"', match.group())
        return ' | '.join(sub_clauses)

    parsed_formulae = []
    if formula_type is MCCTypes.FIREABILITY:
        formulae_file = "CTLFireability.txt"
    else:
        raise ValueError("CTL Cardinality not yet implemented")

    with open(os.path.join(formulae_folder, formulae_file), 'r') as f:
        data = f.read()
        unparsed_formulae = regex.findall(r'is:\s*(.*)\s*end\.', data)
        for unparsed_formula in unparsed_formulae:
            unparsed_formula = unparsed_formula.replace('!', '~')
            parsed_formulae.append(regex.sub(r'is-fireable\((.*?)\)', fireability_parser, unparsed_formula))
    return parsed_formulae


def get_atomic_propositions_from_petri_net(pnml_path):
    with open(pnml_path) as pnml:
        net = loads(pnml.read())
    return sorted([transition.name for transition in net.transition()])


class PetriNetDataset(Dataset):
    petri_net_folder = os.path.join(pathlib.Path(__file__).parent, 'mcc_petri_nets')

    def __init__(self, model, prop_type: MCCTypes, stuttering=True, skip_model_checking=False, **kwargs):
        self.model = model
        self.model_folder = os.path.join(self.petri_net_folder, self.model)
        self.prop_type = prop_type
        self.stuttering = stuttering
        self.skip_model_checking = skip_model_checking
        self.string_params = '_'.join([str(model), str(prop_type), str(stuttering), str(skip_model_checking)])
        self._formulae = parse_petri_net_formulae(self.model_folder, prop_type)
        self._atomic_propositions_set = get_atomic_propositions_from_petri_net(os.path.join(self.model_folder,
                                                                                            'model.pnml'))
        self._mu_formulae = ctl_to_mu_formulae(self._formulae)
        super().__init__(self.string_params, **kwargs)

    def download(self):
        os.makedirs(self.path)
        os.mkdir(self.kripke_path)
        os.mkdir(self.lts_path)

        print("Generating Petri Net")

        with open(os.path.join(self.model_folder, 'model.pnml')) as pnml:
            net = loads(pnml.read())
        ts = StateGraph(net)
        ts.build()

        print("Generating LTS")

        # Obtain the LTS from the Petri net
        lts = pn_to_lts(ts, self.prop_type)

        # Save the LTS to file
        with open(os.path.join(self.lts_path, 'LTS_' + self.model), 'wb') as f:
            pickle.dump(lts, f)

        del lts

        print("Generating Kripke structure")

        # Obtain the Kripke structure from the Petri net
        kripke_structure = pn_to_kripke(ts, self.stuttering, self.prop_type)

        # Save the Kripke structure to file
        with open(os.path.join(self.kripke_path, 'Kripke_' + self.model), 'wb') as f:
            pickle.dump(kripke_structure, f)

        del net, ts

        x, a, y = create_graphs(kripke_structure, self._atomic_propositions_set, self._formulae,
                                self.skip_model_checking, probabilistic=False)

        # We can now save the graph to file as npz
        filename = os.path.join(self.path, 'Graph_' + self.model)
        if self.skip_model_checking:
            np.savez(filename, x=x, a=a)
        else:
            np.savez(filename, x=x, a=a, y=y)

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
