import os
import subprocess
import time
from pyModelChecking import CTL
from pyModelChecking.CTL import Parser

from libmg.compiler import Preset
from libmg.evaluator import save_output_to_csv, PerformanceTest
from CTL import build_model
from datasets.kripke_dataset_utils import ctl_to_mu_formulae
from datasets.pnml_kripke_dataset import PetriNetDataset, MCCTypes


os.environ['TF_CPP_MIN_LOG_LEVEL'] = "0"


# formulae param must be in mu-calculus
class MUCRL2Performance(PerformanceTest):

    @staticmethod
    def model_check_lts(aut_code, action_set, formula):
        ignored_warnings = [b'Cannot determine type of input. Assuming .aut.\r\n'] + [
            b'Cannot determine type of input. '
            b'Assuming .aut.\r\nWarning: the '
            b'modal formula contains an action '
            b'' + str(action).encode('ascii') +
            b' that does not appear in the '
            b'LTS!\r\n' for action in action_set]
        # Create a temporary mcrl2 data file with action specification
        with open('tmp.mcrl2', 'w') as f:
            f.write("act ")
            if len(action_set) > 0:
                for action in action_set[:-1]:
                    f.write(action + ", ")
                f.write(action_set[-1] + ";")
        # Create a temporary formula file
        with open('formula.mcf', 'w') as f:
            f.write(formula)
        # Convert graph to binary string
        aut_code = aut_code.encode('ascii')
        # Create PBES
        start = time.perf_counter()
        pbes = subprocess.run(["lts2pbes", "--formula=formula.mcf", "--data=tmp.mcrl2", "-p"],
                              input=aut_code, capture_output=True)
        end = time.perf_counter()
        pbes_time = end - start
        if pbes.stderr and pbes.stderr not in ignored_warnings:
            print(pbes.stderr)
        # Solve PBES
        start = time.perf_counter()
        result = subprocess.run(["pbessolve", "--threads=8"], input=pbes.stdout, capture_output=True)
        end = time.perf_counter()
        exec_time = end - start
        os.remove("tmp.mcrl2")
        os.remove("formula.mcf")
        return True if 'true' in result.stdout.decode('ascii') else False, pbes_time, exec_time

    def __call__(self, dataset, formulae=None):
        tot_pbes_time = 0
        tot_exec_time = 0
        if formulae:
            formulae = ctl_to_mu_formulae(formulae)
        for formula in formulae or dataset.mu_calculus_formulae:
            for lts in dataset.labelled_transition_systems:
                _, pbes_time, exec_time = MUCRL2Performance.model_check_lts(lts, dataset.atomic_proposition_set,
                                                                            formula)
                tot_pbes_time += pbes_time
                tot_exec_time += exec_time
        print("Using mCRL2", tot_pbes_time, tot_exec_time, sep=' ')
        return tot_pbes_time, tot_exec_time


class PyModelCheckingPerformance(PerformanceTest):
    def __call__(self, dataset, formulae=None):
        parser = Parser()
        tot = 0.0
        for formula in formulae or dataset.formulae:
            phi = parser(formula)
            for kripke_structure in dataset.kripke_structures:
                start = time.perf_counter()
                CTL.modelcheck(kripke_structure, phi)
                end = time.perf_counter()
                tot += end - start
        print("Using pyModelChecking", tot, sep=' ')
        return tot


class CTLModelPerformance(PerformanceTest):
    def __init__(self, *args, split=False, **kwargs):
        self.split = split
        super().__init__(*args, **kwargs)

    def __call__(self, dataset):
        loader = self.loader_constructor(dataset)
        models = []
        total_compile_time = 0
        if self.split:
            for formula in dataset.formulae:
                model, compile_time = self.model_constructor(dataset, [formula])
                models.append(model)
                total_compile_time += compile_time
        else:
            model, compile_time = self.model_constructor(dataset, dataset.formulae)
            models.append(model)
            total_compile_time = compile_time
        tot = 0.0
        for x, y in loader.load():
            for model in models:
                start = time.perf_counter()
                model(x)
                end = time.perf_counter()
                tot += end - start
        print("Using model.__call__ and tf.function", tot, sep=' ')
        return total_compile_time, tot


class CTLPredictPerformance(PerformanceTest):
    def __init__(self, *args, split=False, **kwargs):
        self.split = split
        super().__init__(*args, **kwargs)

    def __call__(self, dataset):
        loader = self.loader_constructor(dataset)
        models = []
        total_compile_time = 0
        if self.split:
            for formula in dataset.formulae:
                model, compile_time = self.model_constructor(dataset, [formula])
                models.append(model)
                total_compile_time += compile_time
        else:
            model, compile_time = self.model_constructor(dataset, dataset.formulae)
            models.append(model)
            total_compile_time = compile_time
        tot = 0.0
        for model in models:
            start = time.perf_counter()
            model.predict(loader.load(), steps=loader.steps_per_epoch)
            end = time.perf_counter()
            tot += end - start
        print("Using model.predict", tot, sep=' ')
        return total_compile_time, tot


if __name__ == '__main__':
    dataset_name = "Philosophers-PT-000005"
    dataset = PetriNetDataset(dataset_name, MCCTypes.FIREABILITY, skip_model_checking=True)
    preset = Preset(single=True, edges=False)
    save_output_to_csv([dataset],
                       [CTLPredictPerformance(lambda dataset, formulas: build_model(dataset, formulas,
                                                                                    config=preset.suggested_config,
                                                                                    optimize='predict',
                                                                                    return_compilation_time=True),
                                              lambda dataset: preset.suggested_loader(dataset, epochs=1), split=False),
                        ],
                       ['split predict compile', 'split predict exe', 'full predict compile', 'full predict exe'],
                       dataset_name + "_updated")
