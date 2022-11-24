import tensorflow as tf
import tensorflow_addons as tfa
from pyModelChecking import CTL, Bool
from libmg import Sigma, Phi
from libmg import GNNCompiler, FixPointConfig, CompilationConfig, NodeConfig

from sources.datasets.dataset_utils import get_tf_data_type
from sources.logics.propositional_logic.propositional_logic import true, false, And, Or, Not, make_atomic_propositions

# Broadcast setup
p3 = Phi(lambda i, e, j: j)
Max = Sigma(lambda m, i, n, x: tf.cast(tf.math.segment_max(tf.cast(m, tf.uint8), i), tf.bool))


def to_mG(expr):
    def _to_mG(phi):
        if isinstance(phi, CTL.Bool):
            if phi == Bool(True):
                return "true"
            elif phi == Bool(False):
                return "false"
            else:
                raise ValueError("Error Parsing formula: " + phi)
        elif isinstance(phi, CTL.AtomicProposition):
            return str(phi)
        elif isinstance(phi, CTL.Not):
            return '(' + _to_mG(phi.subformula(0)) + ' ; not)'
        elif isinstance(phi, CTL.Or):
            # check for equal terms and keep order
            sub_formulas = list(set(phi.subformulas()))
            return '((' + ' || '.join([_to_mG(sub_formula) for sub_formula in sub_formulas]) + ') ; or)'
        elif isinstance(phi, CTL.And):
            # check for equal terms and keep order
            sub_formulas = list(set(phi.subformulas()))
            return '((' + ' || '.join([_to_mG(sub_formula) for sub_formula in sub_formulas]) + ') ; and)'
        elif isinstance(phi, CTL.Imply):
            return _to_mG(CTL.Or(CTL.Not(phi.subformula(0)), phi.subformula(1)))
        elif isinstance(phi, CTL.E):
            sub_phi = phi.subformula(0)
            if isinstance(sub_phi, CTL.X):
                return "( (" + _to_mG(sub_phi.subformula(0)) + ") ; |p3> or )"
            elif isinstance(sub_phi, CTL.G):
                return "( nu X,b . ( ( (" + _to_mG(sub_phi.subformula(0)) + ") || (X ; |p3> or)) ; and) )"
            elif isinstance(sub_phi, CTL.F):
                return _to_mG(CTL.EU(CTL.Bool(True), sub_phi.subformula(0)))
            elif isinstance(sub_phi, CTL.U):
                return "( mu X,b . (((((" + _to_mG(
                    sub_phi.subformula(0)) + ") || (X ; |p3> or) ); and) || (" + _to_mG(
                    sub_phi.subformula(1)) + ") ) ; or) )"
        elif isinstance(phi, CTL.A):
            sub_phi = phi.subformula(0)
            if isinstance(sub_phi, CTL.X):
                return _to_mG(CTL.Not(CTL.EX(CTL.Not(sub_phi.subformula(0)))))
            elif isinstance(sub_phi, CTL.G):
                return _to_mG(CTL.Not(CTL.EU(CTL.Bool(True), CTL.Not(sub_phi.subformula(0)))))
            elif isinstance(sub_phi, CTL.F):
                return _to_mG(CTL.Not(CTL.EG(CTL.Not(sub_phi.subformula(0)))))
            elif isinstance(sub_phi, CTL.U):
                return _to_mG(CTL.Not(CTL.Or(CTL.EU(CTL.Not(sub_phi.subformula(1)),
                                                    CTL.Not(CTL.Or(sub_phi.subformula(0), sub_phi.subformula(1)))),
                                             CTL.EG(CTL.Not(sub_phi.subformula(1))))))
        else:
            raise ValueError("Error parsing formula ", phi)

    return _to_mG(CTL.Parser()(expr))


def build_model(dataset, formulae=None, config=CompilationConfig.xai_config, optimize=None,
                return_compilation_time=False):
    n_atomic_propositions = len(dataset.atomic_proposition_set)
    if n_atomic_propositions > 64:
        data_type = tf.uint8
        data_size = n_atomic_propositions
        funcs = make_atomic_propositions(dataset.atomic_proposition_set, 'one_hot', data_type)
    else:
        data_type = get_tf_data_type(dataset.atomic_proposition_set)
        data_size = 1
        funcs = make_atomic_propositions(dataset.atomic_proposition_set, 'bitstring', data_type)
    compiler = GNNCompiler(psi_functions={'true': true, 'false': false,
                                          'not': Not, 'and': And, 'or': Or} | funcs,
                           sigma_functions={'or': Max},
                           phi_functions={'p3': p3},
                           bottoms={'b': FixPointConfig(1, False)},
                           tops={'b': FixPointConfig(1, True)},
                           config=config(NodeConfig(data_type, data_size), tf.uint8))
    if formulae is None:
        expr = " || ".join([to_mG(formula) for formula in dataset.formulae])
    else:
        expr = " || ".join([to_mG(formula) for formula in formulae])
    return compiler.compile(expr, loss=tfa.metrics.HammingLoss(mode='multilabel'), optimize=optimize,
                            return_compilation_time=return_compilation_time)
