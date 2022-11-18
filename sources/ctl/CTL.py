import tensorflow as tf
import tensorflow_addons as tfa
from pyModelChecking import CTL, Bool
from libmg.layers import PsiLocal, Sigma, Phi
from libmg.compiler import GNNCompiler, FixPointConfig, Bottom, Top, CompilationConfig

from sources.datasets.dataset_utils import to_one_hot, get_tf_data_type

# Broadcast setup
false = PsiLocal(lambda x: tf.zeros((tf.shape(x)[0], 1), dtype=tf.bool))
true = PsiLocal(lambda x: tf.ones((tf.shape(x)[0], 1), dtype=tf.bool))
And = PsiLocal(lambda x: tf.math.reduce_all(x, axis=1, keepdims=True))
Or = PsiLocal(lambda x: tf.math.reduce_any(x, axis=1, keepdims=True))
Not = PsiLocal(lambda x: tf.math.logical_not(x))
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
        funcs = {atom_prop: PsiLocal(lambda x, v=atom_prop: tf.cast(
            tf.math.reduce_sum(x * to_one_hot(v, dataset.atomic_proposition_set, data_type), axis=1, keepdims=True),
            dtype=tf.bool))
                 for atom_prop in dataset.atomic_proposition_set}
    else:
        data_type = get_tf_data_type(dataset.atomic_proposition_set)
        data_size = 1
        funcs = {atom_prop: PsiLocal(lambda x, v=atom_prop: tf.cast(
            tf.bitwise.bitwise_and(x, to_one_hot(v, dataset.atomic_proposition_set, data_type)), tf.bool))
                 for atom_prop in dataset.atomic_proposition_set}
    compiler = GNNCompiler(psi_functions={'true': true, 'false': false,
                                          'not': Not, 'and': And, 'or': Or} | funcs,
                           sigma_functions={'or': Max},
                           phi_functions={'p3': p3},
                           bottoms={'b': FixPointConfig(Bottom(1, False))},
                           tops={'b': FixPointConfig(Top(1, True))},
                           config=config(data_type, data_size, tf.uint8))
    if formulae is None:
        expr = " || ".join([to_mG(formula) for formula in dataset.formulae])
    else:
        expr = " || ".join([to_mG(formula) for formula in formulae])
    return compiler.compile(expr, loss=tfa.metrics.HammingLoss(mode='multilabel'), optimize=optimize,
                            return_compilation_time=return_compilation_time)
