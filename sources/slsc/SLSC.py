import tensorflow as tf
import tensorflow_addons as tfa
from lark import Lark, Transformer
from libmg import Phi, Sigma, FunctionDict
from libmg import GNNCompiler, FixPointConfig, CompilationConfig, NodeConfig

from sources.dataset_utils import get_tf_data_type
from sources.propositional_logic import b_true, b_false, b_and, b_or, b_not, make_atomic_propositions

slsc_grammar = r"""
        ?s_formula: "true"                              -> true
                 | "false"                              -> false
                 | a_prop
                 | "not" s_formula                      -> not_formula
                 | s_formula ( "or" s_formula )+        -> or_formula
                 | s_formula ( "and" s_formula )+       -> and_formula
                 | "N" s_formula                        -> near_formula
                 | s_formula "R" s_formula              -> reachability_formula
                 | "(" s_formula ")"

        a_prop: /[a-zA-Z_][a-zA-Z_0-9]*/

        %import common.WS
        %ignore WS
        """

p3 = Phi(lambda i, e, j: j)
Max = Sigma(lambda m, i, n, x: tf.cast(tf.math.segment_max(tf.cast(m, tf.uint8), i), tf.bool))


def to_mG(phi):
    class SLSCToMuGNN(Transformer):
        def true(self, args):
            return 'true'

        def false(self, args):
            return 'false'

        def a_prop(self, args):
            return str(args[0])

        def not_formula(self, args):
            return '(' + str(args[0]) + ');not'

        def or_formula(self, args):
            return '((' + str(args[0]) + ') || (' + str(args[1]) + '));or'

        def and_formula(self, args):
            return '((' + str(args[0]) + ') || (' + str(args[1]) + '));and'

        def near_formula(self, args):
            return '(' + str(args[0]) + ');|p3>or'

        def reachability_formula(self, args):
            return "mu X,b . (((((" + str(args[0]) + ") || (X;|p3>or));and) || (" + str(args[1]) + "));or)"

    parser = Lark(slsc_grammar, start='s_formula')
    return SLSCToMuGNN().transform(parser.parse(phi))


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
    compiler = GNNCompiler(psi_functions=FunctionDict({'true': b_true, 'false': b_false,
                                                       'not': b_not, 'and': b_and, 'or': b_or} | funcs),
                           sigma_functions=FunctionDict({
                               'or': Sigma(lambda m, i, n, x: tf.cast(
                                   tf.math.unsorted_segment_max(tf.cast(m, tf.uint8), i, n),
                                   tf.bool))}),
                           phi_functions=FunctionDict({'p3': p3}),
                           bottoms={'b': FixPointConfig(1, False)},
                           tops={'b': FixPointConfig(1, True)},
                           config=config(NodeConfig(data_type, data_size), tf.uint8))
    if formulae is None:
        expr = " || ".join(['(' + to_mG(formula) + ')' for formula in dataset.formulae])
    else:
        expr = " || ".join(['(' + to_mG(formula) + ')' for formula in formulae])
    return compiler.compile(expr, loss=tfa.metrics.HammingLoss(mode='multilabel'), optimize=optimize,
                            return_compilation_time=return_compilation_time)
