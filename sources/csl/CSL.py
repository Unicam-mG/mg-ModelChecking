import tensorflow as tf
import tensorflow_addons as tfa
import re
import sys
from libmg import PsiLocal, Phi, Sigma, CompilationConfig, GNNCompiler, FunctionDict, FixPointConfig, NodeConfig, \
    EdgeConfig
import stormpy
from stormpy import ComparisonType, BooleanBinaryStateFormula, UnaryBooleanStateFormula, BooleanLiteralFormula

from sources.csl.fox_glynn_algorithm import fox_glynn_algorithm
from sources.dataset_utils import get_tf_data_type
from sources.propositional_logic import b_not, b_or, b_true, b_and, b_false, make_atomic_propositions
from sources.pctl.PCTL import pmode, probgr, probgreq, proble, probleq, summation, plus, mul

sys.setrecursionlimit(10000)

# neXt is |prod>sum like PCTL but our edges are labelled differently [rate, emb(c), unif(c)]
# prod with emb(c)
prod_emb = Phi(lambda i, e, j: tf.math.multiply(e[:, 1:2], j))

# prod with unif(c)
prod_unif = Phi(lambda i, e, j: tf.math.multiply(e[:, 2:3], j))

# const-prod
prod_const = lambda y: PsiLocal(lambda x: tf.math.multiply(x, float(y)))


# TODO: infinity [0, inf] and [t, inf]
def to_mG(expr, max_exit_rate):
    def _to_mG(phi):
        if phi.is_probability_operator is True:
            if phi.has_bound is True:
                threshold = phi.threshold_expr.evaluate_as_double()
                match phi.comparison_type:
                    case ComparisonType.GEQ:
                        return '(' + _to_mG(phi.subformula) + ');probgreq[' + str(threshold) + ']'
                    case ComparisonType.GREATER:
                        return '(' + _to_mG(phi.subformula) + ');probgr[' + str(threshold) + ']'
                    case ComparisonType.LESS:
                        return '(' + _to_mG(phi.subformula) + ');proble[' + str(threshold) + ']'
                    case ComparisonType.LEQ:
                        return '(' + _to_mG(phi.subformula) + ');probleq[' + str(threshold) + ']'
            else:
                return '(' + _to_mG(phi.subformula) + ')'
        elif phi.is_bounded_until_formula is True:
            ub = phi.upper_bound_expression.evaluate_as_double()
            # stormpy API doesn't have anything to get this value in another way
            lb = float(re.search(r'\[(.+),', str(phi)).group(1))
            phi1 = _to_mG(phi.left_subformula)
            phi2 = _to_mG(phi.right_subformula)
            phi1_and_not_phi2 = '((' + phi1 + ') || (' + phi2 + ');not);and'

            # perform fox-glynn statically
            qt = max_exit_rate * ub
            weights, left, right, total_weight = fox_glynn_algorithm(qt, 1.0e-300, 1.0e+300, 0.001)
            for i in range(left, right + 1):
                weights[i - left] = weights[i - left] / total_weight

            if left == 0:
                output = phi2 + ';%;xk[' + str(weights[0]) + ']'
                rx_step = phi2
                for i in range(1, right):
                    output = '(' + '(' + output + ')' + ' || ' + '(' + '(((' + phi1_and_not_phi2 + ';%) || (' + rx_step + ';%;|x>+));*) || (' + phi2 + ';%));+;xk[' + str(weights[i]) + ']' + ')' + ';+'
                    rx_step = '(' + '(((' + phi1_and_not_phi2 + ';%) || ((' + rx_step + ');%;|x>+));*) || (' + phi2 + ';%));+'
            else:
                raise NotImplementedError("TODO: left > 0")
            return output
        elif phi.is_until_formula is True:
            phi1 = _to_mG(phi.left_subformula)
            phi2 = _to_mG(phi.right_subformula)
            phi1_and_not_phi2 = '((' + phi1 + ') || (' + phi2 + ');not);and'
            return 'mu X,f . ((((' + phi2 + ');%) || ((((' + phi1_and_not_phi2 + ');%) || (X;|*>+));*));+)'
        elif isinstance(phi, BooleanBinaryStateFormula):
            string_phi = str(phi)
            if '&' in string_phi:
                tokens = str(phi).split('&')
                return '((' + _to_mG(stormpy.parse_properties(tokens[0])[0].raw_formula) + ') || (' + _to_mG(
                    stormpy.parse_properties(tokens[1])[0].raw_formula) + '));and'
            elif '|' in string_phi:
                tokens = str(phi).split('|')
                return '((' + _to_mG(stormpy.parse_properties(tokens[0])[0].raw_formula) + ') || (' + _to_mG(
                    stormpy.parse_properties(tokens[1])[0].raw_formula) + '));or'
            else:
                raise ValueError("Unable to parse:", str(phi))
        elif isinstance(phi, UnaryBooleanStateFormula):
            return '(' + _to_mG(phi.subformula) + ');not'
        elif hasattr(phi, 'label'):  # ground case
            return phi.label
        elif isinstance(phi, BooleanLiteralFormula):  # ground case
            return str(phi)
        else:  # for some reason, X must be parsed using this case
            return '(' + _to_mG(stormpy.parse_properties(str(phi)[1:])[0].raw_formula) + ');%;|*>+'

    expr = stormpy.parse_properties(expr)[0].raw_formula
    return _to_mG(expr)


def build_model(dataset, max_exit_rate, formulae=None, config=CompilationConfig.xaei_config, optimize=None,
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
                                                       'not': b_not, 'and': b_and, 'or': b_or,
                                                       '%': pmode, '+': plus, '*': mul,
                                                       'probgreq': probgreq, 'probgr': probgr, 'proble': proble,
                                                       'probleq': probleq, 'xk': prod_const} | funcs),
                           sigma_functions=FunctionDict({'+': summation}),
                           phi_functions=FunctionDict({'*': prod_emb, 'x': prod_unif}),
                           bottoms={'f': FixPointConfig(1, 0.0, 0.001)},
                           tops={},
                           config=config(NodeConfig(data_type, data_size), EdgeConfig(tf.float32, 3), tf.uint8))
    if formulae is None:
        expr = " || ".join([to_mG(formula, max_exit_rate) for formula in dataset.formulae])
    else:
        expr = " || ".join([to_mG(formula, max_exit_rate) for formula in formulae])
    return compiler.compile(expr, loss=tfa.metrics.HammingLoss(mode='multilabel'), optimize=optimize,
                            return_compilation_time=return_compilation_time)
