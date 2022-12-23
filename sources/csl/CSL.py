import math
from fractions import Fraction
import tensorflow as tf
import tensorflow_addons as tfa
import re
import sys
from libmg import PsiLocal, Phi, Sigma, CompilationConfig, GNNCompiler, FunctionDict, FixPointConfig, NodeConfig, \
    EdgeConfig, PsiGlobal

from libmg.layers import PreImage
import stormpy
from stormpy import ComparisonType, BooleanBinaryStateFormula, UnaryBooleanStateFormula, BooleanLiteralFormula, \
    LongRunAvarageOperator

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

transpose_summation = Sigma(lambda m, i, n, x: tf.math.unsorted_segment_sum(m, i, n))

p2rate = Phi(lambda i, e, j: e[:, 0:1])

reciprocal = PsiLocal(tf.math.reciprocal)

mx = PsiLocal(lambda x: tf.math.reduce_prod(x, axis=1, keepdims=True))

total_mean_holding_time = PsiGlobal(single_op=lambda x: tf.math.reduce_sum(x, keepdims=True))



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
            ub = math.inf if '>' in str(phi) else phi.upper_bound_expression.evaluate_as_double()
            # stormpy API doesn't have anything to get this value in another way
            lb = float(Fraction((re.search(r'>=(.+)\s', str(phi)) or re.search(r'\[(.+),', str(phi))).group(1))) if phi.has_lower_bound else 0
            phi1 = _to_mG(phi.left_subformula)
            phi2 = _to_mG(phi.right_subformula)
            phi1_and_not_phi2 = '((' + phi1 + ') || (' + phi2 + ');not);and'

            if lb == 0 and ub == math.inf:
                return 'mu X,f . ((((' + phi2 + ');%) || ((((' + phi1_and_not_phi2 + ');%) || (X;|*>+));*));+)'
            else:
                # build formula
                if lb == 0:
                    # perform fox-glynn statically
                    qt = max_exit_rate * ub
                    weights, left, right, total_weight = fox_glynn_algorithm(qt, 1.0e-300, 1.0e+300, 0.0001)
                    for i in range(left, right + 1):
                        weights[i - left] = weights[i - left] / total_weight
                    terms = [phi2 + ';%']
                    for i in range(1, right):
                        terms.append('(' + '(((' + phi1_and_not_phi2 + ';%) || ((' + terms[i - 1] + ');|x>+));*) || (' + phi2 + ';%));+')
                elif ub == math.inf:
                    # perform fox-glynn statically
                    qt = max_exit_rate * lb
                    weights, left, right, total_weight = fox_glynn_algorithm(qt, 1.0e-300, 1.0e+300, 0.0001)
                    for i in range(left, right + 1):
                        weights[i - left] = weights[i - left] / total_weight
                    probphi1Uphi2 = _to_mG(stormpy.parse_properties('P=? [ ' + str(phi.left_subformula) + ' U ' + str(phi.right_subformula) + ' ]')[0].raw_formula)
                    terms = ['((' + probphi1Uphi2 + ') || (' + phi1 + ';%));*']
                    for i in range(1, right):
                        terms.append('(((' + phi1 + ';%) || ((' + terms[i - 1] + ');|x>+));*)')
                else:
                    # perform fox-glynn statically
                    qt = max_exit_rate * lb
                    weights, left, right, total_weight = fox_glynn_algorithm(qt, 1.0e-300, 1.0e+300, 0.0001)
                    for i in range(left, right + 1):
                        weights[i - left] = weights[i - left] / total_weight
                    probphi1Uphi2 = _to_mG(stormpy.parse_properties('P=? [ ' + str(phi.left_subformula) + ' U[0, ' + str(ub-lb) + '] ' + str(phi.right_subformula) + ' ]')[0].raw_formula)
                    terms = ['((' + probphi1Uphi2 + ') || (' + phi1 + ';%));*']
                    for i in range(1, right):
                        terms.append('(((' + phi1 + ';%) || ((' + terms[i - 1] + ');|x>+));*)')
                # summation
                output = terms[left] + ';xk[' + str(weights[0]) + ']'
                for i in range(left + 1, right):
                    output = '(' + '(' + output + ')' + ' || ' + '(' + terms[i] + ';xk[' + str(
                        weights[i - left]) + '])' + ');+'

                return output
        elif phi.is_until_formula is True:
            phi1 = _to_mG(phi.left_subformula)
            phi2 = _to_mG(phi.right_subformula)
            phi1_and_not_phi2 = '((' + phi1 + ') || (' + phi2 + ');not);and'
            return 'mu X,f . ((((' + phi2 + ');%) || ((((' + phi1_and_not_phi2 + ');%) || (X;|*>+));*));+)'
        elif isinstance(phi, LongRunAvarageOperator):
            output = '(((' + _to_mG(phi.subformula) +');%) || (((((mu X,p . (X;<*|+T)) || (|p2rate>+;^-1));mx) || (((mu X,p . (X;<*|+T)) || (|p2rate>+;^-1));mx;tmht;^-1));mx));mx;tmht'
            if phi.has_bound is True:
                threshold = phi.threshold_expr.evaluate_as_double()
                match phi.comparison_type:
                    case ComparisonType.GEQ:
                        return '(' + output + ');probgreq[' + str(threshold) + ']'
                    case ComparisonType.GREATER:
                        return '(' + output + ');probgr[' + str(threshold) + ']'
                    case ComparisonType.LESS:
                        return '(' + output + ');proble[' + str(threshold) + ']'
                    case ComparisonType.LEQ:
                        return '(' + output + ');probleq[' + str(threshold) + ']'
            else:
                return '(' + output + ')'
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
                                                       'probleq': probleq, 'xk': prod_const, 'mx': mx,
                                                       '^-1': reciprocal,
                                                       'tmht': total_mean_holding_time} | funcs),
                           sigma_functions=FunctionDict({'+': summation, '+T': transpose_summation}),
                           phi_functions=FunctionDict({'*': prod_emb, 'x': prod_unif, 'p2rate': p2rate}),
                           bottoms={'f': FixPointConfig(1, 0.0, 0.0001), 'p': FixPointConfig(1, 0.01, 0.00001)},
                           tops={},
                           config=config(NodeConfig(data_type, data_size), EdgeConfig(tf.float32, 3), tf.uint8))

    if formulae is None:
        expr = " || ".join(['(' + to_mG(formula, max_exit_rate) + ')' for formula in dataset.formulae])
    else:
        expr = " || ".join(['(' + to_mG(formula, max_exit_rate) + ')' for formula in formulae])
    return compiler.compile(expr, loss=tfa.metrics.HammingLoss(mode='multilabel'), optimize=optimize,
                            return_compilation_time=return_compilation_time)
