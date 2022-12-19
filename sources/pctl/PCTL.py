import tensorflow as tf
import tensorflow_addons as tfa
import stormpy
from libmg import PsiLocal, Phi, Sigma, CompilationConfig, GNNCompiler, FunctionDict, FixPointConfig, NodeConfig, \
    EdgeConfig
from stormpy import BooleanLiteralFormula, UnaryBooleanStateFormula, BooleanBinaryStateFormula, ComparisonType

from sources.dataset_utils import get_tf_data_type
from sources.propositional_logic import make_atomic_propositions, b_not, b_or, b_true, b_and, b_false

pmode = PsiLocal(lambda x: tf.cast(x, dtype=tf.float32))

# prob operator
probgr = lambda y: PsiLocal(lambda x: tf.math.greater(x, float(y)))
probgreq = lambda y: PsiLocal(lambda x: tf.math.greater_equal(x, float(y)))
proble = lambda y: PsiLocal(lambda x: tf.math.less(x, float(y)))
probleq = lambda y: PsiLocal(lambda x: tf.math.less_equal(x, float(y)))

# neXt is |prod>sum
prod = Phi(lambda i, e, j: tf.math.multiply(e, j))  # transform bools to floats
summation = Sigma(lambda m, i, n, x: tf.math.segment_sum(m, i))

# bounded Until is phi2 + [(phi1 and not phi2) * X(phi1 U^{k-1} phi2)]
# unbounded Until is the LFP of the bounded until formula
# reach from phi1 = phi1 * X(phi2)
plus = PsiLocal(lambda x: tf.math.reduce_sum(x, axis=1, keepdims=True))
mul = PsiLocal(lambda x: tf.math.reduce_prod(x, axis=1, keepdims=True))


def to_mG(expr):
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
            steps = phi.upper_bound_expression.evaluate_as_int()
            phi1 = _to_mG(phi.left_subformula)
            phi2 = _to_mG(phi.right_subformula)
            phi1_and_not_phi2 = '((' + phi1 + ') || (' + phi2 + ');not);and'
            output = '(' + phi2 + ');%'
            for _ in range(steps):
                output = '(((' + phi2 + ');%) || (((' + phi1_and_not_phi2 + ');%) || (' + output + ';|*>+));*);+'
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


def build_model(dataset, formulae=None, config=CompilationConfig.xaei_config, optimize=None,
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
                                                       'probleq': probleq} | funcs),
                           sigma_functions=FunctionDict({'+': summation}),
                           phi_functions=FunctionDict({'*': prod}),
                           bottoms={'f': FixPointConfig(1, 0.0, 0.001)},
                           tops={},
                           config=config(NodeConfig(data_type, data_size), EdgeConfig(tf.float32, 1), tf.uint8))
    if formulae is None:
        expr = " || ".join([to_mG(formula) for formula in dataset.formulae])
    else:
        expr = " || ".join([to_mG(formula) for formula in formulae])
    return compiler.compile(expr, loss=tfa.metrics.HammingLoss(mode='multilabel'), optimize=optimize,
                            return_compilation_time=return_compilation_time)
