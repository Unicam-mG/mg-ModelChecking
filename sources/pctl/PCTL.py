import tensorflow as tf
from libmg import PsiLocal, Phi, Sigma

from sources.propositional_logic import true, false, And, Or, Not, make_atomic_propositions

# neXt
prod = Phi(lambda i, e, j: tf.math.multiply(e, j))
summation = Sigma(lambda m, i, n, x: tf.math.segment_sum(m, i))

# bounded Until #
# reach from phi1 = phi1 * X(phi2)
