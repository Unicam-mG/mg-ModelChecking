import tensorflow as tf
from libmg import PsiLocal, Phi, Sigma

from sources.propositional_logic import b_not, b_or, b_true, b_and, b_false, make_atomic_propositions
from sources.pctl.PCTL import pmode, probgr, probgreq, proble, probleq, summation, plus, mul

# neXt is |prod>sum like PCTL but our edges are labelled differently [rate, emb(c), unif(c)]
# prod with emb(c)
prod_emb = Phi(lambda i, e, j: tf.math.multiply(e[:, 1:2], j))

# prod with unif(c)
prod_unif = Phi(lambda i, e, j: tf.math.multiply(e[:, 2:3], j))

# const-prod
prod_const = lambda y: PsiLocal(lambda x: tf.math.multiply(x, float(y)))