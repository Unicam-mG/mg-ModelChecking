import tensorflow as tf
from libmg import PsiLocal, Phi, Sigma

pmode = PsiLocal(lambda x: tf.cast(x, dtype=tf.float32))

# prob operator
probgr = lambda y: PsiLocal(lambda x: tf.math.greater(x, float(y)))
probgreq = lambda y: PsiLocal(lambda x: tf.math.greater_equal(x, float(y)))
proble = lambda y: PsiLocal(lambda x: tf.math.less(x, float(y)))
probleq = lambda y: PsiLocal(lambda x: tf.math.less_equal(x, float(y)))

# neXt is |prod>sum
prod = Phi(lambda i, e, j: tf.math.multiply(e, j))  # transform bools to floats
summation = Sigma(lambda m, i, n, x: tf.math.segment_sum(m, i))

# bounded Until #
# reach from phi1 = phi1 * X(phi2)
plus = PsiLocal(lambda x: tf.math.reduce_sum(x, axis=1, keepdims=True))
mul = PsiLocal(lambda x: tf.math.reduce_prod(x, axis=1, keepdims=True))


#rx = PsiLocal(lambda x: tf.where(x[:, 0:1], tf.constant(1.0, dtype=tf.float32), tf.where(x[:, 1:2], tf.constant(0.5, tf.float32), tf.constant(0.0, tf.float32))))
prod2 = Phi(lambda i, e, j: tf.math.multiply(e, tf.cast(j[:, 1:2], tf.float32)))  # transform bools to floats
rx = Sigma(lambda m, i, n, x: tf.where(x[:, 1:2], tf.constant(1.0, dtype=tf.float32), tf.where(x[:, 0:1], tf.math.segment_sum(m, i), tf.constant(0.0, tf.float32))))