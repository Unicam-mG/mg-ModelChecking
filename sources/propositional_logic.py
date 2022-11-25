import tensorflow as tf
from libmg import PsiLocal

from sources.dataset_utils import to_one_hot

false = PsiLocal(lambda x: tf.zeros((tf.shape(x)[0], 1), dtype=tf.bool))
true = PsiLocal(lambda x: tf.ones((tf.shape(x)[0], 1), dtype=tf.bool))
And = PsiLocal(lambda x: tf.math.reduce_all(x, axis=1, keepdims=True))
Or = PsiLocal(lambda x: tf.math.reduce_any(x, axis=1, keepdims=True))
Not = PsiLocal(lambda x: tf.math.logical_not(x))


def make_atomic_propositions(atomic_proposition_set, mode, data_type):
    if mode == 'one_hot':
        funcs = {atom_prop: PsiLocal(lambda x, v=atom_prop: tf.cast(
            tf.math.reduce_sum(x * to_one_hot(v, atomic_proposition_set, data_type), axis=1, keepdims=True),
            dtype=tf.bool))
                 for atom_prop in atomic_proposition_set}
    else:
        funcs = {atom_prop: PsiLocal(lambda x, v=atom_prop: tf.cast(
            tf.bitwise.bitwise_and(x, to_one_hot(v, atomic_proposition_set, data_type)), tf.bool))
                 for atom_prop in atomic_proposition_set}
    return funcs
