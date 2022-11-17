import numpy as np
import tensorflow as tf


def get_np_data_type(atomic_propositions_set):
    if len(atomic_propositions_set) <= 8:
        return np.uint8  # supports up to 8 atomic propositions
    elif len(atomic_propositions_set) <= 16:
        return np.uint16  # supports up to 16 atomic propositions
    elif len(atomic_propositions_set) <= 32:
        return np.uint32  # supports up to 32 atomic propositions
    elif len(atomic_propositions_set) <= 64:
        return np.uint64  # supports up to 64 atomic propositions
    else:
        raise ValueError("Too many atomic propositions...")


def get_tf_data_type(atomic_propositions_set):
    if len(atomic_propositions_set) <= 8:
        return tf.uint8  # supports up to 8 atomic propositions
    elif len(atomic_propositions_set) <= 16:
        return tf.uint16  # supports up to 16 atomic propositions
    elif len(atomic_propositions_set) <= 32:
        return tf.uint32  # supports up to 32 atomic propositions
    elif len(atomic_propositions_set) <= 64:
        return tf.uint64  # supports up to 64 atomic propositions
    else:
        raise ValueError("Too many atomic propositions...")


def to_one_hot(label, label_set, data_type):
    label_set = sorted(label_set)
    index = label_set.index(label)
    if len(label_set) > 64:
        vec = [0] * len(label_set)
        vec[index] = 1
        return tf.constant(vec, dtype=tf.uint8)
    else:
        return tf.constant(2 ** index, dtype=data_type)
