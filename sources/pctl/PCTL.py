import tensorflow as tf
from libmg import PsiLocal, Phi, Sigma, SingleGraphLoader, GNNCompiler, CompilationConfig, NodeConfig, EdgeConfig

from sources.pctl.datasets.DebugDTMCDataset import DebugDTMCDataset
from sources.propositional_logic import true, false, And, Or, Not, make_atomic_propositions


# neXt is |prod>sum
prod = Phi(lambda i, e, j: tf.math.multiply(e, tf.cast(j, dtype=tf.float32)))  # transform bools to floats
summation = Sigma(lambda m, i, n, x: tf.math.segment_sum(m, i))

# bounded Until #
# reach from phi1 = phi1 * X(phi2)


if __name__ == '__main__':
    datasetex = DebugDTMCDataset("P=? [ X (!\"try\" | \"succ\" ) ]", qualitative=False)
    loader = SingleGraphLoader(datasetex, epochs=1)
    atom_props = make_atomic_propositions(['try', 'succ', 'fail'], mode='bitset', data_type=tf.uint8)
    compiler = GNNCompiler({'not': Not, 'or': Or} | atom_props, {'sum': summation}, {'prod': prod}, {}, {},
                           CompilationConfig.xae_config(NodeConfig(tf.uint8, 1), EdgeConfig(tf.float32, 1), tf.uint8))
    model = compiler.compile("((try;not) || succ);or;|prod>sum")
    x, y = loader.load().__iter__().__next__()
    print(model.call(x))
    print(datasetex[0].y)
    #datasetbu = DebugDTMCDataset("P=? [ \"try\" U<=2 \"succ\" ]", qualitative=False)
    #print(datasetbu[0].y)
    #datasetuu = DebugDTMCDataset("P>0.99 [ \"try\" U \"succ\" ]")