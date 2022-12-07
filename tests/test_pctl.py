import tensorflow as tf
from libmg import SingleGraphLoader, GNNCompiler, CompilationConfig, NodeConfig, EdgeConfig, FunctionDict, \
    FixPointConfig

from sources.pctl.PCTL import summation, prod, pmode, plus, mul, probgr, probgreq, rx, prod2
from sources.pctl.datasets.DebugDTMCDataset import DebugDTMCDataset
from sources.propositional_logic import make_atomic_propositions, b_not, b_or, b_true, b_and


class RandomKripkeTest(tf.test.TestCase):

    def test_prob(self):
        datasetprob = DebugDTMCDataset("P>=0.9 [ X (!\"try\" | \"succ\" ) ]", qualitative=True)
        loader = SingleGraphLoader(datasetprob, epochs=1)
        atom_props = make_atomic_propositions(['try', 'succ', 'fail'], mode='bitset', data_type=tf.uint8)
        compiler = GNNCompiler(FunctionDict({'not': b_not, 'or': b_or, 'pmode': pmode, 'probgreq': probgreq} | atom_props),
                               FunctionDict({'sum': summation}),
                               FunctionDict({'prod': prod}), {}, {},
                               CompilationConfig.xae_config(NodeConfig(tf.uint8, 1), EdgeConfig(tf.float32, 1),
                                                            tf.uint8))
        model = compiler.compile("((try;not) || succ);or;pmode;|prod>sum;probgreq[0.9]")
        x, y = loader.load().__iter__().__next__()
        self.assertAllEqual(model.call(x), y)

    def test_next(self):
        datasetex = DebugDTMCDataset("P=? [ X (!\"try\" | \"succ\" ) ]", qualitative=False)
        loader = SingleGraphLoader(datasetex, epochs=1)
        atom_props = make_atomic_propositions(['try', 'succ', 'fail'], mode='bitset', data_type=tf.uint8)
        compiler = GNNCompiler(FunctionDict({'not': b_not, 'or': b_or, 'pmode': pmode} | atom_props),
                               FunctionDict({'sum': summation}),
                               FunctionDict({'prod': prod}), {}, {},
                               CompilationConfig.xae_config(NodeConfig(tf.uint8, 1), EdgeConfig(tf.float32, 1),
                                                            tf.uint8))
        model = compiler.compile("((try;not) || succ);or;pmode;|prod>sum")
        x, y = loader.load().__iter__().__next__()
        self.assertAllEqual(model.call(x), y)

    def test_bounded_until(self):
        atom_props = make_atomic_propositions(['try', 'succ', 'fail'], mode='bitset', data_type=tf.uint8)
        compiler = GNNCompiler(FunctionDict({'not': b_not, 'or': b_or, 'pmode': pmode, 'plus': plus, 'mul': mul, 'true': b_true, 'and': b_and} | atom_props),
                               FunctionDict({'sum': summation, 'rx': rx}), FunctionDict({'prod': prod, 'prod2': prod2}), {}, {},
                               CompilationConfig.xae_config(NodeConfig(tf.uint8, 1), EdgeConfig(tf.float32, 1),
                                                            tf.uint8))

        datasetbu1 = DebugDTMCDataset("P=? [ \"try\" U<=2 \"succ\" ]", qualitative=False)
        loader1 = SingleGraphLoader(datasetbu1, epochs=1)
        model = compiler.compile('((succ;pmode) || (((try;pmode) || (((succ;pmode) || (((try;pmode) || (succ;pmode;|prod>sum));mul));plus;|prod>sum));mul));plus')
        # try if-else version
        model = compiler.compile('(try || succ);|prod2>rx')
        # remember to set phi1 and not phi2
        x, y = loader1.load().__iter__().__next__()
        self.assertAllClose(y, model.call(x))

        datasetbu2 = DebugDTMCDataset("P=? [ true U<=2 \"succ\" ]", qualitative=False)
        loader2 = SingleGraphLoader(datasetbu2, epochs=1)
        model = compiler.compile('((succ;pmode) || (((((true || succ;not);and);pmode) || (((succ;pmode) || (((((true || succ;not);and);pmode) || (succ;pmode;|prod>sum));mul));plus;|prod>sum));mul));plus')
        x, y = loader2.load().__iter__().__next__()
        self.assertAllClose(model.call(x), y)

    def test_unbounded_until(self):
        atom_props = make_atomic_propositions(['try', 'succ', 'fail'], mode='bitset', data_type=tf.uint8)
        compiler = GNNCompiler(FunctionDict({'not': b_not, 'or': b_or, 'pmode': pmode, 'plus': plus, 'mul': mul, 'true': b_true, 'and': b_and} | atom_props),
                               FunctionDict({'sum': summation}), FunctionDict({'prod': prod}), {'f': FixPointConfig(1, 0.0, 0.001)}, {},
                               CompilationConfig.xae_config(NodeConfig(tf.uint8, 1), EdgeConfig(tf.float32, 1),
                                                            tf.uint8))
        datasetuu = DebugDTMCDataset("P=? [ \"try\" U \"succ\" ]", qualitative=False)
        loader = SingleGraphLoader(datasetuu, epochs=1)
        model = compiler.compile('mu X,f . (((succ;pmode) || (((try;pmode) || (X;|prod>sum));mul));plus)')
        x, y = loader.load().__iter__().__next__()
        self.assertAllClose(model.call(x), y)


if __name__ == '__main__':
    tf.test.main()