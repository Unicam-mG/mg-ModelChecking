import tensorflow as tf
from libmg import SingleGraphLoader, GNNCompiler, CompilationConfig, NodeConfig, EdgeConfig, FunctionDict, \
    FixPointConfig

from sources.csl.datasets.DebugCTMCDataset import DebugCTMCDataset
from sources.csl.CSL import b_not, b_or, b_true, b_and, make_atomic_propositions, pmode, prod_emb, summation, plus, mul, \
    prod_unif, prod_const
from sources.csl.fox_glynn_algorithm import fox_glynn_algorithm


class CSLTest(tf.test.TestCase):

    def test_next(self):
        datasetex = DebugCTMCDataset("P=? [ X (\"full\") ]", qualitative=False)
        loader = SingleGraphLoader(datasetex, epochs=1)
        atom_props = make_atomic_propositions(['empty', 'full'], mode='bitset', data_type=tf.uint8)
        compiler = GNNCompiler(FunctionDict({'not': b_not, 'or': b_or, 'pmode': pmode} | atom_props),
                               FunctionDict({'sum': summation}),
                               FunctionDict({'prod': prod_emb}), {}, {},
                               CompilationConfig.xae_config(NodeConfig(tf.uint8, 1), EdgeConfig(tf.float32, 1),
                                                            tf.uint8))
        model = compiler.compile("full;pmode;|prod>sum")
        x, y = loader.load().__iter__().__next__()
        self.assertAllEqual(y, model.call(x))

    def test_unbounded_until(self):
        atom_props = make_atomic_propositions(['empty', 'full'], mode='bitset', data_type=tf.uint8)
        compiler = GNNCompiler(FunctionDict(
            {'not': b_not, 'or': b_or, 'pmode': pmode, 'plus': plus, 'mul': mul, 'true': b_true,
             'and': b_and} | atom_props),
                               FunctionDict({'sum': summation}), FunctionDict({'prod': prod_emb}),
                               {'f': FixPointConfig(1, 0.0, 0.001)}, {},
                               CompilationConfig.xae_config(NodeConfig(tf.uint8, 1), EdgeConfig(tf.float32, 1),
                                                            tf.uint8))
        datasetuu = DebugCTMCDataset("P=? [ !\"empty\" U \"full\" ]", qualitative=False)
        loader = SingleGraphLoader(datasetuu, epochs=1)
        model = compiler.compile(
            'mu X,f . (((full;pmode) || (((((empty;not) || (full;not));and;pmode) || (X;|prod>sum));mul));plus)')
        x, y = loader.load().__iter__().__next__()
        self.assertAllClose(y, model.call(x), atol=0.001)

    def test_interval_until(self):
        atom_props = make_atomic_propositions(['empty', 'full'], mode='bitset', data_type=tf.uint8)
        compiler = GNNCompiler(FunctionDict(
            {'not': b_not, 'or': b_or, 'pmode': pmode, 'plus': plus, 'mul': mul, 'true': b_true, 'and': b_and,
             'prodc': prod_const} | atom_props),
                               FunctionDict({'sum': summation}), FunctionDict({'prod': prod_unif}),
                               {'f': FixPointConfig(1, 0.0, 0.001)}, {},
                               CompilationConfig.xae_config(NodeConfig(tf.uint8, 1), EdgeConfig(tf.float32, 1),
                                                            tf.uint8))

        # case 0, t
        datasetiu1 = DebugCTMCDataset("P=? [ true U[0, 7.5] \"full\" ]", qualitative=False)
        loader1 = SingleGraphLoader(datasetiu1, epochs=1)

        # do a static fox-glynn
        qt = 7.5 * 4.5
        weights, left, right, total_weight = fox_glynn_algorithm(qt, 1.0e-300, 1.0e+300, 1.0e-10)
        for i in range(left, right + 1):
            weights[i - left] = weights[i - left] / total_weight
        print(weights, left, right, total_weight)

        model = compiler.compile('((full;pmode;prodc[' + str(weights[0]) + ']) || ( ((full;not;pmode) || (full;pmode;|prod>sum)) ; mul; prodc[' + str(weights[1]) + ']));plus')
        x, y = loader1.load().__iter__().__next__()
        print(model.call(x), y)
        # self.assertAllClose(y, model.call(x))

        """
        datasetbu2 = DebugCTMCDataset("P=? [ true U<=2 \"succ\" ]", qualitative=False)
        loader2 = SingleGraphLoader(datasetbu2, epochs=1)
        model = compiler.compile('((succ;pmode) || (((((true || succ;not);and);pmode) || (((succ;pmode) || (((((true || succ;not);and);pmode) || (succ;pmode;|prod>sum));mul));plus;|prod>sum));mul));plus')
        x, y = loader2.load().__iter__().__next__()
        self.assertAllClose(model.call(x), y)
        """

    def test_steady_state(self):
        pass


if __name__ == '__main__':
    tf.test.main()
