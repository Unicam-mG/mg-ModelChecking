import tensorflow as tf
from libmg import SingleGraphLoader, GNNCompiler, CompilationConfig, NodeConfig, EdgeConfig, FunctionDict, \
    FixPointConfig

from sources.csl.datasets.DebugCTMCDataset import DebugCTMCDataset
from sources.csl.CSL import b_not, b_or, b_true, b_and, make_atomic_propositions, pmode, prod_emb, summation, plus, mul, \
    prod_unif, prod_const, to_mG, build_model
from sources.csl.fox_glynn_algorithm import fox_glynn_algorithm


class CSLToMgTest(tf.test.TestCase):
    def test_parse(self):
        self.assertEqual('empty', to_mG('\"empty\"'))
        self.assertEqual('true', to_mG('true'))
        self.assertEqual('false', to_mG('false'))
        self.assertEqual('(empty);not', to_mG('!\"empty\"'))
        self.assertEqual('((empty) || (full));and', to_mG('\"empty\" & \"full\"'))
        self.assertEqual('((empty) || (full));or', to_mG('\"empty\" | \"full\"'))
        self.assertEqual('((full);%;|*>+);probgr[0.5]', to_mG('P>0.5 [ X \"full\"]'))
        self.assertEqual('((full);%;|*>+);probgreq[0.5]', to_mG('P>=0.5 [ X \"full\"]'))
        self.assertEqual('((full);%;|*>+);proble[0.5]', to_mG('P<0.5 [ X \"full\"]'))
        self.assertEqual('((full);%;|*>+);probleq[0.5]', to_mG('P<=0.5 [ X \"full\"]'))
        self.assertEqual('((full);%;|*>+)', to_mG('P=? [ X \"full\"]'))
        self.assertEqual('(mu X,f . ((((full);%) || (((((((empty);not) || (full);not);and);%) || (X;|*>+));*));+))',
                         to_mG('P=? [ !\"empty\" U \"full\" ]'))

        '''
        self.assertEqual('((succ);%)', to_mG('P=? [ \"try\" U<=0 \"succ\" ]'))
        
        self.assertEqual('((((succ);%) || (((((try) || (succ);not);and);%) || ((succ);%;|*>+));*);+)',
                         to_mG('P=? [ \"try\" U<=1 \"succ\" ]'))
        self.assertEqual('((((succ);%) || (((((try) || (succ);not);and);%) || ((((succ);%) || (((((try) || ('
                         'succ);not);and);%) || ((succ);%;|*>+));*);+;|*>+));*);+)',
                         to_mG('P=? [ \"try\" U<=2 \"succ\" ]'))
        '''


class CSLTest(tf.test.TestCase):

    def test_next(self):
        datasetex = DebugCTMCDataset("P=? [ X (\"full\") ]", qualitative=False)
        loader = SingleGraphLoader(datasetex, epochs=1)
        model = build_model(datasetex, max_exit_rate=4.5, config=CompilationConfig.xae_config)
        x, y = loader.load().__iter__().__next__()
        self.assertAllEqual(y, model.call(x))

    def test_unbounded_until(self):
        datasetuu = DebugCTMCDataset("P=? [ !\"empty\" U \"full\" ]", qualitative=False)
        loader = SingleGraphLoader(datasetuu, epochs=1)
        model = build_model(datasetuu, max_exit_rate=4.5, config=CompilationConfig.xae_config)
        x, y = loader.load().__iter__().__next__()
        self.assertAllClose(y, model.call(x), atol=0.001)

    def test_interval_until(self):
        # case 0, t
        datasetiu1 = DebugCTMCDataset("P=? [ true U[0, 7.5] \"full\" ]", qualitative=False)
        loader1 = SingleGraphLoader(datasetiu1, epochs=1)

        model = build_model(datasetiu1, max_exit_rate=4.5, config=CompilationConfig.xae_config)

        x, y = loader1.load().__iter__().__next__()
        self.assertAllClose(y, model.call(x))

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
