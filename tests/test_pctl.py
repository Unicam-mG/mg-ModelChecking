import tensorflow as tf
from libmg import SingleGraphLoader, CompilationConfig

from sources.pctl.PCTL import to_mG, build_model
from sources.pctl.datasets.DebugDTMCDataset import DebugDTMCDataset


class PCTLToMgTest(tf.test.TestCase):
    def test_parse(self):
        self.assertEqual('try', to_mG('\"try\"'))
        self.assertEqual('true', to_mG('true'))
        self.assertEqual('false', to_mG('false'))
        self.assertEqual('(try);not', to_mG('!\"try\"'))
        self.assertEqual('((try) || (succ));and', to_mG('\"try\" & \"succ\"'))
        self.assertEqual('((try) || (succ));or', to_mG('\"try\" | \"succ\"'))
        self.assertEqual('((succ);%;|*>+);probgr[0.5]', to_mG('P>0.5 [ X \"succ\"]'))
        self.assertEqual('((succ);%;|*>+);probgreq[0.5]', to_mG('P>=0.5 [ X \"succ\"]'))
        self.assertEqual('((succ);%;|*>+);proble[0.5]', to_mG('P<0.5 [ X \"succ\"]'))
        self.assertEqual('((succ);%;|*>+);probleq[0.5]', to_mG('P<=0.5 [ X \"succ\"]'))
        self.assertEqual('((succ);%;|*>+)', to_mG('P=? [ X \"succ\"]'))
        self.assertEqual('((succ);%)', to_mG('P=? [ \"try\" U<=0 \"succ\" ]'))
        self.assertEqual('((((succ);%) || (((((try) || (succ);not);and);%) || ((succ);%;|*>+));*);+)',
                         to_mG('P=? [ \"try\" U<=1 \"succ\" ]'))
        self.assertEqual('((((succ);%) || (((((try) || (succ);not);and);%) || ((((succ);%) || (((((try) || ('
                         'succ);not);and);%) || ((succ);%;|*>+));*);+;|*>+));*);+)',
                         to_mG('P=? [ \"try\" U<=2 \"succ\" ]'))
        self.assertEqual('(mu X,f . ((((succ);%) || ((((((try) || (succ);not);and);%) || (X;|*>+));*));+))',
                         to_mG('P=? [ \"try\" U \"succ\" ]'))


class PCTLTest(tf.test.TestCase):

    def test_prob(self):
        datasetprob = DebugDTMCDataset("P>=0.9 [ X (!\"try\" | \"succ\" ) ]", qualitative=True)
        loader = SingleGraphLoader(datasetprob, epochs=1)
        model = build_model(datasetprob, config=CompilationConfig.xae_config)
        x, y = loader.load().__iter__().__next__()
        self.assertAllEqual(y, model.call(x))

    def test_next(self):
        datasetex = DebugDTMCDataset("P=? [ X (!\"try\" | \"succ\" ) ]", qualitative=False)
        loader = SingleGraphLoader(datasetex, epochs=1)
        model = build_model(datasetex, config=CompilationConfig.xae_config)
        x, y = loader.load().__iter__().__next__()
        self.assertAllEqual(y, model.call(x))

    def test_bounded_until(self):
        datasetbu1 = DebugDTMCDataset("P=? [ \"try\" U<=2 \"succ\" ]", qualitative=False)
        loader1 = SingleGraphLoader(datasetbu1, epochs=1)
        model = build_model(datasetbu1, config=CompilationConfig.xae_config)
        x, y = loader1.load().__iter__().__next__()
        self.assertAllClose(y, model.call(x))

        datasetbu2 = DebugDTMCDataset("P=? [ true U<=2 \"succ\" ]", qualitative=False)
        loader2 = SingleGraphLoader(datasetbu2, epochs=1)
        model = build_model(datasetbu2, config=CompilationConfig.xae_config)
        x, y = loader2.load().__iter__().__next__()
        self.assertAllClose(y, model.call(x))

    def test_unbounded_until(self):
        datasetuu = DebugDTMCDataset("P=? [ \"try\" U \"succ\" ]", qualitative=False)
        loader = SingleGraphLoader(datasetuu, epochs=1)
        model = build_model(datasetuu, config=CompilationConfig.xae_config)
        x, y = loader.load().__iter__().__next__()
        self.assertAllClose(y, model.call(x), atol=0.001)


if __name__ == '__main__':
    tf.test.main()
