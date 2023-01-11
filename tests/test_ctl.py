import tensorflow as tf

from libmg import CompilationConfig
from libmg import MultipleGraphLoader, SingleGraphLoader
from sources.ctl.CTL import build_model, to_mG
from sources.ctl.datasets.pnml_kripke_dataset import PetriNetDataset, MCCTypes
from sources.ctl.datasets.random_kripke_dataset import RandomKripkeDataset


class CTLToMgTest(tf.test.TestCase):
    def test_parse(self):
        self.assertEqual('try', to_mG('try'))
        self.assertEqual('true', to_mG('true'))
        self.assertEqual('false', to_mG('false'))
        self.assertEqual('(try);not', to_mG('~try'))
        self.assertIn(to_mG('try & succ'), {'((try) || (succ));and', '((succ) || (try));and'})
        self.assertIn(to_mG('try | succ'), {'((try) || (succ));or', '((succ) || (try));or'})
        self.assertEqual('(try);|p3>or', to_mG('E X try'))
        self.assertEqual('nu X:bool[1] = true . (((try) || (X;|p3>or));and)', to_mG('E G try'))
        self.assertEqual('mu X:bool[1] = false . (((((try) || (X;|p3>or));and) || (succ));or)', to_mG('E try U succ'))


class RandomKripkeTest(tf.test.TestCase):

    def setUp(self):
        super().setUp()
        self.formulae = ['true', 'false', 'a', 'not a', 'a and b', 'a or b', 'a --> b', 'E X a', 'E G a', 'E F a',
                         'E (a U b)', 'A X a', 'A G a', 'A F a', 'A (a U b)']
        self.atomic_propositions = ['a', 'b', 'c']
        self.dataset = RandomKripkeDataset(100, 1000, 100, self.atomic_propositions,
                                           self.formulae, name='Parser_Test_Dataset', skip_model_checking=False,
                                           probabilistic=False)
        self.model = build_model(self.dataset)

    def test_correctness(self):
        d_loader = MultipleGraphLoader(self.dataset, node_level=True, batch_size=1, shuffle=False, epochs=1)
        for inputs, y in d_loader.load():
            outputs = self.model.call([inputs], training=False)
            self.assertAllEqual(outputs, y)


class PetriNetTest(tf.test.TestCase):

    def setUp(self):
        super().setUp()
        self.dataset = PetriNetDataset('Dekker-PT-010', MCCTypes.FIREABILITY)

    def test_correctness(self):
        model = build_model(self.dataset, config=CompilationConfig.xa_config)
        d_loader = SingleGraphLoader(self.dataset, epochs=1)
        for inputs, y in d_loader.load():
            outputs = model.call([inputs], training=False)
            self.assertAllEqual(outputs, y)

    def test_correctness_large_scale(self):
        models = []
        for formula in self.dataset.formulae:
            models.append(build_model(self.dataset, [formula], config=CompilationConfig.xa_config))
        d_loader = SingleGraphLoader(self.dataset, epochs=1)
        for inputs, y in d_loader.load():
            for i in range(len(models)):
                outputs = models[i](inputs)
                self.assertAllEqual(outputs, y[:, i:i+1])

    def test_predict(self):
        model = build_model(self.dataset, config=CompilationConfig.xa_config)
        output_loader = SingleGraphLoader(self.dataset, epochs=1)
        inputs, y = output_loader.load().__iter__().__next__()
        d_loader = SingleGraphLoader(self.dataset, epochs=1)
        outputs = model.predict(d_loader.load(), steps=d_loader.steps_per_epoch)
        self.assertAllEqual(outputs, y)


if __name__ == '__main__':
    tf.test.main()
