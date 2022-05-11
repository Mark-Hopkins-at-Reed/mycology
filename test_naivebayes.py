import unittest
import pandas as pd
from naivebayes import JointDistribution
from naivebayes import add_k_smoothing, extract_signature, train_cpt, train_naive_bayes

class TestNaiveBayes(unittest.TestCase):

    def test_add_k_smoothing(self):
        event_counts = {('a',): 4, ('b',): 2, ('c',): 1}
        dist = add_k_smoothing(event_counts, k=1)
        self.assertEqual(dist, {('a',): 0.5, ('b',): 0.3, ('c',): 0.2})

    def test_joint_dist(self):
        event_counts = {('a',): 4, ('b',): 2, ('c',): 1}
        dist = JointDistribution(['letter'], event_counts, lambda c: add_k_smoothing(c, k=1))
        self.assertEqual(dist.prob(['a']), 0.5)
        self.assertEqual(dist.prob(['b']), 0.3)
        self.assertEqual(dist.prob(['c']), 0.2)

    def test_extract_signature(self):
        data = pd.read_csv('mushrooms/agaricus-lepiota.csv')
        signature = extract_signature(data)
        self.assertEqual(signature['poisonous'], ('e', 'p'))
        self.assertEqual(signature['veil-color'], ('n', 'o', 'w', 'y'))

    def test_train_cpt(self):
        data = pd.read_csv('mushrooms/agaricus-lepiota.csv')
        cpt = train_cpt(data, 'cap-color', ['poisonous'])
        print(cpt.prob('w', ['p']))
        print(cpt.prob('r', ['p']))

    def test_train_naive_bayes(self):
        data = pd.read_csv('mushrooms/agaricus-lepiota.csv')
        naive_bayes = train_naive_bayes(data, 'poisonous')
        print(naive_bayes.posterior({'cap-color': 'e'}))


if __name__ == "__main__":
    unittest.main()   