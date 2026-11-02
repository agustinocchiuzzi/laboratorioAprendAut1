import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from test_ablacion import holm, comparar_una


class AblationStatisticsTests(unittest.TestCase):
    def test_holm_preserves_input_order_and_monotonic_adjustment(self):
        actual = holm([.04, .01, .03], 3)
        for value, expected in zip(actual, [.06, .03, .06]):
            self.assertAlmostEqual(value, expected)

    def test_paired_sensitivity_matches_seed_keys_not_insertion_order(self):
        full = {s: float(20+s) for s in range(8)}
        removal = {s: float(s) for s in reversed(range(8))}
        result = comparar_una(full, removal)
        self.assertEqual(result['n_pares'], 8)
        self.assertEqual(result['completo_gana'], 8)
        self.assertEqual(result['W'], 0)
        self.assertAlmostEqual(result['p_wilcoxon'], .0078125)
        self.assertEqual(result['prob_superioridad'], 1)


if __name__ == '__main__':
    unittest.main()
