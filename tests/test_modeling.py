import importlib.util
import unittest

HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None


@unittest.skipUnless(HAS_SKLEARN, "scikit-learn is required for model-generation checks")
class ModelingTests(unittest.TestCase):
    def test_synthetic_generator_is_reproducible_and_has_expected_features(self):
        from med_reserve.modeling import generate_synthetic_data

        first = generate_synthetic_data(samples=50, seed=7)
        second = generate_synthetic_data(samples=50, seed=7)
        self.assertTrue(first.equals(second))
        self.assertEqual(set(first.columns), {"Age", "SpO2", "Heart_Rate", "Comorbidities", "Triage_Level"})
        self.assertTrue(set(first["Triage_Level"].unique()).issubset({0, 1, 2}))

    def test_small_sample_is_rejected(self):
        from med_reserve.modeling import generate_synthetic_data

        with self.assertRaises(ValueError):
            generate_synthetic_data(samples=10)


if __name__ == "__main__":
    unittest.main()

