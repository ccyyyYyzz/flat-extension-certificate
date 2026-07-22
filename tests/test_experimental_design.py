from __future__ import annotations

import unittest
import numpy as np

from cyz_m.experimental_design import greedy_null_sheet_design, ramsey_frequency_budget


class ExperimentalDesignTests(unittest.TestCase):
    def test_greedy_design_is_target_blind_and_full_rank(self) -> None:
        rng = np.random.default_rng(91)
        candidates = rng.normal(size=(120, 3))
        design = greedy_null_sheet_design(candidates, sample_count=14)
        self.assertEqual(len(design.selected_indices), 14)
        self.assertEqual(design.linear_singular_values.size, 3)
        self.assertEqual(design.quadratic_singular_values.size, 6)
        self.assertTrue(np.isfinite(design.linear_condition_number))
        self.assertTrue(np.isfinite(design.quadratic_condition_number))

    def test_ramsey_budget_uses_optimal_time_for_dephasing_model(self) -> None:
        budget = ramsey_frequency_budget(0.02, dephasing_rate=0.5, contrast=0.9)
        self.assertAlmostEqual(budget.interrogation_time, 2.0)
        self.assertGreater(budget.required_shots, 0)
        self.assertLessEqual(budget.frequency_standard_error, 0.02)


if __name__ == "__main__":
    unittest.main()
