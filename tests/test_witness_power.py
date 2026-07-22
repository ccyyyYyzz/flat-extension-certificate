from __future__ import annotations

import unittest
import numpy as np

from cyz_m.witness_power import (
    estimate_rank_power,
    estimate_signature_power,
    estimate_spatial_rank_power,
)


class WitnessPowerTests(unittest.TestCase):
    def test_rank_power_improves_with_shots(self) -> None:
        true = np.asarray([[1.5, 0.0, 0.0], [0.0, 1.2, 0.0], [0.0, 0.0, 1.0], [0.5, -0.3, 0.2]])
        result = estimate_rank_power(true, shots_per_setting=[300, 4000], trials=60)
        self.assertGreaterEqual(result[1].certification_power, result[0].certification_power)
        self.assertLessEqual(result[1].false_extra_rank_rate, 0.15)

    def test_signature_power_recovers_low_noise_inertia(self) -> None:
        rng = np.random.default_rng(12)
        q = rng.normal(size=(80, 3))
        metric = np.diag([1.0, 1.3, 0.8])
        result = estimate_signature_power(
            q,
            [0.1, -0.05, 0.03],
            metric,
            noise_standard_deviations=[0.0, 0.03],
            trials=50,
        )
        self.assertEqual(result[0].correct_inertia_rate, 1.0)
        self.assertGreater(result[1].correct_inertia_rate, 0.9)

    def test_spatial_rank_power_uses_an_independent_rank_deficient_null(self) -> None:
        rng = np.random.default_rng(18)
        q = rng.normal(size=(90, 3))
        result = estimate_spatial_rank_power(
            q,
            [0.08, -0.03, 0.02],
            np.diag([1.4, 1.0, 0.7]),
            noise_standard_deviations=[0.02],
            # 60 trials make the empirical q95 threshold a high-variance
            # estimator: ~1 in 19 seeds inflates the false-pass rate above
            # 0.15 with no estimator defect (mean rate across seeds ~ alpha).
            # 200 trials lock the threshold onto the true quantile.
            trials=200,
        )[0]
        self.assertGreater(result.full_rank_power, 0.8)
        self.assertLess(result.rank_deficient_false_pass_rate, 0.15)


if __name__ == "__main__":
    unittest.main()
