from __future__ import annotations

import unittest
import numpy as np

from cyz_m.spacetime_witness import stable_cad_certificate


class StableCADGuardTests(unittest.TestCase):
    def test_extra_response_copy_does_not_receive_one_target_certificate(self) -> None:
        base = np.asarray([[1.0, 0.0, 0.0]])
        extra = np.asarray([[0.0, 1.0, 0.0]])
        full = np.vstack([base, extra])
        certificate = stable_cad_certificate(base, [extra], full)
        self.assertFalse(certificate.certified_isomorphism)
        self.assertGreater(certificate.universality_gap, 0.6)
        self.assertAlmostEqual(certificate.base_atomic_kernel_gap, 1.0)
        self.assertEqual(certificate.descended_lower_singular_bound, 0.0)
        # Structural rejection (non-vacuous error bound): reported as 'failed',
        # distinct from the eps>=gamma 'inconclusive' branch.
        self.assertFalse(certificate.error_bound_vacuous)
        self.assertEqual(certificate.certificate_status, "failed")


if __name__ == "__main__":
    unittest.main()
