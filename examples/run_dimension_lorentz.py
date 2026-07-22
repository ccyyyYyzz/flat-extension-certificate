from __future__ import annotations

import numpy as np

from cyz_m.dimension_lorentz import (
    audit_cone_universality,
    audit_sl2c_lorentz,
    co_marginal_canonical_drift,
    derive_dimension_selection,
    minkowski_metric,
    sl2c_boost,
    weyl_cone_metric,
)


def main() -> None:
    dimension = derive_dimension_selection(spectral_estimate=3.04, spectral_tolerance=0.1)
    print("target-blind spatial dimension:", dimension.selected_spatial_dimension)
    print("spacetime dimension:", dimension.selected_spacetime_dimension)
    print("spectral cross-check:", dimension.spectral_consistent)

    boost = audit_sl2c_lorentz(sl2c_boost(0.8, axis=2))
    print("Lorentz metric defect:", boost.metric_defect)
    print("proper/orthochronous:", boost.proper, boost.orthochronous)

    velocity = np.asarray([[1.0, 0.1, 0.0], [0.0, 0.9, 0.1], [0.0, 0.0, 1.1]])
    cone = weyl_cone_metric(velocity)
    print("Weyl cone signature:", (cone.signature_positive, cone.signature_negative))

    universality = audit_cone_universality([minkowski_metric(), 2.0 * minkowski_metric()])
    print("shared cone up to scale:", universality.universal)

    drift = co_marginal_canonical_drift(3.8)
    print(
        "co-marginal drift:",
        drift.normalized_gauge_drift,
        drift.normalized_yukawa_drift,
        drift.normalized_quartic_drift,
    )


if __name__ == "__main__":
    main()
