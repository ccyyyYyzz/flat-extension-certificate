"""Demonstrate event-individuation relativity and its forbidden boundary."""
from __future__ import annotations

import numpy as np

from cyz_m import (
    audit_event_basis_change,
    hidden_tensor_events,
    mix_event_frame,
    search_resolved_factorizations,
    similarity_star_defect,
    state_weighted_event_gram,
    transpose_choi_min_eigenvalue,
)


def main() -> None:
    events, _, _ = hidden_tensor_events(2, 2, seed=61)
    sectors = ((0, 1), (2, 3))
    base = search_resolved_factorizations(events, sectors)

    block_gl = np.zeros((4, 4), dtype=np.complex128)
    block_gl[:2, :2] = np.asarray([[1.0, 0.4j], [0.2, 1.0]])
    block_gl[2:, 2:] = np.asarray([[1.0, 0.3], [-0.1j, 1.0]])
    audit = audit_event_basis_change(block_gl, sectors)
    changed = search_resolved_factorizations(mix_event_frame(events, block_gl), sectors)

    state = np.eye(4, dtype=np.complex128) / 4.0
    metric = state_weighted_event_gram(events, state)
    metric_audit = audit_event_basis_change(block_gl, sectors, metric=metric)

    cross_sector = np.eye(4, dtype=np.complex128)
    cross_sector[0, 2] = 0.25
    cross_audit = audit_event_basis_change(cross_sector, sectors)

    assert base.best is not None and changed.best is not None
    print("base cut:", base.best.left_events, "|", base.best.right_events)
    print("block-GL allowed (bare sector):", audit.allowed)
    print("same score after frame change:", abs(base.best.score - changed.best.score) < 1e-10)
    print("same block-GL after metric calibration:", metric_audit.allowed)
    print("cross-sector mixing allowed:", cross_audit.allowed)
    print("nonunitary similarity *-defect:", similarity_star_defect(np.diag([2, 1, 1, 1])))
    print("transpose Choi minimum eigenvalue:", transpose_choi_min_eigenvalue(2))


if __name__ == "__main__":
    main()
