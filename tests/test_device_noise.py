"""Tests for cyz_m.device_noise: CP/TP, ideal limit, and noise monotonicity."""

from __future__ import annotations

import dataclasses

import numpy as np

from cyz_m.device_noise import (
    PlatformModel,
    kraus_depolarizing_1q,
    kraus_depolarizing_2q,
    kraus_dephasing_1q,
    dephasing_p_from_zz_kick,
    assert_cptp,
    choi_min_eig,
    tp_defect,
    comb_choi_on_S,
    validate_all_channels,
    noisy_separator,
    noisy_slope_at_zero,
    noisy_atomic_response,
    noisy_correlator,
    noisy_comb_output,
    correlator_outcome_probs,
    correlator_from_probs,
    partial_trace,
    ideal_separator,
    dm,
    PAULI,
    embed1,
)

DELTA_GRID = np.linspace(0.0, 0.5, 11)
PHI_KICK_GRID = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1]

_kp = (np.array([1, 0], complex) + np.array([0, 1], complex)) / np.sqrt(2)


def _noiseless(mid: str = "ion") -> PlatformModel:
    if mid == "ion":
        return PlatformModel(
            name="noiseless-ion", citation="-", mid_circuit="ion",
            err_1q=0.0, err_2q=0.0, prep_error=0.0, readout_error=0.0,
            ion_dephasing=0.0, ion_crosstalk=0.0,
        )
    return PlatformModel(
        name="noiseless-ibm", citation="-", mid_circuit="ibm",
        err_1q=0.0, err_2q=0.0, prep_error=0.0, readout_error=0.0,
        T2=1.0e30, mcm_window=0.0,
    )


# --------------------------------------------------------------------------- #
# CP / TP of every channel                                                    #
# --------------------------------------------------------------------------- #
def test_primitive_channels_cptp():
    for p in [0.0, 1e-4, 1.83e-3, 0.1, 0.5, 1.0]:
        assert_cptp(kraus_depolarizing_1q(p), 2, f"depol1q({p})")
        assert_cptp(kraus_depolarizing_2q(p), 4, f"depol2q({p})")
        assert_cptp(kraus_dephasing_1q(p), 2, f"deph({p})")


def test_zz_kick_channel_cptp():
    for phi in PHI_KICK_GRID + [0.5, 1.0, np.pi]:
        p = dephasing_p_from_zz_kick(phi)
        assert 0.0 - 1e-15 <= p <= 1.0 + 1e-15
        assert_cptp(kraus_dephasing_1q(p), 2, f"zz_kick({phi})")
    # zero kick is the identity channel
    assert abs(dephasing_p_from_zz_kick(0.0)) < 1e-15


def test_platform_channels_cptp():
    for platform in (PlatformModel.quantinuum_h2(), PlatformModel.ibm_heron_r2()):
        report = validate_all_channels(platform)
        assert len(report) >= 5
        for name, mn, tp in report:
            assert mn >= -1e-12, f"{platform.name}:{name} not CP (min eig {mn})"
            assert tp <= 1e-10, f"{platform.name}:{name} not TP (defect {tp})"


def test_full_comb_choi_cptp_over_grid():
    for platform in (PlatformModel.quantinuum_h2(), PlatformModel.ibm_heron_r2()):
        for delta in [0.0, 0.1, 0.3, 0.5]:
            for phi in [0.0, 0.05, 0.1]:
                J = comb_choi_on_S(platform, delta, phi)
                Jh = (J + J.conj().T) / 2.0
                mn = float(np.min(np.linalg.eigvalsh(Jh)).real)
                tr_out = partial_trace(J, keep=[0], dims=[4, 4])
                tp = float(np.max(np.abs(tr_out - np.eye(4))))
                assert mn >= -1e-12, f"{platform.name} comb Choi not CP ({mn})"
                assert tp <= 1e-10, f"{platform.name} comb not TP ({tp})"


# --------------------------------------------------------------------------- #
# ideal limit: zero noise reproduces sin(delta)                               #
# --------------------------------------------------------------------------- #
def test_ideal_limit_separator_is_sin():
    for mid in ("ion", "ibm"):
        model = _noiseless(mid)
        for d in DELTA_GRID:
            v = noisy_separator(model, float(d), 0.0)
            assert abs(v - np.sin(d)) < 1e-12, f"{mid} E({d})={v} != sin"


def test_ideal_limit_via_four_outcome_probs():
    """Zero-readout 4-outcome correlator path equals sin(delta) too."""
    model = _noiseless("ion")
    for d in DELTA_GRID:
        rho_S = noisy_comb_output(model, dm(_kp), dm(_kp), float(d), 0.0)
        p = correlator_outcome_probs(rho_S, "Y", "X", 0.0)
        c = correlator_from_probs(p)
        assert abs(np.sum(p) - 1.0) < 1e-12
        assert abs(c - np.sin(d)) < 1e-12


def test_ideal_slope_is_one():
    for mid in ("ion", "ibm"):
        s = noisy_slope_at_zero(_noiseless(mid), 0.0)
        assert abs(s - 1.0) < 1e-8, f"{mid} ideal slope {s} != 1"


def test_atomic_family_is_dark_even_with_noise():
    """The single-time IC atomic family is delta-invariant (dark) under noise."""
    for platform in (PlatformModel.quantinuum_h2(), PlatformModel.ibm_heron_r2()):
        a0 = noisy_atomic_response(platform, 0.0, 0.05)
        assert len(a0) == 48
        for d in [0.05, 0.2, 0.5]:
            ad = noisy_atomic_response(platform, d, 0.05)
            assert np.max(np.abs(ad - a0)) < 1e-10, platform.name


def test_separator_zero_at_delta_zero():
    for platform in (PlatformModel.quantinuum_h2(), PlatformModel.ibm_heron_r2()):
        for phi in PHI_KICK_GRID:
            assert abs(noisy_separator(platform, 0.0, phi)) < 1e-12


# --------------------------------------------------------------------------- #
# monotonicity: more noise => smaller slope                                   #
# --------------------------------------------------------------------------- #
def test_noise_reduces_slope_below_ideal():
    for platform in (PlatformModel.quantinuum_h2(), PlatformModel.ibm_heron_r2()):
        s = noisy_slope_at_zero(platform, 0.0)
        assert 0.0 < s < 1.0, f"{platform.name} slope {s} not in (0,1)"


def test_slope_monotone_in_two_qubit_error():
    base = PlatformModel.quantinuum_h2()
    slopes = []
    for scale in [0.0, 1.0, 3.0, 10.0]:
        m = dataclasses.replace(base, err_2q=1.83e-3 * scale)
        slopes.append(noisy_slope_at_zero(m, 0.0))
    for a, b in zip(slopes, slopes[1:]):
        assert b < a + 1e-12, f"slope not decreasing with 2Q error: {slopes}"
    assert slopes[0] > slopes[-1]


def test_slope_monotone_in_ion_dephasing():
    base = PlatformModel.quantinuum_h2()
    slopes = [noisy_slope_at_zero(dataclasses.replace(base, ion_dephasing=p), 0.0)
              for p in [0.0, 2e-4, 4e-4, 5e-3, 5e-2]]
    for a, b in zip(slopes, slopes[1:]):
        assert b < a + 1e-12, f"slope not decreasing with dephasing: {slopes}"


def test_ibm_slope_monotone_in_phi_kick():
    ibm = PlatformModel.ibm_heron_r2()
    slopes = [noisy_slope_at_zero(ibm, phi) for phi in PHI_KICK_GRID]
    for a, b in zip(slopes, slopes[1:]):
        assert b < a + 1e-9, f"IBM slope not decreasing in phi_kick: {slopes}"
    assert slopes[0] > slopes[-1]


def test_ion_slope_independent_of_phi_kick():
    ion = PlatformModel.quantinuum_h2()
    slopes = [noisy_slope_at_zero(ion, phi) for phi in PHI_KICK_GRID]
    assert max(slopes) - min(slopes) < 1e-12, "ion slope must not depend on phi_kick"


def test_readout_error_reduces_correlator():
    base = PlatformModel.quantinuum_h2()
    hi = dataclasses.replace(base, readout_error=0.05)
    c_lo = noisy_separator(base, 0.2, 0.0)
    c_hi = noisy_separator(hi, 0.2, 0.0)
    assert abs(c_hi) < abs(c_lo)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        passed += 1
        print(f"PASS {fn.__name__}")
    print(f"\n{passed}/{len(fns)} device_noise tests passed")


# Repo convention: expose the checks to unittest discovery as one TestCase.
import unittest


class DeviceNoiseTests(unittest.TestCase):
    def test_primitive_channels_cptp(self): test_primitive_channels_cptp()
    def test_zz_kick_channel_cptp(self): test_zz_kick_channel_cptp()
    def test_platform_channels_cptp(self): test_platform_channels_cptp()
    def test_full_comb_choi_cptp_over_grid(self): test_full_comb_choi_cptp_over_grid()
    def test_ideal_limit_separator_is_sin(self): test_ideal_limit_separator_is_sin()
    def test_ideal_limit_via_four_outcome_probs(self): test_ideal_limit_via_four_outcome_probs()
    def test_ideal_slope_is_one(self): test_ideal_slope_is_one()
    def test_atomic_family_is_dark_even_with_noise(self): test_atomic_family_is_dark_even_with_noise()
    def test_separator_zero_at_delta_zero(self): test_separator_zero_at_delta_zero()
    def test_noise_reduces_slope_below_ideal(self): test_noise_reduces_slope_below_ideal()
    def test_slope_monotone_in_two_qubit_error(self): test_slope_monotone_in_two_qubit_error()
    def test_slope_monotone_in_ion_dephasing(self): test_slope_monotone_in_ion_dephasing()
    def test_ibm_slope_monotone_in_phi_kick(self): test_ibm_slope_monotone_in_phi_kick()
    def test_ion_slope_independent_of_phi_kick(self): test_ion_slope_independent_of_phi_kick()
    def test_readout_error_reduces_correlator(self): test_readout_error_reduces_correlator()
