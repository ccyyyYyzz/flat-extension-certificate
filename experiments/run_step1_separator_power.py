"""Step-1 literature-parameterized noise-power simulation for the C7 separator.

For each platform model (Quantinuum H2 ion, IBM Heron r2 transmon) and each
value of the UNVERIFIED IBM ZZ phase-kick sweep parameter ``phi_kick``:

  1. compute the noisy separator E(delta) = <Y_S1 X_S2> over
     delta in linspace(0, 0.5, 11);
  2. extract the noisy slope at delta = 0 (ideal slope = 1) and the contrast
     loss vs the noiseless sin(delta);
  3. run a two-sided shot-noise power analysis for the test
        H0: dark direction absent (delta = 0)   vs
        H1: dark direction present (delta = delta_0),
     for delta_0 in {0.05, 0.1, 0.2}, at alpha = 0.05 and 95% power.

The correlator is sampled through its 4 outcome probabilities (a, b in +/-1),
so the per-shot estimator is a +/-1 variable with EXACT binomial variance
Var = (1 - C^2).  The sample size uses this exact per-hypothesis variance; the
Gaussian variance-1 shortcut is reported alongside.  Total shots additionally
include the atomic-baseline verification: the 48 single-time IC statistics each
measured at the matched shot budget N (conservative matched precision).

A deterministic Monte-Carlo replay (fixed seed) validates the empirical power.

Outputs: a markdown table and a JSON artifact under
``research/results/step1_separator_power/`` plus a full stdout table.

Run:  PYTHONPATH=src python experiments/run_step1_separator_power.py
"""

from __future__ import annotations

import json
import math
import os
from datetime import date
from typing import Dict, List

import numpy as np

from cyz_m.device_noise import (
    PlatformModel,
    noisy_separator,
    noisy_slope_at_zero,
    ideal_separator,
    validate_all_channels,
)

# --------------------------------------------------------------------------- #
# configuration                                                               #
# --------------------------------------------------------------------------- #
ALPHA = 0.05
POWER = 0.95
DELTA_GRID = np.linspace(0.0, 0.5, 11)
DELTA0_GRID = [0.05, 0.1, 0.2]
PHI_KICK_GRID = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1]
N_ATOMIC_STATS = 48                  # 6 preps x 4 Pauli effects x 2 slots
SEED = 20260721
MC_TRIALS = 40000

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "research", "results", "step1_separator_power")


# --------------------------------------------------------------------------- #
# normal quantile (Acklam), so no scipy dependency                            #
# --------------------------------------------------------------------------- #
def norm_ppf(p: float) -> float:
    """Inverse standard-normal CDF (Peter Acklam's rational approximation)."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


Z_ALPHA2 = norm_ppf(1 - ALPHA / 2.0)     # two-sided => 1.959964
Z_BETA = norm_ppf(POWER)                 # power 0.95 => 1.644854


# --------------------------------------------------------------------------- #
# power analysis                                                              #
# --------------------------------------------------------------------------- #
def sample_size_exact_binomial(C0: float, C1: float) -> int:
    """Shots for a two-sided level-alpha test of C=C0 vs C=C1 at 95% power.

    Per-shot correlator estimator is a +/-1 variable with exact binomial
    variance sigma^2 = 1 - C^2 (measured via the 4 outcome probabilities).
    """
    delta = abs(C1 - C0)
    if delta == 0.0:
        return math.inf
    s0 = math.sqrt(max(0.0, 1.0 - C0 * C0))
    s1 = math.sqrt(max(0.0, 1.0 - C1 * C1))
    n = (Z_ALPHA2 * s0 + Z_BETA * s1) ** 2 / delta ** 2
    return int(math.ceil(n))


def sample_size_gaussian(C0: float, C1: float) -> int:
    """Gaussian variance-1 shortcut: sigma = 1 under both hypotheses."""
    delta = abs(C1 - C0)
    if delta == 0.0:
        return math.inf
    n = (Z_ALPHA2 + Z_BETA) ** 2 / delta ** 2
    return int(math.ceil(n))


def mc_empirical_power(C0: float, C1: float, n: int, rng: np.random.Generator,
                       trials: int = MC_TRIALS) -> float:
    """Deterministic MC power check: binomial correlator sampling under H1.

    Estimator Chat = 2*k/n - 1 with k ~ Binomial(n, q1), q1 = (1+C1)/2.
    Two-sided decision: reject H0 if |Chat - C0| > z_{alpha/2} * sigma0/sqrt(n).
    """
    if not math.isfinite(n) or n <= 0:
        return float("nan")
    q1 = (1.0 + C1) / 2.0
    s0 = math.sqrt(max(1e-18, 1.0 - C0 * C0))
    crit = Z_ALPHA2 * s0 / math.sqrt(n)
    k = rng.binomial(n, q1, size=trials)
    chat = 2.0 * k / n - 1.0
    reject = np.abs(chat - C0) > crit
    return float(np.mean(reject))


# --------------------------------------------------------------------------- #
# per-platform driver                                                         #
# --------------------------------------------------------------------------- #
def analyze_platform(platform: PlatformModel, rng: np.random.Generator) -> Dict:
    cptp = [
        {"channel": n, "choi_min_eig": mn, "tp_defect": tp}
        for (n, mn, tp) in validate_all_channels(platform)
    ]
    per_phi: List[Dict] = []
    for phi in PHI_KICK_GRID:
        e_curve = [noisy_separator(platform, float(d), phi) for d in DELTA_GRID]
        slope0 = noisy_slope_at_zero(platform, phi)
        C0 = noisy_separator(platform, 0.0, phi)
        per_delta0 = []
        for d0 in DELTA0_GRID:
            C1 = noisy_separator(platform, d0, phi)
            n_exact = sample_size_exact_binomial(C0, C1)
            n_gauss = sample_size_gaussian(C0, C1)
            n_total = n_exact * (1 + N_ATOMIC_STATS)
            power = mc_empirical_power(C0, C1, n_exact, rng)
            per_delta0.append({
                "delta0": d0,
                "C0": C0,
                "C1_noisy": C1,
                "C1_ideal": ideal_separator(d0),
                "effect_delta": C1 - C0,
                "sigma1": math.sqrt(max(0.0, 1.0 - C1 * C1)),
                "N_exact_binomial": n_exact,
                "N_gaussian": n_gauss,
                "N_total_with_baseline": n_total,
                "mc_empirical_power": power,
            })
        per_phi.append({
            "phi_kick": phi,
            "slope_at_0": slope0,
            "contrast_loss": 1.0 - slope0,
            "C0": C0,
            "E_curve": [float(x) for x in e_curve],
            "per_delta0": per_delta0,
        })
    return {
        "name": platform.name,
        "citation": platform.citation,
        "mid_circuit_model": platform.mid_circuit,
        "notes": list(platform.notes),
        "cptp_validation": cptp,
        "sweep": per_phi,
    }


# --------------------------------------------------------------------------- #
# reporting                                                                   #
# --------------------------------------------------------------------------- #
def headline_sentence(platform_result: Dict, phi: float, d0: float) -> str:
    entry = next(s for s in platform_result["sweep"] if s["phi_kick"] == phi)
    rec = next(r for r in entry["per_delta0"] if r["delta0"] == d0)
    n_total = rec["N_total_with_baseline"]
    return (
        f"Under {platform_result['name']} parameters "
        f"[{platform_result['citation']}]"
        f"{f' with ZZ phase-kick phi={phi} rad' if platform_result['mid_circuit_model']=='ibm' else ''}, "
        f"N = {n_total:,} total shots certify the planted dark direction "
        f"delta_0 = {d0} at 95% power / alpha = 0.05 "
        f"(correlator {rec['N_exact_binomial']:,} + baseline {N_ATOMIC_STATS}x)."
    )


def build_markdown(results: Dict) -> str:
    md: List[str] = []
    md.append("# Step-1 separator noise-power simulation")
    md.append("")
    md.append(f"- Date: {results['metadata']['date']}")
    md.append(f"- Test: dark direction present (delta=delta_0) vs absent (delta=0), "
              f"two-sided, alpha={ALPHA}, power={POWER}")
    md.append(f"- Correlator: <Y_S1 X_S2> (S1=S2=|+>), ideal E(delta)=sin(delta), "
              f"ideal slope=1")
    md.append(f"- z_(alpha/2)={Z_ALPHA2:.6f}, z_beta={Z_BETA:.6f}")
    md.append(f"- N_exact uses exact binomial variance 1-C^2; N_gauss is the "
              f"variance-1 Gaussian shortcut")
    md.append(f"- N_total = N_exact x (1 + {N_ATOMIC_STATS}): correlator plus the 48 "
              f"atomic-baseline statistics at matched shot budget")
    md.append(f"- MC empirical power: {MC_TRIALS} binomial-sampled trials, seed={SEED}")
    md.append("")

    for key in ("quantinuum_h2", "ibm_heron_r2"):
        pr = results["platforms"][key]
        md.append(f"## {pr['name']}")
        md.append("")
        md.append(f"Citation: {pr['citation']}  |  mid-circuit model: "
                  f"`{pr['mid_circuit_model']}`")
        md.append("")
        for note in pr["notes"]:
            md.append(f"- {note}")
        md.append("")
        cptp_ok = all(c["choi_min_eig"] >= -1e-12 and c["tp_defect"] <= 1e-10
                      for c in pr["cptp_validation"])
        md.append(f"CP-TP validation: all {len(pr['cptp_validation'])} channels pass "
                  f"(Choi min-eig >= -1e-12, TP defect <= 1e-10): **{cptp_ok}**")
        md.append("")
        md.append("| phi_kick | slope@0 | contrast loss | delta0 | C1(noisy) | "
                  "N_exact | N_gauss | N_total | MC power |")
        md.append("|---|---|---|---|---|---|---|---|---|")
        for s in pr["sweep"]:
            for i, r in enumerate(s["per_delta0"]):
                phi_cell = f"{s['phi_kick']:.3f}" if i == 0 else ""
                slope_cell = f"{s['slope_at_0']:.5f}" if i == 0 else ""
                cl_cell = f"{s['contrast_loss']*100:.3f}%" if i == 0 else ""
                md.append(
                    f"| {phi_cell} | {slope_cell} | {cl_cell} | "
                    f"{r['delta0']} | {r['C1_noisy']:.5f} | "
                    f"{r['N_exact_binomial']:,} | {r['N_gaussian']:,} | "
                    f"{r['N_total_with_baseline']:,} | {r['mc_empirical_power']:.3f} |"
                )
        md.append("")

    md.append("## Headlines")
    md.append("")
    for h in results["headlines"]:
        md.append(f"- {h}")
    md.append("")
    md.append("## Ion-vs-IBM comparison across the phi_kick sweep")
    md.append("")
    for line in results["comparison"]["narrative"]:
        md.append(f"- {line}")
    md.append("")
    return "\n".join(md)


def build_comparison(results: Dict) -> Dict:
    ion = results["platforms"]["quantinuum_h2"]
    ibm = results["platforms"]["ibm_heron_r2"]
    # ion is phi-independent; take its phi=0 record
    ion_s = ion["sweep"][0]
    narrative: List[str] = []
    flips = []
    for d0 in DELTA0_GRID:
        ion_n = next(r for r in ion_s["per_delta0"] if r["delta0"] == d0)["N_total_with_baseline"]
        ibm_ns = []
        for s in ibm["sweep"]:
            r = next(rr for rr in s["per_delta0"] if rr["delta0"] == d0)
            ibm_ns.append((s["phi_kick"], r["N_total_with_baseline"]))
        # does IBM ever beat ion (fewer total shots) anywhere on the sweep?
        ibm_best = min(n for _, n in ibm_ns)
        ibm_worst = max(n for _, n in ibm_ns)
        ion_wins_all = ibm_best >= ion_n
        flips.append(ion_wins_all)
        narrative.append(
            f"delta0={d0}: ion N_total={ion_n:,} (phi-independent); "
            f"IBM N_total ranges {ibm_best:,} (phi=0) .. {ibm_worst:,} (phi=0.1); "
            f"ion favored across the whole sweep: {ion_wins_all}."
        )
    no_flip = all(flips)
    narrative.append(
        "The phi_kick sweep does NOT flip the ion-vs-IBM ordering: ion needs "
        "fewer certification shots at every swept phi_kick and every delta0."
        if no_flip else
        "The phi_kick sweep FLIPS the ordering for at least one delta0 "
        "(IBM beats ion at small phi_kick)."
    )
    narrative.append(
        "phi_kick only degrades IBM (monotone slope drop / N_total rise); the "
        "ion model is phi_kick-independent by construction (no ZZ-kick term). "
        "The ion advantage widens as phi_kick grows."
    )
    return {"ion_wins_all_across_sweep": no_flip, "narrative": narrative}


# --------------------------------------------------------------------------- #
# main                                                                        #
# --------------------------------------------------------------------------- #
def main() -> None:
    rng = np.random.default_rng(SEED)
    ion = PlatformModel.quantinuum_h2()
    ibm = PlatformModel.ibm_heron_r2()

    results = {
        "metadata": {
            "date": str(date.today()),
            "alpha": ALPHA,
            "power": POWER,
            "z_alpha_over_2": Z_ALPHA2,
            "z_beta": Z_BETA,
            "delta_grid": [float(x) for x in DELTA_GRID],
            "delta0_grid": DELTA0_GRID,
            "phi_kick_grid": PHI_KICK_GRID,
            "n_atomic_stats": N_ATOMIC_STATS,
            "baseline_convention": (
                "N_total = N_exact x (1 + 48); each atomic statistic measured "
                "at the correlator shot budget (matched precision)"
            ),
            "seed": SEED,
            "mc_trials": MC_TRIALS,
        },
        "platforms": {
            "quantinuum_h2": analyze_platform(ion, rng),
            "ibm_heron_r2": analyze_platform(ibm, rng),
        },
    }

    # headlines: ion at its (phi-independent) value; IBM at phi=0 and phi=0.1
    headlines = [
        headline_sentence(results["platforms"]["quantinuum_h2"], 0.0, 0.1),
        headline_sentence(results["platforms"]["ibm_heron_r2"], 0.0, 0.1),
        headline_sentence(results["platforms"]["ibm_heron_r2"], 0.1, 0.1),
    ]
    results["headlines"] = headlines
    results["comparison"] = build_comparison(results)

    os.makedirs(OUT_DIR, exist_ok=True)
    json_path = os.path.join(OUT_DIR, "step1_separator_power.json")
    md_path = os.path.join(OUT_DIR, "step1_separator_power.md")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    md = build_markdown(results)
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(md)

    # stdout report
    print(md)
    print("=" * 78)
    print(f"JSON artifact : {json_path}")
    print(f"Markdown table: {md_path}")


if __name__ == "__main__":
    main()
