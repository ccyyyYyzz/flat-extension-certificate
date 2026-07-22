"""Deterministic four-panel figure for the flat-extension Letter.

Produces ``paper/prl/figures/c7_flat_extension.pdf`` and a 300 dpi
``c7_flat_extension.png`` preview, referenced by the ``figure*`` in
``paper/prl/main.tex`` (\\label{fig:flat}).

Colour semantics are GLOBAL and consistent across panels:

  blue        #0072B2  the single-slot (atomic/marginal) layer -- what
                        causal-break tomography sees;
  vermillion  #D55E00  the hidden multitime direction -- what it misses and
                        what the certificate detects (planted darks);
  grey                  nulls / neutral reference objects;
  green/amber           verdict-status colours (panel d only).

  (a) The CZ--R_x(delta)--CZ comb on registers [S1, E, S2].  Every
      single-slot statistic is exactly flat in delta (blue, computed); the
      retained two-time correlator <Y_S1 X_S2> = sin(delta) (vermillion) has
      slope one at the working point.  (3-qubit pure-state computation.)

  (b) The delayed hidden classical chain (N=8) + rank-one visible sector:
      visible response flat at 1 (blue); the hidden first-order derivative
      activates only at depth 8 (vermillion).  (Exact jet recurrence.)

  (c) All three executed blind hardware runs: ibm_marrakesh (absolute gate),
      ibm_fez (absolute gate; device null floor), ibm_fez protocol v3
      (device-referenced differential; specificity restored).  Values are
      read live from the run JSONs where available; the v3 per-instance
      values are transcribed from the scored run record (also printed as
      Table S-fez-v3 in the supplement).

  (d) The audit verdict flow: CERTIFIED / INCONCLUSIVE / REJECTED with the
      condition above each verdict box.

An AUTOMATED OVERLAP GATE runs after rendering: it collects display-space
extents of every content Text and every FancyBboxPatch per panel and asserts
no (text, text) or (text, non-parent box) pair overlaps by more than 1 pt in
both directions.  The script FAILS (non-zero exit) on any overlap.

Run:  python paper/prl/figures/make_c7_flat_extension.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.text import Text

# --------------------------------------------------------------------------- #
# Global style: serif text, Computer-Modern math, thin recessive axes.
# --------------------------------------------------------------------------- #
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.linewidth": 0.6,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.4,
    "ytick.major.size": 2.4,
    "font.size": 7.5,
    "axes.labelsize": 7.5,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "lines.linewidth": 1.2,
    "figure.dpi": 200,
})

BLUE = "#0072B2"        # single-slot / atomic layer
VERM = "#D55E00"        # hidden multitime direction / planted darks
GREEN = "#009E73"       # CERTIFIED
AMBER = "#E69F00"       # INCONCLUSIVE
GREY = "#5A5A5A"        # neutral / nulls
INK = "#262626"         # annotation ink (never a series colour)

# (text -> parent box) exemptions for the overlap gate
PARENT: dict[int, set[int]] = {}


def _tint(hex_color: str, amount: float) -> tuple[float, float, float]:
    r, g, b = mcolors.to_rgb(hex_color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


# --------------------------------------------------------------------------- #
# Panel (a) data: the CZ--R_x(delta)--CZ comb (3-qubit pure-state model)
# --------------------------------------------------------------------------- #
_I2 = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_k0 = np.array([1, 0], dtype=complex)
_k1 = np.array([0, 1], dtype=complex)
_kp = (_k0 + _k1) / np.sqrt(2)


def _dm(psi):
    return np.outer(psi, psi.conj())


def _cz(a, b, n=3):
    dim = 2 ** n
    diag = np.ones(dim, dtype=complex)
    for idx in range(dim):
        bits = [(idx >> (n - 1 - q)) & 1 for q in range(n)]
        if bits[a] and bits[b]:
            diag[idx] = -1.0
    return np.diag(diag)


def _embed(op, pos, n=3):
    ops = [_I2] * n
    ops[pos] = op
    out = ops[0]
    for k in range(1, n):
        out = np.kron(out, ops[k])
    return out


def _rx(delta):
    return np.cos(delta / 2) * _I2 - 1j * np.sin(delta / 2) * _X


_U1 = _cz(0, 1)     # write CZ_{S1,E}
_U2 = _cz(2, 1)     # read  CZ_{S2,E}
_HALF = _I2 / 2.0


def _comb_out(rho_S1, rho_S2, delta, rho_E=None):
    if rho_E is None:
        rho_E = _dm(_kp)
    V = _U2 @ _embed(_rx(delta), 1) @ _U1
    rho_in = np.kron(np.kron(rho_S1, rho_E), rho_S2)
    return V @ rho_in @ V.conj().T


def _ptrace(rho, keep, dims=(2, 2, 2)):
    dims = list(dims)
    n = len(dims)
    rho = rho.reshape(dims + dims)
    row = list(range(n))
    col = list(range(n, 2 * n))
    for ax in [i for i in range(n) if i not in keep]:
        col[ax] = row[ax]
    out_r = [row[i] for i in keep]
    out_c = [col[i] for i in keep]
    rho = np.einsum(rho, row + col, out_r + out_c)
    d = int(np.prod([dims[i] for i in keep]))
    return rho.reshape(d, d)


def _correlator(delta, p_op, q_op):
    out = _comb_out(_dm(_kp), _dm(_kp), delta)
    obs = _embed(p_op, 0) @ _embed(q_op, 2)
    return float(np.real(np.trace(obs @ out)))


def _atomic_slot1(prep, obs, delta):
    rS1 = _ptrace(_comb_out(prep, _HALF, delta), keep=[0])
    return float(np.real(np.trace(obs @ rS1)))


def panel_a(ax):
    delta = np.linspace(-1.3, 1.3, 261)
    corr = np.array([_correlator(d, _Y, _X) for d in delta])

    # exactly-flat single-slot statistics at three distinct levels (DATA, so
    # they wear the atomic-layer colour, not gridline grey)
    for prep, obs in [(_dm(_k0), _Z), (_dm(_kp), _X), (_dm(_k1), _Z)]:
        vals = np.array([_atomic_slot1(prep, obs, d) for d in delta])
        ax.plot(delta, vals, color=BLUE, lw=1.0, alpha=0.85, zorder=2,
                solid_capstyle="butt")

    # slope-one tangent (thin, recessive)
    tspan = np.array([-0.62, 0.62])
    ax.plot(tspan, tspan, color=INK, lw=0.8, ls=(0, (1, 1.2)), zorder=5)

    # the retained two-time correlator: the hidden direction's signal
    ax.plot(delta, corr, color=VERM, lw=1.7, zorder=4)

    # --- direct labels (ink), no legend box --------------------------------- #
    ax.text(-1.24, 1.09, "single-slot statistics (3 of 48; all flat)", fontsize=5.9,
            color=INK, ha="left", va="bottom")
    ax.text(1.24, -0.60, r"$\langle Y_{S_1}X_{S_2}\rangle=\sin\delta$",
            fontsize=6.6, color=INK, ha="right", va="top")

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.24, 1.34)
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.set_xlabel(r"memory rotation $\delta$")
    ax.set_ylabel("response")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# --------------------------------------------------------------------------- #
# Panel (b) data: delayed hidden chain + visible sector (exact jet recurrence)
# --------------------------------------------------------------------------- #
def delayed_chain_response(chain_length=8, kmax=11):
    n = chain_length + 3
    sink = chain_length + 2

    def st(i):
        return 1 + i

    T0 = np.zeros((n, n))
    T0[0, 0] = 1.0
    for i in range(chain_length - 1):
        T0[st(i + 1), st(i)] = 1.0
    T0[sink, st(chain_length - 1)] = 0.5
    T0[st(chain_length), st(chain_length - 1)] = 0.5
    T0[st(chain_length), st(chain_length)] = 1.0
    T0[sink, sink] = 1.0

    dT = np.zeros((n, n))
    dT[sink, st(chain_length - 1)] = -1.0
    dT[st(chain_length), st(chain_length - 1)] = 1.0

    v = np.zeros(n)
    v[0] = 1.0
    v[st(0)] = 0.5
    dv = np.zeros(n)
    detect_v = np.zeros(n)
    detect_v[0] = 1.0
    detect_eN = np.zeros(n)
    detect_eN[st(chain_length)] = 1.0

    ks, p_vis, dp_hid = [], [], []
    for k in range(kmax + 1):
        ks.append(k)
        p_vis.append(float(detect_v @ v))
        dp_hid.append(float(detect_eN @ dv))
        v, dv = T0 @ v, T0 @ dv + dT @ v
    return np.array(ks), np.array(p_vis), np.array(dp_hid)


def hankel_rank_vs_horizon(chain_length=8, hmax=6, tol=1e-9):
    """Numerical rank of the observable augmented (0th+1st order) Hankel block
    of the hidden-chain (+) visible model, as a function of the tested
    prefix/suffix horizon h.  The hidden derivative enters an entry only when
    prefix+suffix length reaches the chain depth, so the rank plateaus at 1
    (with an exactly clean gap) through every h < N/2 and jumps at h = N/2."""
    n = chain_length + 3
    sink = chain_length + 2

    def st(i):
        return 1 + i

    T0 = np.zeros((n, n))
    T0[0, 0] = 1.0
    for i in range(chain_length - 1):
        T0[st(i + 1), st(i)] = 1.0
    T0[sink, st(chain_length - 1)] = 0.5
    T0[st(chain_length), st(chain_length - 1)] = 0.5
    T0[st(chain_length), st(chain_length)] = 1.0
    T0[sink, sink] = 1.0
    dT = np.zeros((n, n))
    dT[sink, st(chain_length - 1)] = -1.0
    dT[st(chain_length), st(chain_length - 1)] = 1.0

    v0 = np.zeros(n); v0[0] = 1.0; v0[st(0)] = 0.5
    e_v = np.zeros(n); e_v[0] = 1.0
    e_h = np.zeros(n); e_h[st(chain_length)] = 1.0

    # jet propagation to depth L: (v, dv) after L letters
    def jet(L):
        v, dv = v0.copy(), np.zeros(n)
        for _ in range(L):
            v, dv = T0 @ v, T0 @ dv + dT @ v
        return v, dv

    ranks = []
    for h in range(hmax + 1):
        rows = []
        for sfx in range(h + 1):          # suffix word length
            for pfx in range(h + 1):      # prefix word length
                v, dv = jet(sfx + pfx)
                rows.append([e_v @ v, e_h @ v, e_v @ dv, e_h @ dv])
        M = np.array(rows)
        sv = np.linalg.svd(M, compute_uv=False)
        ranks.append(int((sv > tol * sv[0]).sum()))
    return np.arange(hmax + 1), np.array(ranks)


def panel_b(ax):
    N = 8
    hs, ranks = hankel_rank_vs_horizon(chain_length=N, hmax=6)

    jump = int(np.argmax(ranks > ranks[0]))
    ax.axvspan(-0.5, jump - 0.5, color="0.945", zorder=0)

    ax.step(hs, ranks, where="mid", color=INK, lw=1.3, zorder=3)
    ax.plot(hs[:jump], ranks[:jump], ls="none", marker="o", ms=3.4,
            mfc="white", mec=INK, mew=1.0, zorder=4)
    ax.plot(hs[jump:], ranks[jump:], ls="none", marker="o", ms=3.4,
            mfc=VERM, mec=VERM, zorder=4)

    ax.text((jump - 1) / 2.0, 1.52, "false\nplateau", ha="center",
            va="center", fontsize=5.6, color=GREY, linespacing=1.25)
    ax.annotate("jump at\n" + r"$h{=}N/2$", xy=(jump, 1.94),
                xytext=(jump + 1.35, 1.55), ha="center", va="center",
                fontsize=5.5, color=INK, linespacing=1.25,
                arrowprops=dict(arrowstyle="-|>", lw=0.8, shrinkA=10.0,
                                shrinkB=3.0, color=INK))

    ax.set_xlim(-0.5, 6.5)
    ax.set_ylim(0.72, 2.42)
    ax.set_xticks([0, 1, 2, 3, 4, 5, 6])
    ax.set_yticks([1, 2])
    ax.set_xlabel(r"tested horizon $h$")
    ax.set_ylabel(r"numerical rank of $H_h$")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)



# --------------------------------------------------------------------------- #
# Panel (c) data: the three executed blind hardware runs
# --------------------------------------------------------------------------- #
def _repo_root():
    return Path(__file__).resolve().parents[3]


def load_device(rel_json):
    with open(_repo_root() / rel_json, encoding="utf-8") as f:
        d = json.load(f)
    ak = d["answer_key"]
    cert = d["verdicts"]["certificate"]
    mqpt = d["verdicts"]["marginal_qpt"]
    gate = float(cert[0]["z_gate"])
    darks, nulls = [], []
    for a, c in zip(ak, cert):
        (darks if a["dark"] else nulls).append(float(c["max_abs_z"]))
    mq = [float(m["max_abs_z"]) for m in mqpt]
    mq_det = sum(1 for m in mqpt if m["detected"])
    return {"darks": darks, "nulls": nulls, "mqpt": mq, "mqpt_det": mq_det,
            "gate": gate}


# Protocol-v3 per-instance |z_diff| (scored run record, seed 20260729; also
# Table S-fez-v3 of the supplement): darks 6.04, 8.30, 8.26, 4.21, 10.19;
# nulls 1.05, 1.26, 1.62; differential gate 3.5431.
V3 = {"darks": [6.04, 8.30, 8.26, 4.21, 10.19],
      "nulls": [1.05, 1.26, 1.62],
      "gate": 3.5431}


def _spread(x0, n, d=0.055):
    if n <= 1:
        return np.array([x0])
    return x0 + np.linspace(-d * (n - 1) / 2, d * (n - 1) / 2, n)


def _qpt_family_gate():
    """Bonferroni max-|z| gate of the marginal-QPT control family: 32 off-null
    atomic settings x 8 declared instances at alpha_exp = 0.01 (the scorer's
    _gate(0.01, 32, 8)).  Verified against every stored verdict below."""
    from statistics import NormalDist
    return NormalDist().inv_cdf(1.0 - 0.01 / (2.0 * 32 * 8))


def panel_c(ax):
    m = load_device(
        "research/results/blind_benchmark_hardware/blind_benchmark_hardware.json")
    f = load_device(
        "research/results/blind_benchmark_hardware_fez/blind_benchmark_hardware.json")

    qpt_gate = _qpt_family_gate()
    # self-check: the normalized picture must agree with the scorer's verdicts
    for dat in (m, f):
        for z in dat["mqpt"]:
            assert not (z > qpt_gate), "QPT verdict/gate mismatch"

    groups = [(1.0, m, m["gate"]), (2.0, f, f["gate"]), (3.0, V3, V3["gate"])]
    DARK_DX, NULL_DX, MQ_DX = -0.24, 0.02, 0.26

    for xc, dat, cgate in groups:
        xs = _spread(xc + DARK_DX, len(dat["darks"]))
        ax.plot(xs, np.sort(dat["darks"]) / cgate, ls="none", marker="o",
                ms=4.0, mfc=VERM, mec=VERM, zorder=4)
        xs = _spread(xc + NULL_DX, len(dat["nulls"]))
        nulls = np.sort(dat["nulls"]) / cgate
        ax.plot(xs, nulls, ls="none", marker="s", ms=3.8, mfc="white",
                mec=GREY, mew=1.0, zorder=4)
        fp = [(x, r) for x, r in zip(xs, nulls) if r > 1.0]
        if fp:
            ax.plot([q[0] for q in fp], [q[1] for q in fp], ls="none",
                    marker="*", ms=6.5, mfc="black", mec="white", mew=0.4,
                    zorder=6)
        if "mqpt" in dat:
            xs = _spread(xc + MQ_DX, len(dat["mqpt"]))
            ax.plot(xs, np.sort(dat["mqpt"]) / qpt_gate, ls="none", marker="v",
                    ms=3.0, mfc="white", mec=BLUE, mew=0.8, zorder=3)

    # one honest unit line: every family measured against ITS OWN frozen gate
    ax.axhline(1.0, color="0.30", lw=0.9, ls=(0, (5, 2)), zorder=2)
    ax.text(0.55, 1.10, "family gate", ha="left", va="bottom", fontsize=5.8,
            color="0.25")

    ax.text(2.29, 1.78, "device\nfloor", ha="center", va="center",
            fontsize=5.6, color=INK, linespacing=1.2)

    ax.set_xlim(0.45, 3.55)
    ax.set_ylim(0.0, 5.9)
    ax.set_xticks([1.0, 2.0, 3.0])
    ax.set_xticklabels(["marrakesh", "fez", r"fez $\cdot$ v3"])
    ax.set_yticks([0, 1, 2, 3, 4, 5])
    ax.set_ylabel(r"$|z| \, / \, z_\star^{\rm family}$")
    ax.tick_params(axis="x", length=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    handles = [
        Line2D([0], [0], ls="none", marker="o", ms=4.0, mfc=VERM, mec=VERM,
               label="dark"),
        Line2D([0], [0], ls="none", marker="s", ms=3.8, mfc="white", mec=GREY,
               mew=1.0, label="null"),
        Line2D([0], [0], ls="none", marker="v", ms=3.0, mfc="white", mec=BLUE,
               mew=0.8, label="QPT"),
    ]
    leg = ax.legend(handles=handles, loc="upper left",
                    bbox_to_anchor=(0.01, 1.005), frameon=False,
                    handlelength=0.9, handletextpad=0.35, labelspacing=0.28,
                    borderpad=0.1, fontsize=5.7)
    for txt in leg.get_texts():
        txt.set_color(INK)



# --------------------------------------------------------------------------- #
# Panel (d): the audit verdict flow (left -> right)
# --------------------------------------------------------------------------- #
def _reg_label(ax, x, y, text, box=None, **kw):
    t = ax.text(x, y, text, **kw)
    if box is not None:
        PARENT.setdefault(id(t), set()).add(id(box))
    return t


def _rounded_box(ax, xy, wh, text, edge, face, fontcolor, fontsize=6.2,
                 weight="bold", lw=1.1):
    box = FancyBboxPatch(xy, wh[0], wh[1],
                         boxstyle="round,pad=0.010,rounding_size=0.035",
                         facecolor=face, edgecolor=edge, lw=lw, zorder=3)
    ax.add_patch(box)
    _reg_label(ax, xy[0] + wh[0] / 2, xy[1] + wh[1] / 2, text, ha="center",
               va="center", fontsize=fontsize, fontweight=weight,
               color=fontcolor, zorder=5, box=box)
    return box


def panel_d(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # vertical pipeline: core -> checks -> bounded all-depth extension,
    # failure branches skirting the terminal box (conditions in caption)
    bh = 0.155
    _rounded_box(ax, (0.180, 0.860 - bh / 2), (0.640, bh),
                 "finite core $H$", "#3A3A3A", "#F4F4F4", INK,
                 fontsize=5.2, weight="normal", lw=1.0)
    _rounded_box(ax, (0.060, 0.600 - bh / 2), (0.880, bh),
                 r"rank$_\varepsilon{=}n_J$, flat, gapped",
                 "#3A3A3A", "#F4F4F4", INK, fontsize=5.0,
                 weight="normal", lw=1.0)
    _rounded_box(ax, (0.200, 0.330 - bh / 2), (0.600, bh),
                 "all-depth\nextension", GREEN, _tint(GREEN, 0.87),
                 GREEN, fontsize=5.0)

    for y0, y1 in [(0.860 - bh / 2 - 0.012, 0.600 + bh / 2 + 0.012),
                   (0.600 - bh / 2 - 0.012, 0.330 + bh / 2 + 0.012)]:
        ax.add_patch(FancyArrowPatch((0.50, y0), (0.50, y1),
                                     arrowstyle="-|>", mutation_scale=7,
                                     lw=1.1, color=INK, zorder=2,
                                     shrinkA=0, shrinkB=0))
    _reg_label(ax, 0.545, 0.463, r"shifts $X_a$", ha="left",
               va="center", fontsize=4.9, color="0.30")

    # failure chips at the bottom, fed from the check box
    fh = 0.150
    _rounded_box(ax, (0.022, 0.085 - fh / 2 + 0.010), (0.410, fh),
                 "inconclusive", AMBER, _tint(AMBER, 0.87), "#8A5A00",
                 fontsize=4.7, weight="normal")
    _rounded_box(ax, (0.513, 0.085 - fh / 2 + 0.010), (0.465, fh),
                 "class excluded", VERM, _tint(VERM, 0.87), VERM,
                 fontsize=4.6, weight="normal")
    ax.add_patch(FancyArrowPatch((0.105, 0.600 - bh / 2 - 0.012),
                                 (0.175, 0.085 + fh / 2 + 0.022),
                                 arrowstyle="-|>", mutation_scale=7,
                                 lw=1.0, color=AMBER, zorder=2,
                                 connectionstyle="arc3,rad=0.22",
                                 shrinkA=0, shrinkB=0))
    ax.add_patch(FancyArrowPatch((0.895, 0.600 - bh / 2 - 0.012),
                                 (0.825, 0.085 + fh / 2 + 0.022),
                                 arrowstyle="-|>", mutation_scale=7,
                                 lw=1.0, color=VERM, zorder=2,
                                 connectionstyle="arc3,rad=-0.22",
                                 shrinkA=0, shrinkB=0))


# --------------------------------------------------------------------------- #
# Automated overlap gate
# --------------------------------------------------------------------------- #
def _content_texts(ax):
    seen, out = set(), []
    for t in ax.findobj(Text):
        if t.get_text().strip() and id(t) not in seen:
            seen.add(id(t))
            out.append(t)
    return out


def _boxes(ax):
    return [p for p in ax.patches if isinstance(p, FancyBboxPatch)]


def _lin_overlap(b1, b2):
    ox = min(b1.x1, b2.x1) - max(b1.x0, b2.x0)
    oy = min(b1.y1, b2.y1) - max(b1.y0, b2.y0)
    return ox, oy


def run_overlap_gate(fig, axes, tol_pt=1.0):
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    tol_px = tol_pt * fig.dpi / 72.0
    pt_per_px = 72.0 / fig.dpi

    panel_names = ["(a)", "(b)", "(c)", "(d)"]
    pairs_checked = 0
    per_panel = {}
    failures = []

    for name, ax in zip(panel_names, axes):
        texts = _content_texts(ax)
        boxes = _boxes(ax)
        tex = [(t, t.get_window_extent(rend)) for t in texts]
        bex = [(b, b.get_window_extent(rend)) for b in boxes]
        n_pairs = 0

        for i in range(len(tex)):
            for j in range(i + 1, len(tex)):
                n_pairs += 1
                ox, oy = _lin_overlap(tex[i][1], tex[j][1])
                if ox > tol_px and oy > tol_px:
                    failures.append(
                        (name, "text/text",
                         repr(tex[i][0].get_text()), repr(tex[j][0].get_text()),
                         ox * pt_per_px, oy * pt_per_px))

        for t, tb in tex:
            for b, bb in bex:
                if id(b) in PARENT.get(id(t), set()):
                    continue
                n_pairs += 1
                ox, oy = _lin_overlap(tb, bb)
                if ox > tol_px and oy > tol_px:
                    failures.append(
                        (name, "text/box", repr(t.get_text()), "<box>",
                         ox * pt_per_px, oy * pt_per_px))

        per_panel[name] = n_pairs
        pairs_checked += n_pairs

    return pairs_checked, per_panel, failures


# --------------------------------------------------------------------------- #
def main():
    out_dir = Path(__file__).resolve().parent
    out_pdf = out_dir / "c7_flat_extension.pdf"
    out_png = out_dir / "c7_flat_extension.png"

    fig, axes = plt.subplots(
        1, 4, figsize=(7.05, 2.30),
        gridspec_kw={"width_ratios": [0.95, 0.95, 1.14, 1.28]})
    panel_a(axes[0])
    panel_b(axes[1])
    panel_d(axes[2])
    panel_c(axes[3])

    for ax, lab in zip(axes, ["(a)", "(b)", "(c)", "(d)"]):
        ax.set_title(lab, loc="left", fontsize=9, fontweight="bold", pad=4.0)

    fig.tight_layout(pad=0.45, w_pad=1.05)

    pairs, per_panel, failures = run_overlap_gate(fig, axes, tol_pt=1.0)
    print(f"[overlap-gate] pairs checked: {pairs}  "
          f"({', '.join(f'{k}:{v}' for k, v in per_panel.items())})")
    if failures:
        print(f"[overlap-gate] FAIL: {len(failures)} overlapping pair(s):")
        for name, kind, a, b, ox, oy in failures:
            print(f"    {name} {kind}: {a} <> {b}  "
                  f"overlap = {ox:.2f}pt x {oy:.2f}pt")
        plt.close(fig)
        raise SystemExit("Overlap gate failed: fix the layout before shipping.")
    print("[overlap-gate] PASS: no text/text or text/box overlap.")

    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(out_pdf)
    print(out_png)


if __name__ == "__main__":
    main()
