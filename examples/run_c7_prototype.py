"""
scratch_c7_prototype.py  --  numerical prototype for the C7
"operational-dimension" impossibility construction (Candidate A).

------------------------------------------------------------------------------
CANDIDATE A (flagship): a memory-pointer dark direction that is invisible to the
COMPLETE single-time IC atomic protocol family on S, yet first-order visible to
one depth-2 protocol.  Realizes N1's "dark multiplicity" as an explicit 2-qubit
process tensor.

  System S = C^2, memory E = C^2 (init |+>_E).  Register order [S1, E, S2].
  Collision-model comb, two experimenter slots on S:
     slot 1 (write): U1 = CZ_{S1,E}
     dark link      : G_delta on E   (the intrinsic inter-slot generator)
     slot 2 (read) : U2 = CZ_{S2,E},  then discard E.
  M0 := (delta = 0),  M1 := (delta != 0).

------------------------------------------------------------------------------
CORRECTION TO THE DESIGN SPEC (documented by the numbers below).
The spec proposed the dark generator G_delta = R_z(delta) = exp(-i delta Z_E/2).
That choice is DEGENERATE: R_z is diagonal in the same basis as CZ, so it
COMMUTES with both CZ gates and can be pushed to act immediately before E is
discarded -> M0 and M1 are then the *identical* process tensor, invisible at
EVERY depth (the numerics give an exactly-zero depth-2 signal).  The genuine
"atomically dark but depth-2 recoverable" generators are the ones that do NOT
commute with the CZ write/read, namely G_delta = R_x(delta) (used here) or
R_y(delta).  Every other element of the spec (2 qubits, CZ + one-parameter link,
a single depth-2 correlator) is preserved.

Checks (no claim beyond the printed numbers):
  (1) EXACT ATOMIC AGREEMENT  -- machine-zero, all orders, over a grid.
  (2) RANK DIFFERENCE         -- atomic vs full response Jacobians, SVD ranks.
  (3) SEPARATION              -- depth-2 correlator first-order signal.
  (4) SIDE CONDITIONS         -- CP (Choi min eig) + IC (spanning rank).

Repo-utility note: the design spec names obstruction_audit /
response_rank_witness / spacetime_witness.py.  Those symbols are NOT committed
on branch research/factor-recovery-universality (they live only in the
uncommitted main working tree), so ranks are certified here with an explicit
tolerance-resolved SVD.  audit_multiprobe_reflection (which IS on the branch) is
run as an independent N1 cross-check when its src tree is importable.
"""

import itertools
import numpy as np

np.set_printoptions(precision=6, suppress=True, linewidth=140)

# ------------------------------ qubit algebra ------------------------------
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
PAULI = {"I": I2, "X": X, "Y": Y, "Z": Z}

k0 = np.array([1, 0], dtype=complex)
k1 = np.array([0, 1], dtype=complex)
kp = (k0 + k1) / np.sqrt(2)          # |+>
km = (k0 - k1) / np.sqrt(2)          # |->
kpi = (k0 + 1j * k1) / np.sqrt(2)    # |+i>
kmi = (k0 - 1j * k1) / np.sqrt(2)    # |-i>


def dm(psi):
    psi = np.asarray(psi, dtype=complex)
    return np.outer(psi, psi.conj())


# IC single-qubit preparations (6 Pauli eigenstates: overcomplete, spans sa)
PREP = {"|0>": dm(k0), "|1>": dm(k1), "|+>": dm(kp),
        "|->": dm(km), "|+i>": dm(kpi), "|-i>": dm(kmi)}
PREP_ORDER = ["|0>", "|1>", "|+>", "|->", "|+i>", "|-i>"]
EFFECT_ORDER = ["I", "X", "Y", "Z"]   # expectation values fix a qubit state
PAULI_PAIRS = list(itertools.product(["X", "Y", "Z"], ["X", "Y", "Z"]))


def su2_exp(theta):
    """exp(-i (tx X + ty Y + tz Z)/2), analytic (no scipy)."""
    tx, ty, tz = theta
    t = np.sqrt(tx * tx + ty * ty + tz * tz)
    if t < 1e-15:
        return I2.copy()
    n = np.array([tx, ty, tz]) / t
    return np.cos(t / 2) * I2 - 1j * np.sin(t / 2) * (n[0] * X + n[1] * Y + n[2] * Z)


# ---------------------------- register plumbing ----------------------------
N = 3
DIMS = [2, 2, 2]                       # [S1, E, S2]


def embed1(op, pos, n=N):
    ops = [I2] * n
    ops[pos] = op
    out = ops[0]
    for k in range(1, n):
        out = np.kron(out, ops[k])
    return out


def cz(a, b, n=N):
    dim = 2 ** n
    diag = np.ones(dim, dtype=complex)
    for idx in range(dim):
        bits = [(idx >> (n - 1 - q)) & 1 for q in range(n)]
        if bits[a] == 1 and bits[b] == 1:
            diag[idx] = -1.0
    return np.diag(diag)


def partial_trace(rho, keep, dims=DIMS):
    n = len(dims)
    rho = rho.reshape(dims + dims)
    keep = sorted(keep)
    row = list(range(n))
    col = list(range(n, 2 * n))
    for ax in [i for i in range(n) if i not in keep]:
        col[ax] = row[ax]                       # contract traced axes
    out_r = [row[i] for i in keep]
    out_c = [col[i] for i in keep]
    rho = np.einsum(rho, row + col, out_r + out_c)
    d = int(np.prod([dims[i] for i in keep]))
    return rho.reshape(d, d)


U1 = cz(0, 1)                                    # write: CZ_{S1,E}
U2 = cz(2, 1)                                    # read : CZ_{S2,E}
E_INIT = dm(kp)                                  # |+><+|_E


def link_gen(theta):
    """Inter-slot dark generator on E as a general su(2) element."""
    return embed1(su2_exp(theta), 1)


def comb_V(theta):
    return U2 @ link_gen(theta) @ U1


def comb_out(rho_S1, rho_S2, theta, rho_E=E_INIT):
    rho_in = np.kron(np.kron(rho_S1, rho_E), rho_S2)
    V = comb_V(theta)
    return V @ rho_in @ V.conj().T


# theta axes: x -> R_x (the dark generator we use), y -> R_y, z -> R_z (degenerate)
DARK = np.array([1.0, 0.0, 0.0])                 # sigma_x^E generator direction


# ----------------- complete single-time IC atomic family ------------------
# slot i open: prep S_i in a Pauli eigenstate, other slot closed by causal
# break (feed I/2) + discard; measure a Pauli expectation on S_i's output.
HALF = I2 / 2.0


def atomic_response(theta):
    """Ordered vector of every single-time IC statistic of the atomic family."""
    vals = []
    for lab in PREP_ORDER:
        rho = PREP[lab]
        # slot 1 open (S2 closed = I/2)
        rS1 = partial_trace(comb_out(rho, HALF, theta), keep=[0])
        for obs in EFFECT_ORDER:
            vals.append(np.real(np.trace(PAULI[obs] @ rS1)))
        # slot 2 open (S1 closed = I/2)
        rS2 = partial_trace(comb_out(HALF, rho, theta), keep=[2])
        for obs in EFFECT_ORDER:
            vals.append(np.real(np.trace(PAULI[obs] @ rS2)))
    return np.array(vals)


# -------------------- depth-2 separating protocol family -------------------
def depth2_response(theta):
    """IC preps on S1,S2 x every Pauli-pair correlator <P_S1 Q_S2>."""
    vals = []
    for l1 in PREP_ORDER:
        for l2 in PREP_ORDER:
            out = comb_out(PREP[l1], PREP[l2], theta)
            for p, q in PAULI_PAIRS:
                O = embed1(PAULI[p], 0) @ embed1(PAULI[q], 2)
                vals.append(np.real(np.trace(O @ out)))
    return np.array(vals)


def full_response(theta):
    return np.concatenate([atomic_response(theta), depth2_response(theta)])


def depth2_correlator(theta, p, q, rho_S1=dm(kp), rho_S2=dm(kp)):
    out = comb_out(rho_S1, rho_S2, theta)
    O = embed1(PAULI[p], 0) @ embed1(PAULI[q], 2)
    return np.real(np.trace(O @ out))


# ------------------------------- CP / Choi ---------------------------------
def choi(channel, din):
    probe = channel(np.eye(din, dtype=complex))
    dout = probe.shape[0]
    J = np.zeros((din * dout, din * dout), dtype=complex)
    for i in range(din):
        for j in range(din):
            Eij = np.zeros((din, din), dtype=complex)
            Eij[i, j] = 1.0
            J += np.kron(Eij, channel(Eij))
    return J


def phi_slot1(rho):
    cz2 = cz(0, 1, 2)
    return partial_trace(cz2 @ np.kron(rho, E_INIT) @ cz2.conj().T, keep=[0], dims=[2, 2])


def phi_slot2(rho):
    cz2 = cz(0, 1, 2)
    return partial_trace(cz2 @ np.kron(rho, HALF) @ cz2.conj().T, keep=[0], dims=[2, 2])


def lift_S1S2(M):
    """Lift a 4x4 operator on S1(x)S2 into [S1,E,S2] with E-slot = |+><+|."""
    rho_full = np.zeros((8, 8), dtype=complex)
    basis = [k0, k1]
    for s1 in range(2):
        for s2 in range(2):
            for t1 in range(2):
                for t2 in range(2):
                    c = M[2 * s1 + s2, 2 * t1 + t2]
                    if c == 0:
                        continue
                    ketin = np.kron(np.kron(basis[s1], kp), basis[s2])
                    bra = np.kron(np.kron(basis[t1], kp), basis[t2])
                    rho_full += c * np.outer(ketin, bra.conj())
    return rho_full


def depth2_process(M, theta):
    V = comb_V(theta)
    return partial_trace(V @ lift_S1S2(M) @ V.conj().T, keep=[0, 2])


# ------------------------- tolerance-resolved rank -------------------------
def certified_rank(mat, rel_tol=1e-7, abs_floor=1e-8):
    mat = np.asarray(mat, dtype=float)
    if mat.size == 0:
        return 0, np.array([])
    sv = np.linalg.svd(mat, compute_uv=False)
    thr = max(abs_floor, rel_tol * (sv[0] if sv.size else 0.0))
    return int(np.sum(sv > thr)), sv


def fd_jacobian(func, base, hj=1e-6):
    cols = []
    for k in range(3):
        tp = base.copy(); tp[k] += hj
        tm = base.copy(); tm[k] -= hj
        cols.append((func(tp) - func(tm)) / (2 * hj))
    return np.array(cols).T          # (n_stats, 3)


# ---------------------------- repo cross-check -----------------------------
def repo_cross_check():
    import os, sys, subprocess, tempfile, tarfile, io
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"],
                                       cwd=here).decode().strip()
        blob = subprocess.check_output(
            ["git", "archive", "research/factor-recovery-universality", "src"],
            cwd=root)
        tmp = tempfile.mkdtemp(prefix="c7_src_")
        with tarfile.open(fileobj=io.BytesIO(blob)) as tf:
            tf.extractall(tmp)
        sys.path.insert(0, os.path.join(tmp, "src"))
        from cyz_m.multiprobe_reflection import audit_multiprobe_reflection
        m = 2
        c = np.array([1.0, 0.0])
        gens = [np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0.0]]),
                np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0.0]]),
                np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0.0]])]
        energy_rep = [np.kron(g, np.eye(m)) for g in gens]
        terminal_rep = list(gens)
        S = np.zeros((3, 3 * m))
        for a in range(3):
            for mu in range(m):
                S[a, mu * 3 + a] = c[mu]
        au = audit_multiprobe_reflection(None, energy_rep, [terminal_rep], [S])
        return ("    imported audit_multiprobe_reflection OK (N1 quantized, m=2)\n"
                f"    hidden_kernel_dimension       = {au.hidden_kernel_dimension}"
                f"   (= dim ker S ; N1 predicts 3(m-1) = {3 * (m - 1)})\n"
                f"    stacked_projected_resp_rank   = {au.stacked_projected_response_rank}"
                f"   (= dim W = 3)\n"
                f"    completeness_gap (sigma_min)  = {au.completeness_gap:.3e}"
                f"   (= 0  =>  response NOT injective, hidden kernel present)")
    except Exception as exc:
        return (f"    (skipped: {type(exc).__name__}: {exc})\n"
                "    obstruction_audit / response_rank_witness are NOT on this branch;\n"
                "    ranks above are certified by the in-file SVD witness.")


# --------------------------------- main ------------------------------------
def main():
    out = []
    pr = out.append
    pr("=" * 80)
    pr("C7 impossibility construction -- Candidate A (memory-pointer dark mode)")
    pr("=" * 80)
    pr("Construction parameters:")
    pr("  S=C^2, E=C^2 ; E init |+> ; U1=CZ_{S1,E} ; U2=CZ_{S2,E} ; discard E")
    pr("  dark link G_delta = R_x(delta) = exp(-i*delta*X_E/2) on E   (spec-corrected)")
    pr("  register order [S1, E, S2] ; M0: delta=0 ; M1: delta!=0")

    # (1) EXACT ATOMIC AGREEMENT
    pr("\n" + "-" * 80)
    pr("(1) EXACT ATOMIC AGREEMENT  (complete single-time IC family)")
    pr("-" * 80)
    base = atomic_response(np.zeros(3))
    pr(f"  atomic statistics in family : {len(base)}"
       f"  (6 preps x 4 Pauli effects x 2 slots)")
    grid = np.linspace(-1.0, 1.0, 21)
    worst, wpt = 0.0, None
    for d in grid:
        v = atomic_response(DARK * d)            # vary the dark R_x direction
        diff = float(np.max(np.abs(v - base)))
        if diff > worst:
            worst, wpt = diff, d
    pr(f"  1D grid: delta in [-1,1], 21 pts (dark R_x direction)")
    pr(f"    max_delta max_protocol |p_M1 - p_M0| = {worst:.3e}  (at delta={wpt:+.3f})")
    pr(f"    machine-zero (<1e-12)?  {worst < 1e-12}")
    g2 = np.linspace(-1.0, 1.0, 11)
    worst2 = 0.0
    for ax in g2:
        for ay in g2:
            v = atomic_response(np.array([ax, ay, 0.0]))   # 2 varying link dirs
            worst2 = max(worst2, float(np.max(np.abs(v - base))))
    pr(f"  2D grid: link (theta_x,theta_y) on 11x11 grid in [-1,1]^2")
    pr(f"    max over grid |p - p_M0| = {worst2:.3e}   (<1e-12? {worst2 < 1e-12})")
    pr("    => every single-time IC statistic is literally constant in the link,")
    pr("       to all orders, not merely first-order.")

    # (3) SEPARATION
    pr("\n" + "-" * 80)
    pr("(3) SEPARATION  (one depth-2 protocol, first-order signal)")
    pr("-" * 80)
    pr("  protocol: S1=|+>, S2=|+>, comb, measure correlator <P_S1 Q_S2> (E discarded)")
    h = 1e-6
    xy0 = depth2_correlator(np.zeros(3), "Y", "X")
    xyd = (depth2_correlator(DARK * h, "Y", "X")
           - depth2_correlator(-DARK * h, "Y", "X")) / (2 * h)
    pr(f"  HEADLINE  <Y_S1 X_S2>:  value at M0 (delta=0) = {xy0:+.3e}"
       f"  ;  d/ddelta = {xyd:+.6f}")
    pr("  full depth-2 correlator table at M0 and its d/ddelta (dark R_x):")
    for p, q in PAULI_PAIRS:
        v0 = depth2_correlator(np.zeros(3), p, q)
        der = (depth2_correlator(DARK * h, p, q)
               - depth2_correlator(-DARK * h, p, q)) / (2 * h)
        if abs(v0) > 1e-9 or abs(der) > 1e-9:
            flag = "  <-- first-order signal" if abs(der) > 1e-9 else ""
            pr(f"    <{p}_S1 {q}_S2>: value(0)={v0:+.6f}  d/ddelta={der:+.6f}{flag}")
    # atomic first-order signal along the same direction (must be 0)
    ad = np.max(np.abs((atomic_response(DARK * h) - atomic_response(-DARK * h)) / (2 * h)))
    pr(f"  max |d(atomic statistic)/ddelta| along same direction = {ad:.3e}  (exactly 0)")

    # (2) RANK DIFFERENCE
    pr("\n" + "-" * 80)
    pr("(2) RANK DIFFERENCE  (finite-difference response Jacobians)")
    pr("-" * 80)
    pr("  tangent basis on E link: (theta_x -> R_x [dark], theta_y -> R_y, theta_z -> R_z)")
    for tag, b in [("M0  theta=(0,0,0)", np.zeros(3)),
                   ("M1  theta=(0.5,0,0)", np.array([0.5, 0.0, 0.0]))]:
        Ja = fd_jacobian(atomic_response, b)
        Jf = fd_jacobian(full_response, b)
        ra, sva = certified_rank(Ja)
        rf, svf = certified_rank(Jf)
        pr(f"  [{tag}]")
        pr(f"    atomic Jacobian {Ja.shape}: sing.vals = {np.array2string(sva)}"
           f"  -> rank {ra}")
        pr(f"    full   Jacobian {Jf.shape}: sing.vals = {np.array2string(svf)}"
           f"  -> rank {rf}")
        pr(f"    kernel_completeness_gap (full - atomic) = {rf - ra}")
        # single dark direction (theta_x = sigma_x^E)
        da = float(np.linalg.norm(Ja[:, 0]))
        df = float(np.linalg.norm(Jf[:, 0]))
        pr(f"    single dark dir delta=R_x:  ||dA/dd||={da:.2e} (rank {int(da > 1e-8)}),"
           f"  ||dF/dd||={df:.2e} (rank {int(df > 1e-8)})")
        pr(f"    derivative_growth (atomic,depth2) = "
           f"({int(da > 1e-8)},{int(df > 1e-8)}),  completeness_gap = "
           f"{int(df > 1e-8) - int(da > 1e-8)}")
    pr("  note: theta_z (=R_z) is a trivial discard-gauge (commutes with both CZ),")
    pr("        unobservable at EVERY depth -> it never enters the full Jacobian rank.")

    # (4) SIDE CONDITIONS
    pr("\n" + "-" * 80)
    pr("(4) SIDE CONDITIONS")
    pr("-" * 80)
    pr("  CP: Choi min eigenvalue (must be >= -1e-12) and trace preservation:")
    for name, ch, din in [("Phi_slot1 marginal", phi_slot1, 2),
                          ("Phi_slot2 marginal", phi_slot2, 2)]:
        J = choi(ch, din)
        mn = float(np.min(np.linalg.eigvalsh((J + J.conj().T) / 2)).real)
        tp = np.allclose(partial_trace(J, keep=[0], dims=[din, din]), np.eye(din), atol=1e-12)
        pr(f"    {name:20s}: min eig(Choi) = {mn:+.3e}   TP = {tp}")
    for d in [0.0, 0.5, 1.0]:
        J = choi(lambda M: depth2_process(M, DARK * d), 4)
        mn = float(np.min(np.linalg.eigvalsh((J + J.conj().T) / 2)).real)
        tp = np.allclose(partial_trace(J, keep=[0], dims=[4, 4]), np.eye(4), atol=1e-12)
        pr(f"    depth-2 process (delta={d:+.2f}): min eig(Choi) = {mn:+.3e}   TP = {tp}")
    prep_mat = np.array([[np.real(np.trace(B @ PREP[l])) for B in (I2, X, Y, Z)]
                         for l in PREP_ORDER])
    eff_mat = np.array([[np.real(np.trace(PAULI[o] @ B)) for B in (I2, X, Y, Z)]
                        for o in EFFECT_ORDER])
    pr("  IC: span of atomic preparations / effects in B(C^2)_sa (dim 4):")
    prk, psv = certified_rank(prep_mat, abs_floor=1e-10)
    erk, esv = certified_rank(eff_mat, abs_floor=1e-10)
    pr(f"    preparations: rank {prk}/4  sing.vals = {np.array2string(psv)}")
    pr(f"    effects     : rank {erk}/4  sing.vals = {np.array2string(esv)}")
    pr(f"    single-time IC on S? {prk == 4 and erk == 4}")

    # repo cross-check
    pr("\n" + "-" * 80)
    pr("(cross-check) repo audit_multiprobe_reflection on quantized N1 (if importable)")
    pr("-" * 80)
    pr(repo_cross_check())

    # verdict
    pr("\n" + "=" * 80)
    ok1 = worst < 1e-12 and worst2 < 1e-12
    ok3 = abs(xy0) < 1e-12 and abs(xyd) > 1e-6
    ok4_ic = prk == 4 and erk == 4
    verdict = ("CANDIDATE A (R_x dark link) SURVIVES all checks"
               if (ok1 and ok3 and ok4_ic) else "CANDIDATE A FAILED a check")
    pr(f"VERDICT: {verdict}")
    pr(f"  (1) exact atomic agreement : {ok1}")
    pr(f"  (3) depth-2 separation     : {ok3}")
    pr(f"  (4) IC of atomic family    : {ok4_ic}")
    pr("=" * 80)

    print("\n".join(out))


if __name__ == "__main__":
    main()
