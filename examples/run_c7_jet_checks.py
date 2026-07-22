"""
run_c7_jet_checks.py -- self-contained numerical checks (numpy only) for the
"jet certificate" claims of the C7 operational-dimension construction.

Four independent checks, each ending in asserts:

  CHECK 1  Delayed-activation counterexample on a classical 4-state chain.
           Scalar derivative of the outcome probability plateaus (S_1=S_2={0})
           then grows at k=3 -> scalar plateaus are NOT closure certificates.

  CHECK 2  The jet recurrence fixes it.  Tracking the augmented zero-jet + first
           -jet span J_k, J_k grows strictly (dim 1,2,3,4,5) then permanently
           stabilizes; its single non-growth step is TRUE permanent closure,
           unlike the false scalar plateau.

  CHECK 3  Qubit comb (CZ - R_x(delta) - CZ).  Already at ZEROTH order the
           depth-2 connected correlator is nonzero, so the augmented Hankel does
           not plateau at depth 1 (no false pass; depth-2 probing is demanded).

  CHECK 4  Purification / Kraus rank.  For a general SU(2) link G the induced
           (S1,S2) channel has Choi rank exactly 2 (one-qubit environment is
           minimal), reproduced by K0=(g00 I + g01 Z1)/sqrt2,
           K1=Z2(g10 I + g11 Z1)/sqrt2.

  CHECK 5  Augmented jet-Hankel flat-extension audit (Theorem T2''',
           EB-R4-CYZM-8M1X5T, GitHub issue #5).  A finite rank plateau is NOT a
           closure certificate.  The audit returns INCONCLUSIVE for a delayed-
           chain plateau without an order bound, and GROWTH_DETECTED whenever a
           one-letter extension escapes the span (the delayed mode at depth N,
           and the CZ-R_x-CZ comb from atomic to depth 2).  Certification has two
           routes: (b) a known-order realization, and (d) the RANK-SATURATION
           route (Lemma 2.4) -- a two-letter order-3 realization whose length-<=1
           block already saturates Hankel rank 3 == declared 3 certifies WITHOUT
           exhausting horizon 3 or covering every word through length n_J-1.

Run:  PYTHONPATH=src python examples/run_c7_jet_checks.py
"""

import numpy as np

from cyz_m.jet_hankel import (
    build_augmented_jet_hankel,
    build_jet_letter,
    flat_extension_audit,
)

np.set_printoptions(precision=6, suppress=True, linewidth=140)

# ------------------------------ qubit algebra ------------------------------
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
PAULI = {"I": I2, "X": X, "Y": Y, "Z": Z}

k0 = np.array([1, 0], dtype=complex)
k1 = np.array([0, 1], dtype=complex)
kp = (k0 + k1) / np.sqrt(2)
km = (k0 - k1) / np.sqrt(2)
kpi = (k0 + 1j * k1) / np.sqrt(2)
kmi = (k0 - 1j * k1) / np.sqrt(2)


def dm(psi):
    psi = np.asarray(psi, dtype=complex)
    return np.outer(psi, psi.conj())


PREP = {"|0>": dm(k0), "|1>": dm(k1), "|+>": dm(kp),
        "|->": dm(km), "|+i>": dm(kpi), "|-i>": dm(kmi)}
PREP_ORDER = ["|0>", "|1>", "|+>", "|->", "|+i>", "|-i>"]


def rank_tol(mat, tol=1e-9):
    mat = np.asarray(mat, dtype=complex)
    if mat.size == 0:
        return 0
    sv = np.linalg.svd(mat, compute_uv=False)
    thr = max(tol, tol * (sv[0] if sv.size else 0.0))
    return int(np.sum(sv > thr))


# ===========================================================================
# CHECK 1 -- delayed-activation counterexample (classical 4-state chain)
# ===========================================================================
def T_classical(theta):
    """Column-stochastic 4x4 transition matrix. Columns index the source state.
       1->2, 2->3, 3->(1/2-theta)*3+(1/2+theta)*4, 4->4."""
    T = np.zeros((4, 4))
    T[1, 0] = 1.0                      # 1 -> 2
    T[2, 1] = 1.0                      # 2 -> 3
    T[2, 2] = 0.5 - theta             # 3 -> 3
    T[3, 2] = 0.5 + theta             # 3 -> 4
    T[3, 3] = 1.0                      # 4 -> 4
    return T


def check1():
    lines = ["=" * 78,
             "CHECK 1  delayed-activation counterexample (classical 4-state chain)",
             "=" * 78]
    v0 = np.array([1.0, 0.0, 0.0, 0.0])       # init: state 1
    eff = np.array([0.0, 0.0, 0.0, 1.0])      # effect: indicator of state 4

    def p_of_theta(theta, k):
        M = np.linalg.matrix_power(T_classical(theta), k)
        return float(eff @ (M @ v0))

    h = 1e-6
    derivs = {}
    lines.append("  p_k(theta) = <state4 | T(theta)^k | state1>")
    lines.append(f"  {'k':>3} | {'p_k(0)':>12} | {'dp_k/dtheta':>14} | scalar rank S_k")
    lines.append("  " + "-" * 60)
    for k in range(1, 6):
        p0 = p_of_theta(0.0, k)
        d = (p_of_theta(h, k) - p_of_theta(-h, k)) / (2 * h)
        derivs[k] = d
        Sk = 0 if abs(d) < 1e-9 else 1
        lines.append(f"  {k:>3} | {p0:>12.6f} | {d:>14.6f} | {{{'0' if Sk==0 else 'R'}}}  (rank {Sk})")

    d1, d2, d3 = derivs[1], derivs[2], derivs[3]
    lines.append("")
    lines.append(f"  d/dtheta p_1 = {d1:+.6e}   (expect 0)")
    lines.append(f"  d/dtheta p_2 = {d2:+.6e}   (expect 0)")
    lines.append(f"  d/dtheta p_3 = {d3:+.6e}   (expect 1)")
    lines.append("  scalar derivative-rank sequence S_k: S_1=S_2={0} (PLATEAU) then S_3={R} (GROWTH)")
    lines.append("  => a scalar plateau is NOT a closure certificate.")

    ok = abs(d1) < 1e-8 and abs(d2) < 1e-8 and abs(d3 - 1.0) < 1e-6
    lines.append(f"  VERDICT CHECK 1: {'PASS' if ok else 'FAIL'}")
    assert abs(d1) < 1e-8, "d/dtheta p_1 not zero"
    assert abs(d2) < 1e-8, "d/dtheta p_2 not zero"
    assert abs(d3 - 1.0) < 1e-6, "d/dtheta p_3 not 1"
    return "\n".join(lines), (d1, d2, d3)


# ===========================================================================
# CHECK 2 -- jet recurrence: augmented zero-jet + first-jet span J_k
# ===========================================================================
def check2():
    lines = ["", "=" * 78,
             "CHECK 2  jet recurrence -- augmented span J_k (zero-jet + first-jet)",
             "=" * 78]
    T0 = T_classical(0.0)
    dT = (T_classical(1e-6) - T_classical(-1e-6)) / (2e-6)   # dT/dtheta at 0
    # analytic dT: only column 3 nonzero = [0,0,-1,1]
    dT_exact = np.zeros((4, 4)); dT_exact[2, 2] = -1.0; dT_exact[3, 2] = 1.0
    assert np.allclose(dT, dT_exact, atol=1e-6)
    dT = dT_exact

    KMAX = 12
    v0 = np.array([1.0, 0.0, 0.0, 0.0])
    dv = np.zeros(4)
    aug_vectors = []           # augmented (v0_j, dv_j) in R^8
    dimJ = []
    scalar_deriv = []          # dp_k/dtheta = eff . dv_k  (eff = state 4)
    eff = np.array([0.0, 0.0, 0.0, 1.0])
    zerojet_hist = []

    aug_vectors.append(np.concatenate([v0, dv]))
    dimJ.append(rank_tol(np.array(aug_vectors).T))
    scalar_deriv.append(float(eff @ dv))
    zerojet_hist.append(v0.copy())

    for k in range(1, KMAX + 1):
        v0_new = T0 @ v0
        dv_new = T0 @ dv + dT @ v0
        v0, dv = v0_new, dv_new
        aug_vectors.append(np.concatenate([v0, dv]))
        dimJ.append(rank_tol(np.array(aug_vectors).T))
        scalar_deriv.append(float(eff @ dv))
        zerojet_hist.append(v0.copy())

    lines.append("  lifted recurrence at theta0=0:")
    lines.append("     v0_{k+1} = T(0) v0_k")
    lines.append("     dv_{k+1} = T(0) dv_k + (dT/dtheta) v0_k")
    lines.append("")
    lines.append(f"  {'k':>3} | {'dim J_k':>8} | {'grew?':>6} | {'zero-jet v0_k (state4 comp)':>28} | {'scalar dp_k':>12}")
    lines.append("  " + "-" * 76)
    for k in range(0, 6):
        grew = "-" if k == 0 else ("yes" if dimJ[k] > dimJ[k - 1] else "NO")
        lines.append(f"  {k:>3} | {dimJ[k]:>8} | {grew:>6} | {zerojet_hist[k][3]:>28.6f} | {scalar_deriv[k]:>12.6f}")

    dimJ_06 = dimJ[:6]
    lines.append("")
    lines.append(f"  dim J_k for k=0..5           : {dimJ_06}")
    lines.append(f"  dim J_k for k=0..{KMAX} (full)  : {dimJ}")

    # find first non-growth step and confirm it is permanent
    first_flat = next(k for k in range(1, len(dimJ)) if dimJ[k] == dimJ[k - 1])
    permanent = all(dimJ[k] == dimJ[first_flat] for k in range(first_flat, len(dimJ)))
    lines.append(f"  first non-growth step of J_k : k={first_flat-1}->{first_flat} "
                 f"(dim stays {dimJ[first_flat]})")
    lines.append(f"  that non-growth is PERMANENT  : {permanent}  (dim never grows again)")

    # contrast: the k=1->2 step is strictly GROWING even though dv is still 0
    lines.append(f"  at k=1->2: dim J grows {dimJ[1]}->{dimJ[2]} while first-jet dv stays 0")
    lines.append("            (the NEW zero-jet component e_3 is what J correctly captures)")

    # scalar derivatives keep changing AFTER J has closed
    changed_after = any(abs(scalar_deriv[k] - scalar_deriv[k - 1]) > 1e-9
                        for k in range(first_flat + 1, len(scalar_deriv)))
    lines.append(f"  scalar dp_k sequence          : "
                 f"{[round(s,4) for s in scalar_deriv[:8]]} ...")
    lines.append(f"  scalar dp_k still changing after J closed (k>={first_flat}) : {changed_after}")
    lines.append("  => the SINGLE non-growth step of J_k is the genuine permanent-closure")
    lines.append("     certificate; the scalar sequence has a FALSE early plateau (k=1,2)")
    lines.append("     and never gives a clean finite closure signal.")

    ok = (dimJ_06 == [1, 2, 3, 4, 5, 5] and permanent and changed_after)
    lines.append(f"  VERDICT CHECK 2: {'PASS' if ok else 'FAIL'}")
    assert dimJ_06 == [1, 2, 3, 4, 5, 5], f"dim J_k mismatch: {dimJ_06}"
    assert permanent, "J non-growth step not permanent"
    assert changed_after, "scalar derivatives unexpectedly constant"
    return "\n".join(lines), dimJ_06


# ===========================================================================
# comb plumbing (shared by checks 3 and 4)
# ===========================================================================
def su2_exp(nx, ny, nz, angle):
    """exp(-i angle (nx X + ny Y + nz Z)/2), n a unit vector (or angle scales)."""
    return np.cos(angle / 2) * I2 - 1j * np.sin(angle / 2) * (nx * X + ny * Y + nz * Z)


def cz3(a, b):
    """CZ between qubits a,b in a 3-qubit register (order [S1,E,S2])."""
    n = 3
    dim = 8
    diag = np.ones(dim, dtype=complex)
    for idx in range(dim):
        bits = [(idx >> (n - 1 - q)) & 1 for q in range(n)]
        if bits[a] == 1 and bits[b] == 1:
            diag[idx] = -1.0
    return np.diag(diag)


U1 = cz3(0, 1)     # write CZ_{S1,E}
U2 = cz3(2, 1)     # read  CZ_{S2,E}


def link_E(G):
    """Embed 2x2 gate G on E (qubit 1) of the 3-qubit register."""
    return np.kron(I2, np.kron(G, I2))


def comb_V(G):
    return U2 @ link_E(G) @ U1


def embed_S1S2(M):
    """Embed a 4x4 operator M on (S1,S2) into [S1,E,S2] with E = |+><+|."""
    M4 = M.reshape(2, 2, 2, 2)              # [s1,s2,s1',s2']
    Ep = 0.5 * np.ones((2, 2), dtype=complex)  # |+><+| = 0.5 J
    full6 = np.einsum('abcd,ef->aebcfd', M4, Ep)   # [s1,e,s2,s1',e',s2']
    return full6.reshape(8, 8)


def comb_channel(M, G):
    """Induced (S1,S2)->(S1,S2) channel of the comb, tracing out E."""
    V = comb_V(G)
    out_full = V @ embed_S1S2(M) @ V.conj().T
    out6 = out_full.reshape(2, 2, 2, 2, 2, 2)   # [s1,e,s2,s1',e',s2']
    out4 = np.einsum('aebAeB->abAB', out6)      # trace e; -> [s1,s2,s1',s2']
    return out4.reshape(4, 4)


def choi_of(channel):
    """16x16 Choi = sum_ij Eij (x) channel(Eij), din=4."""
    J = np.zeros((16, 16), dtype=complex)
    for i in range(4):
        for j in range(4):
            Eij = np.zeros((4, 4), dtype=complex)
            Eij[i, j] = 1.0
            J += np.kron(Eij, channel(Eij))
    return J


# ===========================================================================
# CHECK 3 -- augmented Hankel does not plateau at depth 1 (connected correlator)
# ===========================================================================
def check3():
    lines = ["", "=" * 78,
             "CHECK 3  qubit comb: nonzero ZEROTH-order depth-2 connected correlator",
             "=" * 78]
    G0 = I2.copy()  # delta = 0
    V = comb_V(G0)

    def full_out(rho_S1, rho_S2):
        rho_in = np.kron(np.kron(rho_S1, dm(kp)), rho_S2)   # E = |+>
        return V @ rho_in @ V.conj().T

    def expect(op8, rho8):
        return float(np.real(np.trace(op8 @ rho8)))

    def emb(op, pos):
        ops = [I2, I2, I2]; ops[pos] = op
        return np.kron(ops[0], np.kron(ops[1], ops[2]))

    lines.append("  delta = 0 (G = I). connected correlator")
    lines.append("     <P_S1 Q_S2>_c = <P_S1 Q_S2> - <P_S1><Q_S2>")
    lines.append(f"  {'input S1':>8} {'input S2':>8} | {'P':>2} {'Q':>2} | "
                 f"{'<PQ>':>10} {'<P><Q>':>10} {'connected':>11}")
    lines.append("  " + "-" * 66)

    best = (0.0, None)
    reported = []
    for l1 in ["|0>", "|+>"]:
        for l2 in ["|0>", "|+>"]:
            rho = full_out(PREP[l1], PREP[l2])
            for pq in [("Z", "Z"), ("X", "X"), ("Z", "X"), ("X", "Z")]:
                p, q = pq
                pqv = expect(emb(PAULI[p], 0) @ emb(PAULI[q], 2), rho)
                pv = expect(emb(PAULI[p], 0), rho)
                qv = expect(emb(PAULI[q], 2), rho)
                conn = pqv - pv * qv
                if abs(conn) > 1e-9:
                    reported.append((l1, l2, p, q, pqv, pv * qv, conn))
                if abs(conn) > abs(best[0]):
                    best = (conn, (l1, l2, p, q, pqv, pv, qv))
    for (l1, l2, p, q, pqv, prod, conn) in reported:
        lines.append(f"  {l1:>8} {l2:>8} | {p:>2} {q:>2} | "
                     f"{pqv:>10.6f} {prod:>10.6f} {conn:>+11.6f}")

    conn, wit = best
    l1, l2, p, q, pqv, pv, qv = wit
    lines.append("")
    lines.append(f"  HEADLINE witness: inputs S1={l1}, S2={l2}")
    lines.append(f"     <{p}_S1 {q}_S2>        = {pqv:+.6f}")
    lines.append(f"     <{p}_S1> = {pv:+.6f} , <{q}_S2> = {qv:+.6f} , product = {pv*qv:+.6f}")
    lines.append(f"     connected correlator = {conn:+.6f}   (NONZERO at zeroth order)")
    lines.append("  => depth-2 zeroth-order statistics are NOT products of depth-1 stats,")
    lines.append("     so the augmented (0th+1st) Hankel does not plateau at depth 1;")
    lines.append("     the jet certificate correctly demands depth-2 probing (no false pass).")

    ok = abs(conn) > 1e-6
    lines.append(f"  VERDICT CHECK 3: {'PASS' if ok else 'FAIL'}")
    assert abs(conn) > 1e-6, "no nonzero zeroth-order connected correlator found"
    return "\n".join(lines), conn


# ===========================================================================
# CHECK 4 -- purification Kraus rank of the induced (S1,S2) channel
# ===========================================================================
def kraus_channel(G):
    """Analytic 2-Kraus form. g_ij = <i|G|j>."""
    g00, g01 = G[0, 0], G[0, 1]
    g10, g11 = G[1, 0], G[1, 1]
    Z1 = np.kron(Z, I2)      # Z on S1
    Z2 = np.kron(I2, Z)      # Z on S2
    Id4 = np.eye(4, dtype=complex)
    K0 = (g00 * Id4 + g01 * Z1) / np.sqrt(2)
    K1 = Z2 @ (g10 * Id4 + g11 * Z1) / np.sqrt(2)
    return [K0, K1]


def check4():
    lines = ["", "=" * 78,
             "CHECK 4  purification / Kraus rank of the induced (S1,S2) channel",
             "=" * 78]
    rng = np.random.default_rng(20260721)

    links = [("R_x(0.4)", su2_exp(1, 0, 0, 0.4))]
    for i in range(5):
        angle = rng.uniform(0, 2 * np.pi)
        axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
        G = su2_exp(axis[0], axis[1], axis[2], angle)
        links.append((f"rand SU(2) #{i+1}", G))

    lines.append("  For each link G: Choi rank of induced channel, and max |Choi_full - Choi_Kraus|.")
    lines.append("  Kraus:  K0 = (g00 I + g01 Z1)/sqrt2 ,  K1 = Z2 (g10 I + g11 Z1)/sqrt2")
    lines.append(f"  {'link':>16} | {'Choi rank':>9} | {'lead eig(Choi)':>28} | {'max dev':>10} | {'TP':>4}")
    lines.append("  " + "-" * 78)

    all_ok = True
    ranks = []
    devs = []
    for name, G in links:
        J_full = choi_of(lambda M: comb_channel(M, G))
        Kr = kraus_channel(G)
        J_kraus = choi_of(lambda M: sum(K @ M @ K.conj().T for K in Kr))
        Jh = (J_full + J_full.conj().T) / 2
        eigs = np.sort(np.linalg.eigvalsh(Jh).real)[::-1]
        r = int(np.sum(eigs > 1e-9))
        dev = float(np.max(np.abs(J_full - J_kraus)))
        # trace preservation: partial trace over output = I_4
        Jt = J_full.reshape(4, 4, 4, 4)
        tp_out = np.einsum('aibi->ab', Jt)   # trace output index -> should be I_in
        tp = np.allclose(tp_out, np.eye(4), atol=1e-9)
        ranks.append(r); devs.append(dev)
        lead = np.array2string(eigs[:2], precision=4)
        lines.append(f"  {name:>16} | {r:>9} | {lead:>28} | {dev:>10.2e} | {str(tp):>4}")
        all_ok = all_ok and (r == 2) and (dev < 1e-10) and tp

    lines.append("")
    lines.append(f"  all Choi ranks = {ranks}  (all == 2 => one-qubit environment is minimal)")
    lines.append(f"  max Choi deviation over all links = {max(devs):.2e}")
    lines.append("  => claimed 2-Kraus form reproduces the channel exactly; rank 2 confirmed.")
    ok = all_ok
    lines.append(f"  VERDICT CHECK 4: {'PASS' if ok else 'FAIL'}")
    assert all(r == 2 for r in ranks), f"Choi rank not all 2: {ranks}"
    assert max(devs) < 1e-10, f"Kraus reproduction deviation too large: {max(devs)}"
    return "\n".join(lines), (ranks, max(devs))


# ===========================================================================
# CHECK 5 -- augmented jet-Hankel flat-extension audit (R3 D repair)
# ===========================================================================
def _words_up_to(length, letter="T"):
    return [tuple([letter] * k) for k in range(length + 1)]


def _delayed_chain_hankel(chain_length, prefix_len, suffix_len):
    """Visible rank-1 subsystem direct-summed with a hidden delayed chain e_0..e_N."""
    n = chain_length + 3           # visible(0), e_0..e_N (1..N+1), sink (N+2)
    sink = chain_length + 2

    def st(i):
        return 1 + i

    def T(theta):
        M = np.zeros((n, n))
        M[0, 0] = 1.0                                   # visible self-loop
        for i in range(chain_length - 1):
            M[st(i + 1), st(i)] = 1.0                   # e_i -> e_{i+1}
        M[sink, st(chain_length - 1)] = 0.5 - theta     # e_{N-1} -> sink
        M[st(chain_length), st(chain_length - 1)] = 0.5 + theta  # e_{N-1} -> e_N
        M[st(chain_length), st(chain_length)] = 1.0     # e_N absorbing
        M[sink, sink] = 1.0
        return M

    T0 = T(0.0)
    dT = np.zeros((n, n))
    dT[sink, st(chain_length - 1)] = -1.0
    dT[st(chain_length), st(chain_length - 1)] = 1.0

    v0 = np.zeros(n); v0[0] = 1.0; v0[st(0)] = 0.5
    detect_v = np.zeros(n); detect_v[0] = 1.0
    detect_eN = np.zeros(n); detect_eN[st(chain_length)] = 1.0
    letter = build_jet_letter(T0, dT)
    return build_augmented_jet_hankel(
        (v0, np.zeros(n)), {"T": letter},
        [(detect_v, np.zeros(n)), (detect_eN, np.zeros(n))],
        _words_up_to(prefix_len), _words_up_to(suffix_len),
    )


def _four_state_hankel(prefix_len, suffix_len):
    T0 = T_classical(0.0)
    dT = np.zeros((4, 4)); dT[2, 2] = -1.0; dT[3, 2] = 1.0
    letter = build_jet_letter(T0, dT)
    v0 = np.array([1.0, 0.0, 0.0, 0.0])
    eff = np.array([0.0, 0.0, 0.0, 1.0])
    return build_augmented_jet_hankel(
        (v0, np.zeros(4)), {"T": letter}, [(eff, np.zeros(4))],
        _words_up_to(prefix_len), _words_up_to(suffix_len),
    )


def _saturating_two_letter_hankel(words):
    """Two-letter, purely zeroth-order realization of minimal order 3. The seed
    reaches all of R^3 through the length-<=1 words {(), (A,), (B,)} and the
    readout observes all of R^3 through the same suffixes, so that block already
    SATURATES at Hankel rank 3 -- without exhausting horizon 3 or covering every
    word through length n_J-1=2. It is the rank-saturation-without-horizon-
    exhaustion witness of Lemma 2.4 (Theorem T2''', change #2)."""
    A0 = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [0.0, -1.0, -2.0]])
    B0 = np.array([[0.0, -1.0, 2.0], [0.0, -2.0, 0.0], [1.0, -2.0, 1.0]])
    letA = build_jet_letter(A0, np.zeros((3, 3)))
    letB = build_jet_letter(B0, np.zeros((3, 3)))
    seed = np.array([1.0, 0.0, 0.0])
    readout = np.array([-2.0, 0.0, 2.0])
    word_list = [tuple(w) for w in words]
    return build_augmented_jet_hankel(
        (seed, np.zeros(3)), {"A": letA, "B": letB},
        [(readout, np.zeros(3))], word_list, word_list,
    )


def _comb_hankel(delta0, depth):
    """CZ-R_x(delta)-CZ comb: memory qubit E (|0> after the S1-write branch)
       evolving under the R_x link, read by S2; jet taken w.r.t. delta."""
    basis = [I2, X, Y, Z]

    def coords(rho):
        return np.array([np.real(np.trace(op @ rho)) for op in basis])

    def dRx(angle):
        return -0.5 * np.sin(angle / 2) * I2 - 0.5j * np.cos(angle / 2) * X

    U = su2_exp(1, 0, 0, delta0)
    dU = dRx(delta0)
    T0 = np.zeros((4, 4)); dT = np.zeros((4, 4))
    for col, pin in enumerate(basis):
        rho = 0.5 * pin
        T0[:, col] = coords(U @ rho @ U.conj().T)
        dT[:, col] = coords(dU @ rho @ U.conj().T + U @ rho @ dU.conj().T)
    letter = build_jet_letter(T0, dT)
    seed = coords(0.5 * (I2 + Z))                        # |0>
    read_y = np.array([0.0, 0.0, 1.0, 0.0])
    read_z = np.array([0.0, 0.0, 0.0, 1.0])
    return build_augmented_jet_hankel(
        (seed, np.zeros(4)), {"L": letter},
        [(read_y, np.zeros(4)), (read_z, np.zeros(4))],
        _words_up_to(depth, "L"), _words_up_to(depth, "L"),
    )


def check5():
    lines = ["", "=" * 78,
             "CHECK 5  augmented jet-Hankel flat-extension audit (R3 repair D)",
             "=" * 78]
    lines.append(f"  {'fixture':>34} | {'rank':>4} | {'max resid':>11} | status")
    lines.append("  " + "-" * 74)

    results = {}

    a_plateau = flat_extension_audit(
        _delayed_chain_hankel(8, 2, 2), declared_order_bound=None)
    results["a_plateau"] = a_plateau
    lines.append(f"  {'(a) delayed chain h=2, no bound':>34} | {a_plateau.numerical_rank:>4} | "
                 f"{a_plateau.max_residual:>11.2e} | {a_plateau.status}")

    a_grow = flat_extension_audit(
        _delayed_chain_hankel(8, 7, 0), declared_order_bound=None)
    results["a_grow"] = a_grow
    lines.append(f"  {'(a) delayed chain, extension @N=8':>34} | {a_grow.numerical_rank:>4} | "
                 f"{a_grow.max_residual:>11.2e} | {a_grow.status}")

    b_cert = flat_extension_audit(
        _four_state_hankel(5, 5), declared_order_bound=5)
    results["b_cert"] = b_cert
    lines.append(f"  {'(b) 4-state order 5, bound 5':>34} | {b_cert.numerical_rank:>4} | "
                 f"{b_cert.max_residual:>11.2e} | {b_cert.status}")

    c_grow = flat_extension_audit(_comb_hankel(0.7, 1), declared_order_bound=None)
    results["c_grow"] = c_grow
    lines.append(f"  {'(c) CZ-Rx-CZ comb, atomic->depth2':>34} | {c_grow.numerical_rank:>4} | "
                 f"{c_grow.max_residual:>11.2e} | {c_grow.status}")

    # (d) RANK-SATURATION ROUTE (Lemma 2.4): the two-letter order-3 realization,
    # fed only the length-<=1 words {(), (A,), (B,)}, saturates at Hankel rank 3.
    # rank == declared 5? no -- declared 3. The block neither reaches horizon 3
    # nor covers every word through length n_J-1=2, yet rank == n_J certifies.
    sat_block = [(), ("A",), ("B",)]
    d_sat = flat_extension_audit(
        _saturating_two_letter_hankel(sat_block), declared_order_bound=3)
    results["d_sat"] = d_sat
    lines.append(f"  {'(d) 2-letter order 3, |w|<=1, bnd 3':>34} | {d_sat.numerical_rank:>4} | "
                 f"{d_sat.max_residual:>11.2e} | {d_sat.status}")
    lines.append(f"       rank-saturation route: horizons_exhausted={d_sat.horizons_exhaust_bound}, "
                 f"all_words_covered={d_sat.all_words_covered}, certified_order={d_sat.certified_order}")

    ok = (
        a_plateau.status == "inconclusive_no_order_bound"
        and a_grow.status == "growth_detected"
        and b_cert.status == "certified_for_declared_class"
        and c_grow.status == "growth_detected"
        and d_sat.status == "certified_for_declared_class"
        and d_sat.certified_order == 3
        and not d_sat.horizons_exhaust_bound
        and not d_sat.all_words_covered
    )
    lines.append("")
    lines.append("  => a finite plateau alone is INCONCLUSIVE; certification needs a declared")
    lines.append("     order bound EITHER exhausted by rank saturation (rank == n_J, Lemma 2.4,")
    lines.append("     no horizon requirement) or by full all-words coverage; growth always falsifies.")
    lines.append(f"  VERDICT CHECK 5: {'PASS' if ok else 'FAIL'}")
    assert a_plateau.status == "inconclusive_no_order_bound", a_plateau.status
    assert a_grow.status == "growth_detected", a_grow.status
    assert b_cert.status == "certified_for_declared_class", b_cert.status
    assert c_grow.status == "growth_detected", c_grow.status
    assert d_sat.status == "certified_for_declared_class", d_sat.status
    assert d_sat.certified_order == 3, d_sat.certified_order
    assert not d_sat.horizons_exhaust_bound, "saturation route must not need horizon exhaustion"
    assert not d_sat.all_words_covered, "saturation route must not rely on all-words coverage"
    return "\n".join(lines), results


# --------------------------------- main ------------------------------------
def main():
    out1, r1 = check1()
    out2, r2 = check2()
    out3, r3 = check3()
    out4, r4 = check4()
    out5, r5 = check5()
    print(out1)
    print(out2)
    print(out3)
    print(out4)
    print(out5)
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"  CHECK 1  scalar dp_1,dp_2,dp_3 = {tuple(round(x,6) for x in r1)}   -> PASS")
    print(f"  CHECK 2  dim J_k (k=0..5) = {r2}                        -> PASS")
    print(f"  CHECK 3  zeroth-order connected correlator = {r3:+.6f}    -> PASS")
    print(f"  CHECK 4  Choi ranks = {r4[0]}, max dev = {r4[1]:.2e}   -> PASS")
    print(f"  CHECK 5  jet-Hankel audit: (a) {r5['a_plateau'].status} / "
          f"{r5['a_grow'].status}, (b) {r5['b_cert'].status}, (c) {r5['c_grow'].status}, "
          f"(d) {r5['d_sat'].status}[order {r5['d_sat'].certified_order}]   -> PASS")
    print("  ALL CHECKS PASS")


if __name__ == "__main__":
    main()
