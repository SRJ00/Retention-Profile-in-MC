"""
Per-row verification of the numerical table in Remark 7.6 of
"Row-Based KL Contraction and Localization in Finite Markov Chains" (v2).

For every chain family in the table we:
  (i)   build the kernel exactly as specified in the manuscript,
  (ii)  compute pi via the dominant left eigenvector of P,
  (iii) compute r(x), M, rho(P),
  (iv)  estimate eta_KL(P) by maximizing F(mu) = KL(muP||pi)/KL(mu||pi) over
        an aggressive candidate pool:
            - every point mass delta_x  (boundary candidates),
            - the local-quadratic candidate mu_pi + eps * pi * phi_2 from
              Proposition 5.x (eigenfunction-perturbation candidate),
            - family-specific structured candidates (cycle arcs, hypercube
              faces, barbell clique mixtures, path arcs),
            - Dirichlet random restarts, and
            - L-BFGS-B (always) plus Nelder-Mead (n <= 30)
              on the softmax parameterisation.
  (v)   check that rho and eta/M land in the intervals quoted in the table.

Output: one line per chain instance + a PASS / FAIL verdict per table row.

Reproducibility: seed = 20260511 (matches manuscript Section "Numerical
methodology").
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np
from scipy import optimize

warnings.filterwarnings("ignore")

SEED = 20260511
RNG = np.random.default_rng(SEED)

# Tolerance used when checking whether a computed quantity lies inside the
# range printed in the table (intervals printed to 2 decimals).
INT_TOL = 0.015          # for ratios printed to 2 decimals
RHO_TOL_EXACT = 5e-4     # for rho values asserted to be exactly 1 or 1/2
RHO_TOL_RANGE = 0.01     # for rho ranges (path, BD, barbell)


# ---------------------------------------------------------------------------
# Core numerics
# ---------------------------------------------------------------------------

def kl(mu: np.ndarray, nu: np.ndarray) -> float:
    mask = mu > 1e-300
    return float(np.sum(mu[mask] * np.log(mu[mask] / nu[mask])))


def stationary(P: np.ndarray) -> np.ndarray:
    w, V = np.linalg.eig(P.T)
    i = int(np.argmin(np.abs(w - 1.0)))
    pi = np.real(V[:, i])
    pi = np.abs(pi)
    pi /= pi.sum()
    # Polish with one power-iteration step to reduce eigensolver noise.
    for _ in range(50):
        pi_new = pi @ P
        pi_new /= pi_new.sum()
        if np.max(np.abs(pi_new - pi)) < 1e-14:
            break
        pi = pi_new
    return pi


def retention(P: np.ndarray, pi: np.ndarray) -> np.ndarray:
    n = P.shape[0]
    r = np.zeros(n)
    for x in range(n):
        denom = np.log(1.0 / pi[x])
        r[x] = kl(P[x], pi) / denom if denom > 1e-15 else 0.0
    return r


def rho_M(P: np.ndarray, pi: np.ndarray) -> Tuple[float, float, np.ndarray]:
    r = retention(P, pi)
    M = float(np.max(r))
    if M < 1e-15:
        return 1.0, 0.0, r
    rbar = float(np.dot(pi, r))
    return rbar / M, M, r


def ratio(mu: np.ndarray, P: np.ndarray, pi: np.ndarray) -> float:
    mu = np.clip(mu, 1e-300, None)
    mu = mu / mu.sum()
    d0 = kl(mu, pi)
    if d0 < 1e-12:
        return 0.0
    mup = mu @ P
    mup = np.clip(mup, 1e-300, None)
    mup = mup / mup.sum()
    return kl(mup, pi) / d0


# ---------------------------------------------------------------------------
# Global search for eta_KL(P)
# ---------------------------------------------------------------------------

def _softmax_neg(P: np.ndarray, pi: np.ndarray) -> Callable[[np.ndarray], float]:
    def f(theta: np.ndarray) -> float:
        theta = theta - theta.max()
        mu = np.exp(theta)
        mu /= mu.sum()
        return -ratio(mu, P, pi)
    return f


def _optimize_from(theta0: np.ndarray, neg: Callable, methods: Tuple[str, ...]) -> float:
    best = 0.0
    for m in methods:
        try:
            if m == "L-BFGS-B":
                res = optimize.minimize(
                    neg, theta0, method=m,
                    options={"maxiter": 5000, "ftol": 1e-15, "gtol": 1e-12},
                )
            else:
                res = optimize.minimize(
                    neg, theta0, method=m,
                    options={"maxiter": 10000, "xatol": 1e-12, "fatol": 1e-14},
                )
            best = max(best, -float(res.fun))
        except Exception:
            pass
    return best


def eta_kl(
    P: np.ndarray,
    pi: np.ndarray,
    extra_candidates: Optional[List[np.ndarray]] = None,
    n_dirichlet: int = 120,
    use_nelder: Optional[bool] = None,
) -> Tuple[float, str]:
    """
    Strong lower bound on eta_KL(P). Returns (best_ratio, witness_tag).
    """
    n = P.shape[0]
    if use_nelder is None:
        use_nelder = n <= 30

    neg = _softmax_neg(P, pi)
    methods_full = ("L-BFGS-B", "Nelder-Mead") if use_nelder else ("L-BFGS-B",)
    methods_quick = ("L-BFGS-B",)

    best = 0.0
    witness = "init"

    def consider(mu: np.ndarray, tag: str, methods: Tuple[str, ...]):
        nonlocal best, witness
        mu = np.clip(mu, 1e-300, None)
        mu = mu / mu.sum()
        # Direct evaluation.
        v = ratio(mu, P, pi)
        if v > best:
            best, witness = v, f"{tag}-direct"
        # Polish via softmax.
        theta0 = np.log(np.clip(mu, 1e-12, None))
        v2 = _optimize_from(theta0, neg, methods)
        if v2 > best:
            best, witness = v2, f"{tag}-polish"

    # (1) Boundary: every point mass.
    for i in range(n):
        e = np.zeros(n)
        e[i] = 1.0
        consider(e, f"delta_{i}", methods_quick)

    # (2) Local-quadratic candidate: mu = pi + eps * pi * phi_2 (top non-trivial
    # right eigenfunction of P in L^2(pi)).  We construct it via the symmetrized
    # operator when reversible, otherwise we just use right eigvecs of P.
    try:
        # Symmetrise to A = diag(sqrt(pi)) P diag(1/sqrt(pi)); right eigvecs of
        # P correspond to A's eigvecs scaled by 1/sqrt(pi).
        sp = np.sqrt(pi)
        A = (sp[:, None] / sp[None, :]) * P
        w, V = np.linalg.eig(A)
        # Sort by |w| descending; skip the trivial eigenvalue 1.
        order = np.argsort(-np.abs(w))
        for k in order:
            if abs(w[k] - 1.0) < 1e-8:
                continue
            phi = np.real(V[:, k]) / sp
            phi = phi - np.dot(pi, phi)  # ensure mean-zero under pi
            if np.linalg.norm(phi) < 1e-12:
                continue
            phi /= np.sqrt(np.dot(pi, phi**2))
            for eps in (0.05, 0.1, 0.2, 0.4, 0.6, 0.8):
                mu = pi * (1.0 + eps * phi)
                if np.any(mu <= 0):
                    continue
                consider(mu, f"eig{k}_eps{eps}", methods_quick)
            break  # only the leading nontrivial eigenfunction
    except Exception:
        pass

    # (3) Family-specific structured candidates.
    if extra_candidates:
        for j, mu in enumerate(extra_candidates):
            consider(mu, f"struct_{j}", methods_full)

    # (4) Dirichlet random restarts (concentrated and diffuse).
    for k in range(n_dirichlet):
        alpha_scale = RNG.choice([0.1, 0.3, 1.0, 3.0, 10.0])
        mu = RNG.dirichlet(np.full(n, alpha_scale))
        consider(mu, f"dir_{k}", methods_quick)

    # (5) A few random softmax inits (heavy-tailed) to break out of plateaus.
    for k in range(60):
        scale = RNG.choice([0.5, 2.0, 5.0, 15.0])
        theta0 = RNG.standard_normal(n) * scale
        v = _optimize_from(theta0, neg, methods_quick)
        if v > best:
            best, witness = v, f"sm_rand_{k}"

    return best, witness


# ---------------------------------------------------------------------------
# Chain constructors (exactly as in the manuscript)
# ---------------------------------------------------------------------------

def complete_graph(n: int) -> np.ndarray:
    P = np.full((n, n), 1.0 / (n - 1))
    np.fill_diagonal(P, 0.0)
    return P


def lazy_star(n: int) -> np.ndarray:
    """Lazy star: center 0, leaves 1..n-1.
    P(0,0)=1/2, P(0,i)=1/(2(n-1)); P(i,i)=P(i,0)=1/2."""
    P = np.zeros((n, n))
    P[0, 0] = 0.5
    for i in range(1, n):
        P[0, i] = 0.5 / (n - 1)
        P[i, i] = 0.5
        P[i, 0] = 0.5
    return P


def lazy_cycle(n: int) -> np.ndarray:
    P = np.zeros((n, n))
    for i in range(n):
        P[i, i] = 0.5
        P[i, (i + 1) % n] = 0.25
        P[i, (i - 1) % n] = 0.25
    return P


def lazy_path(n: int) -> np.ndarray:
    """Lazy path with non-reflecting boundary as specified in footnote of
    Remark 7.6: endpoints transition to their unique neighbor with prob 1/2.
    Interior: P(i,i)=1/2, P(i,i+-1)=1/4."""
    P = np.zeros((n, n))
    P[0, 0] = 0.5
    P[0, 1] = 0.5
    P[n - 1, n - 1] = 0.5
    P[n - 1, n - 2] = 0.5
    for i in range(1, n - 1):
        P[i, i] = 0.5
        P[i, i - 1] = 0.25
        P[i, i + 1] = 0.25
    return P


def biased_bd(n: int, lam: float = 0.85) -> np.ndarray:
    """Biased birth--death chain exactly as in the manuscript footnote.
    Interior (1<=x<=n-2): P(x,x)=1/2, P(x,x+1)=lam/2, P(x,x-1)=(1-lam)/2.
    Boundary, non-lazy: P(0,0)=1-lam, P(0,1)=lam,
                        P(n-1,n-1)=lam, P(n-1,n-2)=1-lam."""
    P = np.zeros((n, n))
    P[0, 0] = 1.0 - lam
    P[0, 1] = lam
    P[n - 1, n - 1] = lam
    P[n - 1, n - 2] = 1.0 - lam
    for x in range(1, n - 1):
        P[x, x] = 0.5
        P[x, x + 1] = lam / 2.0
        P[x, x - 1] = (1.0 - lam) / 2.0
    return P


def two_state_symmetric(a: float) -> np.ndarray:
    return np.array([[1 - a, a], [a, 1 - a]])


def hypercube(d: int) -> np.ndarray:
    """Standard lazy random walk on Q_d = {0,1}^d:
       P(x,x) = 1/2, and for each of the d neighbours y (Hamming distance 1):
       P(x,y) = 1/(2d).
    NOTE: This is NOT the tensor product of the K_2-base [[1/2,1/2],[1/2,1/2]]
    (which has every row equal to pi and hence M=0). The manuscript's
    Remark 7.6 row for Q_d quotes a non-trivial eta/M range (1.00-1.13),
    which corresponds to this standard "stay-or-flip-uniform-coordinate"
    construction. The tensor-product remark and the table row are therefore
    referring to different chains; this is an inconsistency in the
    manuscript.  We use the construction that the *table* implicitly
    assumes."""
    n = 1 << d
    P = np.zeros((n, n))
    for x in range(n):
        P[x, x] = 0.5
        for i in range(d):
            y = x ^ (1 << i)
            P[x, y] = 0.5 / d
    return P


def barbell_lazy(m: int) -> np.ndarray:
    n = 2 * m
    A = np.zeros((n, n))
    for i in range(m):
        for j in range(m):
            if i != j:
                A[i, j] = 1
    for i in range(m, n):
        for j in range(m, n):
            if i != j:
                A[i, j] = 1
    A[m - 1, m] = 1
    A[m, m - 1] = 1
    deg = A.sum(axis=1)
    # Standard lazy RW: with prob 1/2 stay, else move to uniform neighbour.
    P = 0.5 * np.eye(n) + 0.5 * (A / deg[:, None])
    return P


# ---------------------------------------------------------------------------
# Family-specific structured candidates
# ---------------------------------------------------------------------------

def candidates_cycle(n: int) -> List[np.ndarray]:
    cands = []
    # Arcs of various lengths.
    for L in {1, 2, 3, max(2, n // 4), max(2, n // 2)}:
        mu = np.zeros(n)
        mu[:L] = 1.0
        cands.append(mu)
    # Antipodal pair.
    mu = np.zeros(n)
    mu[0] = 0.5
    mu[n // 2] = 0.5
    cands.append(mu)
    return cands


def candidates_path(n: int) -> List[np.ndarray]:
    cands = []
    for L in {1, 2, 3, max(2, n // 4), max(2, n // 2)}:
        mu = np.zeros(n)
        mu[:L] = 1.0
        cands.append(mu.copy())
        mu = np.zeros(n)
        mu[-L:] = 1.0
        cands.append(mu)
    # Two endpoints.
    mu = np.zeros(n)
    mu[0] = 0.5
    mu[-1] = 0.5
    cands.append(mu)
    return cands


def candidates_barbell(m: int) -> List[np.ndarray]:
    n = 2 * m
    cands = []
    mu = np.zeros(n); mu[:m] = 1.0 / m; cands.append(mu)        # left clique uniform
    mu = np.zeros(n); mu[m:] = 1.0 / m; cands.append(mu)        # right clique uniform
    mu = np.zeros(n); mu[0] = 1.0; cands.append(mu)             # far-left vertex
    mu = np.zeros(n); mu[-1] = 1.0; cands.append(mu)            # far-right vertex
    mu = np.zeros(n); mu[m - 1] = 1.0; cands.append(mu)         # bridge endpoint L
    mu = np.zeros(n); mu[m] = 1.0; cands.append(mu)             # bridge endpoint R
    mu = np.zeros(n); mu[:m] = 0.5 / m; mu[m:] = 0.5 / m; cands.append(mu)
    return cands


def candidates_hypercube(d: int) -> List[np.ndarray]:
    n = 1 << d
    cands = []
    # Single vertex.
    e = np.zeros(n); e[0] = 1.0; cands.append(e)
    # Face: fix coord 0 = 0.
    mu = np.zeros(n)
    for i in range(n):
        if (i & 1) == 0:
            mu[i] = 1.0
    cands.append(mu)
    # Antipodal pair.
    mu = np.zeros(n); mu[0] = 0.5; mu[n - 1] = 0.5; cands.append(mu)
    # Parity-balanced (top eigenfunction sign).
    mu = np.zeros(n)
    for i in range(n):
        if bin(i).count("1") % 2 == 0:
            mu[i] = 1.0
    cands.append(mu)
    return cands


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

@dataclass
class Row:
    name: str
    instances: List[Tuple[str, np.ndarray, List[np.ndarray]]]
    rho_expected: Tuple[float, float]      # inclusive interval
    ratio_expected: Tuple[float, float]
    rho_tol: float = RHO_TOL_RANGE
    ratio_tol: float = INT_TOL
    n_dirichlet: int = 120


def build_rows() -> List[Row]:
    rows: List[Row] = []

    # K_n
    inst = []
    for n in (4, 8, 12, 16, 20):
        P = complete_graph(n)
        inst.append((f"K_{n}", P, []))
    rows.append(Row("Complete K_n", inst, (1.0, 1.0), (1.0, 1.0),
                    rho_tol=RHO_TOL_EXACT, ratio_tol=INT_TOL))

    # Star
    inst = []
    for n in (5, 10, 20, 35, 50):
        P = lazy_star(n)
        inst.append((f"S_{n}", P, []))
    rows.append(Row("Lazy star S_n", inst, (0.5, 0.5), (1.0, 1.0),
                    rho_tol=RHO_TOL_EXACT, ratio_tol=INT_TOL))

    # Biased BD lam=0.85
    inst = []
    for n in (10, 20, 30, 40):
        P = biased_bd(n, 0.85)
        # candidates: every point mass (already done) plus arc clusters
        cands = candidates_path(n)
        inst.append((f"BD_{n}_l0.85", P, cands))
    rows.append(Row("Biased BD lambda=0.85", inst, (0.24, 0.27), (1.01, 1.04)))

    # 2-state symmetric
    inst = []
    for a in (0.1, 0.2, 0.35):
        P = two_state_symmetric(a)
        inst.append((f"2state_a{a}", P, []))
    rows.append(Row("Two-state symmetric", inst, (1.0, 1.0), (1.21, 1.37),
                    rho_tol=RHO_TOL_EXACT, ratio_tol=INT_TOL))

    # Hypercube
    inst = []
    for d in (2, 3, 4, 5, 6, 7):
        P = hypercube(d)
        cands = candidates_hypercube(d)
        inst.append((f"Q_{d}", P, cands))
    rows.append(Row("Hypercube Q_d", inst, (1.0, 1.0), (1.00, 1.13),
                    rho_tol=RHO_TOL_EXACT, ratio_tol=INT_TOL))

    # Lazy cycle
    inst = []
    for n in (10, 20, 30, 40, 50):
        P = lazy_cycle(n)
        cands = candidates_cycle(n)
        inst.append((f"C_{n}", P, cands))
    rows.append(Row("Lazy cycle C_n", inst, (1.0, 1.0), (1.35, 1.49),
                    rho_tol=RHO_TOL_EXACT, ratio_tol=INT_TOL))

    # Lazy path
    inst = []
    for n in (10, 20, 30, 40, 50):
        P = lazy_path(n)
        cands = candidates_path(n)
        inst.append((f"P_{n}", P, cands))
    rows.append(Row("Lazy path P_n", inst, (0.87, 0.95), (1.28, 1.47)))

    # Barbell
    inst = []
    for m in (5, 6, 7, 8, 9):
        P = barbell_lazy(m)
        cands = candidates_barbell(m)
        inst.append((f"B_{m}", P, cands))
    rows.append(Row("Barbell B_m", inst, (0.96, 0.99), (2.33, 2.44)))

    return rows


def in_interval(v: float, lo: float, hi: float, tol: float) -> bool:
    return (lo - tol) <= v <= (hi + tol)


def main():
    rows = build_rows()
    t0 = time.time()

    overall_pass = True
    summary = []

    for row in rows:
        print("=" * 78)
        print(f"ROW: {row.name}")
        print(f"  expected rho in [{row.rho_expected[0]:.2f}, {row.rho_expected[1]:.2f}], "
              f"ratio in [{row.ratio_expected[0]:.2f}, {row.ratio_expected[1]:.2f}]")
        print("-" * 78)

        row_rho_min, row_rho_max = +np.inf, -np.inf
        row_ratio_min, row_ratio_max = +np.inf, -np.inf
        per_inst = []
        all_pass = True

        for label, P, cands in row.instances:
            pi = stationary(P)
            rho, M, _r = rho_M(P, pi)
            eta, witness = eta_kl(P, pi, extra_candidates=cands,
                                  n_dirichlet=row.n_dirichlet)
            r_over_M = eta / M if M > 1e-15 else float("nan")

            rho_ok = in_interval(rho, row.rho_expected[0], row.rho_expected[1], row.rho_tol)
            ratio_ok = (np.isnan(r_over_M)
                        or in_interval(r_over_M, row.ratio_expected[0],
                                       row.ratio_expected[1], row.ratio_tol))
            inst_pass = rho_ok and ratio_ok
            all_pass = all_pass and inst_pass

            row_rho_min = min(row_rho_min, rho)
            row_rho_max = max(row_rho_max, rho)
            row_ratio_min = min(row_ratio_min, r_over_M if not np.isnan(r_over_M) else +np.inf)
            row_ratio_max = max(row_ratio_max, r_over_M if not np.isnan(r_over_M) else -np.inf)

            flag = "OK " if inst_pass else "BAD"
            print(f"  [{flag}] {label:14s}  rho={rho:.4f}  M={M:.4e}  "
                  f"eta>={eta:.4f}  eta/M={r_over_M:.4f}  witness={witness}")
            per_inst.append({
                "label": label, "rho": rho, "M": M, "eta": eta,
                "eta_over_M": r_over_M, "witness": witness,
                "rho_ok": rho_ok, "ratio_ok": ratio_ok,
            })

        verdict = "PASS" if all_pass else "FAIL"
        overall_pass = overall_pass and all_pass
        print(f"  ROW VERDICT: {verdict}   observed rho in "
              f"[{row_rho_min:.3f}, {row_rho_max:.3f}], "
              f"eta/M in [{row_ratio_min:.3f}, {row_ratio_max:.3f}]")
        summary.append({
            "row": row.name,
            "expected_rho": row.rho_expected,
            "observed_rho": [row_rho_min, row_rho_max],
            "expected_ratio": row.ratio_expected,
            "observed_ratio": [row_ratio_min, row_ratio_max],
            "verdict": verdict,
            "instances": per_inst,
        })

    print("=" * 78)
    elapsed = time.time() - t0
    print(f"TOTAL TIME: {elapsed:.1f}s   OVERALL: "
          f"{'PASS' if overall_pass else 'FAIL'}")

    with open("verify_table_remark_7_6_results.json", "w") as f:
        json.dump({"seed": SEED, "elapsed_sec": elapsed,
                   "overall_pass": overall_pass, "rows": summary}, f, indent=2)
    print("Detailed results -> verify_table_remark_7_6_results.json")

    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()
