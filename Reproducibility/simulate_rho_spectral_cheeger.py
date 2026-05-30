"""
Numerical exploration: rho(P) vs spectral gap, Cheeger constant, and mixing time.

Generates a library of Markov chains (symmetric/asymmetric, fast/slow mixing),
computes rho(P), spectral gap, Cheeger constant, mixing time estimate,
and plots the relationships.
"""

import numpy as np
from itertools import combinations
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

# ═══════════════════════════════════════════════════════════════════════════════
# Core computations
# ═══════════════════════════════════════════════════════════════════════════════

def compute_stationary(P, tol=1e-12):
    """Stationary distribution via left eigenvector."""
    eigvals, eigvecs = np.linalg.eig(P.T)
    idx = np.argmin(np.abs(eigvals - 1.0))
    pi = np.real(eigvecs[:, idx])
    pi = np.abs(pi)
    pi /= pi.sum()
    return pi


def kl_divergence(p, q):
    """D_KL(p || q) in nats."""
    mask = p > 1e-300
    return np.sum(p[mask] * np.log(p[mask] / q[mask]))


def compute_retention_profile(P, pi):
    """Compute r(x), rbar_pi, M, rho(P)."""
    n = P.shape[0]
    r = np.zeros(n)
    for x in range(n):
        kl_row = kl_divergence(P[x], pi)
        denom = np.log(1.0 / pi[x])
        r[x] = kl_row / denom if denom > 1e-15 else 0.0
    M = np.max(r)
    rbar = np.dot(pi, r)
    rho = rbar / M if M > 1e-15 else 1.0
    return r, rbar, M, rho


def spectral_gap(P):
    """Absolute spectral gap gamma = 1 - max_{i>=2} |lambda_i|."""
    eigvals = np.sort(np.abs(np.linalg.eigvals(P)))[::-1]
    lambda_star = eigvals[1] if len(eigvals) > 1 else 0.0
    return 1.0 - lambda_star, eigvals


def cheeger_constant(P, pi):
    """
    Exact Cheeger constant for small chains (brute force over all subsets).
    For large chains, uses a heuristic sweep.
    """
    n = P.shape[0]
    if n > 18:
        return _cheeger_sweep(P, pi)

    best_h = np.inf
    best_A = None
    for size in range(1, n // 2 + 1):
        for A in combinations(range(n), size):
            A_set = set(A)
            pi_A = sum(pi[x] for x in A_set)
            if pi_A > 0.5 + 1e-12:
                continue
            flow = sum(pi[x] * P[x, y] for x in A_set for y in range(n) if y not in A_set)
            h = flow / pi_A
            if h < best_h:
                best_h = h
                best_A = A_set
    return best_h, best_A


def _cheeger_sweep(P, pi):
    """Heuristic Cheeger via Fiedler vector sweep."""
    n = P.shape[0]
    # Build reversibilized chain and Laplacian
    L = np.diag(np.ones(n)) - P
    eigvals, eigvecs = np.linalg.eigh(np.diag(np.sqrt(pi)) @ L @ np.diag(1.0 / np.sqrt(pi)))
    idx = np.argsort(eigvals)
    fiedler = eigvecs[:, idx[1]]
    order = np.argsort(fiedler)

    best_h = np.inf
    best_A = None
    for k in range(1, n):
        A_set = set(order[:k])
        pi_A = sum(pi[x] for x in A_set)
        if pi_A > 0.5 + 1e-12:
            continue
        if pi_A < 1e-15:
            continue
        flow = sum(pi[x] * P[x, y] for x in A_set for y in range(n) if y not in A_set)
        h = flow / pi_A
        if h < best_h:
            best_h = h
            best_A = A_set
    return best_h, best_A


def mixing_time_estimate(P, pi, eps=0.25, max_t=100000):
    """Estimate TV mixing time by simulation."""
    n = P.shape[0]
    # Start from worst state (minimum pi)
    x0 = np.argmin(pi)
    mu = np.zeros(n)
    mu[x0] = 1.0
    for t in range(1, max_t + 1):
        mu = mu @ P
        tv = 0.5 * np.sum(np.abs(mu - pi))
        if tv <= eps:
            return t
    return max_t


def mutual_information(P, pi):
    """I(X_0; X_1) for the stationary one-step pair."""
    return sum(pi[x] * kl_divergence(P[x], pi) for x in range(len(pi)))


def sum_lambda_sq(P):
    """Sum of lambda_i^2 for i >= 2."""
    eigvals = np.linalg.eigvals(P)
    eigvals_sorted = sorted(np.abs(eigvals), reverse=True)
    return sum(v**2 for v in eigvals_sorted[1:])


# ═══════════════════════════════════════════════════════════════════════════════
# Chain generators
# ═══════════════════════════════════════════════════════════════════════════════

def complete_graph(n):
    """Simple RW on K_n (non-lazy). Uniform pi, vertex-transitive."""
    P = np.full((n, n), 1.0 / (n - 1))
    np.fill_diagonal(P, 0.0)
    pi = np.full(n, 1.0 / n)
    return P, pi, f"K_{n}"


def lazy_complete_graph(n):
    """Lazy RW on K_n. Uniform pi, vertex-transitive."""
    P = np.full((n, n), 1.0 / (2 * (n - 1)))
    np.fill_diagonal(P, 0.5)
    pi = np.full(n, 1.0 / n)
    return P, pi, f"Lazy K_{n}"


def cycle_graph(n, lazy=True):
    """RW on cycle. Uniform pi, vertex-transitive."""
    P = np.zeros((n, n))
    for i in range(n):
        if lazy:
            P[i, i] = 0.5
            P[i, (i + 1) % n] = 0.25
            P[i, (i - 1) % n] = 0.25
        else:
            P[i, (i + 1) % n] = 0.5
            P[i, (i - 1) % n] = 0.5
    pi = np.full(n, 1.0 / n)
    tag = "Lazy " if lazy else ""
    return P, pi, f"{tag}C_{n}"


def path_graph(n, lazy=True):
    """RW on path 0--1--...--n-1. Non-uniform pi."""
    P = np.zeros((n, n))
    for i in range(n):
        neighbors = []
        if i > 0:
            neighbors.append(i - 1)
        if i < n - 1:
            neighbors.append(i + 1)
        if lazy:
            P[i, i] = 0.5
            for j in neighbors:
                P[i, j] = 0.5 / len(neighbors)
        else:
            for j in neighbors:
                P[i, j] = 1.0 / len(neighbors)
    pi = compute_stationary(P)
    tag = "Lazy " if lazy else ""
    return P, pi, f"{tag}Path_{n}"


def hypercube(d):
    """RW on {0,1}^d. Uniform pi, vertex-transitive."""
    n = 2**d
    P = np.zeros((n, n))
    for x in range(n):
        P[x, x] = 0.5  # lazy
        for bit in range(d):
            y = x ^ (1 << bit)
            P[x, y] = 0.5 / d
    pi = np.full(n, 1.0 / n)
    return P, pi, f"Hypercube_d{d}"


def barbell_graph(m):
    """Two K_m cliques connected by a single edge. n=2m."""
    n = 2 * m
    # Adjacency: clique 0..m-1, clique m..2m-1, edge m-1 <-> m
    adj = np.zeros((n, n))
    for i in range(m):
        for j in range(i + 1, m):
            adj[i, j] = adj[j, i] = 1
    for i in range(m, n):
        for j in range(i + 1, n):
            adj[i, j] = adj[j, i] = 1
    adj[m - 1, m] = adj[m, m - 1] = 1
    deg = adj.sum(axis=1)
    # Lazy simple RW
    P = np.zeros((n, n))
    for x in range(n):
        P[x, x] = 0.5
        for y in range(n):
            if adj[x, y]:
                P[x, y] = 0.5 / deg[x]
    pi = compute_stationary(P)
    return P, pi, f"Barbell_{m}+{m}"


def lollipop_graph(m, k):
    """K_m clique attached to path of length k. n=m+k."""
    n = m + k
    adj = np.zeros((n, n))
    # Clique on 0..m-1
    for i in range(m):
        for j in range(i + 1, m):
            adj[i, j] = adj[j, i] = 1
    # Path m-1 -- m -- m+1 -- ... -- m+k-1
    for i in range(m - 1, m + k - 1):
        adj[i, i + 1] = adj[i + 1, i] = 1
    deg = adj.sum(axis=1)
    P = np.zeros((n, n))
    for x in range(n):
        if deg[x] > 0:
            P[x, x] = 0.5
            for y in range(n):
                if adj[x, y]:
                    P[x, y] = 0.5 / deg[x]
        else:
            P[x, x] = 1.0
    pi = compute_stationary(P)
    return P, pi, f"Lollipop_{m}+{k}"


def star_graph(n):
    """Star with center 0 and n-1 leaves. Lazy RW."""
    adj = np.zeros((n, n))
    for i in range(1, n):
        adj[0, i] = adj[i, 0] = 1
    deg = adj.sum(axis=1)
    P = np.zeros((n, n))
    for x in range(n):
        P[x, x] = 0.5
        for y in range(n):
            if adj[x, y]:
                P[x, y] = 0.5 / deg[x]
    pi = compute_stationary(P)
    return P, pi, f"Star_{n}"


def two_cluster(m, p_intra=0.5, p_inter=0.01):
    """Two clusters of size m with different internal/cross edge probabilities."""
    n = 2 * m
    adj = np.zeros((n, n))
    rng = np.random.RandomState(123)
    # Intra-cluster edges
    for i in range(m):
        for j in range(i + 1, m):
            if rng.random() < p_intra:
                adj[i, j] = adj[j, i] = 1
    for i in range(m, n):
        for j in range(i + 1, n):
            if rng.random() < p_intra:
                adj[i, j] = adj[j, i] = 1
    # Inter-cluster edges
    for i in range(m):
        for j in range(m, n):
            if rng.random() < p_inter:
                adj[i, j] = adj[j, i] = 1
    # Ensure connected: add at least one cross edge
    if adj[:m, m:].sum() == 0:
        adj[0, m] = adj[m, 0] = 1
    deg = adj.sum(axis=1)
    P = np.zeros((n, n))
    for x in range(n):
        if deg[x] > 0:
            P[x, x] = 0.5
            for y in range(n):
                if adj[x, y]:
                    P[x, y] = 0.5 / deg[x]
        else:
            P[x, x] = 1.0
    pi = compute_stationary(P)
    return P, pi, f"TwoCluster_{m}({p_inter:.2f})"


def doubly_stochastic_random(n, seed=42):
    """Random doubly stochastic (uniform pi) via Sinkhorn."""
    rng = np.random.RandomState(seed)
    A = rng.exponential(1.0, (n, n)) + 0.01
    for _ in range(300):
        A = A / A.sum(axis=1, keepdims=True)
        A = A / A.sum(axis=0, keepdims=True)
    P = A / A.sum(axis=1, keepdims=True)
    pi = np.full(n, 1.0 / n)
    return P, pi, f"DoublyStoch_{n}"


def birth_death_biased(n, lam=0.7):
    """Birth-death chain with bias. Non-uniform pi."""
    P = np.zeros((n, n))
    for i in range(n):
        if i == 0:
            P[i, 0] = 1.0 - lam
            P[i, 1] = lam
        elif i == n - 1:
            P[i, n - 2] = 1.0 - lam
            P[i, n - 1] = lam
        else:
            P[i, i + 1] = lam * 0.5
            P[i, i - 1] = (1.0 - lam) * 0.5
            P[i, i] = 0.5
    pi = compute_stationary(P)
    return P, pi, f"BD_biased_{n}(λ={lam})"


def ehrenfest(n):
    """Ehrenfest urn model on {0,...,n}. n+1 states."""
    N = n + 1
    P = np.zeros((N, N))
    for k in range(N):
        if k < n:
            P[k, k + 1] = (n - k) / n
        if k > 0:
            P[k, k - 1] = k / n
    # Make lazy for aperiodicity
    P = 0.5 * np.eye(N) + 0.5 * P
    pi = compute_stationary(P)
    return P, pi, f"Ehrenfest_{n}"


# ═══════════════════════════════════════════════════════════════════════════════
# Main experiment
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_chain(P, pi, name, compute_cheeger=True):
    """Compute all quantities for one chain."""
    n = P.shape[0]

    # Retention profile
    r, rbar, M, rho = compute_retention_profile(P, pi)

    # Spectral gap
    gap, eigvals = spectral_gap(P)

    # Cheeger constant
    if compute_cheeger and n <= 18:
        h, A_star = cheeger_constant(P, pi)
    elif compute_cheeger:
        h, A_star = _cheeger_sweep(P, pi)
    else:
        h, A_star = np.nan, None

    # Mixing time
    t_mix = mixing_time_estimate(P, pi, eps=0.25)

    # Mutual information
    mi = mutual_information(P, pi)

    # Sum of lambda_i^2
    slsq = sum_lambda_sq(P)

    # Spectral upper bound on rbar
    h_pi = -np.sum(pi * np.log(pi + 1e-300))
    rbar_spectral_ub = slsq / max(np.log(1.0 / np.max(pi)), 1e-15)

    return {
        'name': name, 'n': n,
        'r': r, 'rbar': rbar, 'M': M, 'rho': rho,
        'gap': gap, 'eigvals': eigvals,
        'cheeger': h, 'cheeger_set': A_star,
        't_mix': t_mix,
        'mi': mi, 'sum_lam_sq': slsq,
        'rbar_spectral_ub': rbar_spectral_ub,
        'H_pi': h_pi,
        'is_uniform': np.allclose(pi, 1.0 / n, atol=1e-6),
    }


def main():
    print("=" * 80)
    print("Exploring rho(P) vs Spectral Gap, Cheeger Constant, Mixing Time")
    print("=" * 80)

    # ─── Build chain library ──────────────────────────────────────────────
    chains = []

    # Vertex-transitive (rho = 1)
    for n in [5, 8, 12, 16]:
        chains.append(complete_graph(n))
    for n in [6, 10, 16, 24]:
        chains.append(cycle_graph(n, lazy=True))
    for d in [3, 4, 5]:
        chains.append(hypercube(d))
    chains.append(lazy_complete_graph(8))
    chains.append(lazy_complete_graph(16))

    # Bottleneck chains (rho potentially < 1)
    for m in [4, 6, 8]:
        chains.append(barbell_graph(m))
    for m, k in [(5, 5), (6, 4), (8, 4)]:
        chains.append(lollipop_graph(m, k))
    for n in [6, 10, 16]:
        chains.append(star_graph(n))

    # Path graphs
    for n in [5, 8, 12, 16]:
        chains.append(path_graph(n, lazy=True))

    # Two-cluster with varying bottleneck
    for p_inter in [0.005, 0.02, 0.05, 0.15, 0.4]:
        chains.append(two_cluster(6, p_intra=0.5, p_inter=p_inter))

    # Random doubly stochastic
    for n in [5, 8, 12]:
        chains.append(doubly_stochastic_random(n, seed=n * 7))

    # Birth-death biased
    for n in [5, 8, 12]:
        for lam in [0.55, 0.7, 0.85]:
            chains.append(birth_death_biased(n, lam))

    # Ehrenfest urn
    for n in [6, 10, 14]:
        chains.append(ehrenfest(n))

    # ─── Analyze all chains ───────────────────────────────────────────────
    results = []
    for P, pi, name in chains:
        try:
            res = analyze_chain(P, pi, name)
            results.append(res)
            print(f"  {name:30s}  n={res['n']:3d}  ρ={res['rho']:.4f}  "
                  f"γ={res['gap']:.4f}  h={res['cheeger']:.4f}  "
                  f"t_mix={res['t_mix']:5d}  M={res['M']:.4f}  "
                  f"r̄={res['rbar']:.4f}")
        except Exception as e:
            print(f"  [ERROR] {name}: {e}")

    # ─── Classification ───────────────────────────────────────────────────
    categories = []
    for res in results:
        if res['is_uniform'] and abs(res['rho'] - 1.0) < 0.05:
            categories.append('vertex-trans')
        elif abs(res['rho'] - 1.0) < 0.05:
            categories.append('uniform-rho')
        elif res['rho'] < 0.5:
            categories.append('bottleneck')
        else:
            categories.append('moderate')

    cat_colors = {
        'vertex-trans': 'royalblue',
        'uniform-rho': 'green',
        'bottleneck': 'red',
        'moderate': 'orange'
    }
    cat_labels = {
        'vertex-trans': r'Vertex-transitive ($\rho\approx1$, unif $\pi$)',
        'uniform-rho': r'Non-VT, $\rho\approx1$',
        'bottleneck': r'Bottleneck ($\rho<0.5$)',
        'moderate': r'Moderate ($0.5\leq\rho<0.95$)'
    }
    colors = [cat_colors[c] for c in categories]

    # ─── Plotting ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # Extract arrays
    rhos = np.array([r['rho'] for r in results])
    gaps = np.array([r['gap'] for r in results])
    cheegers = np.array([r['cheeger'] for r in results])
    tmixs = np.array([r['t_mix'] for r in results])
    Ms = np.array([r['M'] for r in results])
    rbars = np.array([r['rbar'] for r in results])

    # --- Plot 1: rho vs spectral gap ---
    ax = axes[0, 0]
    ax.scatter(gaps, rhos, c=colors, s=60, edgecolors='black', linewidths=0.5, zorder=3)
    ax.set_xlabel(r'Spectral gap $\gamma$', fontsize=12)
    ax.set_ylabel(r'$\rho(P)$', fontsize=12)
    ax.set_title(r'$\rho(P)$ vs Spectral Gap', fontsize=13)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)

    # --- Plot 2: rho vs Cheeger ---
    ax = axes[0, 1]
    valid_h = ~np.isnan(cheegers)
    ax.scatter(cheegers[valid_h], rhos[valid_h], c=[colors[i] for i in range(len(colors)) if valid_h[i]],
               s=60, edgecolors='black', linewidths=0.5, zorder=3)
    ax.set_xlabel(r'Cheeger constant $h_P$', fontsize=12)
    ax.set_ylabel(r'$\rho(P)$', fontsize=12)
    ax.set_title(r'$\rho(P)$ vs Cheeger Constant', fontsize=13)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)

    # --- Plot 3: rho vs mixing time ---
    ax = axes[0, 2]
    ax.scatter(tmixs, rhos, c=colors, s=60, edgecolors='black', linewidths=0.5, zorder=3)
    ax.set_xlabel(r'$t_{\mathrm{mix}}(1/4)$', fontsize=12)
    ax.set_ylabel(r'$\rho(P)$', fontsize=12)
    ax.set_title(r'$\rho(P)$ vs Mixing Time', fontsize=13)
    ax.set_xscale('log')
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)

    # --- Plot 4: M vs spectral gap ---
    ax = axes[1, 0]
    ax.scatter(gaps, Ms, c=colors, s=60, edgecolors='black', linewidths=0.5, zorder=3)
    ax.set_xlabel(r'Spectral gap $\gamma$', fontsize=12)
    ax.set_ylabel(r'$M = \max_x r(x)$', fontsize=12)
    ax.set_title(r'Worst-case retention $M$ vs Spectral Gap', fontsize=13)
    ax.set_xlim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    # --- Plot 5: rbar vs sum(lambda_i^2) ---
    ax = axes[1, 1]
    slsqs = np.array([r['sum_lam_sq'] for r in results])
    h_pis = np.array([r['H_pi'] for r in results])
    rbar_ubs = slsqs / np.maximum(np.log(1.0 / np.array([np.max(compute_stationary(chains[i][0])) if i < len(chains) else 1.0/results[i]['n'] for i in range(len(results))])), 1e-15)

    ax.scatter(slsqs, rbars, c=colors, s=60, edgecolors='black', linewidths=0.5,
               zorder=3, label=r'$\bar{r}_\pi$ (actual)')
    # Upper bound line
    max_val = max(np.max(slsqs), np.max(rbars)) * 1.1
    bound_x = np.linspace(0, np.max(slsqs) * 1.2, 100)
    # For uniform pi chains
    ln_ns = np.array([np.log(r['n']) for r in results])
    ax.set_xlabel(r'$\sum_{i\geq 2}\lambda_i^2$', fontsize=12)
    ax.set_ylabel(r'$\bar{r}_\pi$', fontsize=12)
    ax.set_title(r'$\bar{r}_\pi$ vs Spectral $\sum\lambda_i^2$', fontsize=13)
    ax.grid(True, alpha=0.3)

    # --- Plot 6: M vs Cheeger (lower bound test) ---
    ax = axes[1, 2]
    ax.scatter(cheegers[valid_h], Ms[valid_h], c=[colors[i] for i in range(len(colors)) if valid_h[i]],
               s=60, edgecolors='black', linewidths=0.5, zorder=3)
    ax.set_xlabel(r'Cheeger constant $h_P$', fontsize=12)
    ax.set_ylabel(r'$M = \max_x r(x)$', fontsize=12)
    ax.set_title(r'$M$ vs Cheeger Constant', fontsize=13)
    ax.grid(True, alpha=0.3)

    # Global legend
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=v,
                              markersize=10, markeredgecolor='black', label=cat_labels[k])
                       for k, v in cat_colors.items()]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=11,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig("rho_vs_spectral_cheeger_mixing.png", dpi=150, bbox_inches='tight')
    print(f"\nMain figure saved: rho_vs_spectral_cheeger_mixing.png")
    plt.close(fig)

    # ═══════════════════════════════════════════════════════════════════════
    # Figure 2: Detailed retention profiles for selected chains
    # ═══════════════════════════════════════════════════════════════════════
    fig2, axes2 = plt.subplots(2, 3, figsize=(17, 10))
    selected = ['K_8', 'Lazy C_16', 'Hypercube_d4',
                'Barbell_6+6', 'Lollipop_6+4', 'Star_10']
    for idx_ax, name in enumerate(selected):
        ax = axes2[idx_ax // 3, idx_ax % 3]
        res = next((r for r in results if r['name'] == name), None)
        if res is None:
            ax.set_title(f'{name} (not found)')
            continue
        states = np.arange(res['n'])
        ax.bar(states, res['r'], color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.axhline(y=res['rbar'], color='red', linestyle='--', linewidth=1.5,
                    label=rf'$\bar{{r}}_\pi={res["rbar"]:.3f}$')
        ax.axhline(y=res['M'], color='darkred', linestyle='-', linewidth=1.5,
                    label=rf'$M={res["M"]:.3f}$')
        ax.set_xlabel('State $x$', fontsize=10)
        ax.set_ylabel('$r(x)$', fontsize=10)
        ax.set_title(rf'{name}: $\rho={res["rho"]:.3f}$, $\gamma={res["gap"]:.3f}$', fontsize=11)
        ax.legend(fontsize=8, loc='upper right')
        ax.set_ylim(0, min(max(res['M'] * 1.3, 0.1), 1.05))
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig("retention_profiles_selected.png", dpi=150, bbox_inches='tight')
    print(f"Profiles figure saved: retention_profiles_selected.png")
    plt.close(fig2)

    # ═══════════════════════════════════════════════════════════════════════
    # Figure 3: Spectral bound validation (uniform pi chains only)
    # ═══════════════════════════════════════════════════════════════════════
    fig3, axes3 = plt.subplots(1, 2, figsize=(13, 5.5))

    uniform_results = [r for r in results if r['is_uniform']]
    if uniform_results:
        u_rbars = np.array([r['rbar'] for r in uniform_results])
        u_slsq = np.array([r['sum_lam_sq'] for r in uniform_results])
        u_lnn = np.array([np.log(r['n']) for r in uniform_results])
        u_ub = u_slsq / u_lnn
        u_lb = u_slsq / (2 * u_lnn)
        u_names = [r['name'] for r in uniform_results]

        # Bound validation
        ax = axes3[0]
        x_pos = np.arange(len(uniform_results))
        width = 0.25
        ax.barh(x_pos - width, u_lb, width, color='green', alpha=0.6, label=r'Lower: $\frac{\sum\lambda_i^2}{2\ln n}$')
        ax.barh(x_pos, u_rbars, width, color='steelblue', alpha=0.8, label=r'$\bar{r}_\pi$ (exact)')
        ax.barh(x_pos + width, u_ub, width, color='red', alpha=0.6, label=r'Upper: $\frac{\sum\lambda_i^2}{\ln n}$')
        ax.set_yticks(x_pos)
        ax.set_yticklabels(u_names, fontsize=8)
        ax.set_xlabel(r'Value', fontsize=11)
        ax.set_title(r'Spectral bounds on $\bar{r}_\pi$ (uniform $\pi$)', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, axis='x', alpha=0.3)

        # Scatter: rbar vs spectral bound
        ax = axes3[1]
        ax.scatter(u_ub, u_rbars, s=80, c='steelblue', edgecolors='black', zorder=3)
        for i, nm in enumerate(u_names):
            ax.annotate(nm, (u_ub[i], u_rbars[i]), fontsize=6,
                        textcoords="offset points", xytext=(4, 4))
        max_v = max(np.max(u_ub), np.max(u_rbars)) * 1.1
        ax.plot([0, max_v], [0, max_v], 'k--', alpha=0.5, label='$y=x$ (bound saturated)')
        ax.plot([0, max_v], [0, max_v / 2], 'g--', alpha=0.5, label=r'$y=x/2$ (lower bound)')
        ax.set_xlabel(r'Spectral UB: $\sum\lambda_i^2/\ln n$', fontsize=11)
        ax.set_ylabel(r'$\bar{r}_\pi$', fontsize=11)
        ax.set_title('Spectral Bound Tightness', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("spectral_bound_validation.png", dpi=150, bbox_inches='tight')
    print(f"Spectral bounds figure saved: spectral_bound_validation.png")
    plt.close(fig3)

    # ═══════════════════════════════════════════════════════════════════════
    # Summary table
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n\n{'=' * 120}")
    print(f"{'Chain':<30} {'n':>3} {'ρ(P)':>7} {'M':>7} {'r̄_π':>7} "
          f"{'γ':>7} {'h_P':>7} {'t_mix':>6} {'Σλ²':>8} {'I(X;Y)':>8} {'Type':>14}")
    print(f"{'=' * 120}")
    for i, res in enumerate(results):
        print(f"{res['name']:<30} {res['n']:>3} {res['rho']:>7.4f} {res['M']:>7.4f} "
              f"{res['rbar']:>7.4f} {res['gap']:>7.4f} {res['cheeger']:>7.4f} "
              f"{res['t_mix']:>6d} {res['sum_lam_sq']:>8.4f} {res['mi']:>8.4f} "
              f"{categories[i]:>14}")
    print(f"{'=' * 120}")

    # ─── Key findings summary ─────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    vt = [r for r, c in zip(results, categories) if c == 'vertex-trans']
    bn = [r for r, c in zip(results, categories) if c == 'bottleneck']

    if vt:
        vt_gaps = [r['gap'] for r in vt]
        print(f"\n1. Vertex-transitive chains: ρ(P) ≈ 1 always.")
        print(f"   Spectral gaps range from {min(vt_gaps):.4f} to {max(vt_gaps):.4f}")
        print(f"   => ρ(P) is independent of spectral gap for symmetric chains.")

    if bn:
        print(f"\n2. Bottleneck chains: ρ(P) < 0.5")
        for r in bn:
            print(f"   {r['name']:25s} ρ={r['rho']:.4f}  γ={r['gap']:.4f}  h={r['cheeger']:.4f}")

    # Check spectral bound
    if uniform_results:
        violations = sum(1 for r in uniform_results
                         if r['rbar'] > r['sum_lam_sq'] / np.log(r['n']) + 1e-6)
        print(f"\n3. Spectral upper bound r̄_π ≤ Σλ²/ln(n): "
              f"{'VALIDATED' if violations == 0 else f'{violations} VIOLATIONS'} "
              f"across {len(uniform_results)} uniform-π chains.")

    # Correlation analysis
    print(f"\n4. Correlations (Spearman rank):")
    from scipy.stats import spearmanr
    rho_gap_corr, rho_gap_p = spearmanr(rhos, gaps)
    rho_h_corr, rho_h_p = spearmanr(rhos[valid_h], cheegers[valid_h])
    rho_tmix_corr, rho_tmix_p = spearmanr(rhos, tmixs)
    M_gap_corr, M_gap_p = spearmanr(Ms, gaps)
    print(f"   ρ(P) vs γ:      r_s = {rho_gap_corr:+.3f}  (p = {rho_gap_p:.4f})")
    print(f"   ρ(P) vs h_P:    r_s = {rho_h_corr:+.3f}  (p = {rho_h_p:.4f})")
    print(f"   ρ(P) vs t_mix:  r_s = {rho_tmix_corr:+.3f}  (p = {rho_tmix_p:.4f})")
    print(f"   M vs γ:         r_s = {M_gap_corr:+.3f}  (p = {M_gap_p:.4f})")


if __name__ == "__main__":
    main()
