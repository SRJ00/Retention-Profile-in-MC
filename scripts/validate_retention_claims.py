"""Independent, standard-library checks for claims in main-revised-adv.tex."""

from __future__ import annotations

import math


TOL = 1e-12


def assert_close(left: float, right: float, tol: float = TOL) -> None:
    if not math.isclose(left, right, rel_tol=tol, abs_tol=tol):
        raise AssertionError(f"{left!r} is not close to {right!r}")


def kl(p: list[float], q: list[float]) -> float:
    return sum(px * math.log(px / qx) for px, qx in zip(p, q) if px > 0.0)


def binary_kl(q: float, p: float) -> float:
    return q * math.log(q / p) + (1.0 - q) * math.log((1.0 - q) / (1.0 - p))


def matvec(matrix: list[list[float]], vector: list[float]) -> list[float]:
    return [sum(row[j] * vector[j] for j in range(len(vector))) for row in matrix]


def counterexample_matrix(
    epsilon: float, k: int, n_b: int, delta: float
) -> tuple[list[float], list[list[float]]]:
    n = k + n_b
    pi = [epsilon / k] * k + [(1.0 - epsilon) / n_b] * n_b
    p = [[0.0] * n for _ in range(n)]
    for x in range(k):
        p[x][:k] = [(1.0 - delta) / k] * k
        p[x][k:] = [delta / n_b] * n_b
    for x in range(k, n):
        p[x][:k] = [epsilon * delta / (k * (1.0 - epsilon))] * k
        p[x][k:] = [
            (1.0 - epsilon - epsilon * delta) / (n_b * (1.0 - epsilon))
        ] * n_b
    return pi, p


def block_formulas(
    epsilon: float, k: int, n_b: int, delta: float
) -> tuple[float, float, float, float, float]:
    d_a = binary_kl(1.0 - delta, epsilon)
    alpha = epsilon * delta / (1.0 - epsilon)
    d_b = binary_kl(alpha, epsilon)
    r_a = d_a / math.log(k / epsilon)
    r_b = d_b / math.log(n_b / (1.0 - epsilon))
    r_set = d_a / math.log(1.0 / epsilon)
    return d_a, d_b, r_a, r_b, r_set


def check_counterexample_matrix() -> None:
    epsilon, k, n_b, delta = 0.08, 11, 23, 0.005
    pi, p = counterexample_matrix(epsilon, k, n_b, delta)
    n = len(pi)

    if not all(entry > 0.0 for row in p for entry in row):
        raise AssertionError("counterexample matrix is not strictly positive")
    for row in p:
        assert_close(sum(row), 1.0)
    for y in range(n):
        assert_close(sum(pi[x] * p[x][y] for x in range(n)), pi[y])
    for x in range(n):
        for y in range(n):
            assert_close(pi[x] * p[x][y], pi[y] * p[y][x])

    d_a, d_b, r_a_formula, r_b_formula, r_set_formula = block_formulas(
        epsilon, k, n_b, delta
    )
    assert_close(kl(p[0], pi), d_a)
    assert_close(kl(p[k], pi), d_b)
    assert_close(kl(p[0], pi) / math.log(k / epsilon), r_a_formula)
    assert_close(kl(p[k], pi) / math.log(n_b / (1.0 - epsilon)), r_b_formula)

    pi_a = [1.0 / k] * k + [0.0] * n_b
    output = [sum(pi_a[x] * p[x][y] for x in range(n)) for y in range(n)]
    r_set_direct = kl(output, pi) / math.log(1.0 / epsilon)
    assert_close(r_set_direct, r_set_formula)

    m_1 = max(r_a_formula, r_b_formula)
    l_1 = (epsilon * r_a_formula + (1.0 - epsilon) * r_b_formula) / m_1
    print(
        "finite counterexample: "
        f"M1={m_1:.9f}, L1={l_1:.9f}, R(A)={r_set_direct:.9f}"
    )


def check_diagonal_selection() -> None:
    """Numerically realize the N_m-then-delta_m choices used in the proof."""
    worst_l_1 = 0.0
    smallest_witness = 1.0
    for m in range(3, 16):
        epsilon = math.exp(-m)
        k = math.ceil(math.exp(m * m))
        a_m = m / (m + math.log(k))

        n_b = 2
        b_m = math.log(1.0 / (1.0 - epsilon)) / math.log(
            n_b / (1.0 - epsilon)
        )
        while b_m > a_m / m:
            n_b *= 2
            b_m = math.log(1.0 / (1.0 - epsilon)) / math.log(
                n_b / (1.0 - epsilon)
            )

        delta = 1e-2
        while True:
            _, _, r_a, r_b, witness = block_formulas(epsilon, k, n_b, delta)
            if (
                a_m * (1.0 - 1.0 / m) <= r_a <= a_m * (1.0 + 1.0 / m)
                and r_b <= 2.0 * a_m / m
                and witness >= 1.0 - 1.0 / m
            ):
                break
            delta /= 10.0
            if delta < 1e-15:
                raise AssertionError(f"failed to realize diagonal choice for m={m}")

        m_1 = max(r_a, r_b)
        l_1 = (epsilon * r_a + (1.0 - epsilon) * r_b) / m_1
        if not m_1 <= a_m * (1.0 + 1.0 / m) + TOL:
            raise AssertionError("M1 upper estimate failed")
        if not l_1 <= epsilon + 2.0 / (m - 1.0) + TOL:
            raise AssertionError("L1 upper estimate failed")
        worst_l_1 = max(worst_l_1, l_1)
        smallest_witness = min(smallest_witness, witness)

    print(
        "diagonal construction: verified for m=3,...,15 "
        f"(max L1={worst_l_1:.6f}, min set witness={smallest_witness:.6f})"
    )


def check_three_state_example() -> None:
    p_a = [
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0],
    ]
    p_b = [
        [0.5, 0.25, 0.25],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
    ]
    eigenpairs_a = [
        ([1.0, 1.0, 1.0], 1.0),
        ([1.0, -1.0, 0.0], -0.5),
        ([1.0, 0.0, -1.0], -0.5),
    ]
    eigenpairs_b = [
        ([1.0, 1.0, 1.0], 1.0),
        ([0.0, 1.0, -1.0], 0.5),
        ([-1.0, 1.0, 1.0], 0.0),
    ]
    for matrix, eigenpairs in ((p_a, eigenpairs_a), (p_b, eigenpairs_b)):
        for vector, eigenvalue in eigenpairs:
            for left, right in zip(matvec(matrix, vector), vector):
                assert_close(left, eigenvalue * right)

    pi_a = [1.0 / 3.0] * 3
    pi_b = [0.5, 0.25, 0.25]
    r_a = [kl(row, pi_a) / math.log(3.0) for row in p_a]
    r_b = [kl(row, pi_b) / math.log(1.0 / pi_b[x]) for x, row in enumerate(p_b)]
    for value in r_a:
        assert_close(value, math.log(1.5) / math.log(3.0))
    for left, right in zip(r_b, [0.0, 0.25, 0.25]):
        assert_close(left, right)
    assert_close(sum(pi_a[x] * r_a[x] for x in range(3)) / max(r_a), 1.0)
    assert_close(sum(pi_b[x] * r_b[x] for x in range(3)) / max(r_b), 0.5)
    print("three-state spectra and profiles: verified")


def check_binary_example() -> None:
    for index in range(1, 500):
        a = index / 1000.0
        h_b = -a * math.log(a) - (1.0 - a) * math.log(1.0 - a)
        m_1 = (math.log(2.0) - h_b) / math.log(2.0)
        eta = (1.0 - 2.0 * a) ** 2
        if not eta > m_1:
            raise AssertionError(f"binary strict comparison failed at a={a}")
    print("binary strict comparison on a dense grid: verified")


def check_uniform_bottleneck_bound() -> None:
    for p_index in range(1, 101):
        p = p_index / 200.0
        for h_index in range(1, 100):
            h = h_index / 100.0
            lhs = binary_kl(1.0 - h, p) / math.log(1.0 / p)
            h_b = -h * math.log(h) - (1.0 - h) * math.log(1.0 - h)
            rhs = 1.0 - h - h_b / math.log(2.0)
            if lhs + TOL < rhs:
                raise AssertionError("uniform bottleneck simplification failed")
    print("uniform set-bottleneck simplification: verified")


if __name__ == "__main__":
    check_counterexample_matrix()
    check_diagonal_selection()
    check_three_state_example()
    check_binary_example()
    check_uniform_bottleneck_bound()
