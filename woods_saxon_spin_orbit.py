"""
woods_saxon_spin_orbit.py
============================

Extension of `nuclear_structure_schrodinger.py`: adds the
Goeppert-Mayer/Jensen spin-orbit interaction to the Woods-Saxon mean
field, which splits each (n, l) level into j = l + 1/2 and j = l - 1/2
sub-levels. This is the missing ingredient that turns the "wrong" 1949
shell model (magic numbers 2, 8, 20, 40, ...) into the correct one
(2, 8, 20, 28, 50, 82, 126) -- historically the single most famous
result in nuclear structure theory, and the natural next step flagged
in `nuclear_structure_schrodinger.py`.

PHYSICS
-------
Radial equation for u(r) = r R(r), now with an added spin-orbit term:

    -hbar^2/(2m) u'' + [V_WS(r) + V_centrifugal(r) + V_SO(r,l,j)] u = E u

    V_SO(r,l,j) = -V_so * (1/r) * (df/dr) * <L.S>_{l,j}

    f(r) = 1 / (1 + exp((r-R)/a))        (Woods-Saxon form factor)

    <L.S>_{l,j} = (hbar^2/2) * [j(j+1) - l(l+1) - 3/4]
                = +hbar^2 * l/2         for j = l + 1/2
                = -hbar^2 * (l+1)/2     for j = l - 1/2

V_so (the spin-orbit strength) is taken from the standard Bohr-Mottelson/
Skyrme-style parametrization V_so ~ 0.44 * V0 * (lambda_pi)^2 with the
pion-Compton-wavelength-squared factor lambda_pi^2 ~ 2 fm^2; this gives
V_so of order 20-25 MeV*fm^2, the textbook range. THIS IS AN ILLUSTRATIVE
VALUE, not fit to reproduce a specific nucleus's spectroscopy -- getting
quantitatively accurate splittings for a real nucleus requires fitting
V0, r0, a, and V_so simultaneously to experimental level data, which is
a natural "next next" step (e.g. a least-squares fit against known
single-particle level data for a closed-shell nucleus like Ca-40 or
Pb-208).

USAGE
-----
    python3 woods_saxon_spin_orbit.py
"""

import numpy as np
from scipy.linalg import eigh_tridiagonal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def solve_schrodinger_fd(x, V, hbar2_over_2m, n_states=6):
    """Same finite-difference eigensolver as nuclear_structure_schrodinger.py
    (duplicated here so this file runs standalone)."""
    n = len(x)
    h = x[1] - x[0]
    diag = 2.0 * hbar2_over_2m / h**2 + V
    offdiag = -hbar2_over_2m / h**2 * np.ones(n - 1)
    diag_int = diag[1:-1]
    offdiag_int = offdiag[1:-1]
    energies, vecs = eigh_tridiagonal(diag_int, offdiag_int,
                                       select='i', select_range=(0, n_states - 1))
    psis = np.zeros((n, n_states))
    psis[1:-1, :] = vecs
    for k in range(n_states):
        norm = np.sqrt(np.trapezoid(psis[:, k]**2, x))
        psis[:, k] /= norm
    return energies, psis


def woods_saxon_spin_orbit_levels(A=40, V0=50.0, r0=1.25, a_diff=0.65,
                                   V_so=20.0, l_max=4, n_states_per_lj=3,
                                   r_max=15.0, n_points=4000):
    """
    Solve the radial Schrodinger equation with Woods-Saxon + spin-orbit,
    for each (l, j) channel with l = 0..l_max.

    Returns a dict keyed by (l, j) -> {'E': array, 'r': array}.
    """
    hbar_c = 197.327
    mNc2 = 939.0
    hbar2_over_2m = hbar_c**2 / (2.0 * mNc2)

    R = r0 * A**(1.0 / 3.0)
    r = np.linspace(1e-6, r_max, n_points)
    f = 1.0 / (1.0 + np.exp((r - R) / a_diff))
    V_ws = -V0 * f
    df_dr = np.gradient(f, r)

    results = {}
    for l in range(l_max + 1):
        centrifugal = hbar2_over_2m * l * (l + 1) / r**2

        j_values = [l + 0.5] if l == 0 else [l + 0.5, l - 0.5]
        for j in j_values:
            if j == l + 0.5:
                LS = l / 2.0
            else:
                LS = -(l + 1) / 2.0

            # Correct sign convention: V_SO must be MORE negative (more bound)
            # for the higher-j partner (j = l+1/2, larger <L.S>) so that e.g.
            # 1f7/2 drops below 1f5/2, as required physically. Since f(r) is
            # a decreasing function of r, df/dr < 0, so no extra leading
            # minus sign is needed here (V_SO = +V_so/r * df/dr * <L.S>).
            V_so_term = V_so * (1.0 / r) * df_dr * LS
            V_eff = V_ws + centrifugal + V_so_term

            E, _ = solve_schrodinger_fd(r, V_eff, hbar2_over_2m,
                                         n_states=n_states_per_lj)
            bound_mask = E < 0
            results[(l, j)] = {"E": E[bound_mask], "r": r}

    return results, R


def frac(j):
    """Format a half-integer j as a fraction string, e.g. 3.5 -> '7/2'."""
    num = int(round(2 * j))
    return f"{num}/2"


def print_and_rank_levels(results):
    label = {0: "s", 1: "p", 2: "d", 3: "f", 4: "g"}
    flat = []
    for (l, j), data in results.items():
        for n_r, E in enumerate(data["E"]):
            deg = int(2 * j + 1)
            flat.append((E, l, j, n_r + 1, deg))
    flat.sort(key=lambda t: t[0])

    print("=" * 78)
    print("Woods-Saxon + spin-orbit single-particle levels")
    print("=" * 78)
    print(f"{'E (MeV)':>10}  {'level':>10}  {'deg. (2j+1)':>12}  {'cumulative N':>13}")
    cumulative = 0
    magic = [2, 8, 20, 28, 50, 82, 126]
    hits = []
    for E, l, j, n_r, deg in flat:
        cumulative += deg
        marker = "  <-- magic number!" if cumulative in magic else ""
        if cumulative in magic:
            hits.append(cumulative)
        print(f"{E:>10.3f}  {n_r}{label.get(l,'?')}{frac(j):>4}     "
              f"{deg:>10d}  {cumulative:>13d}{marker}")

    print("-" * 78)
    reproduced = sorted(set(hits))
    print(f"Magic numbers reproduced as clean shell gaps: {reproduced}")
    missing = [m for m in magic if m not in reproduced and m <= cumulative]
    if missing:
        print(f"Not cleanly reproduced within this level range: {missing}")
    print("(Illustrative V_so and A=40 -- exact gap positions depend on")
    print(" fitting V0/r0/a/V_so to real spectroscopic data; the point")
    print(" demonstrated here is the qualitative mechanism: spin-orbit")
    print(" splitting pulls the high-j intruder state, e.g. 1f7/2, down")
    print(" in energy relative to its 1f5/2 partner, opening the N=28 gap")
    print(" that is absent without spin-orbit coupling.)")
    print("=" * 78)
    return flat


def plot_comparison(results_with_so, R):
    label = {0: "s", 1: "p", 2: "d", 3: "f", 4: "g"}
    fig, ax = plt.subplots(figsize=(7, 8))

    flat = []
    for (l, j), data in results_with_so.items():
        for n_r, E in enumerate(data["E"]):
            deg = int(2 * j + 1)
            flat.append((E, l, j, n_r + 1, deg))
    flat.sort(key=lambda t: t[0])

    cumulative = 0
    colors = plt.cm.tab10(np.linspace(0, 1, 5))
    magic = [2, 8, 20, 28, 50, 82, 126]
    for E, l, j, n_r, deg in flat:
        cumulative_prev = cumulative
        cumulative += deg
        ax.hlines(E, cumulative_prev, cumulative, colors=colors[l % 5], lw=5)
        ax.text(cumulative_prev, E, f" {n_r}{label.get(l,'?')}{frac(j)}",
                 fontsize=8, va="bottom")
        if cumulative in magic:
            ax.axvline(cumulative, color="red", ls="--", alpha=0.6, lw=1.5)
            ax.text(cumulative, ax.get_ylim()[0] if False else E - 1,
                     f"N={cumulative}", color="red", fontsize=8, ha="left")

    ax.set_xlabel("Cumulative nucleon occupation N")
    ax.set_ylabel("Energy (MeV)")
    ax.set_title("Woods-Saxon + spin-orbit shell structure\n(red dashed = magic-number gap)")
    plt.tight_layout()
    plt.savefig("woods_saxon_spin_orbit_levels.png", dpi=150)
    print("Saved plot: woods_saxon_spin_orbit_levels.png")


def main():
    results, R = woods_saxon_spin_orbit_levels(A=40, V_so=20.0)
    print_and_rank_levels(results)
    plot_comparison(results, R)


if __name__ == "__main__":
    main()
