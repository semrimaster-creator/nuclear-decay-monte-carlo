"""
nuclear_structure_schrodinger.py
==================================

Numerical solution of the time-independent Schrodinger equation as a
matrix eigenvalue problem, first validated against analytically-solvable
textbook potentials, then applied to the Woods-Saxon potential to obtain
realistic single-particle nuclear shell-model energy levels.

RELATION TO THE REST OF THE CV / PROJECT
------------------------------------------
This is the first module in the repository that touches quantum mechanics
directly, but the numerical machinery is exactly the eigenvalue-problem
toolkit already listed on the CV and used in the M.Sc. thesis (ODE
eigenvalue problems, finite-difference/FEM discretization, linear algebra
eigensolvers): the time-independent Schrodinger equation

    -hbar^2/(2m) d^2(psi)/dx^2 + V(x) psi = E psi

is turned into a discrete matrix eigenvalue problem

    H psi_i = E_i psi_i

by finite-difference discretization of the second derivative, exactly the
same discretization strategy (central differences on a uniform mesh) used
for the boundary-value / eigenvalue ODE problems in the CV's stated
skillset. `scipy.linalg.eigh_tridiagonal` (a symmetric tridiagonal
eigensolver) plays the same numerical-linear-algebra role as the FEM
stiffness-matrix eigenvalue analysis used in the thesis's stability study.

PART 1 -- VALIDATION ON ANALYTIC BENCHMARKS
---------------------------------------------
(a) Infinite square well of width L:
        E_n = n^2 * pi^2 * hbar^2 / (2 m L^2),   n = 1, 2, 3, ...
(b) Quantum harmonic oscillator:
        E_n = hbar*omega * (n + 1/2),   n = 0, 1, 2, ...

PART 2 -- WOODS-SAXON NUCLEAR MEAN-FIELD POTENTIAL
-----------------------------------------------------
The radial Schrodinger equation for u(r) = r*R(r) in a spherical
mean-field potential:

    -hbar^2/(2m) u''(r) + [V_WS(r) + hbar^2 l(l+1) / (2 m r^2)] u(r) = E u(r)

with the Woods-Saxon potential

    V_WS(r) = -V0 / (1 + exp((r - R) / a))

and boundary condition u(0) = 0 (regularity at the origin). This is
solved for several orbital angular momenta l = 0, 1, 2, 3 (s, p, d, f) to
obtain single-particle bound-state energies -- the starting point of the
nuclear shell model.

NOTE ON SPIN-ORBIT COUPLING: this base Woods-Saxon (no spin-orbit term)
correctly reproduces the shell gaps at particle numbers 2, 8, and 20, but
NOT the higher magic numbers (28, 50, 82, 126), which require the
Goeppert-Mayer/Jensen spin-orbit interaction to split each (n,l) level
into j = l +/- 1/2 sublevels. This is flagged explicitly below rather than
glossed over, and is the natural next extension of this module.

USAGE
-----
    python3 nuclear_structure_schrodinger.py
"""

import numpy as np
from scipy.linalg import eigh_tridiagonal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# Generic 1D Schrodinger finite-difference eigensolver
# ----------------------------------------------------------------------

def solve_schrodinger_fd(x, V, hbar2_over_2m, n_states=6):
    """
    Solve  -hbar^2/(2m) psi'' + V(x) psi = E psi  on a uniform grid `x`
    (Dirichlet BC: psi = 0 at both ends) via central finite differences.

    Returns (energies[:n_states], wavefunctions[:, :n_states]) with
    wavefunctions normalized so that integral |psi|^2 dx = 1.
    """
    n = len(x)
    h = x[1] - x[0]

    # Central finite-difference for -d^2/dx^2 gives a tridiagonal matrix:
    #   diagonal:      2 * hbar2_over_2m / h^2 + V(x_i)
    #   off-diagonal:  -hbar2_over_2m / h^2
    diag = 2.0 * hbar2_over_2m / h**2 + V
    offdiag = -hbar2_over_2m / h**2 * np.ones(n - 1)

    # Interior points only (Dirichlet BC: psi=0 at the two endpoints)
    diag_int = diag[1:-1]
    offdiag_int = offdiag[1:-1]

    energies, vecs = eigh_tridiagonal(diag_int, offdiag_int,
                                       select='i', select_range=(0, n_states - 1))

    # Pad wavefunctions with the zero boundary values and normalize
    psis = np.zeros((n, n_states))
    psis[1:-1, :] = vecs
    for k in range(n_states):
        norm = np.sqrt(np.trapezoid(psis[:, k]**2, x))
        psis[:, k] /= norm

    return energies, psis


# ----------------------------------------------------------------------
# PART 1a: Infinite square well benchmark
# ----------------------------------------------------------------------

def benchmark_square_well():
    print("=" * 70)
    print("Benchmark 1: Infinite square well")
    print("=" * 70)

    hbar2_over_2m = 1.0   # natural units: hbar = m = 1
    L = 1.0
    n_points = 2000
    x = np.linspace(0, L, n_points)
    V = np.zeros(n_points)  # well interior; Dirichlet BC enforces walls

    n_states = 6
    E_numeric, _ = solve_schrodinger_fd(x, V, hbar2_over_2m, n_states=n_states)
    n_quantum = np.arange(1, n_states + 1)
    E_analytic = (n_quantum**2 * np.pi**2 * hbar2_over_2m) / (L**2)

    print(f"{'n':>3} {'E_analytic':>14} {'E_numeric':>14} {'rel. error':>12}")
    for i in range(n_states):
        rel_err = abs(E_numeric[i] - E_analytic[i]) / E_analytic[i]
        print(f"{i+1:>3} {E_analytic[i]:>14.6f} {E_numeric[i]:>14.6f} "
              f"{100*rel_err:>11.4f}%")
    print()
    return E_analytic, E_numeric


# ----------------------------------------------------------------------
# PART 1b: Harmonic oscillator benchmark
# ----------------------------------------------------------------------

def benchmark_harmonic_oscillator():
    print("=" * 70)
    print("Benchmark 2: Quantum harmonic oscillator")
    print("=" * 70)

    hbar2_over_2m = 0.5  # hbar^2/(2m) = 0.5 for hbar = m = 1
    omega = 1.0  # hbar*omega = 1 in these units (hbar=m=1)
    x_max = 10.0
    n_points = 3000
    x = np.linspace(-x_max, x_max, n_points)
    V = 0.5 * omega**2 * x**2  # V = (1/2) m omega^2 x^2, m=1

    n_states = 6
    E_numeric, _ = solve_schrodinger_fd(x, V, hbar2_over_2m, n_states=n_states)
    n_quantum = np.arange(0, n_states)
    E_analytic = omega * (n_quantum + 0.5)

    print(f"{'n':>3} {'E_analytic':>14} {'E_numeric':>14} {'rel. error':>12}")
    for i in range(n_states):
        rel_err = abs(E_numeric[i] - E_analytic[i]) / E_analytic[i]
        print(f"{i:>3} {E_analytic[i]:>14.6f} {E_numeric[i]:>14.6f} "
              f"{100*rel_err:>11.4f}%")
    print()
    return E_analytic, E_numeric


# ----------------------------------------------------------------------
# PART 2: Woods-Saxon nuclear mean-field potential
# ----------------------------------------------------------------------

def woods_saxon_levels(A=40, V0=50.0, r0=1.25, a_diff=0.65,
                        l_max=3, n_states_per_l=4, r_max=15.0, n_points=4000):
    """
    Solve the radial Schrodinger equation for u(r) = r*R(r) in a
    Woods-Saxon potential, for orbital angular momenta l = 0..l_max.

    Units: energies in MeV, lengths in fm.
    hbar^2/(2m) for a nucleon: (hbar*c)^2 / (2 * m_N c^2)
        hbar*c   = 197.327 MeV*fm
        m_N c^2  = 939.0 MeV  (average nucleon mass)
    """
    hbar_c = 197.327
    mNc2 = 939.0
    hbar2_over_2m = hbar_c**2 / (2.0 * mNc2)  # ~20.7 MeV*fm^2

    R = r0 * A**(1.0 / 3.0)
    r = np.linspace(1e-6, r_max, n_points)  # avoid r=0 singularity in l(l+1)/r^2
    V_ws = -V0 / (1.0 + np.exp((r - R) / a_diff))

    results = {}
    for l in range(l_max + 1):
        centrifugal = hbar2_over_2m * l * (l + 1) / r**2
        V_eff = V_ws + centrifugal

        E, psi = solve_schrodinger_fd(r, V_eff, hbar2_over_2m,
                                       n_states=n_states_per_l)
        # Keep only bound states (E < 0, i.e. below the V=0 continuum)
        bound_mask = E < 0
        results[l] = {
            "E": E[bound_mask],
            "psi": psi[:, bound_mask],
            "r": r,
        }

    return results, R, V0, a_diff, hbar2_over_2m


def print_shell_structure(results):
    print("=" * 70)
    print("Woods-Saxon single-particle levels (no spin-orbit coupling)")
    print("=" * 70)
    label = {0: "s", 1: "p", 2: "d", 3: "f", 4: "g"}

    # Build a flat list of (E, l, n_radial) for sorting into a shell diagram
    flat = []
    for l, data in results.items():
        for n_r, E in enumerate(data["E"]):
            flat.append((E, l, n_r + 1))
    flat.sort(key=lambda t: t[0])

    print(f"{'E (MeV)':>10}  {'level':>8}  {'degeneracy (2(2l+1))':>22}")
    cumulative = 0
    for E, l, n_r in flat:
        deg = 2 * (2 * l + 1)
        cumulative += deg
        print(f"{E:>10.3f}  {n_r}{label.get(l,'?'):>7}  {deg:>10d}"
              f"   (cumulative N = {cumulative})")

    print()
    print("Expected shell-model magic numbers: 2, 8, 20, 28, 50, 82, 126")
    print("Without spin-orbit coupling, only 2, 8, 20 emerge as clean gaps")
    print("above; 28/50/82/126 require the spin-orbit splitting of each")
    print("(n,l) level into j = l+1/2 and j = l-1/2 sub-levels -- the")
    print("natural next extension of this module (Woods-Saxon + spin-orbit,")
    print("i.e. -lambda(r) * L.S, giving the full Goeppert-Mayer/Jensen")
    print("shell model).")
    print("=" * 70)
    return flat


def plot_woods_saxon(results, R, V0, a_diff, hbar2_over_2m):
    label = {0: "s", 1: "p", 2: "d", 3: "f"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    # Left panel: potential + energy levels
    r = results[0]["r"]
    V_ws = -V0 / (1.0 + np.exp((r - R) / a_diff))
    ax = axes[0]
    ax.plot(r, V_ws, "k-", lw=2, label="Woods-Saxon $V(r)$")
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    for l, data in results.items():
        for E in data["E"]:
            ax.hlines(E, 0, R * 2.2, colors=[colors[l]], linestyles="--",
                      alpha=0.7)
        if len(data["E"]) > 0:
            ax.plot([], [], color=colors[l], ls="--",
                     label=f"l={l} ({label.get(l,'?')})")
    ax.set_xlabel("r (fm)")
    ax.set_ylabel("Energy (MeV)")
    ax.set_title(f"Woods-Saxon potential and bound levels (A={40})")
    ax.legend(fontsize=8)
    ax.set_xlim(0, R * 2.2)

    # Right panel: shell diagram (energy vs cumulative occupation)
    ax2 = axes[1]
    flat = []
    for l, data in results.items():
        for n_r, E in enumerate(data["E"]):
            flat.append((E, l, n_r + 1))
    flat.sort(key=lambda t: t[0])
    cumulative = 0
    magic_numbers = [2, 8, 20, 28, 50, 82, 126]
    for E, l, n_r in flat:
        deg = 2 * (2 * l + 1)
        cumulative_prev = cumulative
        cumulative += deg
        ax2.hlines(E, cumulative_prev, cumulative, colors=colors[l], lw=4)
        ax2.text(cumulative_prev, E, f" {n_r}{label.get(l,'?')}", fontsize=8,
                  va="bottom")
        if cumulative in magic_numbers or cumulative_prev in magic_numbers:
            ax2.axvline(cumulative, color="red", ls=":", alpha=0.4)
    ax2.set_xlabel("Cumulative nucleon occupation N")
    ax2.set_ylabel("Energy (MeV)")
    ax2.set_title("Single-particle shell structure\n(red dashed = candidate magic-number gap)")

    plt.tight_layout()
    plt.savefig("woods_saxon_levels.png", dpi=150)
    print("Saved plot: woods_saxon_levels.png")


def plot_benchmarks(sw_analytic, sw_numeric, ho_analytic, ho_numeric):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    n_sw = np.arange(1, len(sw_analytic) + 1)
    axes[0].plot(n_sw, sw_analytic, "ko-", label="Analytic")
    axes[0].plot(n_sw, sw_numeric, "r x", markersize=10, label="Finite-difference")
    axes[0].set_xlabel("n")
    axes[0].set_ylabel(r"$E_n$")
    axes[0].set_title("Infinite square well")
    axes[0].legend()

    n_ho = np.arange(0, len(ho_analytic))
    axes[1].plot(n_ho, ho_analytic, "ko-", label="Analytic")
    axes[1].plot(n_ho, ho_numeric, "r x", markersize=10, label="Finite-difference")
    axes[1].set_xlabel("n")
    axes[1].set_ylabel(r"$E_n$")
    axes[1].set_title("Harmonic oscillator")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("schrodinger_benchmarks.png", dpi=150)
    print("Saved plot: schrodinger_benchmarks.png")


def main():
    sw_analytic, sw_numeric = benchmark_square_well()
    ho_analytic, ho_numeric = benchmark_harmonic_oscillator()
    plot_benchmarks(sw_analytic, sw_numeric, ho_analytic, ho_numeric)

    print()
    results, R, V0, a_diff, hbar2_over_2m = woods_saxon_levels(A=40)
    print_shell_structure(results)
    plot_woods_saxon(results, R, V0, a_diff, hbar2_over_2m)


if __name__ == "__main__":
    main()
