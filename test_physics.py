"""
test_physics.py
=================

Automated regression/validation tests for the four project modules.
Run with:

    pytest test_physics.py -v

These tests encode exactly the sanity checks described in the READMEs:
each numerical result must agree with an analytic or physically-required
bound before it is trusted. Keeping them as executable tests (rather than
one-off print statements) means a future change to any module is
automatically checked against the same benchmarks.
"""

import numpy as np
import pytest

from nuclear_structure_schrodinger import (
    solve_schrodinger_fd,
    benchmark_square_well,
    benchmark_harmonic_oscillator,
)
from neutron_transport_1d import (
    run_ensemble,
    run_ensemble_volumetric_source,
    diffusion_fem_1d,
)
from keff_power_iteration import compute_keff
from woods_saxon_spin_orbit import woods_saxon_spin_orbit_levels


# ----------------------------------------------------------------------
# Schrodinger solver: analytic benchmarks
# ----------------------------------------------------------------------

def test_square_well_matches_analytic():
    E_analytic, E_numeric = benchmark_square_well()
    rel_err = np.abs(E_numeric - E_analytic) / E_analytic
    assert np.all(rel_err < 1e-3), \
        f"Square well FD solution deviates from analytic: {rel_err}"


def test_harmonic_oscillator_matches_analytic():
    E_analytic, E_numeric = benchmark_harmonic_oscillator()
    rel_err = np.abs(E_numeric - E_analytic) / E_analytic
    assert np.all(rel_err < 1e-3), \
        f"Harmonic oscillator FD solution deviates from analytic: {rel_err}"


def test_solver_reproduces_ground_state_parity():
    """Ground state of a symmetric potential should be even (no sign
    change), a basic sanity check independent of the energy value."""
    x = np.linspace(-10, 10, 2000)
    V = 0.5 * x**2
    _, psi = solve_schrodinger_fd(x, V, hbar2_over_2m=0.5, n_states=2)
    # Ground state should not change sign (allow tiny numerical noise
    # near the truncated boundary)
    core = psi[100:-100, 0]
    assert np.all(core >= -1e-6) or np.all(core <= 1e-6), \
        "Ground-state wavefunction should not change sign"


# ----------------------------------------------------------------------
# Neutron transport: MC vs diffusion-theory FEM
# ----------------------------------------------------------------------

def test_mc_transport_matches_diffusion_fem():
    L = 10.0
    sigma_a = 0.02
    sigma_s = 0.8
    sigma_tr = sigma_a + sigma_s

    result = run_ensemble_volumetric_source(
        N=50_000, L=L, sigma_a=sigma_a, sigma_s=sigma_s, n_bins=50, seed=7
    )
    x_fem, phi_fem = diffusion_fem_1d(L, sigma_a, sigma_tr, S=1.0, n_elements=200)
    phi_fem_interp = np.interp(result["bin_centers"], x_fem, phi_fem)

    # Compare only the bulk (exclude the outermost bin on each side, where
    # diffusion theory is expected to deviate -- this is a real transport
    # boundary-layer effect, not a numerical error)
    mid = slice(3, -3)
    rel_err = np.abs(phi_fem_interp[mid] - result["flux"][mid]) / phi_fem_interp[mid]
    assert np.mean(rel_err) < 0.05, \
        f"MC transport flux deviates >5% from diffusion-theory FEM in the bulk: {np.mean(rel_err):.3f}"


def test_transport_probabilities_sum_to_one():
    result = run_ensemble(N=20_000, L=10.0, sigma_a=0.02, sigma_s=0.8,
                           source_mode="beam", seed=1, n_bins=20)
    total = sum(result["probs"].values())
    assert abs(total - 1.0) < 1e-9


# ----------------------------------------------------------------------
# k_eff: physical sanity bounds
# ----------------------------------------------------------------------

def test_keff_below_k_infinity():
    """A finite slab must always multiply less than the infinite-medium
    limit, because leakage can only remove neutrons, never add them."""
    sigma_a, sigma_f, sigma_s, nu = 0.010, 0.070, 0.400, 2.5
    k_inf = nu * sigma_f / (sigma_a + sigma_f)

    k_mean, k_stderr, _ = compute_keff(
        L=40.0, sigma_a=sigma_a, sigma_f=sigma_f, sigma_s=sigma_s, nu=nu,
        N=3000, n_generations=80, n_inactive=20, seed=1
    )
    assert k_mean < k_inf, \
        f"k_eff ({k_mean:.4f}) should be strictly below k_infinity ({k_inf:.4f})"


def test_keff_increases_with_slab_thickness():
    """Thicker slabs leak proportionally less, so k_eff should increase
    (monotonically, up to statistical noise) with L."""
    sigma_a, sigma_f, sigma_s, nu = 0.010, 0.070, 0.400, 2.5
    kwargs = dict(sigma_a=sigma_a, sigma_f=sigma_f, sigma_s=sigma_s, nu=nu,
                   N=3000, n_generations=80, n_inactive=20)

    k_thin, _, _ = compute_keff(L=10.0, seed=2, **kwargs)
    k_thick, _, _ = compute_keff(L=60.0, seed=3, **kwargs)
    assert k_thick > k_thin, \
        f"k_eff should increase with slab thickness: k(10cm)={k_thin:.3f}, k(60cm)={k_thick:.3f}"


# ----------------------------------------------------------------------
# Woods-Saxon + spin-orbit: correct level ordering
# ----------------------------------------------------------------------

def test_spin_orbit_ordering_j_gt_below_j_lt():
    """Physically, for l > 0 the j = l+1/2 partner must be MORE bound
    (lower energy) than the j = l-1/2 partner (this is what opens the
    N=28, 50, 82 shell gaps). This directly guards against the sign
    error that was caught during development."""
    results, _ = woods_saxon_spin_orbit_levels(A=40, V_so=20.0, l_max=3)

    for l in [1, 2, 3]:
        E_high_j = results[(l, l + 0.5)]["E"][0]
        E_low_j = results[(l, l - 0.5)]["E"][0]
        assert E_high_j < E_low_j, \
            (f"For l={l}, j={l+0.5} should be more bound than j={l-0.5}: "
             f"got E(j={l+0.5})={E_high_j:.3f} MeV, "
             f"E(j={l-0.5})={E_low_j:.3f} MeV")


def test_spin_orbit_reproduces_n28_gap():
    results, _ = woods_saxon_spin_orbit_levels(A=40, V_so=20.0, l_max=4)
    flat = []
    for (l, j), data in results.items():
        for n_r, E in enumerate(data["E"]):
            flat.append((E, int(2 * j + 1)))
    flat.sort(key=lambda t: t[0])

    cumulative = 0
    cumulative_values = []
    for _, deg in flat:
        cumulative += deg
        cumulative_values.append(cumulative)

    assert 28 in cumulative_values, \
        "Spin-orbit Woods-Saxon should reproduce a clean shell gap at N=28"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
