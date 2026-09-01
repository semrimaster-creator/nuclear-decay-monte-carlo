"""
keff_power_iteration.py
=========================

Monte Carlo eigenvalue (k_eff) calculation for a bare, homogeneous,
fissile 1D slab, via power iteration -- the standard method used by
production codes (MCNP, Serpent, OpenMC) to find the neutron
multiplication factor.

RELATION TO THE REST OF THE PROJECT
------------------------------------
This module extends `neutron_transport_1d.py` by adding a third reaction
channel (fission) to the collision physics. Conceptually, k_eff is found
by an eigenvalue iteration on the neutron transport operator:

    L * phi = (1/k) * F * phi

where L is the loss operator (leakage + absorption + scattering-out) and
F is the fission production operator. Power iteration is exactly the
"repeatedly apply the operator and renormalize" algorithm used for the
dominant-eigenvalue problem -- the same conceptual tool as the Jacobian
eigenvalue/stability analysis performed on the reaction-diffusion system
in the M.Sc. thesis, now applied to a genuine transport eigenvalue problem
instead of a linearized stability problem.

METHOD
------
1. Start with N fission source neutrons distributed uniformly across the
   slab (a guess for the fission source shape phi_0).
2. Transport each neutron: sample free flight, then branch collision type
   by relative cross section:
        p_scatter  = Sigma_s / Sigma_t
        p_capture  = Sigma_a / Sigma_t      (non-fission absorption)
        p_fission  = Sigma_f / Sigma_t
   On fission, sample the number of secondary neutrons (mean nu, Poisson-
   like via stochastic rounding of nu) and record their birth positions
   (at the fission site) as the next generation's source sites.
3. After simulating all N histories in this generation, the k_eff estimate
   for this generation is:
        k_gen = (number of fission neutrons produced) / N
   (the ratio of "new generation size" to "old generation size").
4. Renormalize the fission-neutron bank back to exactly N sites (uniform
   random resampling with replacement) so population doesn't explode or
   die out -- standard "population control" in MC criticality codes.
5. Repeat for many generations; discard the first few as inactive
   (settling) cycles, then average k_gen over the active cycles to get
   k_eff with an estimate of its statistical uncertainty.

A bisection search on the slab thickness L is then used to find the
*critical thickness* L_c at which k_eff = 1, analogous to finding the
critical point of a dynamical system in bifurcation analysis.

USAGE
-----
    python3 keff_power_iteration.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def sample_free_flight(sigma_t, rng):
    u = rng.random()
    return -np.log(1.0 - u) / sigma_t


def run_generation(source_positions, L, sigma_a, sigma_f, sigma_s, nu, rng):
    """
    Transport one generation of neutrons, starting from `source_positions`
    (fission sites from the previous generation). Returns the list of
    fission-neutron birth positions produced this generation.
    """
    sigma_t = sigma_a + sigma_f + sigma_s
    p_scatter = sigma_s / sigma_t
    p_capture = sigma_a / sigma_t
    # p_fission = sigma_f / sigma_t  (remainder)

    next_gen_positions = []

    for x0 in source_positions:
        x = x0
        mu = rng.uniform(-1.0, 1.0)  # isotropic emission from fission

        while True:
            s = sample_free_flight(sigma_t, rng)
            x_new = x + mu * s

            if x_new < 0.0 or x_new > L:
                break  # leakage -- history ends, no fission neutrons

            x = x_new
            u = rng.random()
            if u < p_scatter:
                mu = rng.uniform(-1.0, 1.0)
                continue
            elif u < p_scatter + p_capture:
                break  # non-fission capture -- history ends
            else:
                # Fission event: sample integer number of secondaries with
                # mean nu via stochastic rounding (e.g. nu=2.5 -> 2 or 3).
                n_sec = int(np.floor(nu)) + (1 if rng.random() < (nu - np.floor(nu)) else 0)
                next_gen_positions.extend([x] * n_sec)
                break  # the fissioning neutron itself is absorbed

    return next_gen_positions


def resample_to_size(positions, N, rng):
    """Population control: resample the fission-neutron bank to exactly N
    sites via random sampling with replacement (standard technique in MC
    criticality codes to keep generation size constant)."""
    if len(positions) == 0:
        return []
    idx = rng.integers(0, len(positions), size=N)
    return [positions[i] for i in idx]


def compute_keff(L, sigma_a, sigma_f, sigma_s, nu, N=5000,
                  n_generations=150, n_inactive=30, seed=1):
    """Run power iteration and return (k_eff_mean, k_eff_stderr, k_history)."""
    rng = np.random.default_rng(seed)
    source_positions = list(rng.uniform(0.0, L, size=N))

    k_history = []
    for gen in range(n_generations):
        next_gen = run_generation(source_positions, L, sigma_a, sigma_f,
                                   sigma_s, nu, rng)
        k_gen = len(next_gen) / N
        k_history.append(k_gen)

        source_positions = resample_to_size(next_gen, N, rng)
        if len(source_positions) == 0:
            # Population died out completely; pad with a fresh uniform
            # guess to avoid a hard crash (k_eff is clearly << 1 here).
            source_positions = list(rng.uniform(0.0, L, size=N))

    active = k_history[n_inactive:]
    k_mean = float(np.mean(active))
    k_stderr = float(np.std(active, ddof=1) / np.sqrt(len(active)))
    return k_mean, k_stderr, k_history


def find_critical_thickness(sigma_a, sigma_f, sigma_s, nu,
                             L_lo=1.0, L_hi=200.0, tol=0.005,
                             N=3000, n_generations=100, n_inactive=20):
    """Bisection search for the slab thickness L at which k_eff = 1."""
    def k_of_L(L, seed):
        k, _, _ = compute_keff(L, sigma_a, sigma_f, sigma_s, nu,
                                N=N, n_generations=n_generations,
                                n_inactive=n_inactive, seed=seed)
        return k

    seed = 100
    k_lo = k_of_L(L_lo, seed)
    k_hi = k_of_L(L_hi, seed + 1)
    print(f"  Bisection bracket check: k_eff({L_lo} cm) = {k_lo:.4f}, "
          f"k_eff({L_hi} cm) = {k_hi:.4f}")

    if k_lo > 1.0 or k_hi < 1.0:
        print("  WARNING: bracket does not contain k_eff=1; "
              "widen L_lo/L_hi. Returning best available estimate.")

    lo, hi = L_lo, L_hi
    for it in range(14):
        mid = 0.5 * (lo + hi)
        k_mid = k_of_L(mid, seed + 2 + it)
        print(f"  iter {it+1:2d}: L = {mid:7.3f} cm  ->  k_eff = {k_mid:.4f}")
        if k_mid < 1.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main():
    # One-speed cross sections loosely representative of a fast-ish
    # fissile assembly (illustrative, not a specific real material).
    sigma_a = 0.010   # non-fission capture, 1/cm
    sigma_f = 0.070   # fission cross section, 1/cm
    sigma_s = 0.400   # scattering, 1/cm
    nu = 2.5           # mean neutrons per fission

    L = 40.0            # slab thickness, cm
    N = 8000
    n_generations = 150
    n_inactive = 30

    print("=" * 70)
    print("Monte Carlo k_eff via power iteration")
    print("=" * 70)
    print(f"Sigma_a = {sigma_a}, Sigma_f = {sigma_f}, Sigma_s = {sigma_s}, "
          f"nu = {nu}")
    print(f"Slab thickness L = {L} cm, N = {N} neutrons/generation, "
          f"{n_generations} generations ({n_inactive} inactive)")
    print("-" * 70)

    k_mean, k_stderr, k_history = compute_keff(
        L, sigma_a, sigma_f, sigma_s, nu, N=N,
        n_generations=n_generations, n_inactive=n_inactive, seed=42
    )
    print(f"k_eff = {k_mean:.4f} +/- {k_stderr:.4f}  "
          f"(averaged over generations {n_inactive}-{n_generations})")

    # Infinite-medium multiplication factor k_infinity for reference
    # (no leakage: pure competition between fission and absorption)
    sigma_t = sigma_a + sigma_f + sigma_s
    k_inf = nu * sigma_f / (sigma_a + sigma_f)
    print(f"k_infinity (no-leakage limit, nu*Sigma_f/Sigma_a_total) "
          f"= {k_inf:.4f}")
    print("-" * 70)

    # Plot generation-by-generation k_eff convergence
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_history, color="steelblue", lw=1, alpha=0.7, label="k per generation")
    ax.axhline(k_mean, color="red", ls="--", label=f"active-cycle mean = {k_mean:.4f}")
    ax.axvline(n_inactive, color="gray", ls=":", label="inactive/active cutoff")
    ax.axhline(1.0, color="black", lw=0.8)
    ax.set_xlabel("Generation")
    ax.set_ylabel(r"$k_{\mathrm{eff}}$ (this generation)")
    ax.set_title(f"Power iteration convergence  (L = {L} cm)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("keff_convergence.png", dpi=150)
    print("Saved plot: keff_convergence.png")
    print("-" * 70)

    # Critical thickness search
    print("Searching for critical slab thickness (k_eff = 1)...")
    L_crit = find_critical_thickness(sigma_a, sigma_f, sigma_s, nu,
                                      L_lo=5.0, L_hi=100.0)
    print(f"\nEstimated critical thickness: L_crit ~ {L_crit:.2f} cm")
    print("=" * 70)


if __name__ == "__main__":
    main()
