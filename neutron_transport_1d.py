"""
neutron_transport_1d.py
========================

Monte Carlo simulation of neutron transport through a 1D homogeneous slab.

RELATION TO THE DECAY PROJECT
------------------------------
This is a direct structural extension of the Gillespie (event-driven) radioactive
decay simulation:

    Decay project:     dt ~ Exponential(lambda)         -> "when does the next
                                                             decay event happen?"
    This project:      ds ~ Exponential(Sigma_t)         -> "how far does the
                                                             neutron fly before its
                                                             next collision?"

Both use the same sampling primitive:  x = -ln(1 - U) / rate
The ensemble-averaging loop (run N independent histories, accumulate statistics,
compare the stochastic result to the deterministic/analytic solution) is also
structurally identical to the decay project's ensemble-average routine.

PHYSICS MODEL
-------------
- Homogeneous, non-multiplying slab of thickness L (cm), infinite in y,z.
- Total macroscopic cross section  Sigma_t = Sigma_a + Sigma_s (1/cm).
- A neutron starts at x = 0 moving in the +x direction (mono-directional beam
  source) OR isotropically, depending on `source_mode`.
- Free-flight distance to next collision:  s ~ Exponential(Sigma_t)
  (mean free path lambda = 1/Sigma_t) -- same primitive as decay time sampling.
- At each collision, sample the interaction type using the branching
  probabilities p_scatter = Sigma_s/Sigma_t, p_absorb = Sigma_a/Sigma_t
  -- exactly analogous to sampling which reaction channel fires in a
  multi-channel Gillespie simulation.
- If scattering: sample a new direction cosine mu' isotropically in the
  lab frame (mu' ~ Uniform(-1,1)), continue flight.
- If absorption: history ends (neutron absorbed).
- If the neutron's position crosses x=0 or x=L: history ends (leakage,
  classified as reflection or transmission).
- A track-length estimator accumulates path length in each spatial bin to
  estimate the scalar flux profile phi(x), which is compared against the
  analytic/FEM solution of the neutron diffusion equation.

DIFFUSION-THEORY VALIDATION
----------------------------
For a purely absorbing/scattering (non-multiplying) slab with a uniform
volumetric source S (n/cm^3/s), steady-state one-speed diffusion theory gives

    -D d^2(phi)/dx^2 + Sigma_a * phi = S,      D = 1 / (3 * Sigma_tr)

with vacuum boundary conditions phi = 0 at the extrapolated boundaries
x = -d and x = L + d, where d = 0.71 / Sigma_tr (linear extrapolation distance).

This script solves that ODE two ways:
  1. Closed-form analytic solution (cosh/sinh), and
  2. A linear 1D Galerkin Finite Element Method (FEM) solve on the same mesh
     -- the identical numerical machinery used in the M.Sc. thesis for the
     reaction-diffusion system (FEM discretization + linear solve).

Both are plotted against the Monte Carlo track-length flux tally.

USAGE
-----
    python3 neutron_transport_1d.py

Produces:
  - console summary: transmission / reflection / absorption probabilities,
    mean number of collisions per history, MC vs diffusion-theory comparison
  - neutron_transport_flux.png : flux profile, MC vs FEM vs analytic
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# 1. MONTE CARLO NEUTRON TRANSPORT (event-driven, Gillespie-style)
# ----------------------------------------------------------------------

def sample_free_flight(sigma_t, rng):
    """Sample distance to next collision ~ Exponential(sigma_t).

    Identical primitive to the decay-time sampling in the Gillespie decay
    code: x = -ln(1-U) / rate.
    """
    u = rng.random()
    return -np.log(1.0 - u) / sigma_t


def run_single_history(L, sigma_a, sigma_s, rng, x0=0.0, mu0=1.0,
                        flux_tally=None, n_bins=None, dx=None):
    """
    Simulate one neutron history in a 1D slab [0, L].

    Returns
    -------
    outcome : str
        'transmitted', 'reflected', or 'absorbed'
    n_collisions : int
        number of collisions (scatters) experienced before termination
    """
    sigma_t = sigma_a + sigma_s
    p_scatter = sigma_s / sigma_t

    x = x0
    mu = mu0
    n_collisions = 0

    while True:
        s = sample_free_flight(sigma_t, rng)
        # Displacement along the slab axis
        x_new = x + mu * s

        # --- Track-length flux tally: accumulate path length within slab ---
        if flux_tally is not None:
            _tally_track_length(flux_tally, x, x_new, mu, L, n_bins, dx)

        # Check leakage before registering the collision
        if x_new < 0.0:
            return "reflected", n_collisions
        if x_new > L:
            return "transmitted", n_collisions

        # Neutron survives to collide at x_new
        x = x_new
        u = rng.random()
        if u < p_scatter:
            # Isotropic scattering in the lab frame (simplified 1D model)
            mu = rng.uniform(-1.0, 1.0)
            n_collisions += 1
            continue
        else:
            return "absorbed", n_collisions


def _tally_track_length(flux_tally, x_start, x_end, mu, L, n_bins, dx):
    """Distribute the *actual path length* (not the x-axis projection) of one
    free flight across spatial bins that lie within [0, L].

    IMPORTANT: the neutron's true path length s and its x-displacement are
    related by  x_end - x_start = mu * s.  A segment spanning an x-interval
    of width w therefore corresponds to a true path length of w / |mu|
    (for oblique directions the neutron travels further per unit x than a
    neutron flying straight along the axis). The standard track-length flux
    estimator requires the true path length, so we must divide by |mu|.
    """
    if mu == 0.0:
        return
    # Clip the segment to the slab boundaries [0, L] for tallying purposes
    lo = min(x_start, x_end)
    hi = max(x_start, x_end)
    lo_clipped = max(lo, 0.0)
    hi_clipped = min(hi, L)
    if hi_clipped <= lo_clipped:
        return

    bin_lo = int(np.floor(lo_clipped / dx))
    bin_hi = int(np.floor(min(hi_clipped, L - 1e-12) / dx))
    bin_lo = max(bin_lo, 0)
    bin_hi = min(bin_hi, n_bins - 1)

    inv_mu = 1.0 / abs(mu)
    for b in range(bin_lo, bin_hi + 1):
        seg_lo = max(lo_clipped, b * dx)
        seg_hi = min(hi_clipped, (b + 1) * dx)
        x_width = max(seg_hi - seg_lo, 0.0)
        true_path_length = x_width * inv_mu
        flux_tally[b] += true_path_length


def run_ensemble(N, L, sigma_a, sigma_s, source_mode="beam", seed=42,
                  n_bins=100):
    """
    Run N independent neutron histories and accumulate ensemble statistics.

    This mirrors the ensemble-averaging structure of the decay project:
    instead of averaging thousands of decay simulations, we average
    thousands of particle histories.
    """
    rng = np.random.default_rng(seed)
    outcomes = {"transmitted": 0, "reflected": 0, "absorbed": 0}
    collisions_list = []

    dx = L / n_bins
    flux_tally = np.zeros(n_bins)

    for _ in range(N):
        if source_mode == "beam":
            x0, mu0 = 0.0, 1.0
        elif source_mode == "isotropic":
            x0, mu0 = 0.0, rng.uniform(-1.0, 1.0)
        else:
            raise ValueError("source_mode must be 'beam' or 'isotropic'")

        outcome, n_coll = run_single_history(
            L, sigma_a, sigma_s, rng, x0=x0, mu0=mu0,
            flux_tally=flux_tally, n_bins=n_bins, dx=dx
        )
        outcomes[outcome] += 1
        collisions_list.append(n_coll)

    probs = {k: v / N for k, v in outcomes.items()}
    mean_collisions = float(np.mean(collisions_list))

    # Track-length estimator: phi(x) ~ (total path length in bin) / (bin volume * N)
    # For a unit cross-sectional area, bin volume = dx * 1 * 1.
    flux = flux_tally / (dx * N)
    bin_centers = (np.arange(n_bins) + 0.5) * dx

    return {
        "probs": probs,
        "mean_collisions": mean_collisions,
        "flux": flux,
        "bin_centers": bin_centers,
        "dx": dx,
    }


# ----------------------------------------------------------------------
# 2. DETERMINISTIC DIFFUSION-THEORY VALIDATION
# ----------------------------------------------------------------------

def diffusion_analytic(x, L, sigma_a, sigma_tr, S=1.0):
    """
    Analytic solution of  -D phi'' + Sigma_a phi = S  on [0, L]
    with phi = 0 at the extrapolated boundaries x = -d and x = L + d.

    phi(x) = (S / Sigma_a) * [1 - cosh((x - L/2 - d + d) ... ] -- solved below
    in closed form using the standard cosh construction.
    """
    D = 1.0 / (3.0 * sigma_tr)
    d = 0.7104 / sigma_tr  # linear extrapolation distance (vacuum BC)
    Lext = L + 2 * d
    kappa = np.sqrt(sigma_a / D)  # inverse diffusion length

    # Particular solution: S / Sigma_a (constant)
    # Homogeneous solution built to vanish at x = -d and x = L + d:
    # phi(x) = (S/Sigma_a) * [1 - cosh(kappa*(x - (L/2))) / cosh(kappa*(Lext/2))]
    x_shift = (x + d) - Lext / 2.0
    phi = (S / sigma_a) * (1.0 - np.cosh(kappa * x_shift) / np.cosh(kappa * Lext / 2.0))
    return np.clip(phi, 0.0, None)


def diffusion_fem_1d(L, sigma_a, sigma_tr, S=1.0, n_elements=200):
    """
    Solve  -D phi'' + Sigma_a phi = S  on [0, L] with linear 1D FEM
    (Galerkin method, linear hat basis functions), vacuum BC applied at
    extrapolated boundaries via a Robin-type reduction: here we solve on
    the extended domain [-d, L+d] with phi = 0 at both ends, then return
    the physical-domain slice.

    This uses the same FEM discretization approach (element stiffness +
    mass matrices, global assembly, linear solve) as the reaction-diffusion
    FEM model in the M.Sc. thesis.
    """
    D = 1.0 / (3.0 * sigma_tr)
    d = 0.7104 / sigma_tr
    x0, x1 = -d, L + d

    nodes = np.linspace(x0, x1, n_elements + 1)
    h = nodes[1] - nodes[0]
    n_nodes = n_elements + 1

    K = np.zeros((n_nodes, n_nodes))  # stiffness (diffusion) + reaction
    F = np.zeros(n_nodes)             # load vector

    # Element stiffness matrix for -D phi'':  (D/h) * [[1,-1],[-1,1]]
    k_stiff = (D / h) * np.array([[1.0, -1.0], [-1.0, 1.0]])
    # Element mass matrix for Sigma_a * phi: (Sigma_a*h/6) * [[2,1],[1,2]]
    k_mass = (sigma_a * h / 6.0) * np.array([[2.0, 1.0], [1.0, 2.0]])
    k_elem = k_stiff + k_mass
    f_elem = S * h / 2.0 * np.array([1.0, 1.0])  # constant source load vector

    for e in range(n_elements):
        i, j = e, e + 1
        K[np.ix_([i, j], [i, j])] += k_elem
        F[[i, j]] += f_elem

    # Dirichlet BC: phi(x0) = phi(x1) = 0
    for boundary_node in [0, n_nodes - 1]:
        K[boundary_node, :] = 0.0
        K[boundary_node, boundary_node] = 1.0
        F[boundary_node] = 0.0

    phi_nodes = np.linalg.solve(K, F)

    # Return only the physical-domain [0, L] slice
    mask = (nodes >= 0.0) & (nodes <= L)
    return nodes[mask], phi_nodes[mask]


# ----------------------------------------------------------------------
# 3. MAIN: run MC transport, compare against diffusion theory
# ----------------------------------------------------------------------

def main():
    # --- Problem parameters (typical light-water-like moderating medium) ---
    L = 10.0          # slab thickness, cm
    sigma_a = 0.02     # macroscopic absorption cross section, 1/cm
    sigma_s = 0.8      # macroscopic scattering cross section, 1/cm
    sigma_tr = sigma_a + sigma_s  # transport cross section (isotropic scattering approx)
    N = 200_000        # number of neutron histories

    print("=" * 70)
    print("1D Monte Carlo Neutron Transport  (Gillespie-style event-driven MC)")
    print("=" * 70)
    print(f"Slab thickness L        = {L} cm")
    print(f"Sigma_a (absorption)    = {sigma_a} 1/cm")
    print(f"Sigma_s (scattering)    = {sigma_s} 1/cm")
    print(f"Sigma_t (total)         = {sigma_a + sigma_s} 1/cm")
    print(f"Number of histories N   = {N:,}")
    print("-" * 70)

    result = run_ensemble(N, L, sigma_a, sigma_s, source_mode="beam", n_bins=100)

    p = result["probs"]
    print(f"Transmission probability : {p['transmitted']:.4f}")
    print(f"Reflection probability   : {p['reflected']:.4f}")
    print(f"Absorption probability   : {p['absorbed']:.4f}")
    print(f"Mean collisions/history  : {result['mean_collisions']:.3f}")
    print("-" * 70)

    # --- Deterministic diffusion-theory comparison ---
    # For an isotropic, uniformly-distributed volumetric source with the
    # same total emission rate as the MC beam source (S normalized to 1
    # neutron per unit time per unit area entering at x=0), we instead
    # validate on a companion problem: uniform volumetric source S=1.
    x_fine = np.linspace(0, L, 200)
    phi_analytic = diffusion_analytic(x_fine, L, sigma_a, sigma_tr, S=1.0)
    x_fem, phi_fem = diffusion_fem_1d(L, sigma_a, sigma_tr, S=1.0, n_elements=200)

    # Run a second MC ensemble with a *uniform volumetric source* to match
    # the diffusion-theory boundary-value problem exactly (birth points
    # sampled uniformly in [0, L], isotropic direction).
    result_vol = run_ensemble_volumetric_source(
        N=N, L=L, sigma_a=sigma_a, sigma_s=sigma_s, n_bins=100, seed=7
    )

    rel_err = np.abs(
        np.interp(result_vol["bin_centers"], x_fem, phi_fem) - result_vol["flux"]
    )
    valid = result_vol["flux"] > 1e-6
    mean_rel_err = np.mean(
        rel_err[valid] / np.interp(result_vol["bin_centers"], x_fem, phi_fem)[valid]
    )
    print(f"MC vs FEM diffusion-theory mean relative error (volumetric-source "
          f"problem): {100 * mean_rel_err:.2f}%")
    print("=" * 70)

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.bar(result["bin_centers"], result["flux"], width=result["dx"] * 0.9,
           alpha=0.5, label="MC track-length tally (beam source)", color="steelblue")
    ax.set_xlabel("x (cm)")
    ax.set_ylabel(r"$\phi(x)$  (flux, arb. units)")
    ax.set_title("Beam source: MC flux profile")
    ax.legend()

    ax2 = axes[1]
    ax2.plot(x_fine, phi_analytic, "k-", lw=2, label="Diffusion theory (analytic)")
    ax2.plot(x_fem, phi_fem, "r--", lw=2, label="Diffusion theory (FEM)")
    ax2.bar(result_vol["bin_centers"], result_vol["flux"],
            width=result_vol["dx"] * 0.9, alpha=0.4,
            label="MC track-length tally", color="steelblue")
    ax2.set_xlabel("x (cm)")
    ax2.set_ylabel(r"$\phi(x)$")
    ax2.set_title("Uniform volumetric source: MC vs. diffusion theory")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("neutron_transport_flux.png", dpi=150)
    print("Saved plot: neutron_transport_flux.png")


def run_ensemble_volumetric_source(N, L, sigma_a, sigma_s, n_bins=100, seed=7):
    """Ensemble run with neutrons born uniformly in [0, L] with isotropic
    direction -- the source configuration matching the diffusion-theory
    boundary value problem solved analytically/FEM above."""
    rng = np.random.default_rng(seed)
    dx = L / n_bins
    flux_tally = np.zeros(n_bins)

    for _ in range(N):
        x0 = rng.uniform(0.0, L)
        mu0 = rng.uniform(-1.0, 1.0)
        run_single_history(L, sigma_a, sigma_s, rng, x0=x0, mu0=mu0,
                            flux_tally=flux_tally, n_bins=n_bins, dx=dx)

    # Source strength S=1 n/cm^3/s over volume L (unit area) means total
    # source rate = L. Each history represents L/N of that source. Flux
    # tally must be scaled accordingly (track length / (dx*N)) * (L / 1)
    # normalized so that total source = L * S with S=1 => scale by L.
    flux = (flux_tally / (dx * N)) * L
    bin_centers = (np.arange(n_bins) + 0.5) * dx
    return {"flux": flux, "bin_centers": bin_centers, "dx": dx}


if __name__ == "__main__":
    main()
