"""
Monte Carlo Simulation of Radioactive Decay (v2)
==================================================

Physical model
---------------
Radioactive decay is a Markovian stochastic process: each of the N(t)
surviving nuclei at time t has the same, constant probability per unit
time of decaying, called the decay constant lambda = ln(2) / T_(1/2). This
memoryless property leads to the population-level linear ODE

    dN/dt = -lambda * N(t)      =>      N(t) = N0 * exp(-lambda * t)

This script simulates the underlying *stochastic* process at the level of
individual nuclei via Monte Carlo, and shows that the ensemble average
converges to the analytic ODE solution above (law of large numbers). This
is the same bridge -- stochastic microscopic rule -> deterministic
macroscopic equation -- that justifies Monte Carlo neutron transport codes
(MCNP, Serpent, Geant4) used throughout nuclear engineering.

What's new in v2 (compared to v1)
----------------------------------
1. Real isotope presets (half-life, decay mode, and emitted-particle
   energy in MeV) for Am-241 (alpha), Cs-137 (beta), and Co-60 (gamma,
   via beta decay followed by gamma emission), so the simulation
   represents an actual radionuclide rather than an abstract N0/lambda.
2. Cumulative released energy E(t) = (number of decays so far) x
   (energy per decay, MeV), plotted alongside N(t) on a secondary axis.
3. Ensemble statistics: in addition to the mean, the sample standard
   deviation across runs is computed at every time point and shown as a
   shaded +/-1 sigma confidence band, with a numerical check against the
   analytic binomial-process variance, Var[N(t)] = N0 p(t) (1 - p(t)),
   where p(t) = exp(-lambda t) is the survival probability of a single
   nucleus.
4. Vectorized ensemble simulation: the discrete-time (binomial) method is
   still inherently *sequential in time* -- N(t) depends on N(t-dt), so
   the time loop cannot be removed -- but the loop over independent
   ensemble runs has been eliminated. All `n_runs` trajectories are now
   advanced together at each time step using a single vectorized NumPy
   binomial draw over an array of shape (n_runs,), instead of an outer
   Python for-loop calling the single-run function n_runs times.

Author: Mohamed Semri
"""

from __future__ import annotations
import argparse
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Real isotope presets: (half_life, unit, decay_mode, energy_MeV, note)
# Half-life is stored in the unit given; energy_MeV is the characteristic
# emitted-particle (or photon) energy per decay event.
# Sources: standard nuclear data tables (e.g. IAEA Nuclear Data Services).
# ---------------------------------------------------------------------------
ISOTOPE_PRESETS = {
    "am-241": {
        "half_life": 432.2, "unit": "years",
        "decay_mode": "alpha", "energy_mev": 5.486,
        "note": "Am-241 -> Np-237 + alpha; smoke-detector source.",
    },
    "cs-137": {
        "half_life": 30.17, "unit": "years",
        "decay_mode": "beta", "energy_mev": 0.512,
        "note": "Cs-137 -> Ba-137m + beta- (average beta energy).",
    },
    "co-60": {
        "half_life": 5.27, "unit": "years",
        "decay_mode": "beta+gamma", "energy_mev": 1.17 + 1.33,
        "note": "Co-60 -> Ni-60 + beta-, followed by two gamma photons "
                "(1.17 and 1.33 MeV) -- energies summed here.",
    },
    "i-131": {
        "half_life": 8.02, "unit": "days",
        "decay_mode": "beta+gamma", "energy_mev": 0.606 + 0.364,
        "note": "I-131 -> Xe-131 + beta- + gamma (medical isotope).",
    },
}


def simulate_binomial(n0: int, decay_constant: float, t_max: float,
                       dt: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Discrete-time binomial Monte Carlo simulation of radioactive decay
    for a SINGLE run. decays ~ Binomial(N, p), p = 1 - exp(-lambda*dt)."""
    n_steps = int(np.ceil(t_max / dt))
    times = np.linspace(0.0, n_steps * dt, n_steps + 1)
    counts = np.empty(n_steps + 1, dtype=np.int64)
    counts[0] = n0

    p = 1.0 - np.exp(-decay_constant * dt)
    n_current = n0
    for i in range(1, n_steps + 1):
        if n_current > 0:
            decays = rng.binomial(n_current, p)
            n_current -= decays
        counts[i] = n_current

    return times, counts


def simulate_gillespie(n0: int, decay_constant: float, t_max: float,
                        rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Exact event-driven (Gillespie) Monte Carlo simulation. Waiting time
    to the next decay ~ Exponential(rate = N * lambda). No discretization
    error: every event time is exact.

    Note on vectorization: this algorithm is inherently sequential -- the
    rate of the next event depends on the outcome (N) of the previous one
    -- so it cannot be vectorized over time within a single trajectory.
    (Vectorizing *across* many independent Gillespie trajectories at once
    is possible in principle via approximate methods such as tau-leaping,
    but that trades exactness for speed and is left as a possible
    extension -- see README.)
    """
    times = [0.0]
    counts = [n0]
    t = 0.0
    n_current = n0

    while n_current > 0:
        rate = n_current * decay_constant
        waiting_time = rng.exponential(1.0 / rate)
        t += waiting_time
        if t > t_max:
            break
        n_current -= 1
        times.append(t)
        counts.append(n_current)

    times.append(t_max)
    counts.append(n_current)
    return np.array(times), np.array(counts)


def analytic_solution(t: np.ndarray, n0: int, decay_constant: float) -> np.ndarray:
    """Closed-form solution of dN/dt = -lambda * N,  N(0) = n0."""
    return n0 * np.exp(-decay_constant * t)


def analytic_std(t: np.ndarray, n0: int, decay_constant: float) -> np.ndarray:
    """Analytic standard deviation of N(t) under the exact binomial decay
    process: each of the N0 initial nuclei independently survives to time
    t with probability p(t) = exp(-lambda t). N(t) is therefore exactly
    Binomial(N0, p(t)), whose variance is N0 * p(t) * (1 - p(t))."""
    p = np.exp(-decay_constant * t)
    return np.sqrt(n0 * p * (1.0 - p))


def run_ensemble_vectorized(n_runs: int, n0: int, decay_constant: float,
                            t_max: float, dt: float,
                            rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized ensemble simulation of the binomial method.

    Instead of looping n_runs times over the single-run function (as in
    v1), all n_runs trajectories are advanced together: at each time step,
    a single call to rng.binomial(n_current_array, p) draws the number of
    decays for *all* runs at once, since NumPy's binomial sampler accepts
    an array of trial counts n. The Python loop that remains is only over
    time steps, which is unavoidable because N(t) depends on N(t-dt).

    Returns
    -------
    times : array of time points, shape (n_steps+1,)
    mean_counts : ensemble mean of N(t) at each time point
    std_counts : ensemble standard deviation of N(t) at each time point
    """
    n_steps = int(np.ceil(t_max / dt))
    times = np.linspace(0.0, n_steps * dt, n_steps + 1)
    p = 1.0 - np.exp(-decay_constant * dt)

    # Shape (n_runs,): current population of each independent trajectory.
    n_current = np.full(n_runs, n0, dtype=np.int64)

    mean_counts = np.empty(n_steps + 1)
    std_counts = np.empty(n_steps + 1)
    mean_counts[0] = n0
    std_counts[0] = 0.0

    for i in range(1, n_steps + 1):
        # Vectorized binomial draw across all n_runs trajectories at once.
        decays = rng.binomial(n_current, p)
        n_current = n_current - decays
        mean_counts[i] = n_current.mean()
        std_counts[i] = n_current.std(ddof=1) if n_runs > 1 else 0.0

    return times, mean_counts, std_counts


def resolve_physical_parameters(args) -> tuple[float, str, str, float, str]:
    """Resolve half-life (in the chosen time unit), decay mode, and
    per-decay energy, either from an isotope preset or from manual
    command-line overrides."""
    if args.isotope is not None:
        preset = ISOTOPE_PRESETS[args.isotope]
        half_life = preset["half_life"]
        unit = preset["unit"]
        decay_mode = preset["decay_mode"]
        energy_mev = preset["energy_mev"]
        note = preset["note"]
    else:
        half_life = args.half_life
        unit = args.time_unit
        decay_mode = args.decay_mode
        energy_mev = args.energy_mev
        note = "Manually specified parameters (no isotope preset)."
    return half_life, unit, decay_mode, energy_mev, note


def main():
    parser = argparse.ArgumentParser(
        description="Monte Carlo simulation of radioactive decay for a "
                    "real or user-specified isotope (v2: statistics, "
                    "decay energy, vectorized ensemble)."
    )
    parser.add_argument("--isotope", type=str, default="cs-137",
                        choices=list(ISOTOPE_PRESETS.keys()) + ["none"],
                        help="Isotope preset to use (default: cs-137). "
                            "Pass 'none' to specify parameters manually.")
    parser.add_argument("--n0", type=int, default=5000,
                        help="Initial number of nuclei (default: 5000)")
    parser.add_argument("--half-life", type=float, default=10.0,
                        help="Manual half-life (used only if --isotope none)")
    parser.add_argument("--time-unit", type=str, default="time units",
                        help="Manual time unit label (used only if --isotope none)")
    parser.add_argument("--decay-mode", type=str, default="unspecified",
                        help="Manual decay mode label (used only if --isotope none)")
    parser.add_argument("--energy-mev", type=float, default=1.0,
                        help="Manual energy per decay in MeV (used only if --isotope none)")
    parser.add_argument("--t-max-half-lives", type=float, default=6.0,
                        help="Total simulated time, in units of half-lives (default: 6.0)")
    parser.add_argument("--dt-fraction", type=float, default=0.01,
                        help="Time step as a fraction of the half-life (default: 0.01)")
    parser.add_argument("--ensemble-runs", type=int, default=200,
                        help="Number of vectorized ensemble trajectories (default: 200)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--out", type=str, default="decay_simulation_v2.png",
                        help="Output plot filename")
    args = parser.parse_args()

    if args.isotope == "none":
        args.isotope = None

    half_life, unit, decay_mode, energy_mev, note = resolve_physical_parameters(args)
    decay_constant = np.log(2.0) / half_life
    t_max = args.t_max_half_lives * half_life
    dt = args.dt_fraction * half_life
    rng = np.random.default_rng(args.seed)

    # --- Single runs of both Monte Carlo methods ---
    t_binom, n_binom = simulate_binomial(args.n0, decay_constant, t_max, dt, rng)
    t_gill, n_gill = simulate_gillespie(args.n0, decay_constant, t_max, rng)

    # --- Vectorized ensemble: mean AND standard deviation ---
    t_ens, n_ens_mean, n_ens_std = run_ensemble_vectorized(
        args.ensemble_runs, args.n0, decay_constant, t_max, dt, rng
    )

    # --- Analytic reference curves ---
    t_fine = np.linspace(0, t_max, 500)
    n_analytic = analytic_solution(t_fine, args.n0, decay_constant)
    std_analytic_at_ens = analytic_std(t_ens, args.n0, decay_constant)

    # --- Cumulative energy released, from the ensemble mean trajectory ---
    decays_so_far = args.n0 - n_ens_mean
    energy_released_mev = decays_so_far * energy_mev

    # --- Quantitative checks ---
    final_analytic = analytic_solution(np.array([t_max]), args.n0, decay_constant)[0]
    final_ensemble = n_ens_mean[-1]
    rel_error = abs(final_ensemble - final_analytic) / max(final_analytic, 1e-9)

    final_std_sim = n_ens_std[-1]
    final_std_analytic = std_analytic_at_ens[-1]
    std_rel_error = (abs(final_std_sim - final_std_analytic) / final_std_analytic
                    if final_std_analytic > 0 else float("nan"))

    print(f"Isotope:                         {args.isotope or 'custom'}  ({note})")
    print(f"Decay mode:                      {decay_mode}")
    print(f"Energy per decay:                {energy_mev:.3f} MeV")
    print(f"Half-life:                       {half_life:g} {unit}")
    print(f"Decay constant (lambda):         {decay_constant:.6g}  [1/{unit}]")
    print(f"Simulated time span:             {t_max:.3g} {unit}  "
        f"({args.t_max_half_lives:g} half-lives)")
    print()
    print(f"Analytic N(t_max):               {final_analytic:.2f}")
    print(f"Ensemble-mean N(t_max):          {final_ensemble:.2f}  "
        f"(n_runs={args.ensemble_runs})")
    print(f"Relative error (mean):           {rel_error:.4%}")
    print()
    print(f"Analytic std N(t_max):           {final_std_analytic:.2f}")
    print(f"Ensemble std N(t_max):           {final_std_sim:.2f}")
    print(f"Relative error (std dev):        {std_rel_error:.4%}")
    print()
    print(f"Total energy released by t_max:  {energy_released_mev[-1]:.2f} MeV "
        f"(ensemble mean, {decays_so_far[-1]:.1f} decays)")

    # --- Plot: population (with 1-sigma band) + cumulative energy ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 9), sharex=True)

    ax1.plot(t_fine, n_analytic, "k-", lw=2.5,
            label=r"Analytic ODE solution: $N(t) = N_0 e^{-\lambda t}$")
    ax1.plot(t_binom, n_binom, "o", ms=2.5, alpha=0.4, color="tab:blue",
            label="Single run — Binomial Monte Carlo")
    ax1.step(t_gill, n_gill, where="post", color="tab:orange", lw=1.1, alpha=0.8,
            label="Single run — Exact (Gillespie) Monte Carlo")
    ax1.plot(t_ens, n_ens_mean, "--", lw=2, color="tab:green",
            label=f"Ensemble mean ({args.ensemble_runs} vectorized runs)")
    ax1.fill_between(t_ens, n_ens_mean - n_ens_std, n_ens_mean + n_ens_std,
                    color="tab:green", alpha=0.2,
                    label=r"Ensemble $\pm 1\sigma$ (simulated)")
    ax1.set_ylabel("Surviving nuclei, N(t)")
    ax1.set_title(f"Radioactive Decay of {args.isotope.upper() if args.isotope else 'Custom Isotope'} "
                f"({decay_mode}, {energy_mev:.3f} MeV/decay): Monte Carlo vs. Analytic")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.plot(t_ens, energy_released_mev, color="tab:red", lw=2,
            label="Cumulative energy released (ensemble mean)")
    ax2.set_xlabel(f"Time [{unit}]")
    ax2.set_ylabel("Cumulative energy released [MeV]")
    ax2.legend(loc="lower right", fontsize=8)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"\nPlot saved to: {args.out}")


if __name__ == "__main__":
    main()
