"""
Monte Carlo Simulation of Radioactive Decay
=============================================

Physical model
---------------
Radioactive decay is a Markovian stochastic process: each of the N(t)
surviving nuclei at time t has the same, constant probability per unit
time of decaying, called the decay constant lambda. This is exactly the
memoryless (exponential) property that turns the population-level
description into a simple linear ODE:

    dN/dt = -lambda * N(t)

whose solution is the familiar exponential decay law:

    N(t) = N0 * exp(-lambda * t)

This script does NOT integrate that ODE directly. Instead, it simulates
the underlying *stochastic* process at the level of individual nuclei,
using a Monte Carlo method, and then shows that the resulting ensemble
average converges to the analytic ODE solution above. This is the same
bridge -- from a stochastic microscopic rule to a deterministic
macroscopic differential equation -- that appears throughout
computational nuclear physics and reactor kinetics (e.g. Monte Carlo
neutron transport codes such as MCNP/Serpent, which simulate individual
neutron histories rather than solving the transport equation directly).

Two complementary Monte Carlo approaches are implemented:

1. discrete-time (binomial) method:
   At each small time step dt, every surviving nucleus decays
   independently with probability p = 1 - exp(-lambda * dt). The number
   of decays in that step is Binomial(N(t), p). Simple, but the choice
   of dt introduces discretization error unless dt << 1/lambda.

2. exact event-driven (Gillespie-style) method:
   Rather than stepping in fixed dt, we draw the *exact* waiting time to
   the next decay event from an Exponential(rate = N(t) * lambda)
   distribution, decrement N by 1, and repeat. This is mathematically
   exact (no discretization error) and is the standard approach used in
   stochastic chemical/nuclear kinetics (the Gillespie algorithm).

Author: Mohamed Semri
"""

from __future__ import annotations
import argparse
import numpy as np
import matplotlib.pyplot as plt


def simulate_binomial(n0: int, decay_constant: float, t_max: float,
                       dt: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Discrete-time binomial Monte Carlo simulation of radioactive decay.

    At each step, decays ~ Binomial(N, p) with p = 1 - exp(-lambda*dt).

    Returns
    -------
    times : array of time points
    counts : array of surviving nuclei N(t) at each time point
    """
    n_steps = int(np.ceil(t_max / dt))
    times = np.linspace(0.0, n_steps * dt, n_steps + 1)
    counts = np.empty(n_steps + 1, dtype=np.int64)
    counts[0] = n0

    p = 1.0 - np.exp(-decay_constant * dt)  # exact per-step decay probability
    n_current = n0
    for i in range(1, n_steps + 1):
        if n_current > 0:
            decays = rng.binomial(n_current, p)
            n_current -= decays
        counts[i] = n_current

    return times, counts


def simulate_gillespie(n0: int, decay_constant: float, t_max: float,
                        rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Exact event-driven (Gillespie) Monte Carlo simulation.

    Waiting time to the next decay ~ Exponential(rate = N * lambda).
    No discretization error: every event time is exact.

    Returns
    -------
    times : array of event times (starts at 0, ends at t_max)
    counts : array of surviving nuclei N(t) immediately after each event
    """
    times = [0.0]
    counts = [n0]
    t = 0.0
    n_current = n0

    while n_current > 0:
        rate = n_current * decay_constant
        # Exact waiting time until the next single-nucleus decay event.
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


def run_ensemble(n_runs: int, n0: int, decay_constant: float, t_max: float,
                  dt: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Run the binomial simulation many times and average, to show
    convergence of the stochastic ensemble mean to the deterministic ODE
    solution (law of large numbers)."""
    times = None
    all_counts = []
    for _ in range(n_runs):
        t_arr, c_arr = simulate_binomial(n0, decay_constant, t_max, dt, rng)
        if times is None:
            times = t_arr
        all_counts.append(c_arr)
    return times, np.mean(all_counts, axis=0)


def main():
    parser = argparse.ArgumentParser(
        description="Monte Carlo simulation of radioactive decay "
                    "(binomial and Gillespie-exact methods)."
    )
    parser.add_argument("--n0", type=int, default=5000,
                        help="Initial number of nuclei (default: 5000)")
    parser.add_argument("--half-life", type=float, default=10.0,
                        help="Half-life in arbitrary time units (default: 10.0)")
    parser.add_argument("--t-max", type=float, default=60.0,
                        help="Total simulated time (default: 60.0)")
    parser.add_argument("--dt", type=float, default=0.1,
                        help="Time step for the binomial method (default: 0.1)")
    parser.add_argument("--ensemble-runs", type=int, default=200,
                        help="Number of repeated runs to average for the "
                            "ensemble-mean convergence plot (default: 200)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--out", type=str, default="decay_simulation.png",
                        help="Output plot filename")
    args = parser.parse_args()

    decay_constant = np.log(2.0) / args.half_life
    rng = np.random.default_rng(args.seed)

    # --- Run both Monte Carlo methods once each ---
    t_binom, n_binom = simulate_binomial(args.n0, decay_constant, args.t_max, args.dt, rng)
    t_gill, n_gill = simulate_gillespie(args.n0, decay_constant, args.t_max, rng)

    # --- Run the ensemble average to demonstrate convergence to the ODE ---
    t_ens, n_ens_mean = run_ensemble(args.ensemble_runs, args.n0, decay_constant,
                                      args.t_max, args.dt, rng)

    # --- Analytic reference curve ---
    t_fine = np.linspace(0, args.t_max, 500)
    n_analytic = analytic_solution(t_fine, args.n0, decay_constant)

    # --- Report a simple quantitative check ---
    final_analytic = analytic_solution(np.array([args.t_max]), args.n0, decay_constant)[0]
    final_ensemble = n_ens_mean[-1]
    rel_error = abs(final_ensemble - final_analytic) / final_analytic
    print(f"Decay constant (lambda):        {decay_constant:.5f}  [1/time unit]")
    print(f"Analytic N(t_max):              {final_analytic:.2f}")
    print(f"Ensemble-mean N(t_max):         {final_ensemble:.2f}  "
        f"(n_runs={args.ensemble_runs})")
    print(f"Relative error (ensemble vs analytic): {rel_error:.4%}")

    # --- Plot everything together ---
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(t_fine, n_analytic, "k-", lw=2.5,
            label=r"Analytic ODE solution: $N(t) = N_0 e^{-\lambda t}$")
    ax.plot(t_binom, n_binom, "o", ms=3, alpha=0.5, color="tab:blue",
            label="Single run — Binomial Monte Carlo")
    ax.step(t_gill, n_gill, where="post", color="tab:orange", lw=1.2, alpha=0.8,
            label="Single run — Exact (Gillespie) Monte Carlo")
    ax.plot(t_ens, n_ens_mean, "--", lw=2, color="tab:green",
            label=f"Ensemble mean over {args.ensemble_runs} Binomial runs")

    ax.set_xlabel("Time")
    ax.set_ylabel("Surviving nuclei, N(t)")
    ax.set_title("Radioactive Decay: Stochastic Monte Carlo vs. Analytic ODE Solution")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"\nPlot saved to: {args.out}")


if __name__ == "__main__":
    main()
