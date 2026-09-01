# Monte Carlo Simulation of Radioactive Decay

A small, self-contained project simulating radioactive decay at the level
of individual nuclei, written as a first step in transitioning from an
applied mathematics background into computational nuclear physics.

## Motivation

My academic background is in applied mathematics — differential equations,
finite element methods, and stochastic differential equations — applied
previously to disease-dynamics modeling. This project is a deliberate,
concrete step toward nuclear physics: it takes the same mathematical
machinery (stochastic processes, ODEs, ensemble averaging) and points it
directly at a nuclear physics problem.

## The physics

Radioactive decay is memoryless: every surviving nucleus has the same
constant probability per unit time, `λ` (the decay constant), of decaying,
independent of how long it has already existed. At the population level,
this leads to the standard exponential decay law via the linear ODE

```
dN/dt = -λ N(t)      ⇒      N(t) = N₀ e^(-λt)
```

That equation is usually where a physics course stops. This project goes
one level deeper and asks: *what microscopic, individual-nucleus rule
produces that macroscopic equation, and how do we simulate it directly?*
This is exactly the question behind Monte Carlo neutron transport codes
(MCNP, Serpent, Geant4) used throughout nuclear engineering and reactor
physics — they simulate individual particle histories rather than solving
the transport equation in closed form.

## Two Monte Carlo methods implemented

1. **Binomial (discrete-time) method** — at each small time step `dt`,
   every surviving nucleus independently decays with probability
   `p = 1 - e^(-λ dt)`. Simple and intuitive, but introduces a small
   discretization error unless `dt ≪ 1/λ`.

2. **Gillespie (exact, event-driven) method** — instead of stepping in
   fixed `dt`, the waiting time to the *next* decay event is drawn exactly
   from an `Exponential(rate = N(t) · λ)` distribution. This has no
   discretization error and is the standard algorithm used in stochastic
   chemical and nuclear kinetics.

The script also runs an **ensemble average** over many independent
binomial simulations to demonstrate the law of large numbers: as the
number of runs grows, the stochastic mean converges to the deterministic
analytic ODE solution — the same convergence argument that justifies using
Monte Carlo methods as a numerical proxy for solving transport/kinetics
equations in the first place.

## Usage

```bash
pip install numpy matplotlib
python monte_carlo_decay.py --n0 5000 --half-life 10 --t-max 60 --ensemble-runs 200
```

Key arguments:

| Argument | Meaning | Default |
|---|---|---|
| `--n0` | Initial number of nuclei | 5000 |
| `--half-life` | Half-life, in arbitrary time units | 10.0 |
| `--t-max` | Total simulated time | 60.0 |
| `--dt` | Time step for the binomial method | 0.1 |
| `--ensemble-runs` | Number of repeated runs averaged for convergence | 200 |
| `--seed` | Random seed (reproducibility) | 42 |
| `--out` | Output plot filename | `decay_simulation.png` |

The script prints a quantitative comparison (decay constant, analytic vs.
ensemble-mean final population, relative error) and saves a plot overlaying:

- the analytic solution,
- one run of the binomial method,
- one run of the exact Gillespie method, and
- the ensemble mean over many binomial runs.

## Example output

With the default parameters, a single run typically shows the
ensemble-mean final population within well under 1% of the analytic
prediction — see `decay_simulation.png`.

## Version 2: real isotopes, statistics, and vectorization

`monte_carlo_decay_v2.py` extends the original script in three ways:

1. **Real isotope presets** — Am-241 (alpha), Cs-137 (beta), Co-60 and I-131
   (beta+gamma), each with an actual half-life and characteristic emitted
   energy in MeV (see `ISOTOPE_PRESETS` in the code), instead of an
   abstract, unitless half-life.
2. **Ensemble statistics and cumulative energy** — the ensemble simulation
   now reports the standard deviation of N(t) alongside the mean, plotted
   as a ±1σ band, and checked numerically against the analytic result for
   a binomial decay process: `Var[N(t)] = N0 · p(t) · (1 − p(t))`, where
   `p(t) = e^(−λt)` is the single-nucleus survival probability. A second
   panel plots the cumulative energy released, `E(t) = (N0 − N(t)) ×
   (energy per decay)`.
3. **Vectorized ensemble simulation** — the *time* loop cannot be removed
   (N(t) depends sequentially on N(t−dt)), but the loop over independent
   ensemble runs has been eliminated: all trajectories are advanced
   together via a single vectorized NumPy binomial draw per time step,
   `rng.binomial(n_current_array, p)`, rather than an outer Python loop
   calling the single-run function `n_runs` times.

Run it with, for example:

```bash
python monte_carlo_decay_v2.py --isotope cs-137
python monte_carlo_decay_v2.py --isotope am-241 --ensemble-runs 500
python monte_carlo_decay_v2.py --isotope none --half-life 12 --time-unit hours --decay-mode beta --energy-mev 0.8
```

## Possible extensions (next steps)

- Add a **branching decay chain** (parent → daughter → granddaughter),
  which turns the single ODE into a coupled linear system — a natural
  bridge to Bateman-equation-style problems in nuclear engineering.
- Replace the exponential decay law with a simple **two-level nuclear
  reaction network** and study equilibrium (secular/transient) behavior.
- Port the core event loop to a compiled backend (Numba/C++) to scale to
  realistic nuclide inventories, as used in production Monte Carlo burnup
  codes.
  <!--
  Paste this section near the top of the repository's main README.md,
  above or in place of the current single-project description. Written
  to be read by a PhD admissions committee in under a minute.
-->

## From radioactive decay to nuclear transport, criticality, and structure

This repository began as a Monte Carlo simulation of radioactive decay
(binomial and Gillespie/event-driven methods, validated against the
analytic decay law). It has since been extended along the three research
directions listed in my research profile: **nuclear transport**,
**nuclear reactions/criticality**, and **nuclear structure**. Each
extension reuses the same core computational pattern — sample a
stochastic event from an exponential law governed by a physical rate,
branch between competing outcomes, average over an ensemble of
histories, and validate the result against an independent deterministic
or analytic solution — applied to a different physical equation each
time.

| Module | Physics | Stochastic sampling | Deterministic validation |
|---|---|---|---|
| `decay_simulation.py` *(original)* | Radioactive decay | `t = -ln(1-U)/λ` | `N(t) = N0 e^{-λt}` |
| `neutron_transport_1d.py` | 1D neutron transport | `s = -ln(1-U)/Σt` | 1-speed diffusion equation (analytic + FEM) |
| `keff_power_iteration.py` | Criticality / k_eff | fission-neutron power iteration | k_eff < k_∞ (leakage bound); k_eff(L) monotonicity |
| `nuclear_structure_schrodinger.py`, `woods_saxon_spin_orbit.py` | Nuclear structure | *(deterministic eigenvalue problem, not stochastic)* | square well & harmonic oscillator analytic energies; correct N=2,8,20,28 shell gaps |

**Validation discipline:** every module includes an automated test suite
(`test_physics.py`, runnable with `pytest`) enforcing these checks, and
every module's development history includes at least one caught-and-fixed
bug (documented in `README_project_extensions.md`) — a track-length
estimator that initially used the wrong path-length convention, a
tridiagonal-matrix indexing error, a ħ²/2m unit-convention mismatch, and a
spin-orbit sign error that initially inverted the j = l ± 1/2 level
ordering. I've kept these in the documentation deliberately: catching
this class of error against an analytic or physical benchmark, rather
than trusting a first working run, is the habit I most want a committee
to see demonstrated in code.

See `README_project_extensions.md` for the full technical writeup of each
module, including the specific physics being tested and the exact
mechanism of each bug that was caught during development.

## Author

Mohamed Semri — semri.master@gmail.com
