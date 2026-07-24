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

## Possible extensions (next steps)

- Add a **branching decay chain** (parent → daughter → granddaughter),
  which turns the single ODE into a coupled linear system — a natural
  bridge to Bateman-equation-style problems in nuclear engineering.
- Replace the exponential decay law with a simple **two-level nuclear
  reaction network** and study equilibrium (secular/transient) behavior.
- Port the core event loop to a compiled backend (Numba/C++) to scale to
  realistic nuclide inventories, as used in production Monte Carlo burnup
  codes.

## Author

Mohamed Semri — semri.master@gmail.com
