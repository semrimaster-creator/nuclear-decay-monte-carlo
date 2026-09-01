# 1D Monte Carlo Neutron Transport — extension of `nuclear-decay-monte-carlo`

## Why this belongs in the same repository

This module is not a new project bolted onto the decay simulation — it is
the **same computational machinery applied to a different rate equation**.

| | Radioactive decay (existing) | Neutron transport (this module) |
|---|---|---|
| Random variable sampled | time to next decay | distance to next collision |
| Sampling law | `Exponential(λ)` | `Exponential(Σt)` |
| Sampling formula | `t = -ln(1-U)/λ` | `s = -ln(1-U)/Σt` |
| Branching decision | (single channel: decay) | scatter vs. absorb, `P = Σs/Σt` |
| Ensemble step | average N decay trajectories | average N particle histories |
| Deterministic check | `N(t) = N0 e^{-λt}` | 1D diffusion equation, solved by FEM |

Both projects follow the same three-stage pattern used throughout Monte
Carlo transport codes (MCNP, Serpent, Geant4, OpenMC): **(1)** sample the
next event from an exponential law governed by a rate, **(2)** branch
between competing event types by relative rate, **(3)** validate the
stochastic result against the deterministic macroscopic equation it is
sampling from.

## What's implemented

1. **Event-driven MC transport** (`run_single_history`, `run_ensemble`) —
   free-flight sampling, isotropic scattering, absorption, leakage
   (reflection/transmission), track-length flux estimator.
2. **Deterministic validation**:
   - closed-form analytic solution of the 1-speed diffusion equation,
   - 1D linear Galerkin **FEM** solve of the same equation (identical
     discretization strategy — element stiffness/mass assembly, global
     linear solve — used for the reaction–diffusion PDE system in the
     M.Sc. thesis).
3. Result: MC and FEM/analytic flux profiles agree to ~2% in the bulk of
   the slab, with the expected transport-theory boundary-layer deviation
   within ~1 mean free path of the vacuum boundaries — a real physical
   effect (diffusion theory is only asymptotically correct far from
   boundaries/sources), not a numerical artifact.

## Planned extensions (next steps)

- **Criticality / k_eff**: add a fission cross section Σf and neutron
  multiplicity ν, track fission-neutron generations, and estimate k_eff by
  power iteration — this is an eigenvalue problem on the same footing as
  the eigenvalue/stability analysis already used for the thesis's
  Jacobian linearization, just applied to the neutron transport operator.
- **Nuclear structure**: numerical solution of the time-independent
  Schrödinger equation as a finite-difference/FEM eigenvalue problem —
  first for a square well and harmonic oscillator (analytic benchmarks),
  then for a Woods–Saxon potential to obtain single-particle shell-model
  energy levels. (Separate module, in progress.)

## Files

- `neutron_transport_1d.py` — full simulation + FEM validation + plotting.
- `neutron_transport_flux.png` — output figure (regenerated on each run).

## Run it

```bash
python3 neutron_transport_1d.py
```
