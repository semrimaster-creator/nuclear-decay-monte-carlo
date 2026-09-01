# Extension roadmap: `nuclear-decay-monte-carlo` → nuclear physics portfolio

This document ties together three new modules built as direct extensions
of the original Gillespie/binomial radioactive-decay Monte Carlo project,
following the three research directions targeted for PhD applications in
computational and theoretical nuclear physics: **transport**,
**reactions/criticality**, and **nuclear structure**.

## 1. `neutron_transport_1d.py` — Monte Carlo neutron transport (bridge: transport)

Event-driven MC transport of neutrons through a 1D slab, using the exact
same exponential free-flight sampling primitive as the original decay
code (`s = -ln(1-U)/Σt` in place of `t = -ln(1-U)/λ`), branching between
scattering and absorption, with a track-length flux estimator validated
against a deterministic 1-speed diffusion-theory solution (both closed-form
and 1D FEM). Agreement in the bulk of the slab: **~2.4%**, with the
expected transport-theory boundary-layer deviation near the vacuum
boundaries correctly reproduced.

*Debugging note kept intentionally in the README*: the first version of
the track-length estimator tallied the x-axis projection of each flight
segment instead of the true 3D path length, which differ by a factor
1/|μ| for oblique trajectories. This produced a ~52% low bias in the
flux; correcting it (dividing by |μ|) brought MC and diffusion theory
into ~2% agreement. This is exactly the kind of estimator subtlety that
shows up (and must be caught) in real Monte Carlo transport codes.

## 2. `keff_power_iteration.py` — k_eff and criticality (bridge: reactions/criticality)

Adds a fission channel (Σf, ν) to the transport physics and finds the
neutron multiplication factor k_eff by **Monte Carlo power iteration** —
the standard algorithm in MCNP/Serpent/OpenMC, and conceptually the same
"iterate an operator to its dominant eigenvalue" idea as the eigenvalue/
stability analysis already performed on the M.Sc. thesis's reaction-
diffusion Jacobian, now applied to a genuine transport eigenvalue problem.
Includes a bisection search for the **critical slab thickness** (k_eff = 1).

Sanity checks passed: k_eff for the finite slab (2.09) falls correctly
below the infinite-medium value k_∞ = νΣf/(Σa+Σf) = 2.19 (leakage
reduces multiplication, as physically required), and the power iteration
converges cleanly after a short settling transient.

## 3. `nuclear_structure_schrodinger.py` — nuclear structure (bridge: structure)

Solves the time-independent Schrödinger equation as a finite-difference
matrix eigenvalue problem (`scipy.linalg.eigh_tridiagonal`) — the same
"discretize, then call a linear-algebra eigensolver" strategy used
throughout the CV's stated ODE/eigenvalue skillset. Validated to <0.002%
against the analytic infinite square well and harmonic oscillator, then
applied to a Woods-Saxon nuclear mean-field potential (A=40) to obtain
single-particle energy levels. The resulting shell gaps at nucleon
numbers **2, 8, and 20** reproduce the first three shell-model magic
numbers correctly; the absence of the higher magic numbers (28, 50, 82,
126) without spin-orbit coupling is flagged explicitly as a known,
physically-understood limitation and the natural next step (adding a
Goeppert-Mayer/Jensen spin-orbit term).

*Debugging notes kept intentionally*: two bugs were caught during
testing — (1) a tridiagonal-matrix indexing error in the finite-difference
Hamiltonian assembly, and (2) a mass/ħ unit-convention mismatch between
the harmonic-oscillator potential and the kinetic-energy prefactor that
initially produced energies systematically high by a factor of √2. Both
were caught by the analytic benchmark checks before trusting the
Woods-Saxon results — illustrating why the validation-before-application
structure matters.

## What this collectively demonstrates for a PhD application

- The same core computational skill (stochastic sampling + ensemble
  averaging, validated against a deterministic/analytic reference) is
  shown to generalize across three distinct areas of nuclear physics:
  transport, reactor/criticality physics, and structure.
- The FEM and eigenvalue-problem machinery from the M.Sc. thesis
  (reaction-diffusion PDEs, Jacobian stability analysis) is shown to
  transfer directly to nuclear transport and quantum-mechanical
  eigenvalue problems — supporting the CV's stated "methodology
  transferability" claim with working code rather than assertion alone.
- Every numerical result is checked against an analytic or independent
  deterministic benchmark before being trusted, and discrepancies (all
  three modules had at least one non-trivial bug during development) were
  diagnosed and fixed rather than hidden — good research practice to
  demonstrate directly in a portfolio repository.

## Files in this delivery

| File | Purpose |
|---|---|
| `neutron_transport_1d.py` | MC neutron transport + diffusion-theory (analytic + FEM) validation |
| `neutron_transport_flux.png` | Output figure |
| `keff_power_iteration.py` | k_eff via MC power iteration + critical thickness search |
| `keff_convergence.png` | Output figure |
| `nuclear_structure_schrodinger.py` | Schrödinger eigenvalue solver: benchmarks + Woods-Saxon |
| `schrodinger_benchmarks.png`, `woods_saxon_levels.png` | Output figures |
| `README_neutron_transport.md` | Detailed notes on the transport module specifically |
| `README_project_extensions.md` | This file |

## Suggested next steps before using this in applications

1. Add a short **abstract/summary** (half a page) to the GitHub repo's
   main README stating the three-bridge structure above in your own
   words — committees read the README before the code.
2. Add unit tests (`pytest`) for the analytic benchmarks so the repo
   demonstrates software-engineering discipline, not just physics.
3. If time allows: implement the spin-orbit extension to the Woods-Saxon
   module to recover the higher magic numbers (28, 50, 82) — this would
   be a strong, self-contained "extra mile" addition specifically
   relevant to nuclear structure theory positions.
4. Cite the specific software this mirrors (MCNP/Serpent/OpenMC for
   transport, standard shell-model codes for structure) in the README to
   show awareness of the production tools in the field.
