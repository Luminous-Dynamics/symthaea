# Intertwined / Multicomponent Superconducting Order

**Contract:** SC-INTERTWINED-ORDER-0.1  
**Status:** synthetic zero-momentum multicomponent static GL evaluator; finite-Q PDW dynamics and stiffness are not evaluated  
**Tracking issue:** https://github.com/Luminous-Dynamics/symthaea/issues/12826

## Why this model family exists

A one-component complex GL field cannot represent every superconducting state. Some relevant systems require several band/orbital/sublattice or symmetry-labelled condensates, interband pair amplitudes, relative-phase locking, and secondary charge/nematic order. Finite-momentum pair-density-wave (PDW) components require still more explicit wave-vector and symmetry conventions.

Source-grounded examples:
- Kong et al., *Cooper-pair density modulation state in an iron-based superconductor*, Nature 640, 55–61 (2025), DOI https://doi.org/10.1038/s41586-025-08703-x. The work reports a lattice-periodic pair-gap modulation in FeTe0.55Se0.45, interpreted in relation to Fe-sublattice asymmetry and nematic distortion in exfoliated flakes. A lattice-periodic modulation is not, by itself, proof of a finite-Q PDW.
- Baldelli et al., *Fragmented superconductivity in the Hubbard model as solitons in Ginzburg–Landau theory*, npj Quantum Materials 10, 22 (2025), DOI https://doi.org/10.1038/s41535-024-00718-3. DMRG/theory study of fragmented pairing in a charge-ordered background; not a universal experimental GL model.
- Samokhin, *Ginzburg-Landau energy of multiband superconductors with interband pairing*, Phys. Rev. B 109, 134508 (2024), DOI https://doi.org/10.1103/PhysRevB.109.134508. A two-band model admits two intraband and an interband pair condensate; phase-sensitive terms can support time-reversal-breaking states.
- Wang et al., *Anomalous superfluid density in pair-density-wave superconductors*, npj Quantum Materials 11, 13 (2026), DOI https://doi.org/10.1038/s41535-025-00843-7. Their calculation finds a broad PDW parameter regime with negative superfluid density, an instability signal. This motivates a stiffness-positivity hard gate.
- Lesser et al., *Emblems of pair density waves: Dual identity of topological defects and their transport signatures*, Phys. Rev. B 113, 214505 (2026), DOI https://doi.org/10.1103/1gyj-r5fr. Fractional-vorticity defects are proposed theoretically for a pure 2D PDW; do not treat them as universal experimentally established defects.

These papers motivate multiple model classes. They do not establish one universal mechanism or make every spatial gap modulation a PDW.

## Component basis and common gauge

Represent the state as a vector of symmetry-labelled pair fields:

\[
\boldsymbol\Psi_i=(\psi_{i1},\ldots,\psi_{iN}), \qquad
\psi_{ia}=\rho_{ia}e^{i\phi_{ia}}.
\]

Each component carries:
- a stable component ID;
- a role (intraband, interband, uniform, finite-Q PDW, sublattice pair-density modulation, or another symmetry component);
- a band/orbital identity and symmetry label;
- a declared pair momentum \(\mathbf Q_a\) and form factor;
- the same Cooper-pair electromagnetic gauge charge.

All charged components transform under one common local electromagnetic U(1):

\[
\psi_{ia}'=e^{i\chi_i}\psi_{ia},
\qquad
A_{ij}'=A_{ij}+\chi_i-\chi_j.
\]

Relative phases are physical. Independently rotating one component is not an electromagnetic gauge transformation when intercomponent couplings are present. A wrong independent-component transformation is therefore an explicit negative control.

## Static functional represented by the first evaluator

The current zero-momentum evaluator implements the declared terms:

\[
F=F_2+F_{4,\mathrm{self}}+F_{4,\mathrm{density}}
+F_{\mathrm{lock}}+F_{\mathrm{secondary}}+F_{\mathrm{gradient}}.
\]

Quadratic mixing:

\[
F_2=\sum_i\sum_{ab} A_{ab}\,
\operatorname{Re}(\psi_{ia}^*\psi_{ib}),
\]

with real symmetric \(A\). It need not be positive definite; negative quadratic eigenvalues can drive ordering.

Self and density quartic terms:

\[
F_{4}=\sum_i\left[
\sum_a\frac{\beta_a}{2}|\psi_{ia}|^4
+\sum_{a<b}\eta_{ab}|\psi_{ia}|^2|\psi_{ib}|^2
\right].
\]

The initial evaluator requires the associated quartic form in the nonnegative amplitude-squared variables to be positive semidefinite. This is a sufficient boundedness condition for this restricted quartic functional, not a necessary condition for every GL theory. Models with stabilizing sixth-order terms or more general copositive but non-PSD quartic forms need a new contract.

Phase locking:

\[
F_{\mathrm{lock}}=-2\sum_i\sum_{(a,b)}J_{ab}
\operatorname{Re}(\psi_{ia}^*\psi_{ib}).
\]

Intercomponent gradient energy on oriented lattice links:

\[
F_{\mathrm{gradient}}=
\sum_{(ij),\mu}\sum_{ab}K_{ab}^{(\mu)}
\operatorname{Re}\left[
(\psi_{ia}-e^{iA_{ij}}\psi_{ja})^*
(\psi_{ib}-e^{iA_{ij}}\psi_{jb})
\right].
\]

Each gradient matrix is required to be real symmetric positive semidefinite. This prevents the declared gradient form from lowering energy without bound along a component-gradient direction.

A real secondary order field \(\rho_{ki}\), such as a charge or nematic modulation, can couple to component density in the current schema:

\[
F_{\mathrm{secondary}}=
\sum_{i,k,a}\lambda_{ka}\rho_{ki}|\psi_{ia}|^2.
\]

The symmetry label and coupling scope are preserved in the input. The evaluator currently applies only this declared density coupling; it does not infer a symmetry-allowed coupling from a field name.

## Implemented restricted evaluator

Binary: `sc_intertwined_order_audit` in `tools/superconductivity-intertwined-order-harness`.

The schema and synthetic 2×2 fixture declare three zero-momentum components:
- band-A intraband pair;
- band-B intraband pair;
- A–B interband pair.

The fixture also includes intercomponent quadratic and gradient mixing, quartic density couplings, phase locking, a spatially varying charge-order proxy, and oriented periodic lattice links.

The Rust evaluator:
- validates schema, unique component/site/edge IDs, site coverage and field-to-component identity;
- checks oriented unit-bond geometry and periodic-image shifts;
- rejects non-finite coefficients/fields and unsupported dimensions;
- rejects nonzero component momentum rather than silently evaluating a PDW as a uniform condensate;
- checks symmetry of quadratic/gradient matrices;
- checks gradient-matrix positive semidefiniteness;
- checks the sufficient quartic-amplitude boundedness condition;
- evaluates each declared energy contribution separately;
- tests the common local U(1) transformation on all components and all link phases;
- includes a negative control showing an independent rotation of one component is not a gauge symmetry;
- checks that simultaneous component-basis permutation and tensor permutation preserve physical energy;
- records exact schema and fixture SHA-256 hashes.

## Implemented thermal pilot and remaining qualification gaps

A separate `--monte-carlo` path now runs finite-size single-component-site Metropolis proposals over the full multicomponent lattice field. Each proposal changes one complex component at one site and uses an exact local energy difference comprising the changed site's quadratic/quartic/phase-lock/secondary-order terms and every incident intercomponent gradient link. A regression test compares this local \(\Delta F\) against full-lattice recomputation for every site/component in nonuniform fields and nonzero gauge links.

The sampler supports deterministic multi-chain seeds and records per-chain:
- acceptance / invalid proposal counts, initial/final energy and raw measured configurations;
- component-resolved mean \(|\psi_a|\) and \(|\psi_a|^2\);
- relative-phase-lock cosine and sine statistics (both invariant under the common electromagnetic U(1));
- quadratic, quartic, gradient and total energy;
- heat-capacity fluctuation point estimate;
- twist-current means, diamagnetic tensor, current covariance and the thermal helicity-modulus point-estimate tensor;
- the minimum eigenvalue of the symmetric stiffness point-estimate tensor;
- block-jackknife standard errors for the helicity-modulus tensor and minimum eigenvalue when at least 20 measurements are available.

The stiffness interpretation is deliberately conservative:
- a negative point estimate or approximate interval with upper endpoint below zero prohibits a stable-superconductor claim for the pilot result;
- a nonnegative point estimate alone is not a pass;
- if the chain is too short, the jackknife uncertainty is `null` and status is `insufficient_samples_for_stiffness_jackknife`;
- when available, the approximate 95% jackknife interval describes finite-chain resampling variability only.

The current CI smoke uses two chains, two warmup sweeps and four measurement sweeps. This is a software plumbing test only and intentionally does not emit a stiffness error bar. The sampler uses a full energy recomputation at every measurement but local \(\Delta F\) for each proposal. It does not yet implement convergence/thermalization criteria, independent-start diagnostics, replica exchange, HMC, autocorrelation-aware blocked analysis beyond the initial block jackknife, or finite-size scaling.

Still not implemented or scientifically qualified:
- finite-Q PDW real-space phase factors, wave-vector stars, commensurability and symmetry-derived coupling selection;
- free-energy minimization and phase diagrams;
- thermodynamic-limit defect energetics and constrained-vortex thermodynamic integration;
- Leggett / relative-phase dynamics, time-reversal symmetry classification and conductivity;
- microscopic derivation/calibration of coefficients from a material or ab initio calculation.

The input declares that a physically labelled superconducting state must have a positive-semidefinite superfluid-stiffness tensor and that a negative eigenvalue is a hard stability gate. The thermal pilot computes only a short-chain point estimate; it is not yet a convergence-qualified stability verdict and must not be interpreted as passing the physical gate.

Finite-Q components are accepted by the schema for provenance and future work, but the initial evaluator rejects nonzero \(\mathbf Q\). The next PDW implementation must define the lattice convention for \(e^{i\mathbf Q\cdot\mathbf r}\), component translation/symmetry transformations, common-gauge-covariant gradients, and allowed couplings before calculating PDW free energies or stiffness.

## Benchmark ladder

1. Recover the one-component GL result when all other components are removed or made non-dynamical in a separately validated limit.
2. Validate two-band/interband analytic limits and common-gauge invariance.
3. Test component-basis permutation, site relabelling and lattice-translation invariance.
4. Test relative-phase locking, frustrated couplings and time-reversal symmetry only with explicit symmetry diagnostics.
5. Add finite-Q components with wave-vector-star and translation-symmetry controls.
6. Compute the full twist-response/stiffness tensor; reject any candidate state with a negative eigenvalue beyond tolerance.
7. Compare scalar GL, multiband GL and PDW GL with matched data, compute and model complexity, using leave-specimen and leave-family holdouts.
8. Use #12811 same-specimen multimodal calibration to test whether added components improve held-out predictions beyond a scalar GL or nematic/structural alternative.

## Claim ceiling

This is a synthetic zero-momentum multicomponent static GL evaluator. It does not establish a PDW, a time-reversal-breaking state, a stable superconducting phase, a microscopic mechanism, a measured Tc, room-temperature superconductivity or an experimentally actionable material design.


## Bare common-twist response (configuration level only)

The current evaluator also differentiates the intercomponent gradient energy under a common total boundary twist. For a positive unit link on axis \(\mu\), define

\[
A_{ij}(\boldsymbol\theta)=A_{ij}+\frac{\theta_\mu}{L_\mu},
\]

and let \(B_{ja}=e^{iA_{ij}}\psi_{ja}\). For the real symmetric intercomponent gradient matrix \(K^{(\mu)}\), the configuration-level first and second derivatives are

\[
J_\mu=2\sum_{(ij)\parallel\mu}\frac{1}{L_\mu}
\operatorname{Im}\left[
\sum_{ab}\psi_{ia}^*K_{ab}^{(\mu)}B_{jb}
\right],
\]

\[
D_{\mu\mu}=2\sum_{(ij)\parallel\mu}\frac{1}{L_\mu^2}
\operatorname{Re}\left[
\sum_{ab}\psi_{ia}^*K_{ab}^{(\mu)}B_{jb}
\right].
\]

For the current axis-aligned unit-edge lattice, off-diagonal spatial twist-curvature components are zero by construction; diagonal entries are retained. The evaluator verifies that both \(J_\mu\) and \(D_{\mu\nu}\) are invariant under a common local U(1) gauge transformation, using nonuniform fields and nonzero links in a dedicated unit test.

**This is not the thermal helicity modulus.** A thermodynamic helicity modulus also requires the current-fluctuation covariance over an equilibrated ensemble:

\[
\Upsilon_{\mu\nu}=\langle D_{\mu\nu}\rangle-
\frac{\operatorname{Cov}(J_\mu,J_\nu)}{T}.
\]

The branch now has a separate finite-temperature Metropolis pilot that samples the zero-momentum complex fields and evaluates the current covariance and thermal helicity-modulus point estimate. It also applies a block jackknife to the current and diamagnetic sample series when at least 20 measurements are available. The four-sample CI smoke deliberately reports `insufficient_samples_for_stiffness_jackknife`; longer runs report an approximate finite-chain interval. Neither outcome proves equilibration, autocorrelation adequacy or finite-size convergence. The stiffness-positivity gate therefore remains a research diagnostic, not a thermodynamic stability certificate and not a reproduction of PDW stability results.

The zero-momentum evaluator also rejects a component labelled `pdw_plus_q` or `pdw_minus_q` with zero momentum, and rejects all nonzero-momentum components. This keeps the role metadata honest until the finite-Q lattice phase factors, translation-symmetry operations, and symmetry-allowed couplings have a separate executable contract.


### Thermal sampler and local update correctness

The random-walk proposal is symmetric in the complex plane:
\[
\psi_{ia}'=\psi_{ia}+\delta_x+i\delta_y,
\qquad
\delta_x,\delta_y\sim U[-s,s],
\]
with acceptance \(\min(1,e^{-\Delta F/T})\). Only one component at one site is changed per proposal, but all quadratic, quartic, phase-locking, secondary-order and incident intercomponent gradient terms are included in \(\Delta F\).

The full-lattice energy is recomputed at every measurement and replaces the incrementally accumulated energy, limiting floating-point drift. The exact local-update regression test compares each site/component proposal's incremental \(\Delta F\) with the full energy difference for all fixture sites/components under nonuniform complex fields and nonzero link phases.

The thermal twist estimator is
\[
\Upsilon_{\mu\nu}=\langle D_{\mu\nu}\rangle-
\operatorname{Cov}(J_\mu,J_\nu)/T.
\]
It is reported as a finite-chain point estimate. The pilot does not yet report blocked jackknife/bootstrap uncertainty for the tensor/eigenvalues. Autocorrelation and multi-size scaling therefore remain required before a negative or positive point estimate can be used as a material-stability statement.


### Relative-phase chirality signal (not a symmetry classification)

For each declared phase-lock pair \((a,b)\), the sampler now reports both

\[
C_{ab}=\left\langle
\frac{\operatorname{Re}(\psi_a^*\psi_b)}
{|\psi_a||\psi_b|}
\right\rangle,
\qquad
S_{ab}=\left\langle
\frac{\operatorname{Im}(\psi_a^*\psi_b)}
{|\psi_a||\psi_b|}
\right\rangle.
\]

Both are invariant under the one common electromagnetic U(1). Under simple time reversal (complex conjugation), \(C_{ab}\) is even and \(S_{ab}\) is odd. The receipt deliberately labels time-reversal classification as `not_inferred_from_relative_phase_sine_without_full_crystal_symmetry_action`: a nonzero relative-phase sine by itself does not establish a time-reversal-breaking phase without checking all crystal/gauge symmetry operations and possible component transformations.

A unit test checks common-gauge invariance and sign reversal of the relative-phase sine under complex conjugation. This is an observable-level control, not a phase classification.
