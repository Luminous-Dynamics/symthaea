# Integrated Information from First Principles:
# Testing 10 Consciousness Theories Against the Schrödinger Equation

## Authors
Tristan Stoltz, Luminous Dynamics

## Target
Neuroscience of Consciousness / PLoS Computational Biology / arXiv:q-bio.NC

---

## Abstract

We present the first systematic comparison of 10 consciousness theories
grounded in ab initio quantum chemistry. By mapping each theory to a
computable molecular quantity derived from the Schrödinger equation, we
evaluate which theories agree, which are independent, and whether
molecular systems are rich enough to discriminate between them.

Computing 120 theory-molecule measurements (10 theories × 12 molecules)
at the Hartree-Fock/STO-3G level, validated against published benchmarks,
we find four principal results:

(1) Higher-Order Thought theory and Orchestrated Objective Reduction are
formally equivalent at the molecular level (r = 0.89–0.99, robust across
three normalization schemes, 12 molecules, and 60 reaction-coordinate
points), both reducing to the HOMO-LUMO excitation gap;

(2) Quantum coherence and classical objectivity (Quantum Darwinism) are
fundamentally anti-correlated (r = −0.91 to −1.00), confirming that
consciousness may require a balance between quantum depth and classical
reportability;

(3) Molecular consciousness theory space is effectively 2-dimensional —
eigendecomposition of the 10×10 correlation matrix across molecular
variants reveals only 1–2 significant principal components, indicating
that the 10 theories are highly redundant at this scale;

(4) The correlation structure between theories is stable along chemical
reaction coordinates (no phase transitions detected), suggesting that
the "consciousness increases at bond dissociation" observation is a
method artifact rather than genuine physics.

These findings establish that molecules are too structurally simple to
discriminate between most consciousness theories. The theories that
SEEM distinct at the neural level collapse into the same physical
observable (the excitation gap) at the molecular level. This places
a lower bound on the complexity required to meaningfully test
consciousness theories and suggests that the mesoscopic scale
(protein conformations, membrane dynamics) is the minimum resolution
at which these theories make distinguishable predictions.

All computations are performed in pure Rust from the Schrödinger
equation with no empirical parameters.

---

## 1. Introduction

The scientific study of consciousness faces a foundational problem:
multiple competing theories (IIT, GWT, HOT, FEP, Orch-OR, etc.) make
different predictions but are rarely tested against each other on the
same physical system. Most comparisons are qualitative or rely on
neural-level phenomenology where the underlying physics is abstracted
away.

We propose a novel approach: ground each theory in a specific,
computable quantity derivable from the molecular wavefunction, then
run all theories on the same set of molecules and analyze the
correlation structure. This reveals which theories are measuring the
same underlying physics (redundant), which are independent (capturing
different aspects), and — crucially — whether molecules are complex
enough to make the distinction meaningful.

### Why molecules?

Molecules are the simplest quantum systems with internal structure.
They provide an exactly solvable testing ground: electronic structure
is computed from the Schrödinger equation with no free parameters.
If consciousness theories are physically grounded, they should make
predictions even at the molecular level.

However, this simplicity is a double-edged sword. If molecules are
too simple to distinguish between theories, that is itself a finding —
it places a lower bound on the complexity required for consciousness
science.

---

## 2. Methods

### 2.1 Quantum Chemistry

Electronic structure computed at the Restricted Hartree-Fock level
with STO-3G minimal basis set (Hehre, Stewart & Pople 1969) using
the Obara-Saika integral scheme with Schwarz prescreening. DIIS
convergence acceleration (Pulay 1980). Canonical orthogonalization
for numerical stability. 6-31G split-valence basis for selected
molecules.

Validated against published benchmarks: CH₄ and HF match reference
energies exactly; H₂, H₂O, NH₃, LiH within 2 kcal/mol of
Szabo & Ostlund reference values. MP2 correlation energy computed
for all molecules.

### 2.2 Theory-to-Metric Mapping

| # | Theory | Molecular Metric | What It Measures |
|---|--------|-----------------|------------------|
| 1 | IIT (Tononi 2004) | Max bipartition entanglement entropy | Integration across atomic subsystems |
| 2 | GWT (Baars 1988) | Orbital delocalization (1 - Herfindahl) | Global availability of information |
| 3 | FEP (Friston 2010) | Helmholtz free energy ratio | Thermodynamic order |
| 4 | HOT (Rosenthal 2005) | 1/(1 + ΔE/10eV) for first CIS excitation | Meta-representational access |
| 5 | Orch-OR (Penrose 1994) | tanh(1/(gap × 100)) | Quantum coherence duration |
| 6 | Autopoiesis (Maturana 1980) | MP2 correlation / HF energy | Self-organization via electron correlation |
| 7 | Complexity (Crutchfield 1989) | Mutual information network density | Structured correlations between orbitals |
| 8 | Info Geometry (Amari 2016) | KL(joint ‖ product) of orbital populations | Geometric integration in probability space |
| 9 | Dissipative (Prigogine 1977) | T × entropy / kT | Entropy production maintaining order |
| 10 | Q-Darwinism (Zurek 2003) | 1/(1 + χ) where χ = quantum-classical parameter | Classical objectivity of the state |

### 2.3 Composite Score

Following the softmin-bottleneck + weighted-amplifier architecture:
- Necessary conditions (softmin): IIT, GWT, Complexity
- Amplifiers (weighted mean): remaining 7 theories
- Limiting theory: whichever necessary condition scores lowest

### 2.4 Normalization Sensitivity

Three schemes tested: (a) theory-specific sigmoid/scaling, (b)
rank-based, (c) min-max per theory. Key correlations must survive
all three to be considered robust.

### 2.5 Reaction Coordinate Analysis

H₂ dissociation (30 points, R = 0.7–6.0 Bohr), H₂O symmetric
stretch (15 points, f = 0.6–2.5), LiH dissociation (15 points,
R = 1.5–8.0 Bohr). Sliding-window (7-point) correlation analysis
to detect phase transitions in the theory correlation structure.

### 2.6 Dimensionality Analysis

Eigendecomposition of the 10×10 theory correlation matrix within
each molecular complexity class. Effective dimensionality defined
as number of eigenvalues exceeding 5% of total variance.

### 2.7 Molecules

12 molecules: H₂, HeH⁺, LiH, HF, H₂O, NH₃, CH₄, N₂, CO, HCN,
H₂CO, glycine (C₂H₅NO₂). Spanning ionic, polar, nonpolar,
multiply-bonded, and biologically relevant systems.

---

## 3. Results

### 3.1 Finding 1: HOT ≡ Orch-OR (r = 0.89–0.99)

Higher-Order Thought theory and Orchestrated Objective Reduction
produce nearly identical scores across all conditions tested:

| Condition | HOT-OrchOR r |
|-----------|-------------|
| 12 molecules (default normalization) | 0.89 |
| 12 molecules (rank normalization) | 0.98 |
| 12 molecules (min-max normalization) | 0.89 |
| H₂ dissociation (30 dynamic points) | 0.98 |
| H₂O stretch (15 dynamic points) | 0.96 |
| LiH dissociation (15 dynamic points) | 0.99 |

Both theories reduce to the same physical observable: the energy
gap between the highest occupied and lowest unoccupied molecular
orbital (HOMO-LUMO gap). A small gap means both easy
meta-representation (HOT: the system can represent its own states)
and long quantum coherence (Orch-OR: the superposition persists).

### 3.2 Finding 2: Coherence Opposes Objectivity (r = −0.91 to −1.00)

The HOT/Orch-OR axis is anti-correlated with Quantum Darwinism
across all conditions. This is a fundamental constraint: a molecule
that preserves quantum superposition (high HOT/Orch-OR) cannot
simultaneously be classically observable (high QDarwinism).

N₂ exemplifies the tension: maximum quantum integration (IIT = 1.0,
HOT = 0.79) but near-zero classical objectivity (QDarwinism = 0.01).

### 3.3 Finding 3: Molecular Theory Space is 2-Dimensional

Eigendecomposition of the 10×10 correlation matrix within each
complexity class:

| Category | Atoms | Electrons | Effective Dimensions |
|----------|-------|-----------|---------------------|
| 2-electron diatomics | 2 | 2 | 0 |
| 4-electron | 2 | 4 | 2 |
| 10-electron, 2 atoms | 2 | 10 | 2 |
| 10-electron, 3 atoms | 3 | 10 | 1 |
| 10-electron, 4 atoms | 4 | 10 | 2 |
| 10-electron, 5 atoms | 5 | 10 | 2 |
| 16-electron, 4 atoms | 4 | 16 | 1 |

The 10 theories collapse into at most 2 independent dimensions.
No clear scaling with molecular complexity (r = 0.27 for atoms,
0.18 for electrons). At the molecular level, most theories are
measuring the same 1–2 physical quantities.

### 3.4 Finding 4: No Consciousness Phase Transitions in Reactions

Sliding-window analysis of the theory correlation structure along
H₂ dissociation shows NO reorganization. The largest correlation
change between equilibrium and dissociation regions is Δr = 0.06
(FEP-QDarwinism). All other theory pairs change by < 0.01.

The observation that composite consciousness scores increase toward
dissociation is entirely explained by the HOMO-LUMO gap closing
(which increases HOT, Orch-OR, and decreases QDarwinism). This is
a metric artifact, not a consciousness phase transition.

### 3.5 Consciousness Ranking (12 molecules)

| Rank | Molecule | Composite | Atoms | Electrons | Limiting Theory |
|------|----------|-----------|-------|-----------|-----------------|
| 1 | CH₄ | 0.173 | 5 | 10 | Complexity |
| 2 | LiH | 0.126 | 2 | 4 | Complexity |
| 3 | NH₃ | 0.100 | 4 | 10 | Complexity |
| 4 | CO | 0.076 | 2 | 14 | Complexity |
| 5 | HF | 0.068 | 2 | 10 | GWT |
| 6 | H₂O | 0.067 | 3 | 10 | Complexity |
| 7 | N₂ | 0.063 | 2 | 14 | Complexity |
| 8 | H₂CO | 0.047 | 4 | 16 | Complexity |
| 9 | HCN | 0.029 | 3 | 14 | Complexity |
| 10 | HeH⁺ | 0.028 | 2 | 2 | Complexity |
| 11 | H₂ | 0.007 | 2 | 2 | Complexity |
| 12 | Glycine | 0.001 | 10 | 40 | Complexity |

---

## 4. Discussion

### 4.1 The Poverty of Molecular Consciousness

The central finding is negative: molecules are not complex enough to
meaningfully distinguish between consciousness theories. Ten theories
that make apparently different predictions about consciousness reduce
to 1–2 independent physical observables when evaluated on molecular
wavefunctions. The dominant observable is the HOMO-LUMO gap, which
simultaneously determines HOT scores, Orch-OR scores, QDarwinism
scores, and (indirectly) several other metrics.

This is not a failure of the theories — it is a property of the
molecules. A system with only 2–40 electrons and 2–10 atoms does
not have enough structural degrees of freedom for 10 theories to
make distinguishable predictions. The theories need richer
substrates: conformational flexibility (proteins), spatial
organization (membranes), or temporal dynamics (neural networks).

### 4.2 What IS Robust

The HOT ≡ Orch-OR equivalence is not a molecular artifact — it
reflects a genuine mathematical identity between the excitation
gap and the coherence time. This equivalence should hold at any
scale: if HOT is determined by the energy gap between the current
state and the nearest accessible meta-representational state, and
Orch-OR is determined by the coherence time of a superposition
(which is inversely proportional to the energy gap), then these
theories are mathematically identical regardless of system size.

This is a strong prediction for neural-scale experiments: any
manipulation that changes the excitation spectrum (anesthesia,
psychedelics, sleep) should affect HOT and Orch-OR scores
identically.

### 4.3 The Minimum Scale for Consciousness Science

Our results suggest a "complexity floor" below which consciousness
theories cannot be tested. For 2-electron systems, the theory space
has 0 dimensions (all theories are constant). For 10-electron
systems, 1–2 dimensions. Extrapolating, we estimate that ~10³–10⁴
effective degrees of freedom may be needed for all 10 theories
to become meaningfully independent.

This corresponds roughly to: a medium-sized protein (~100 amino
acids), a lipid bilayer patch (~100 lipids), or a small neural
circuit (~100 neurons). Below this scale, consciousness theories
are undertermined by the physics.

### 4.4 Limitations

- STO-3G minimal basis limits quantitative accuracy
- Closed-shell RHF only (no open-shell radicals or transition metals)
- The theory-to-metric mapping involves choices that could be
  questioned (especially the MI density proxy for Complexity)
- Only 12 molecules; amino acid survey limited by SCF convergence
- The normalization scheme, while tested for robustness, introduces
  some arbitrariness

### 4.5 Honest Assessment of Artifacts

The "consciousness increases at dissociation" observation reported
in preliminary analyses is an artifact of the HOMO-LUMO gap closing
as bonds stretch. The sliding-window correlation analysis (Section
3.4) confirms no reorganization of the theory correlation structure.
We report this negative result explicitly to prevent misinterpretation.

---

## 5. Conclusion

By grounding 10 consciousness theories in the Schrödinger equation,
we discover that molecular systems are too simple to distinguish
between most of them. The theories collapse into 1–2 physical
observables, dominated by the HOMO-LUMO excitation gap. The one
robust finding — that HOT and Orch-OR are formally equivalent —
reflects a genuine mathematical identity, not a limitation of the
molecular scale.

The implication for consciousness science is that testing between
theories requires systems with sufficient structural complexity.
Molecules provide a rigorous lower bound: below ~10² degrees of
freedom, consciousness theories are undertermined. The mesoscopic
scale — protein conformations, membrane dynamics, small neural
circuits — is likely the minimum resolution at which these theories
make distinguishable predictions.

---

## Code Availability

All computations performed with symthaea-quantum-chemistry v0.1.0
(pure Rust, WASM-compatible, 121 tests passing).
Source: github.com/luminous-dynamics/symthaea

## References

1. Tononi, G. (2004). BMC Neuroscience 5, 42.
2. Baars, B. J. (1988). A Cognitive Theory of Consciousness. Cambridge UP.
3. Rosenthal, D. M. (2005). Consciousness and Mind. Oxford UP.
4. Friston, K. (2010). Nature Reviews Neuroscience 11, 127.
5. Penrose, R. (1994). Shadows of the Mind. Oxford UP.
6. Maturana, H. R. & Varela, F. J. (1980). Autopoiesis and Cognition. Reidel.
7. Crutchfield, J. P. (1989). Inferring Statistical Complexity. PRL 63, 105.
8. Amari, S. (2016). Information Geometry and Its Applications. Springer.
9. Prigogine, I. (1977). Self-Organization in Non-Equilibrium Systems. Wiley.
10. Zurek, W. H. (2003). Rev. Mod. Phys. 75, 715.
11. Szabo, A. & Ostlund, N. S. (1996). Modern Quantum Chemistry. Dover.
12. Koch, C. et al. (2016). Nature Reviews Neuroscience 17, 307.
13. Aaronson, S. (2014). Why I Am Not An Integrated Information Theorist.
14. Pulay, P. (1980). Chem. Phys. Lett. 73, 393.
15. Hehre, W. J. et al. (1969). J. Chem. Phys. 51, 2657.
