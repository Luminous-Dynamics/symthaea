# Temporal Topology Paper - Cross-Reference

**Title:** Temporal Topology: Cognitive Coherence Emerges from Continuous-Time Dynamics

**Primary Location:** `/srv/luminous-dynamics/symthaea/papers/temporal_topology_consciousness/`

**Format:** Nature Letter (2,500 words)

**Status:** Ready for arXiv + Nature submission

---

## Relationship to This Paper Series

This paper complements the existing 15-paper Master Equation series by providing an **empirical wedge** into the academic discourse. While the Master Equation papers (particularly Paper 01) develop the theoretical framework C = f(Φ, B, W, A, R), the Temporal Topology paper focuses narrowly on:

1. **Φ-topology relationship** (260 measurements across 19 topologies)
2. **Chronos vs Kairos** framing (spatializing time vs respecting time)
3. **3D optimality finding** (99.2% of theoretical maximum)
4. **Energy efficiency** (60× reduction vs transformers)

### Strategic Positioning

| Paper | Focus | Target |
|-------|-------|--------|
| Temporal Topology | Empirical demonstration | Nature (Letter) |
| Paper 01 (Master Equation) | Unified theory | Nature Neuroscience |
| Papers 02-15 | Satellite applications | Specialty journals |

The Temporal Topology paper serves as a "Trojan Horse" - presenting empirical findings that implicitly challenge the scaling paradigm without making that the overt thesis.

---

## Key Data Used

All data originates from the same Symthaea codebase:

- **260 λ₂ measurements**: `src/hdc/phi_real.rs` (note: file name is a misnomer—computes algebraic connectivity, not IIT Φ)
- **19 topologies**: `src/hdc/consciousness_topology_generators.rs`
- **Validation**: `examples/tier_3_exotic_topologies.rs`

> **METRIC CLARIFICATION**: The measurements labeled "Φ" in the codebase are actually λ₂
> (algebraic connectivity / Fiedler value), a spectral graph metric. This is **NOT** IIT's
> integrated information (Φ), which requires computing minimum information partition and is
> computationally intractable for n > 12. See `docs/METRIC_DEFINITIONS.md` for details.

The difference is in framing and emphasis, not data.

---

## Files at Primary Location

```
/srv/luminous-dynamics/symthaea/papers/temporal_topology_consciousness/
├── main.md                    # Full manuscript (4,200 words)
├── letter_nature.md           # Nature Letter version (2,487 words)
├── abstract.md                # Standalone abstract
├── supplementary.md           # Extended methods
├── cover_letter_nature.md     # Nature submission cover letter
├── figures/
│   ├── README.md              # Figure specifications
│   └── generate_all_figures.py # Python script for all 4 figures
└── arxiv/
    ├── README.md              # arXiv submission instructions
    ├── main.tex               # LaTeX version
    └── bibliography.bib       # BibTeX references
```

---

## Submission Timeline

1. **arXiv**: Submit immediately to establish priority (cs.AI, q-bio.NC)
2. **Nature**: Submit Letter format after arXiv posts
3. **Paper 01**: Submit to Nature Neuroscience after Nature response

---

## Cross-References

- Paper 01 (this series): `PAPER_01_SUBMISSION_READY.md`
- Temporal Topology: `/srv/luminous-dynamics/symthaea/papers/temporal_topology_consciousness/`
- Consolidated Strategy: `/srv/luminous-dynamics/SYMTHAEA_MYCELIX_STRATEGY.md`

---

*This document links the Temporal Topology paper to the existing publication strategy.*
