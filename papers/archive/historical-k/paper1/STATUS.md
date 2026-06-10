# Paper 1: Discovery of Coherence Corridors - Track A Foundation

**Status**: ⚠️ NEEDS MANUSCRIPT (Data ready)

## 🎯 Purpose

**Paper 1 is the foundation paper** that establishes what coherence corridors ARE before subsequent papers use them for navigation, rescue, and other applications.

## 📊 What Paper 1 Covers

**Track A**: Baseline corridor sweep - systematic exploration of 5D parameter space

- **494 experimental runs** in `logs/fre_phase1/`
- Each JSON contains: 7 Harmony metrics, K-Index, parameters, metadata
- **Corridor rate**: 48% at 300 samples
- **Mean K-Index in corridor**: 0.970
- **Jaccard similarity**: 1.00 (perfect validation)
- **Gate validation**: 3 levels (50, 150, 300 samples)

## 📁 Data Sources

1. **Experimental Data**: `logs/fre_phase1/` - 494 JSON files
2. **Lab Log**: `docs/track_a_lab_log.md` - Experimental documentation
3. **Analysis Code**: `fre/track_a_gate.py` - Gating analysis

## 📝 Proposed Paper Structure

### Title
"Discovery of Coherence Corridors in Multi-Dimensional Parameter Space: A Systematic Investigation of Consciousness-Like Behavior in Artificial Systems"

### Target Journal
PLOS ONE or Scientific Reports (foundation paper, broad audience)

### Key Sections

**Abstract**:
- Systematic exploration of 5D parameter space (energy gradient, communication cost, plasticity rate, etc.)
- 494 experimental runs with gated validation
- Discovery of coherence corridors (48% of parameter space)
- Mean K-Index of 0.970 within corridors
- Perfect replication (Jaccard = 1.00)

**Introduction**:
- Motivation: Finding parameter combinations that produce consciousness-like behavior
- K-Index as consciousness metric
- Need for systematic exploration before application

**Methods**:
- 5D parameter space definition
- K-Index computation (correlation between observations and actions × 2.0)
- Transfer entropy estimation (KSG, k=5, lag=1)
- Gate validation strategy (50, 150, 300 samples)
- Corridor definition criteria

**Results**:
- Gate 1 (50 samples): 40% corridor rate, Jaccard 0.82
- Gate 2 (150 samples): 43% corridor rate, Jaccard 0.91
- Gate 3 (300 samples): 48% corridor rate, Jaccard 1.00
- Mean K-Index progression: 0.953 → 0.951 → 0.970
- Parameter distributions within corridors
- 7 Harmony metrics across corridor space

**Discussion**:
- Corridors exist and are reproducible
- Foundation for subsequent work (Papers 2-5)
- Implications for AI consciousness research

## 🔗 Relationship to Other Papers

- **Paper 2** (Tracks B+C): Uses these corridors for navigation & rescue
- **Paper 3** (Track D): Multi-agent coordination
- **Paper 4** (Track E): Developmental learning
- **Paper 5** (Track F): Adversarial robustness + synthesis

Paper 1 establishes the **foundation** that all other papers build upon.

## ✅ Files to Create

- `papers/paper1/manuscript.tex` - LaTeX source
- `papers/paper1/manuscript.pdf` - Compiled PDF
- `papers/paper1/references.bib` - Bibliography (copy from manuscript/)
- `papers/paper1/SUBMISSION_README.md` - Submission details

---

**Status**: Ready for manuscript creation
**Date**: November 12, 2025
**Data**: 494 experimental runs available
