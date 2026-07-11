# Spark Engine: Anomaly Investigation Roadmap

## Path C+D: Anomaly Investigation + Neutron Source Development

**Date**: 2026-02-05
**Status**: Active Development

---

## Executive Summary

Standard Gamow physics gives `<σv>` ~ 10⁻⁵⁰ cm³/s at room temperature, predicting effectively zero fusion. Yet NASA observed ~10³ neutrons/s. This **50 orders of magnitude gap** is either:

1. A measurement error (unlikely - peer-reviewed, multiple methods)
2. New physics we don't understand (revolutionary)

This roadmap pursues **both paths simultaneously**:
- **Path C**: Investigate the anomaly scientifically
- **Path D**: Develop practical neutron sources assuming the anomaly is real

---

## The Central Mystery

| Quantity | Standard Physics | NASA Observed | Gap |
|----------|-----------------|---------------|-----|
| Neutron rate | ~10⁻⁴⁷ n/s/cm³ | ~10³ n/s/cm² | 10⁵⁰× |
| `<σv>` at 300K | 10⁻⁵⁰ cm³/s | (implies) 10⁰ cm³/s | 10⁵⁰× |
| Q factor | 10⁻⁵⁰ | (implies) ~10⁻³ | 10⁴⁷× |

**Either we're wrong about fundamental nuclear physics, or something extraordinary is happening in the lattice.**

---

## Phase 1: Validation & Characterization (Months 1-6)

### Objective
Confirm or refute the NASA observations with independent measurements.

### Key Experiments

#### 1.1 Independent Replication
- **What**: Reproduce NASA setup at independent facility
- **Why**: Eliminate systematic errors, confirm effect
- **Success**: Observe > 100 n/s from PdD under X-ray trigger
- **Failure**: No signal above background → systematic error identified

#### 1.2 H/D Control Comparison
- **What**: Run identical experiment with PdH instead of PdD
- **Why**: D-D fusion requires deuterium; if signal persists with H, it's not fusion
- **Success**: Signal only with D, not with H
- **Failure**: Signal with H → not fusion, investigate alternative mechanism

#### 1.3 Neutron Spectrum Measurement
- **What**: High-resolution neutron spectroscopy
- **Why**: D-D produces 2.45 MeV neutrons; confirm this signature
- **Success**: Peak at 2.45 MeV confirms D-D fusion
- **Failure**: Wrong energy → different reaction or background

#### 1.4 Branching Ratio Check
- **What**: Measure neutron/proton branch ratio
- **Why**: D-D gives ~50/50 split; deviations indicate lattice effects
- **Success**: 50±5% → consistent with known physics
- **Interesting**: Significant deviation → lattice modifies nuclear reaction

### Resources Required
- Independent lab with neutron detection capability
- PdD sample preparation facility
- X-ray trigger system (~$50K)
- Estimated budget: $200K

---

## Phase 2: Mechanism Investigation (Months 6-18)

### Objective
If Phase 1 confirms anomaly, determine the physical mechanism.

### Hypotheses to Test

#### 2.1 Transient Hot Spots
- **Hypothesis**: X-ray absorption creates local T ~ 10,000K for ~ps
- **Test**: Vary X-ray pulse duration, measure rate scaling
- **Prediction**: Rate ∝ (pulse energy) × (pulse duration)⁻¹/²

#### 2.2 Coherent Phonon Enhancement
- **Hypothesis**: Phonons coherently add to D-D CM energy
- **Test**: Drive with specific acoustic frequencies
- **Prediction**: Resonances at PdD phonon frequencies (56 meV, etc.)

#### 2.3 Super-Screening
- **Hypothesis**: Screening > 309 eV in high-loading conditions
- **Test**: Measure rate vs D/Pd loading ratio
- **Prediction**: Rate increases sharply above D/Pd > 0.7

#### 2.4 Lattice-Modified Nuclear
- **Hypothesis**: Periodic potential affects tunneling
- **Test**: Compare FCC (Pd) vs BCC (V) vs HCP (Ti) hosts
- **Prediction**: Structure-dependent rate differences

### Resources Required
- Ultrafast diagnostics (ps time resolution)
- Multiple host metal samples
- Variable-frequency acoustic driver
- Estimated budget: $500K

---

## Phase 3: Neutron Source Development (Months 12-24)

### Objective
Develop practical neutron source assuming anomaly is real.

### Target Applications

| Application | Rate Needed | Portability | Market Size |
|-------------|-------------|-------------|-------------|
| NAA | 10⁶ n/s | Portable | $50M/year |
| Security | 10⁸ n/s | Transportable | $500M/year |
| BNCT | 10⁹ n/s | Fixed | $200M/year |
| Research | 10⁴ n/s | Portable | $100M/year |

### Design Targets

#### 3.1 Portable NAA Source
- **Spec**: 10⁶ n/s, < 25 kg, < 20 L
- **Design**: ~10 g Pd, X-ray trigger, borated PE shielding
- **Cost target**: < $50K

#### 3.2 Security Screening Unit
- **Spec**: 10⁸ n/s, truck-mounted
- **Design**: ~1 kg Pd, high-power X-ray trigger
- **Cost target**: < $200K

### Competitive Advantage
- No radioactive sources (unlike Cf-252)
- No tritium (unlike D-T generators)
- No high voltage (unlike accelerator sources)
- Compact (unlike reactors)

### Resources Required
- Prototype fabrication facility
- Neutron detection/characterization lab
- Regulatory consultation (NRC/DOE)
- Estimated budget: $1M

---

## Phase 4: Scale-Up & Commercialization (Months 24-48)

### Objective
Transition from lab prototype to commercial product.

### Milestones

#### 4.1 Regulatory Pathway
- DOE/NRC exemption for D-D sources (no fissile material)
- Safety documentation package
- Operating procedures

#### 4.2 Manufacturing
- Pd procurement and quality control
- X-ray trigger standardization
- Shielding fabrication
- System integration

#### 4.3 Market Entry
- Beta units to select customers
- Application-specific optimization
- Service and support infrastructure

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Anomaly is measurement error | 30% | Project pivot | Phase 1 validation |
| Mechanism is non-reproducible | 20% | Delay | Multiple independent tests |
| Regulatory barriers | 40% | Delay | Early NRC engagement |
| Competition from D-T | 50% | Market share | Focus on no-tritium advantage |
| Pd cost volatility | 60% | Margin | Alternative hosts, recycling |

---

## Decision Points

### Gate 1 (Month 6): Validation Complete
- **Go**: Confirmed anomaly → proceed to Phase 2+3
- **Pivot**: Anomaly not confirmed → publish negative result, investigate alternative physics

### Gate 2 (Month 18): Mechanism Identified
- **Go**: Understood mechanism → optimize for applications
- **Continue**: Mechanism unclear but reproducible → engineering approach

### Gate 3 (Month 24): Prototype Viable
- **Go**: Meets specs → commercialization
- **Iterate**: Near-miss → design optimization cycle

---

## Budget Summary

| Phase | Duration | Budget | Key Outputs |
|-------|----------|--------|-------------|
| 1 | 6 mo | $200K | Validation data |
| 2 | 12 mo | $500K | Mechanism understanding |
| 3 | 12 mo | $1M | Working prototypes |
| 4 | 24 mo | $2M | Commercial product |
| **Total** | 48 mo | $3.7M | |

---

## Team Requirements

### Core Team
- **Physics Lead**: Nuclear/plasma physics, tunneling expertise
- **Materials Lead**: Metal hydrides, lattice dynamics
- **Engineering Lead**: Neutron systems, shielding
- **Applications Lead**: NAA, security screening domain knowledge

### Advisors
- NASA LCF team (Steinetz et al.)
- Nuclear physics theorist (tunneling/screening)
- Regulatory consultant (NRC/DOE)

---

## Success Criteria

### Scientific Success
- Reproducible anomalous rate confirmed by 3+ independent labs
- Physical mechanism identified and published
- Theory consistent with observations

### Commercial Success
- Prototype meeting NAA specs ($50K, 10⁶ n/s, < 25 kg)
- First customer deployment within 36 months
- Revenue > $1M/year within 48 months

---

## Why This Matters

If the anomaly is real:
- **Physics**: New understanding of nuclear reactions in condensed matter
- **Technology**: Compact neutron sources without radioactive materials
- **Medicine**: Portable BNCT for cancer treatment
- **Security**: Widespread cargo screening capability
- **Science**: Accessible neutron beams for every university

If standard physics is correct:
- **Physics**: Important negative result clarifying LCF claims
- **Knowledge**: Comprehensive model of what LCF cannot do
- **Foundation**: Clear basis for future fusion research directions

**Either outcome advances science. We proceed.**

---

## Project Structure

```
/srv/luminous-dynamics/
├── spark-engine/              # This project
│   ├── src/                   # Rust physics engine
│   │   ├── physics.rs         # Gamow integration, Q factor
│   │   ├── anomaly.rs         # Path C: investigation framework
│   │   ├── neutron_source.rs  # Path D: source design
│   │   └── ...
│   ├── docs/                  # Documentation
│   │   └── ANOMALY_INVESTIGATION_ROADMAP.md  # This document
│   ├── examples/              # Usage examples
│   └── tests/                 # Test suite
│
└── symthaea/                  # Parent project (moved from nested path)
    └── symthaea-core/         # Original physics crate
        └── src/physics/       # Full physics modules
```

---

## Next Steps

1. **Immediate**: Build and test spark-engine crate
2. **Week 1**: Design Phase 1 validation experiments
3. **Week 2**: Identify independent lab partners
4. **Month 1**: Begin procurement for replication experiment
5. **Month 3**: First independent data

**The gap is too large to ignore. Let's find out what's really happening.**
