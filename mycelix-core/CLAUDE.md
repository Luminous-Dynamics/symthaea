# Mycelix Protocol - Claude Context

**Version**: v5.3 → v6.0 (Active Development)
**Status**: Research + Production Implementation
**Achievement**: 34% Validated Byzantine Tolerance (exceeds classical 33% limit)

---

## Quick Commands

```bash
nix develop                        # Enter environment
cd 0TML && poetry install          # Install dependencies
pytest tests/                      # Run tests
python src/zerotrustml/coordinator.py  # Run FL experiment
```

---

## Project Structure

```
Mycelix-Core/
├── docs/
│   ├── architecture/              # System architecture, charters
│   ├── whitepaper/                # Academic paper (MLSys/ICML 2026)
│   └── grants/                    # NSF CISE, NIH R01 materials
├── 0TML/                          # Zero-TrustML implementation
│   ├── src/zerotrustml/           # Production code
│   ├── tests/                     # Test suite
│   └── docs/                      # Technical documentation
└── mycelix-desktop/               # Desktop app (Tauri)
```

---

## Core Concepts

### The Epistemic Cube (E-N-M)
3-axis framework for classifying all claims:

**E-Axis (Empirical)** - How do we verify this?
- E0: Null (unverifiable) → E4: Publicly Reproducible

**N-Axis (Normative)** - Who agrees this is binding?
- N0: Personal → N3: Axiomatic (constitutional)

**M-Axis (Materiality)** - How long does this matter?
- M0: Ephemeral → M3: Foundational (preserve forever)

Example: Passed MIP = (E0, N2, M3) - belief, network consensus, permanent.

### MATL (45% Byzantine Tolerance)
```
Composite = PoGQ + TCDM + Entropy
Three Modes: Peer (33%) | PoGQ Oracle (45%) | PoGQ+TEE (50%)
```

Breaks 33% BFT limit through reputation-weighted validation.

---

## Key Architecture Decisions

| Decision | Why |
|----------|-----|
| Reputation-weighted validation | Byzantine_Power = Σ(malicious_rep²), low-rep attackers weak |
| 3D Epistemic Model (E/N/M) | Fact, law, permanence are independent dimensions |
| Multi-backend (Postgres/Holochain/ETH) | Different guarantees for different use cases |
| Modular Charters | Epistemic, Governance, Economic, Commons evolve independently |

---

## Status

| Component | Status |
|-----------|--------|
| 0TML Core | ✅ Production (PyPI published) |
| Byzantine Testing | ✅ 35 experiments, 100% detection |
| MATL | ✅ 45% BFT achieved |
| Epistemic Charter v2.0 | 🚧 Active development |
| v6.0 Architecture | 🚧 In progress |
| MLSys/ICML 2026 | 🎯 Jan 15, 2026 submission |

---

## Key Documents

| Purpose | File |
|---------|------|
| Constitution | `docs/architecture/THE MYCELIX SPORE CONSTITUTION (v0.24).md` |
| Epistemic Charter | `docs/architecture/THE EPISTEMIC CHARTER (v2.0).md` |
| System Architecture | `docs/architecture/Mycelix Protocol_ Integrated System Architecture v5.2.md` |
| MATL Architecture | `0TML/docs/06-architecture/matl_architecture.md` |
| 0TML Quickstart | `0TML/docs/01-getting-started/QUICKSTART.md` |
| Roadmap | `docs/architecture/Mycelix_Roadmap_v5.3_to_v6.0.md` |

---

## Resources

- **Site**: https://mycelix.net
- **GitHub**: https://github.com/Luminous-Dynamics/mycelix
- **Improvement Plan**: `docs/05-roadmap/IMPROVEMENT_PLAN_NOV2025.md`

---

*Cultivating a new substrate for collective intelligence* 🍄
