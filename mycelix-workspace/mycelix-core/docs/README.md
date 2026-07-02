# 🍄 Mycelix Protocol Documentation

**A Universal Framework for Agent-Centric Decentralized Systems**

**Byzantine-Resistant Federated Learning + Epistemic Knowledge Graph + Constitutional Governance**

---

> 📘 **Canonical Site** — This repository is published live at [mycelix.net](https://mycelix.net) via MkDocs Material. Always refer to the site for the latest navigation, search, and version switcher.
>
> 🛠️ **Active Improvement Plan** — Execution priorities, owners, and timelines are tracked in [`docs/05-roadmap/IMPROVEMENT_PLAN_NOV2025.md`](./05-roadmap/IMPROVEMENT_PLAN_NOV2025.md). Any structural docs changes should update that plan.

## 🎯 Quick Navigation

### 🌟 Start Here
- **[CLAUDE.md](../CLAUDE.md)** - Complete development context and central navigation hub ⭐
- **[Main README](../README.md)** - Project introduction and quick start
- **[Key Metrics](#-key-achievements)** - Performance benchmarks and achievements

### 👥 By Role

#### 🆕 For New Users
1. **[Executive Summary](#)** - High-level overview (coming soon)
2. **[0TML Quickstart](../0TML/docs/01-getting-started/QUICKSTART.md)** - 5-minute introduction
3. **[Core Concepts](#-core-concepts-epistemic-cube)** - Foundation principles

#### 🏛️ For Researchers & Philosophers
1. **[Spore Constitution v0.24](./architecture/THE%20MYCELIX%20SPORE%20CONSTITUTION%20(v0.24).md)** - Foundational principles
2. **[Epistemic Charter v2.0](./architecture/THE%20EPISTEMIC%20CHARTER%20(v2.0).md)** - 3D truth framework ✨ **LATEST**
3. **[The Four Charters](#-constitutional-framework-the-four-charters)** - Modular governance
4. **[Whitepaper (PoGQ)](./whitepaper/)** - Academic publication

#### 🛠️ For Developers
1. **[0TML Developer Guide](../0TML/docs/03-developer-guide/README.md)** - Build on Zero-TrustML
2. **[SDK Design](./architecture/Designing%20Mycelix%20Protocol%20SDK.md)** - Protocol SDK architecture
3. **[API Documentation](../0TML/docs/04-api/README.md)** - Complete API reference
4. **[Code Examples](../0TML/docs/05-examples/README.md)** - Tutorials and samples

#### 🏗️ For System Architects
1. **[Integrated Architecture v5.2](./architecture/Mycelix%20Protocol_%20Integrated%20System%20Architecture%20v5.2.md)** - Complete system design
2. **[Base Spec v1.0](./architecture/mycelix_base_spec.md)** - Governance-neutral kernel
3. **[MATL Architecture](../0TML/docs/06-architecture/matl_architecture.md)** - Adaptive Trust Layer
4. **[0TML Architecture Hub](../0TML/docs/06-architecture/README.md)** - Technical implementation

#### 💰 For Grant Reviewers
1. **[Architecture Comparison](./grants/ARCHITECTURE_COMPARISON_FOR_GRANTS.md)** - System analysis
2. **[Whitepaper Section 3](./whitepaper/POGQ_WHITEPAPER_SECTION_3_DRAFT.md)** - PoGQ mechanism
3. **[Research Foundation](../0TML/docs/RESEARCH_FOUNDATION.md)** - Academic backing
4. **[Key Metrics](#-key-achievements)** - Performance benchmarks

---

## 📚 Documentation Structure

### 🏛️ Constitutional Framework (The Four Charters)

The Mycelix governance framework is built on a modular charter system, all subordinate to the Spore Constitution:

#### Core Constitution
- **[The Mycelix Spore Constitution v0.24](./architecture/THE%20MYCELIX%20SPORE%20CONSTITUTION%20(v0.24).md)** - Foundational principles and core values
  - Status: Current
  - 61KB, comprehensive constitutional framework
  - Defines agent sovereignty, subsidiarity, and core principles

- **[Constitution v0.22 Complete](./constitution_v022_complete.md)** - Previous version
  - Status: Superseded (retained for reference)

#### The Four Modular Charters

**1. Epistemic Charter v2.0** - Truth & Knowledge Infrastructure ✨ **LATEST**
- **[Epistemic Charter v2.0](./architecture/THE%20EPISTEMIC%20CHARTER%20(v2.0).md)**
  - Revolutionary 3D "Epistemic Cube" framework
  - E-Axis: Empirical Verifiability (E0-E4)
  - N-Axis: Normative Authority (N0-N3)
  - M-Axis: Materiality/State Management (M0-M3)
  - Supersedes v1.0 (see [v1.0 here](./architecture/THE%20EPISTEMIC%20CHARTER%20(v1.0).md))

**2. Governance Charter v1.0** - Decision-Making Processes
- **[Governance Charter v1.0](./architecture/THE%20GOVERNANCE%20CHARTER%20(v1.0).md)**
  - Federated tier system (Hearths → Sectors → Regions → Global)
  - MIP (Mycelix Improvement Proposal) framework
  - Member Redress Council and dispute resolution

**3. Economic Charter v1.0** - Value Flows & Incentives
- **[Economic Charter v1.0](./architecture/THE%20ECONOMIC%20CHARTER%20(v1.0).md)**
  - Reputation-weighted validator economics
  - Stake requirements and slashing conditions
  - Multi-currency exchange architecture

**4. Commons Charter v1.0** - Shared Resources & Stewardship
- **[Commons Charter v1.0](./architecture/THE%20COMMONS%20CHARTER%20(v1.0).md)**
  - Decentralized Knowledge Graph (DKG) as commons
  - Contribution accounting and attribution
  - Audit Guild and Knowledge Council roles

---

### 🏗️ Technical Architecture

#### Current/Canonical Versions
- **[Integrated System Architecture v5.2](./architecture/Mycelix%20Protocol_%20Integrated%20System%20Architecture%20v5.2.md)** (158KB)
  - Status: **CURRENT** (November 2025)
  - Complete system design covering all components
  - Holochain + PostgreSQL + Multi-chain integration

- **[Mycelix Base Spec v1.0](./architecture/mycelix_base_spec.md)** (37KB)
  - Governance-neutral verification kernel
  - DKG, DHT, and proof services
  - Interoperability standards

- **[MATL Architecture v1.0](../0TML/docs/06-architecture/matl_architecture.md)**
  - Mycelix Adaptive Trust Layer
  - 45% Byzantine tolerance mechanism
  - Composite trust scoring

#### Previous Versions (Archived)
- [Integrated System Architecture v4.0](./architecture/Mycelix_Protocol_Integrated_System_Architecture_v4.0.md)
  - Status: Superseded by v5.2
  - Retained for historical reference

#### Specialized Architecture Documents
- **[Holochain Currency Exchange](../0TML/docs/06-architecture/HOLOCHAIN_CURRENCY_EXCHANGE_ARCHITECTURE.md)**
- **[SDK Design](./architecture/Designing%20Mycelix%20Protocol%20SDK.md)** (78KB)
- **[0TML Architecture Index](../0TML/docs/06-architecture/README.md)** - Complete technical docs

---

### 🔬 Zero-TrustML (0TML) Implementation

The 0TML directory contains the production implementation of Byzantine-resistant federated learning:

#### Complete 0TML Documentation
- **[0TML Documentation Hub](../0TML/docs/README.md)** - Master index
  - 10 numbered sections (00-10) covering all aspects
  - Excellent organization model for rest of project

#### Key Sections
- **[00-Overview](../0TML/docs/00-overview/)** - Executive summary and vision
- **[01-Getting Started](../0TML/docs/01-getting-started/)** - Installation and quickstart
- **[02-Core Concepts](../0TML/docs/02-core-concepts/)** - Four primitives, reputation
- **[03-Developer Guide](../0TML/docs/03-developer-guide/)** - Building adapters
- **[04-API](../0TML/docs/04-api/)** - Complete API reference
- **[05-Examples](../0TML/docs/05-examples/)** - Tutorials and samples
- **[06-Architecture](../0TML/docs/06-architecture/)** - Technical design
- **[07-Security](../0TML/docs/07-security/)** - Defense-in-depth
- **[08-Governance](../0TML/docs/08-governance/)** - DAO structure
- **[09-Operations](../0TML/docs/09-operations/)** - Deployment guides
- **[10-Research](../0TML/docs/10-research/)** - Academic papers

---

### 📚 Academic & Research Materials

#### Whitepaper (PoGQ)
Academic paper materials for MLSys/ICML submission (January 2026)

- **[Whitepaper Index](./whitepaper/)** - Complete paper navigation
- **[POGQ Outline](./whitepaper/POGQ_WHITEPAPER_OUTLINE.md)** - 12-page structure
- **[Section 3: Core Mechanism](./whitepaper/POGQ_WHITEPAPER_SECTION_3_DRAFT.md)** ✅ Complete (3800 words)
- **[PoGQ Integration Plan](./whitepaper/POGQ_INTEGRATION_PLAN.md)** - Implementation details

**Progress**: Section 3 complete, 7 sections remaining (~8400 words)

#### Grant Applications
Strategic materials for NSF CISE and NIH R01 applications (June 2026)

- **[Grant Applications Index](./grants/)** - All grant materials
- **[Architecture Comparison for Grants](./grants/ARCHITECTURE_COMPARISON_FOR_GRANTS.md)** - Analysis for reviewers

**Target Funding**: $500K-$1M NSF CISE, $1M-$2M NIH R01

#### Research Foundation
- **[Research Foundation](../0TML/docs/RESEARCH_FOUNDATION.md)** - Academic backing and citations
- **[Byzantine Resistance Research](../0TML/docs/10-research/BYZANTINE_RESISTANCE.md)**
- **[Communication Efficiency](../0TML/docs/10-research/COMMUNICATION_EFFICIENCY.md)**

---

### 🗺️ Roadmap & Evolution

#### Current Development
- **[Roadmap v5.3 → v6.0](./architecture/Mycelix_Roadmap_v5.3_to_v6.0.md)** - Active development trajectory
- **[Version History](./VERSION_HISTORY.md)** - Release history and changes

#### Deployment & Operations
- **[Deployment Guide](./DEPLOYMENT_GUIDE.md)** - Production deployment instructions
- **[Production Operations Runbook](../0TML/docs/PRODUCTION_OPERATIONS_RUNBOOK.md)** - Operational procedures

---

### 🧪 Testing & Validation

- **[Testing Matrix](../0TML/docs/testing/README.md)** - Comprehensive test documentation
  - Unit, integration, and Byzantine attack tests
  - Auto-skip logic for missing dependencies
  - Re-enablement guides

- **[Performance Benchmarks](./performance/)** - Experimental results and metrics

---

## 🎯 Key Achievements

### Revolutionary Performance
- **45% Byzantine Tolerance** - Exceeding classical 33% BFT limit
- **100% Attack Detection Rate** at 45% adversarial ratio
- **+23pp Accuracy Improvement** over Multi-Krum baseline
- **HIPAA-Compliant** for healthcare federated learning

### Production-Ready Infrastructure
- **Multi-Backend Architecture** - PostgreSQL + Holochain + Ethereum + Cosmos
- **Real Cryptographic Security** - Bulletproofs, VRFs, ZK proofs
- **Comprehensive Testing** - 35+ Byzantine attack scenarios
- **Deployment Tools** - PyPI package, CLI tools, Docker support

### Epistemic Innovation
- **3D Epistemic Cube** - First framework classifying truth across E/N/M axes
- **Constitutional Governance** - Modular charter system with clear versioning
- **Agent-Centric Economy** - Holochain-based scalable local-first interactions

---

## 🧠 Core Concepts: Epistemic Cube

The revolutionary **Layered Epistemic Model v2.0** (3D "Epistemic Cube") classifies all claims along three independent axes:

### 1. E-Axis (Empirical Verifiability)
**Question**: How do we know this claim is true?

| Tier | Name | Description | Example |
|------|------|-------------|---------|
| E0 | Null | Unverifiable belief/opinion | "I believe X is good" |
| E1 | Testimonial | Personal attestation (DID-signed) | "I received this product" |
| E2 | Privately Verifiable | Audit Guild attestation | "Books are balanced" |
| E3 | Cryptographically Proven | ZKP/signature validation | "I'm over 18" (ZK proof) |
| E4 | Publicly Reproducible | Open data + open code | Scientific experiments |

### 2. N-Axis (Normative Authority)
**Question**: Who agrees this claim is binding?

| Tier | Name | Description | Example |
|------|------|-------------|---------|
| N0 | Personal | Self only (not a law) | "I'm selling this bike" |
| N1 | Communal | Local/Sector/Regional DAO | Artist royalty rules |
| N2 | Network | Global Mycelix consensus | Passed MIP |
| N3 | Axiomatic | Constitutional/Mathematical | Core principles, 2+2=4 |

### 3. M-Axis (Materiality)
**Question**: Why, where, for how long does this matter?

| Tier | Name | Description | State Management |
|------|------|-------------|------------------|
| M0 | Ephemeral | Transient signal | Discard immediately |
| M1 | Temporal | Valid until state changes | Prune after change |
| M2 | Persistent | Audit/reputation record | Archive after time |
| M3 | Foundational | Constitutional/existential | Preserve forever |

### Classification Examples

| Claim Type | Coordinate | Rationale |
|------------|-----------|-----------|
| A "like" | (E0, N0, M0) | Unverifiable, personal, ephemeral |
| Agora listing | (E1, N1, M1) | Testimonial, communal, temporary |
| Aura copyright | (E3, N1, M3) | Crypto-proven, communal, permanent |
| Passed MIP | (E0, N2, M3) | Belief, network consensus, permanent |
| Mathematics | (E4, N3, M3) | Reproducible, axiomatic, permanent |

**See [Epistemic Charter v2.0](./architecture/THE%20EPISTEMIC%20CHARTER%20(v2.0).md) for complete details.**

---

## 🚀 Project Status

### Phase 10: Byzantine Attack Testing ✅
- ✅ 35 experiment suite (7 attacks × 5 defenses)
- ✅ Comprehensive benchmarking infrastructure
- 🚧 Large-scale testing in progress

### Phase 9: Deployment & Packaging ✅
- ✅ PyPI package published
- ✅ CLI tools complete
- ✅ Production-ready

### v6.0 Development 🚧
- 🚧 Epistemic Charter v2.0 integration
- 🚧 Constitution v0.25 drafting
- 🚧 Enhanced governance framework

---

## 📈 Academic Timeline

| Milestone | Date | Status |
|-----------|------|--------|
| Section 3 Draft (PoGQ) | Oct 14, 2025 | ✅ Complete |
| Byzantine Testing | Nov 2025 | 🚧 In Progress |
| Full Draft | Dec 2025 | ⏳ Pending |
| MLSys/ICML Submission | Jan 15, 2026 | 🎯 Target |
| NSF CISE Grant Application | Jun 2026 | 📋 Planned |
| NIH R01 Application | Jun 2026 | 📋 Planned |

---

## 🗂️ File Organization

```
Mycelix-Core/
├── CLAUDE.md                           # ⭐ Central navigation hub
├── README.md                           # Project introduction
│
├── docs/                               # 📚 All documentation
│   ├── README.md                       # This file - master index
│   │
│   ├── architecture/                   # System architecture
│   │   ├── THE MYCELIX SPORE CONSTITUTION (v0.24).md
│   │   ├── THE EPISTEMIC CHARTER (v2.0).md ✨ LATEST
│   │   ├── THE GOVERNANCE CHARTER (v1.0).md
│   │   ├── THE ECONOMIC CHARTER (v1.0).md
│   │   ├── THE COMMONS CHARTER (v1.0).md
│   │   ├── Mycelix Protocol_ Integrated System Architecture v5.2.md
│   │   ├── mycelix_base_spec.md
│   │   └── Designing Mycelix Protocol SDK.md
│   │
│   ├── whitepaper/                     # Academic paper
│   │   ├── POGQ_WHITEPAPER_OUTLINE.md
│   │   ├── POGQ_WHITEPAPER_SECTION_3_DRAFT.md ✅
│   │   └── POGQ_INTEGRATION_PLAN.md
│   │
│   ├── grants/                         # Grant applications
│   │   └── ARCHITECTURE_COMPARISON_FOR_GRANTS.md
│   │
│   ├── DEPLOYMENT_GUIDE.md             # Deployment instructions
│   └── VERSION_HISTORY.md              # Release history
│
├── 0TML/                               # 🔬 Production implementation
│   ├── src/                            # Source code
│   ├── tests/                          # Test suite
│   └── docs/                           # Technical documentation (00-10)
│       └── README.md                   # Complete 0TML docs index
│
└── [other project files...]            # Development artifacts
```

---

## 🤝 Contributing

We welcome contributions! Key resources:

- **[Contributing Guide](../CONTRIBUTING.md)** - How to contribute
- **[0TML Developer Guide](../0TML/docs/03-developer-guide/README.md)** - Building on the platform
- **[Code Standards](../0TML/docs/03-developer-guide/BEST_PRACTICES.md)** - Best practices

---

## 🔗 External Resources

- **Live Site**: https://mycelix.net
- **GitHub**: https://github.com/Luminous-Dynamics/mycelix
- **Contact**: tristan.stoltz@evolvingresonantcocreationism.com
- **License**: Apache 2.0 (SDK) + Commercial licensing available

---

## 📋 Documentation Health

| Section | Status | Completeness | Notes |
|---------|--------|--------------|-------|
| Constitutional | ✅ Excellent | 100% | All 4 charters + constitution complete |
| Architecture | ✅ Excellent | 98% | v5.2 current, clear versioning |
| 0TML Technical | ✅ Excellent | 100% | All 10 sections complete |
| Whitepaper | 🚧 In Progress | 35% | Section 3 complete (3800/12200 words) |
| Grants | ✅ Good | 80% | Architecture comparison complete |
| Roadmap | ✅ Good | 90% | v5.3→v6.0 clear |

**Overall Documentation Status**: Excellent ✅

---

## 🔄 Recent Updates

- **Nov 10, 2025**: Documentation reorganization complete
  - Created CLAUDE.md central navigation hub
  - Enhanced docs/README.md with comprehensive navigation
  - Integrated Epistemic Charter v2.0 (3D Epistemic Cube)
  - Added clear version markers and supersession tracking

- **Oct 14, 2025**: Whitepaper Section 3 complete (3800 words)
- **Oct 2025**: Constitution v0.24 and all 4 charters v1.0 finalized

---

**Last Updated**: November 10, 2025
**Documentation Reorganization**: Complete ✅
**Next Major Update**: v6.0 Release (Q1 2026)

🍄 **Cultivating Collective Intelligence Through Rigorous Truth Infrastructure** 🍄
