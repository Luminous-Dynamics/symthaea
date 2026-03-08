# Changelog

All notable changes to Epistemic Markets will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive simulation engine for testing market mechanisms
- Performance optimization documentation
- Internationalization framework for 100+ languages
- Developer guide for building on the platform
- Community moderation policies and tools
- Persona-based onboarding journeys

## [0.1.0] - 2026-01-05

### Added

#### Core Market Infrastructure
- **Markets Zome**: Core prediction market functionality with LMSR, CDA, Parimutuel mechanisms
- **Question Markets Zome**: Trade on which questions are worth answering
- **Resolution Zome**: MATL-weighted oracle consensus with 45% Byzantine tolerance
- **Scoring Zome**: Calibration tracking, Brier scores, wisdom metrics
- **Predictions Zome**: Prediction submission with commit-reveal for independence
- **Markets Bridge Zome**: Cross-hApp integration for ecosystem-wide markets

#### E-N-M Epistemic Classification
- **Empirical Axis (E0-E4)**: From subjective to cryptographically verifiable
- **Normative Axis (N0-N3)**: From personal to universal consensus scope
- **Materiality Axis (M0-M3)**: From ephemeral to foundational time horizons
- Automatic resolution mechanism selection based on E-N-M position

#### Multi-Dimensional Stakes
- Monetary staking with configurable currencies
- Reputation staking tied to MATL domains
- Social staking with configurable visibility levels
- Commitment staking for conditional actions
- Time staking for research contributions

#### TypeScript SDK
- `EpistemicMarketsClient` unified interface
- `MarketsClient` for market operations
- `QuestionMarketsClient` for question discovery
- `ResolutionClient` for oracle participation
- Utility functions: `calculateStakeValue`, `calculateBrierScore`, `recommendResolution`

#### Documentation Suite
- **MANIFESTO.md**: Core values and invitation to participate
- **GETTING_STARTED.md**: From first prediction to wisdom keeper
- **DESIGN_PRINCIPLES.md**: 26 principles across 4 categories
- **GOVERNANCE.md**: Decision-making processes and checks
- **ECONOMICS.md**: Value flows, staking, sustainability
- **RITUALS.md**: Daily, weekly, monthly, annual practices
- **FAILURE_MODES.md**: Detection and healing patterns
- **EDGE_CASES.md**: Boundary condition handling
- **PERSONAS.md**: 8 personas with journeys and needs
- **SECURITY.md**: Threat model and protections
- **ADVANCED_MECHANISMS.md**: Belief graphs, attention markets
- **ECOSYSTEM_INTEGRATION.md**: Cross-hApp patterns
- **LONG_TERM_VISION.md**: Civilizational evolution
- **THE_DEEPER_VISION.md**: Eight Harmonies alignment
- **THE_LIVING_PROTOCOL.md**: System as living entity
- **RESEARCH_AGENDA.md**: Open questions
- **COMPARATIVE_ANALYSIS.md**: vs Polymarket, Metaculus, etc.
- **FAQ.md**: Common questions answered
- **CASE_STUDIES.md**: Real-world examples
- **GLOSSARY.md**: Complete vocabulary
- **API_REFERENCE.md**: Full SDK documentation
- **METRICS.md**: What we measure and why
- **ACCESSIBILITY.md**: Universal design
- **MODERATION.md**: Community policies
- **ONBOARDING.md**: Persona journeys
- **INTERNATIONALIZATION.md**: Multi-language support
- **DEVELOPER_GUIDE.md**: Building on platform
- **PERFORMANCE.md**: Optimization guide

#### Genesis Ceremony
- **GENESIS_MARKET.md**: The first prediction about the system itself
- **FOUNDING_WISDOM.md**: Seeds planted at genesis for future generations

#### Testing
- Comprehensive integration test suite (1000+ lines)
- Tests as philosophical teaching moments
- 7 test categories covering all aspects
- Simulation engine for mechanism validation

### Changed
- N/A (initial release)

### Deprecated
- N/A (initial release)

### Removed
- N/A (initial release)

### Fixed
- N/A (initial release)

### Security
- Commit-reveal protocol for prediction independence
- ZK-compatible stake proofs
- Rate limiting and anti-manipulation measures
- MATL-weighted Byzantine tolerance at 45%
- Tamper-evident prediction logs

## [0.0.1] - 2025-12-15

### Added
- Initial project structure
- Basic market data types
- Placeholder zomes

---

## Version Guidelines

### Major Version (X.0.0)
- Breaking changes to core market mechanics
- Fundamental changes to E-N-M classification
- Major restructuring of stake types
- Breaking SDK API changes

### Minor Version (0.X.0)
- New zome functionality
- New market mechanisms
- SDK feature additions
- New documentation sections

### Patch Version (0.0.X)
- Bug fixes
- Documentation corrections
- Performance improvements
- Security patches

---

## Migration Notes

### From 0.0.x to 0.1.0
This is a complete rewrite. No migration path from pre-release versions.

---

## Links

- [Documentation](./README.md)
- [Manifesto](./MANIFESTO.md)
- [Getting Started](./GETTING_STARTED.md)
- [API Reference](./docs/API_REFERENCE.md)

---

*"Each release is a ceremony - a moment when we offer our work to the world and invite it to grow."*
