# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive improvement plan (Phase 1-6)
- Example credentials, FL rounds, and test fixtures
- Architecture diagrams (Mermaid + ASCII)
- Getting started guide
- FAQ documentation
- VS Code workspace configuration

### Changed
- Zomes temporarily converted to library-only crates (removed HDK dependencies for clean builds)
- CI now runs successfully on all checks

### Fixed
- Rust workspace compilation errors
- All tests now pass

## [0.1.0] - 2025-11-15

### Added
- Initial monorepo structure
  - `apps/web`: React + TypeScript web client with Vite
  - `crates/`: Rust libraries (praxis-core, praxis-agg)
  - `zomes/`: Holochain zome scaffolds (learning, FL, credentials, DAO)
  - `schemas/`: W3C VC schemas + DHT entry definitions
  - `docs/`: Protocol, threat model, privacy, ADRs
  - `scripts/`: Development tooling (dev.sh, hc-reset.sh)
- Root configuration files
  - `.gitignore`, `.editorconfig`
  - `Makefile` with build/test/dev commands
  - `CODEOWNERS` for repository ownership
- Governance and policy documents
  - `LICENSE` (Apache-2.0)
  - `CODE_OF_CONDUCT.md` (Contributor Covenant 2.1)
  - `CONTRIBUTING.md` - Comprehensive contribution guidelines
  - `SECURITY.md` - Vulnerability reporting policy
  - `GOVERNANCE.md` - Decision-making framework (fast/normal/slow paths)
- Rust workspace
  - **praxis-core**: Core types, crypto (BLAKE3), provenance tracking
  - **praxis-agg**: Robust aggregation (trimmed mean, median, weighted mean) with tests
- Holochain zome scaffolds
  - **learning_zome**: Courses, learner progress, learning activities
  - **fl_zome**: Federated learning rounds and updates
  - **credential_zome**: W3C Verifiable Credentials
  - **dao_zome**: Governance proposals and voting
- Web application
  - React 18 + TypeScript + Vite
  - ESLint + Prettier configured
  - Sample UI demonstrating platform features
- Verifiable Credentials schemas
  - `EduAchievementCredential.schema.json` (W3C VC spec)
  - JSON-LD context (`praxis-v1.jsonld`)
- Comprehensive documentation
  - **Protocol specification** (`docs/protocol.md`): 6-phase FL lifecycle
  - **Threat model** (`docs/threat-model.md`): 10 attack vectors + mitigations
  - **Privacy model** (`docs/privacy.md`): Privacy protections, GDPR/FERPA compliance
  - **ADR framework** with ADR-0001 (FL Protocol v0)
  - DHT entry definitions
- Development tools
  - `scripts/dev.sh` - One-command development environment startup
  - `scripts/hc-reset.sh` - Holochain state reset utility
- GitHub infrastructure
  - CI workflow (Rust + Web builds, tests, security audits)
  - Release workflow (automated releases with artifacts)
  - Issue templates (bug report, feature request, design proposal)
  - PR template with security/privacy checklist
- Root README.md
  - Quick start guide
  - Architecture overview
  - Complete documentation index
  - Roadmap (v0.1-alpha → v1.0)

### Security
- Implemented gradient clipping for FL privacy
- Commitment-reveal scheme for gradient submissions
- Robust aggregation to mitigate poisoning attacks
- Comprehensive threat model analyzing 10 attack vectors
- Security policy with vulnerability reporting process

## [0.0.0] - 2025-11-14

### Added
- Initial repository creation
- Project planning and design

---

## Release Types

- **Major (X.0.0)**: Breaking changes, incompatible API changes
- **Minor (0.X.0)**: New features, backward-compatible
- **Patch (0.0.X)**: Bug fixes, backward-compatible

## Planned Releases

- **v0.2.0** (Target: Q1 2026): DAO governance, DP support, multi-coordinator verification
- **v0.3.0** (Target: Q2 2026): Mobile app, community pilot (50-100 users)
- **v1.0.0** (Target: Q3 2026): Production-ready FL protocol, external security audit, public launch

---

[Unreleased]: https://github.com/Luminous-Dynamics/mycelix-praxis/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Luminous-Dynamics/mycelix-praxis/releases/tag/v0.1.0
[0.0.0]: https://github.com/Luminous-Dynamics/mycelix-praxis/commit/initial
