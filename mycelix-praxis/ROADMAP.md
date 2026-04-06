# Mycelix Praxis Roadmap

**Status**: Public
**Last Updated**: 2025-11-15
**Version**: 1.0

This roadmap outlines our vision and planned milestones for Mycelix Praxis. Dates are targets and may shift based on community feedback and development progress.

---

## Vision

Build a **privacy-preserving, decentralized education platform** where:
- Learners own their data
- AI personalizes without seeing raw data
- Credentials are portable and verifiable
- Communities govern curricula
- Everyone can contribute

---

## Release Timeline

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  v0.1     v0.2        v0.3           v1.0         v2.0      │
│   │        │           │              │            │        │
│   │        │           │              │            │        │
│   ▼        ▼           ▼              ▼            ▼        │
│  Nov     Q1 2026    Q2 2026       Q3 2026      2027+       │
│  2025                                                       │
│                                                             │
│  Alpha   DAO +      Pilot        Production   Scaling      │
│          DP        (100 users)    Ready      & Growth      │
└─────────────────────────────────────────────────────────────┘
```

---

## v0.1.0 - Alpha Foundation ✅ **DONE** (Nov 2025)

**Status**: Released 🎉

**Goal**: Establish architecture, libraries, and developer onboarding

### Delivered
- ✅ Monorepo structure (apps, crates, zomes, docs)
- ✅ Core Rust libraries (`praxis-core`, `praxis-agg`)
- ✅ Zome data structures (library-only)
- ✅ React web client (UI mockups)
- ✅ W3C VC schemas
- ✅ Comprehensive documentation
  - Protocol specification
  - Threat model
  - Privacy model
  - Architecture diagrams
  - Getting started guide
  - FAQ
- ✅ Example data (credentials, FL rounds, courses)
- ✅ CI/CD pipelines
- ✅ Governance framework (GOVERNANCE.md)
- ✅ VS Code development environment
- ✅ Phase 2 improvements (Nov 2025)
  - FAQ with 30+ questions
  - ROADMAP (this document)
  - VS Code configuration
  - Pre-commit hooks
  - Dependabot

### Metrics
- 59 files created
- 7,800+ lines of code/docs
- 14 passing tests
- 100% CI success rate

---

## v0.2.0 - Holochain Integration (Target: Q1 2026)

**Status**: In Planning

**Goal**: Implement actual Holochain zomes and web client integration

### Features

#### Holochain DNA
- [ ] DNA manifest (`dna/dna.yaml`)
- [ ] Integrity + coordinator zome split
- [ ] HDK integration for all zomes

#### Learning Zome
- [ ] Entry definitions (Course, LearnerProgress, LearningActivity)
- [ ] Validation functions
- [ ] Zome functions (create_course, enroll, update_progress)
- [ ] Links (creator → courses, learner → progress)
- [ ] Integration tests

#### FL Zome
- [ ] Entry definitions (FlRound, FlUpdate)
- [ ] Validation functions (clipping, commitment verification)
- [ ] Zome functions (create_round, join_round, submit_update, aggregate)
- [ ] P2P gradient transfer
- [ ] Integration tests

#### Credential Zome
- [ ] Entry definitions (VerifiableCredential)
- [ ] Validation functions (W3C VC spec compliance)
- [ ] Zome functions (issue_credential, verify_credential)
- [ ] Ed25519 signature verification
- [ ] Revocation lists

#### DAO Zome
- [ ] Entry definitions (Proposal, Vote)
- [ ] Validation functions (voting rules)
- [ ] Zome functions (create_proposal, vote, execute)
- [ ] Fast/normal/slow path logic

#### Web Client
- [ ] Holochain conductor connection
- [ ] Call zome functions
- [ ] Real data (not mocks)
- [ ] Course discovery and enrollment flow
- [ ] FL round participation flow
- [ ] Credential display and sharing

#### Developer Experience
- [ ] Local dev harness (conductor + web)
- [ ] End-to-end testing framework
- [ ] Performance benchmarks

### Success Criteria
- [ ] End-to-end FL round completes successfully (local)
- [ ] Credentials issuable and verifiable
- [ ] Web client fully functional
- [ ] 10+ integration tests
- [ ] Documentation updated

### Timeline
- **Dec 2025**: DNA structure, HDK integration
- **Jan 2026**: Learning + FL zomes implemented
- **Feb 2026**: Credential + DAO zomes implemented
- **Mar 2026**: Web integration, testing, v0.2.0 release

---

## v0.3.0 - Community Pilot (Target: Q2 2026)

**Status**: Planned

**Goal**: Run a pilot with 50-100 real users, gather feedback

### Features

#### Differential Privacy
- [ ] Client-side DP implementation
- [ ] Configurable ε, δ parameters
- [ ] Privacy budget tracking
- [ ] Trade-off visualization (privacy vs. accuracy)

#### Multi-Coordinator Verification
- [ ] Multiple coordinators aggregate independently
- [ ] Compare results for consensus
- [ ] Detect malicious coordinators
- [ ] Fallback mechanisms

#### Enhanced DAO
- [ ] Reputation system
- [ ] Delegated voting
- [ ] Proposal templates
- [ ] Treasury management

#### Mobile App (React Native)
- [ ] iOS and Android support
- [ ] Offline-first architecture
- [ ] Push notifications for FL rounds
- [ ] Credential wallet
- [ ] Course downloads

#### Pilot Program
- [ ] 3-5 sample courses (language learning, skills training)
- [ ] 50-100 beta users
- [ ] Telemetry and analytics (privacy-preserving)
- [ ] User feedback surveys
- [ ] Performance monitoring

### Success Criteria
- [ ] 50+ active users
- [ ] 5+ FL rounds completed
- [ ] 100+ credentials issued
- [ ] User satisfaction > 4.0/5.0
- [ ] No critical bugs
- [ ] DAO proposals created and executed

### Timeline
- **Apr 2026**: DP implementation, multi-coordinator
- **May 2026**: Mobile app development
- **Jun 2026**: Pilot launch, feedback collection
- **Jul 2026**: Iteration based on feedback, v0.3.0 release

---

## v1.0.0 - Production Ready (Target: Q3 2026)

**Status**: Planned

**Goal**: Production-ready system with security audit, ready for public launch

### Features

#### Security
- [ ] External security audit (penetration testing)
- [ ] Formal threat model verification
- [ ] Bug bounty program
- [ ] Incident response plan
- [ ] Security disclosures handled

#### Performance
- [ ] 1,000+ participants per FL round
- [ ] Sub-500ms zome call latency
- [ ] Horizontal scalability validated
- [ ] Load testing completed
- [ ] Performance benchmarks published

#### User Experience
- [ ] Onboarding flow (<5 minutes)
- [ ] Tutorial courses (interactive)
- [ ] Help center and documentation
- [ ] Accessibility audit (WCAG 2.1 AA)
- [ ] i18n support (English, Spanish, French, Chinese)

#### Compliance
- [ ] GDPR compliance audit
- [ ] FERPA compliance audit
- [ ] Privacy policy
- [ ] Terms of service
- [ ] Cookie consent

#### Community
- [ ] 10+ regular contributors
- [ ] 100+ GitHub stars
- [ ] 5+ production courses
- [ ] Active forum/discussion board
- [ ] Monthly community calls

#### Infrastructure
- [ ] Bootstrap nodes deployed
- [ ] CDN for static assets
- [ ] Monitoring and alerting
- [ ] Backup and disaster recovery
- [ ] SLA definitions

### Success Criteria
- [ ] Security audit passed
- [ ] Performance targets met
- [ ] Compliance requirements met
- [ ] 100+ launch day users
- [ ] No critical bugs in first month
- [ ] Positive press coverage

### Timeline
- **Aug 2026**: Security audit, performance testing
- **Sep 2026**: Compliance audits, final polishing
- **Oct 2026**: Public launch, v1.0.0 release

---

## v2.0+ - Scaling & Growth (2027+)

**Status**: Vision

**Goal**: Scale to 10,000+ users, add advanced features

### Potential Features

#### Advanced FL
- [ ] Asynchronous federated learning
- [ ] Hierarchical FL (multiple aggregation layers)
- [ ] Secure aggregation (MPC)
- [ ] Byzantine-robust methods (Krum, Bulyan)
- [ ] Model compression (sparsification, quantization)

#### Advanced Privacy
- [ ] Zero-knowledge proofs for credentials
- [ ] Anonymous credentials (selective disclosure)
- [ ] Mix networks for communication privacy
- [ ] Trusted execution environments (TEE)

#### Ecosystem
- [ ] Third-party integrations (LMS, job platforms)
- [ ] Credential marketplace
- [ ] Course marketplace
- [ ] FL-as-a-Service API
- [ ] White-label deployments

#### Research
- [ ] Academic papers published
- [ ] Conference presentations
- [ ] Research partnerships
- [ ] Benchmark datasets

---

## Community Goals

### Contribution Targets

**2025** (Nov-Dec):
- 5+ contributors
- 10+ merged PRs
- 50+ GitHub stars

**2026 Q1**:
- 10+ contributors
- 30+ merged PRs
- 100+ GitHub stars
- First external PR merged

**2026 Q2**:
- 20+ contributors
- 60+ merged PRs
- 200+ GitHub stars
- Community governance proposal

**2026 Q3+**:
- 50+ contributors
- 100+ merged PRs
- 500+ GitHub stars
- Active community-led initiatives

### Ecosystem Growth

**Courses**:
- 2025: 3 sample courses
- 2026 Q1: 5 courses
- 2026 Q2: 10 courses
- 2026 Q3: 20+ courses

**Users**:
- 2025: <10 (developers)
- 2026 Q1: 10-50 (early adopters)
- 2026 Q2: 50-200 (pilot)
- 2026 Q3: 200-1,000 (launch)
- 2027: 1,000-10,000 (growth)

---

## How to Influence the Roadmap

We welcome community input!

### Propose Features
1. Open a [Discussion](https://github.com/Luminous-Dynamics/mycelix-praxis/discussions) with tag `roadmap`
2. Describe the feature and use case
3. Community discusses and votes
4. Maintainers evaluate feasibility
5. If accepted, issue created and added to milestone

### Vote on Priorities
- Comment on existing roadmap issues
- Upvote (👍) features you want
- Share your use case
- Volunteer to implement!

### Contribute
- Pick an issue from the milestone
- Implement and open PR
- Help us ship faster!

---

## Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Holochain breaking changes | HIGH | MEDIUM | Pin versions, test upgrades carefully |
| Security vulnerabilities | HIGH | LOW | Audit before v1.0, bug bounty |
| Poor user adoption | MEDIUM | MEDIUM | Pilot program, user feedback, marketing |
| Contributor burnout | MEDIUM | MEDIUM | Distribute responsibilities, onboard co-maintainers |
| Regulatory challenges | MEDIUM | LOW | Compliance audits, legal review |
| Performance issues | MEDIUM | MEDIUM | Early performance testing, profiling |

---

## Versioning

We follow [Semantic Versioning](https://semver.org/):

- **Major (X.0.0)**: Breaking changes, incompatible API
- **Minor (0.X.0)**: New features, backward-compatible
- **Patch (0.0.X)**: Bug fixes, backward-compatible

---

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for detailed version history.

---

## Questions?

- **Roadmap feedback**: [Discussions](https://github.com/Luminous-Dynamics/mycelix-praxis/discussions)
- **Feature requests**: [Issues](https://github.com/Luminous-Dynamics/mycelix-praxis/issues/new?template=feature_request.md)
- **General questions**: [FAQ](docs/faq.md)

---

**Last Updated**: 2025-11-15
**Next Review**: 2026-02-15 (quarterly)

**Let's build the future of education together! 🚀**
