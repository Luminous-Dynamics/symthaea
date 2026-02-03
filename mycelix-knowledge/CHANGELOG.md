# Changelog

All notable changes to Mycelix Knowledge will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2024-01-15

### Added

#### Core Zomes
- **claims/** - Claim management with E-N-M epistemic classification
  - Create, update, delete claims
  - Evidence attachment
  - Domain and topic categorization
  - Claim status lifecycle (Draft → Published → Verified)
  - Verification market spawning
  - Dependency registration for belief propagation

- **graph/** - Knowledge graph relationships
  - 8 relationship types (Supports, Contradicts, DerivedFrom, etc.)
  - Weighted relationships
  - Belief propagation algorithm
  - Circular dependency detection
  - Information value ranking
  - Cascade impact calculation

- **query/** - Search and discovery
  - Full-text search
  - Epistemic range queries (E-N-M filtering)
  - Domain and topic filtering
  - Advanced query builder
  - Pagination support

- **inference/** - Credibility and reasoning
  - Enhanced credibility scoring with MATL² integration
  - Evidence strength assessment
  - Author reputation tracking
  - Domain expertise scoring
  - Batch credibility assessment

- **factcheck/** - Fact-checking API
  - Statement verification against knowledge graph
  - 7 verdict types (True through InsufficientEvidence)
  - Confidence scoring
  - Supporting/contradicting claim identification
  - External hApp integration

- **markets_integration/** - Epistemic Markets bridge
  - Bidirectional verification market integration
  - Market value assessment
  - Market resolution callbacks
  - Claim-as-evidence tracking

- **bridge/** - Cross-hApp communication
  - hApp registration and authentication
  - Permission levels (Full, Standard, Limited)
  - Domain access control
  - External claim submission

#### TypeScript SDK (@mycelix/knowledge-sdk)
- Full client for all zomes
- Type-safe interfaces
- Service layer abstractions
  - KnowledgeService for high-level workflows
  - BeliefGraphService for graph operations
- Utility functions
  - `toDiscreteEpistemic()` - E-N-M discretization
  - `calculateInformationValue()` - IV computation
  - `recommendVerification()` - Market recommendations
  - `calculateCompositeCredibility()` - Score aggregation
  - `determineVerdict()` - Fact-check verdict logic

#### Documentation (28 files)
- **Concepts**: E-N-M classification, belief graphs, credibility engine, information value
- **Integration**: Epistemic Markets, MATL, fact-check API, cross-hApp bridge
- **Tutorials**: 5 step-by-step guides
- **Reference**: API reference, zome functions, entry types, error codes
- **Governance**: Epistemic charter, claim moderation, design principles
- **Operations**: Security, performance, accessibility, metrics
- **Root**: README, getting started, contributing, manifesto

#### Simulations
- Knowledge simulation engine
- 5 scenario simulations:
  - Verification cascade
  - Contradiction detection
  - Cross-hApp fact-check
  - Byzantine resilience (45% adversarial tolerance)
  - Information value ranking

#### Testing
- SDK unit tests
- Holochain integration tests (Tryorama)
- Multi-agent test scenarios

#### Infrastructure
- GitHub Actions CI/CD pipeline
- Rust formatting and linting
- TypeScript type checking
- Documentation link checking
- Automated releases
- npm publishing

### Technical Details

- **Holochain HDK**: 0.6.x
- **TypeScript**: 5.3+
- **Node.js**: 20+
- **Rust**: 2021 edition

### Architecture

```
Knowledge hApp
├── claims/       - Claim lifecycle management
├── graph/        - Relationship and belief graphs
├── query/        - Search and discovery
├── inference/    - Credibility computation
├── factcheck/    - External verification API
├── markets_integration/ - Epistemic Markets bridge
└── bridge/       - Cross-hApp communication
```

### Dependencies

- `@holochain/client`: ^0.18.0
- `@mycelix/epistemic-markets-sdk`: ^0.1.0 (peer)

---

## Version History

| Version | Date | Highlights |
|---------|------|------------|
| 0.1.0 | 2024-01-15 | Initial release with full feature set |

---

## Migration Guides

### From 0.0.x (Pre-release)

No migration needed - this is the first release.

### Future Migrations

Migration guides will be added here as breaking changes are introduced.

---

## Contributing

See [CONTRIBUTING.md](./docs/CONTRIBUTING.md) for contribution guidelines.

## License

MIT License - see [LICENSE](./LICENSE) for details.
