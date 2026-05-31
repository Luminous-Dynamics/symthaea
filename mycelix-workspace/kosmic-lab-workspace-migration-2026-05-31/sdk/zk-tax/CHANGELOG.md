# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- 58 jurisdiction support (up from 19 G20 countries)
- Proof caching layer with LRU eviction and TTL
- Global cache singleton for application-wide caching
- TypeScript SDK with full type definitions
- Python bindings via PyO3/maturin
- Comprehensive benchmarks with Criterion
- GitHub Actions CI/CD pipeline
- Fuzz testing infrastructure with cargo-fuzz
- New examples: business entities, subnational taxes, caching

### Improved
- Enhanced TypeScript types with region information
- Better cache configuration presets (high_performance, strict, persistent)
- Expanded compression options

## [0.1.0] - 2024-12-XX

### Added
- Initial release
- Zero-knowledge tax bracket proofs using RISC Zero zkVM
- 19 G20 country support with 2024/2025 tax brackets
- US state income tax brackets (all 50 states)
- Swiss cantonal tax brackets (all 26 cantons)
- Canadian provincial tax brackets
- Filing status support: Single, MFJ, MFS, HOH
- Proof types:
  - TaxBracketProof - Basic bracket membership
  - RangeProof - Income within bounds
  - BatchProof - Multi-year history
  - EffectiveTaxRateProof - Average tax rate
  - CrossJurisdictionProof - Multiple countries
  - DeductionProof - Itemized deductions
  - CompositeProof - Combined proofs
  - ProofChain - Audit trail with SHA3-256 linking
- WASM bindings for browser usage
- CLI tool for command-line proof generation
- Axum HTTP server example
- Solidity verifier contract
- Holochain zome integration
- Dev mode for fast testing
- GPU acceleration support (CUDA/Metal)
- Structured tracing/logging support
- no_std compatibility with alloc feature

### Features
- `prover` - Full RISC Zero zkVM integration
- `verifier-only` - Lightweight verification
- `wasm` - WebAssembly support
- `cli` - Command-line interface
- `server` - Axum HTTP server
- `anchoring` - Blockchain proof anchoring
- `entity` - Business entity taxation
- `service` - Rate limiting and metrics
- `aggregation` - Merkle tree batching
- `tracing-support` - Observability
- `cuda`/`metal` - GPU acceleration
- `full` - All features enabled

## Links

- [Documentation](https://docs.rs/mycelix-zk-tax)
- [Repository](https://github.com/Luminous-Dynamics/mycelix)
- [npm Package](https://www.npmjs.com/package/@mycelix/zk-tax)
- [PyPI Package](https://pypi.org/project/mycelix-zk-tax)
