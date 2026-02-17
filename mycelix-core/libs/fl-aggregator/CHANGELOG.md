# Changelog

All notable changes to fl-aggregator will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- zkSTARK proof system with Winterfell integration
  - RangeProof: Value bounds verification
  - MembershipProof: Merkle tree membership
  - GradientIntegrityProof: FL gradient validation
  - IdentityAssuranceProof: Assurance level verification
  - VoteEligibilityProof: Governance voting eligibility
- Property-based testing with proptest
- Real zstd compression for distributed cache (proofs-compressed feature)
- Prometheus metrics for proof operations
- Timestamped proofs with validity periods
- Recursive proof composition for batch verification
- GPU acceleration framework (proofs-gpu-wgpu feature, placeholder)
- WebAssembly bindings (proofs-wasm feature)
- gRPC service integration (proofs-grpc feature)
- Blockchain anchoring support (proofs-anchoring feature)
- Rate limiting for proof generation
- Distributed proof generation coordinator
- HSM integration for key management
- MPC threshold signatures
- Comprehensive audit logging

### Changed
- Improved clippy compliance (0 warnings)
- Enhanced error handling with detailed ProofError types

### Fixed
- Benchmark edge case with out-of-range values
- Compression marker collision in distributed cache
- Various unused import warnings

## [0.1.0] - 2025-01-10

### Added
- Initial release of fl-aggregator
- Byzantine-fault-tolerant gradient aggregation
  - FedAvg, Krum, Trimmed Mean, Median, FLTrust defenses
- Gradient compression (TopK, Random-K, Quantization)
- Async aggregator with Tokio runtime
- HTTP API with Axum
- Python bindings with PyO3
- PostgreSQL and local storage backends
- Holochain integration (optional)
- Privacy module with differential privacy

[Unreleased]: https://github.com/Luminous-Dynamics/Mycelix-Core/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Luminous-Dynamics/Mycelix-Core/releases/tag/v0.1.0
