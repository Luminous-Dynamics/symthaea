# Changelog

All notable changes to Mycelix Mail will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- **HDK 0.6 Migration**: Updated all Holochain zomes to HDK 0.6.0 / HDI 0.7.0 APIs
  - `GetLinksInputBuilder` → `LinkQuery::try_new()` with `GetStrategy::default()`
  - `remote_signal` → `send_remote_signal` with explicit encoding
  - `delete_link` now requires `GetOptions` second parameter
  - `hash_entry(&string)` → Path-based anchors
  - `EntryCreationAction` → `Create` in validation callbacks
  - `BTreeSet` → `HashSet` for `GrantedFunctions`
  - `Signature` → `Vec<u8>` conversion for entry storage

### Added
- **Holochain TypeScript SDK** (`@mycelix/sdk/holochain`)
  - Full zome client implementations for all 12 coordinator zomes
  - React hooks: `useInbox`, `useSendEmail`, `useContacts`, `useTrustScore`, `useSearch`, `useSyncState`
  - Signal-based real-time updates via `MycelixProvider`
  - TypeScript types matching all Rust entry types
- Federation security model documentation
- Rate limiting for all zome calls with trust-based multipliers
- Stake requirements for trust attestations (Sybil resistance)

## [0.1.0] - 2026-01-03

### Added

#### Core Features
- **End-to-End Encrypted Email**: All messages encrypted using X3DH key exchange and Double Ratchet algorithm
- **MATL Trust System**: Multi-dimensional trust attestation with stake-weighted scoring
- **Decentralized Architecture**: Built on Holochain with no central server
- **Offline-First Design**: CRDT-based sync for seamless offline operation

#### Holochain Zomes
- `messages` - Email composition, storage, and threading
- `trust` - MATL trust attestation and scoring
- `contacts` - Address book management
- `profiles` - User profile management
- `keys` - Key management and rotation
- `capabilities` - Capability-based access control
- `sync` - CRDT synchronization
- `federation` - Cross-cell communication
- `search` - Local search indexing
- `backup` - Data export/import
- `scheduler` - Delayed and recurring emails
- `audit` - Compliance logging

#### TypeScript Client (`@mycelix/holochain-client`)
- Full-featured client library for Holochain integration
- React hooks: `useInbox`, `useSendEmail`, `useTrustNetwork`, `useContacts`
- Vue composables: `useMessages`, `useTrust`, `useContacts`
- Svelte stores with reactive updates
- Signal-based real-time updates via `SignalHub`
- Comprehensive service layer:
  - `CryptoService` - Client-side encryption
  - `CacheService` - LRU caching with TTL
  - `HealthService` - Connection monitoring
  - `BatchService` - Operation batching
  - `SpamFilterService` - Client-side spam detection
  - `SchedulerService` - Email scheduling
  - `SearchService` - Full-text search with filters
  - `MetricsService` - Performance monitoring
  - `AuditLogService` - Compliance logging

#### UI Components
- `TrustBadge` - Visual trust level indicator
- `TrustNetworkGraph` - Interactive trust network visualization
- `Avatar` - User avatar with status
- `EmailListItem` - Email list entry with swipe actions
- `SettingsPanel` - Comprehensive settings UI
- `MobileEmailList` - Mobile-optimized email list with gestures

#### Mobile Support
- Touch gesture handling (swipe, pinch, long-press)
- Pull-to-refresh functionality
- Mobile-optimized components
- PWA support with service worker

#### Developer Experience
- Storybook component documentation
- Playwright E2E test suite
- Comprehensive TypeScript types
- React hooks for all features
- Vue composables for all features

#### DevOps
- Docker Compose setup for development
- Production Docker configuration
- Makefile with 40+ build targets
- GitHub Actions CI/CD pipeline
- Multi-node test network scripts

#### Documentation
- Architecture guide
- Getting started guide
- Security audit report
- Federation security model
- API reference

### Security
- Ed25519 keypair authentication via Lair Keystore
- AES-256-GCM symmetric encryption
- X3DH key exchange for session establishment
- Double Ratchet algorithm for forward secrecy
- Rate limiting on all zome calls
- Stake requirements for trust attestations
- Replay attack prevention with nonce tracking
- TLS 1.3 for WebSocket connections

### Performance
- LRU caching for frequently accessed data
- Trust score memoization
- Batch operations for bulk actions
- Lazy loading of email bodies
- Background sync scheduling

---

## Version History

### Semantic Versioning

This project uses Semantic Versioning:

- **MAJOR** version for incompatible API changes
- **MINOR** version for backwards-compatible functionality additions
- **PATCH** version for backwards-compatible bug fixes

### Pre-release Versions

- `0.x.x` - Initial development phase (API may change)
- `1.0.0` - First stable release (planned)

---

## Migration Guides

### Upgrading to 0.1.0

This is the initial release. No migration required.

---

## Deprecation Notices

None at this time.

---

## Contributors

See [CONTRIBUTORS.md](./CONTRIBUTORS.md) for a list of contributors.

---

[Unreleased]: https://github.com/luminous-dynamics/mycelix-mail/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/luminous-dynamics/mycelix-mail/releases/tag/v0.1.0
