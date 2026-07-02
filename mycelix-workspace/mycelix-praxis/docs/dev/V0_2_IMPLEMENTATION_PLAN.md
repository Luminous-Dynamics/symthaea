# v0.2.0 Holochain Integration - Implementation Plan

**Status**: In Progress
**Target**: Q1 2026
**Goal**: Implement actual Holochain zomes with HDK integration and web client connectivity

---

## Prerequisites

- [x] v0.1.0-alpha complete
- [x] Tests passing (28/28)
- [x] Core types and aggregation libraries working
- [ ] Holochain tooling installed (hc, holochain, lair-keystore)

---

## Phase 1: DNA Setup & Infrastructure (Week 1-2)

### 1.1 Create DNA Manifest
- [ ] Create `dna/` directory structure
- [ ] Write `dna/dna.yaml` with integrity/coordinator zome declarations
- [ ] Configure network settings
- [ ] Set up development conductor config

### 1.2 Add Holochain Dependencies
- [ ] Update zome Cargo.toml files with HDK dependencies
- [ ] Add `hdi` (Holochain Development Interface) to integrity zomes
- [ ] Add `hdk` (Holochain Development Kit) to coordinator zomes
- [ ] Configure WASM build targets

### 1.3 Development Environment
- [ ] Set up local conductor for development
- [ ] Create sandbox configuration
- [ ] Write helper scripts for DNA packaging
- [ ] Document development workflow

---

## Phase 2: Learning Zome Implementation (Week 3-4)

### 2.1 Integrity Zome (learning_integrity)
- [ ] Define `Course` entry type with HDI macros
- [ ] Define `LearnerProgress` entry type (private)
- [ ] Define `LearningActivity` entry type (private)
- [ ] Write entry validation functions
- [ ] Write link validation functions
- [ ] Add unit tests for validation logic

### 2.2 Coordinator Zome (learning_coordinator)
- [ ] Implement `create_course()`
- [ ] Implement `update_course()`
- [ ] Implement `get_course()`
- [ ] Implement `list_courses()`
- [ ] Implement `enroll()`
- [ ] Implement `update_progress()`
- [ ] Implement `get_progress()`
- [ ] Implement `record_activity()`
- [ ] Add integration tests

---

## Phase 3: FL Zome Implementation (Week 5-6)

### 3.1 Integrity Zome (fl_integrity)
- [ ] Define `FlRound` entry type
- [ ] Define `FlUpdate` entry type
- [ ] Write validation for FL round lifecycle
- [ ] Write validation for gradient commitments
- [ ] Write validation for clipping verification
- [ ] Add unit tests

### 3.2 Coordinator Zome (fl_coordinator)
- [ ] Implement `create_round()`
- [ ] Implement `join_round()`
- [ ] Implement `submit_update()`
- [ ] Implement `aggregate_updates()` using praxis-agg
- [ ] Implement `get_round_status()`
- [ ] Implement `list_rounds()`
- [ ] Add P2P gradient transfer logic
- [ ] Add integration tests

---

## Phase 4: Credential Zome Implementation (Week 7)

### 4.1 Integrity Zome (credential_integrity)
- [ ] Define `VerifiableCredential` entry type
- [ ] Define `CredentialStatus` entry type (for revocation)
- [ ] Write W3C VC spec validation
- [ ] Write signature verification validation
- [ ] Add unit tests

### 4.2 Coordinator Zome (credential_coordinator)
- [ ] Implement `issue_credential()`
- [ ] Implement `verify_credential()`
- [ ] Implement `revoke_credential()`
- [ ] Implement `get_credential()`
- [ ] Implement `list_credentials()`
- [ ] Add Ed25519 signature support
- [ ] Add integration tests

---

## Phase 5: DAO Zome Implementation (Week 8)

### 5.1 Integrity Zome (dao_integrity)
- [ ] Define `Proposal` entry type
- [ ] Define `Vote` entry type
- [ ] Write validation for proposal types (fast/normal/slow)
- [ ] Write validation for voting rules
- [ ] Add unit tests

### 5.2 Coordinator Zome (dao_coordinator)
- [ ] Implement `create_proposal()`
- [ ] Implement `vote()`
- [ ] Implement `execute()` for approved proposals
- [ ] Implement fast/normal/slow path logic
- [ ] Implement `get_proposal()`
- [ ] Implement `list_proposals()`
- [ ] Add integration tests

---

## Phase 6: Web Client Integration (Week 9-10)

### 6.1 Holochain Client Setup
- [ ] Install `@holochain/client` npm package
- [ ] Create connection manager utility
- [ ] Set up WebSocket connection to conductor
- [ ] Create zome call wrapper functions

### 6.2 Replace Mock Data
- [ ] Replace `mockCourses.ts` with real zome calls
- [ ] Replace `mockRounds.ts` with real zome calls
- [ ] Replace `mockCredentials.ts` with real zome calls
- [ ] Update components to handle real data
- [ ] Add loading states and error handling

### 6.3 Real Features
- [ ] Course discovery and enrollment flow
- [ ] FL round participation flow
- [ ] Credential display and sharing
- [ ] User profile management

---

## Phase 7: End-to-End Testing (Week 11)

### 7.1 Integration Tests
- [ ] Write test for complete FL round lifecycle
- [ ] Write test for course enrollment and progress
- [ ] Write test for credential issuance and verification
- [ ] Write test for DAO proposal and voting

### 7.2 Performance Testing
- [ ] Benchmark FL aggregation with 10+ participants
- [ ] Test DHT sync times
- [ ] Measure zome call latencies

### 7.3 Developer Experience
- [ ] Create local dev harness script
- [ ] Write comprehensive setup documentation
- [ ] Create troubleshooting guide

---

## Phase 8: Release Preparation (Week 12)

### 8.1 Documentation
- [ ] Update README with real installation instructions
- [ ] Update Protocol docs with actual zome APIs
- [ ] Create API reference documentation
- [ ] Write migration guide from v0.1

### 8.2 Release Artifacts
- [ ] Build production DNA bundles
- [ ] Package web client
- [ ] Create Docker compose for easy deployment
- [ ] Tag v0.2.0-beta release

---

## Success Criteria

- [ ] End-to-end FL round completes successfully (local conductor)
- [ ] Credentials are issuable and verifiable
- [ ] Web client fully functional with real Holochain backend
- [ ] 10+ integration tests passing
- [ ] Documentation updated and accurate
- [ ] Performance meets targets (<500ms zome calls)

---

## Tech Stack

- **Holochain**: 0.2+ (latest stable)
- **HDI/HDK**: Latest compatible version
- **Rust**: Stable (1.70+)
- **Node.js**: 18+ for web client
- **@holochain/client**: Latest stable

---

## Development Commands

```bash
# Build all zomes to WASM
make zomes

# Run conductor with DNA
holochain -c conductor-config.yaml

# Package DNA
hc dna pack dna/

# Run integration tests
cargo test --test integration

# Start web client
cd apps/web && npm run dev
```

---

## Next Actions (This Week)

1. **Set up Holochain tooling** in development environment
2. **Create DNA manifest** with basic structure
3. **Start with Learning Zome** integrity implementation
4. **Write first integration test** for course creation

---

**Last Updated**: 2025-12-10
**Maintainer**: Mycelix Praxis Team
**Status**: Ready to start Phase 1
