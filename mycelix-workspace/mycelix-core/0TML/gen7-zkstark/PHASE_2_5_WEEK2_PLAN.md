# Phase 2.5 Week 2 Plan: Coordinator Integration & Testing

**Start Date**: TBD (After Week 1 completion)
**Target Duration**: 5 days
**Status**: Planning

---

## 🎯 Week 2 Objectives

Integrate Dilithium authentication into the coordinator and perform comprehensive E7.5 testing with authenticated proofs.

### Success Criteria
- ✅ Client registry operational (PostgreSQL/SQLite)
- ✅ Nonce tracking prevents all replay attacks
- ✅ Password-encrypted key management working
- ✅ Performance benchmarks meet targets (<10ms proof time)
- ✅ E7.5 integration test passes (5 rounds, 10 clients)

---

## 📋 Detailed Task Breakdown

### Day 1-2: Client Registry & Nonce Tracking

#### Task 1.1: Database Schema Design
**File**: `src/zerotrustml/gen7/client_registry.py`

**Schema Requirements**:
```sql
CREATE TABLE clients (
    client_id BLOB PRIMARY KEY,          -- 32 bytes (SHA-256 of public key)
    public_key BLOB NOT NULL,            -- 2592 bytes (Dilithium5)
    registered_at INTEGER NOT NULL,      -- Unix timestamp
    last_seen INTEGER,                   -- Unix timestamp
    total_rounds INTEGER DEFAULT 0,      -- Participation count
    reputation_score REAL DEFAULT 1.0    -- For future PoGQ integration
);

CREATE TABLE nonces (
    client_id BLOB NOT NULL,
    nonce BLOB NOT NULL,                 -- 32 bytes
    round_number INTEGER NOT NULL,
    timestamp INTEGER NOT NULL,
    PRIMARY KEY (client_id, nonce),
    FOREIGN KEY (client_id) REFERENCES clients(client_id)
);

CREATE INDEX idx_nonces_round ON nonces(round_number);
CREATE INDEX idx_nonces_timestamp ON nonces(timestamp);
```

**Implementation**:
- SQLite for local testing
- PostgreSQL for production
- Connection pooling (SQLAlchemy)
- Automatic cleanup (delete nonces > 1 hour old)

#### Task 1.2: Client Registration API
**File**: `src/zerotrustml/gen7/client_registry.py`

**Methods**:
```python
class ClientRegistry:
    def register_client(self, public_key: bytes) -> bytes:
        """Register new client, return client_id"""

    def get_client(self, client_id: bytes) -> Optional[Client]:
        """Retrieve client by ID"""

    def is_nonce_used(self, client_id: bytes, nonce: bytes) -> bool:
        """Check if nonce already used"""

    def mark_nonce_used(
        self,
        client_id: bytes,
        nonce: bytes,
        round_number: int,
        timestamp: int
    ) -> None:
        """Store used nonce"""

    def cleanup_old_nonces(self, max_age_seconds: int = 3600) -> int:
        """Remove expired nonces, return count deleted"""
```

**Testing**:
- Unit tests for each method
- Concurrent access stress test (100 simultaneous registrations)
- Nonce collision test (attempt replay)

#### Task 1.3: Integration with Coordinator
**File**: `src/zerotrustml/gen7/authenticated_gradient_proof.py`

**Update** `AuthenticatedGradientCoordinator.verify_proof()`:
```python
def verify_proof(
    self,
    auth_proof: AuthenticatedProof,
) -> Tuple[bool, str]:
    # 1. Check client registered
    client = self.registry.get_client(auth_proof.client_id)
    if client is None:
        return False, "Client not registered"

    # 2. Check nonce not used (NEW)
    if self.registry.is_nonce_used(auth_proof.client_id, auth_proof.nonce):
        return False, "Nonce already used (replay attack detected)"

    # 3. Verify proof via Rust (existing)
    is_valid, error = auth_proof.verify(...)
    if not is_valid:
        return False, error

    # 4. Mark nonce as used (NEW)
    self.registry.mark_nonce_used(
        auth_proof.client_id,
        auth_proof.nonce,
        auth_proof.round_number,
        auth_proof.timestamp
    )

    # 5. Update client stats (NEW)
    self.registry.update_participation(auth_proof.client_id)

    return True, "Proof verified and nonce recorded"
```

---

### Day 3: Key Management

#### Task 3.1: Password-Encrypted Keypair Storage
**File**: `src/zerotrustml/gen7/key_management.py`

**Requirements**:
- Password-based encryption (PBKDF2 + AES-256-GCM)
- Store encrypted secret key only (public key derivable)
- Salt + IV stored with ciphertext
- Argon2id for key derivation (recommended over PBKDF2)

**Implementation**:
```python
class KeyManager:
    def save_keypair(
        self,
        keypair: DilithiumKeypair,
        password: str,
        path: str = "~/.ztml/client.key"
    ) -> None:
        """Encrypt and save secret key"""

    def load_keypair(
        self,
        password: str,
        path: str = "~/.ztml/client.key"
    ) -> DilithiumKeypair:
        """Decrypt and load secret key"""

    def change_password(
        self,
        old_password: str,
        new_password: str,
        path: str = "~/.ztml/client.key"
    ) -> bool:
        """Re-encrypt with new password"""
```

**File Format** (binary):
```
[Header: 8 bytes "ZTMLKEY\x01"]
[Salt: 32 bytes]
[IV: 16 bytes]
[Ciphertext: variable (secret_key encrypted)]
[Auth tag: 16 bytes (GCM)]
```

**Testing**:
- Correct password decrypt
- Wrong password reject
- Password change
- Corrupted file detection

---

### Day 4: Performance Benchmarks

#### Task 4.1: Proof Generation Benchmarks
**File**: `experiments/benchmark_phase2_5.py`

**Metrics to Measure**:
```python
def benchmark_proof_generation():
    """Measure proof generation time breakdown"""

    # 1. zkSTARK proof generation (from Phase 2)
    # 2. Dilithium signature generation (NEW)
    # 3. Message construction overhead (NEW)
    # 4. Total end-to-end time

    # Target: <10ms total (Phase 2 was 4.8ms)
    # Expected: ~6-8ms (zkSTARK 4.8ms + Dilithium 1-3ms)
```

**Test Cases**:
- 1000 proof generations (get mean + std dev)
- Gradient sizes: 100KB, 1MB, 10MB
- Model sizes: 10 params, 1000 params, 100K params

#### Task 4.2: Verification Benchmarks
**File**: `experiments/benchmark_phase2_5.py`

**Metrics**:
```python
def benchmark_verification():
    """Measure verification time breakdown"""

    # 1. Dilithium signature verification (NEW)
    # 2. Database nonce lookup (NEW)
    # 3. zkSTARK verification (from Phase 2)
    # 4. Total coordinator overhead

    # Target: <3ms total (Phase 2 was ~1ms STARK verify)
    # Expected: ~2ms (Dilithium 1ms + nonce 0.1ms + STARK 1ms)
```

#### Task 4.3: Throughput Testing
**File**: `experiments/benchmark_phase2_5.py`

**Scenario**:
- 100 concurrent clients
- Each submits 10 proofs
- Measure coordinator throughput (proofs/second)

**Target**: >100 proofs/second verification

---

### Day 5: E7.5 Integration Testing

#### Task 5.1: Full Federated Learning Round
**File**: `experiments/test_e7_5_authenticated.py`

**Test Configuration**:
```python
config = {
    "num_clients": 10,
    "num_rounds": 5,
    "dataset": "MNIST",
    "model": "SimpleCNN",
    "aggregation": "FedAvg",
    "authentication": "zk-DASTARK"  # NEW
}
```

**Test Flow**:
```python
def test_authenticated_federated_learning():
    # Setup
    coordinator = AuthenticatedGradientCoordinator()
    clients = [AuthenticatedGradientClient() for _ in range(10)]

    # Register all clients
    for client in clients:
        coordinator.register_client(client.keypair.get_public_key())

    # Run 5 rounds
    for round_num in range(5):
        # 1. Each client generates authenticated proof
        proofs = []
        for client in clients:
            proof = client.generate_proof(
                gradient=local_gradient,
                model_params=global_model,
                local_data=client_data,
                local_labels=client_labels,
                round_number=round_num
            )
            proofs.append(proof)

        # 2. Coordinator verifies all proofs
        verified_gradients = []
        for proof in proofs:
            is_valid, error = coordinator.verify_proof(proof)
            assert is_valid, f"Proof verification failed: {error}"
            verified_gradients.append(extract_gradient(proof))

        # 3. Aggregate verified gradients
        global_model = aggregate(verified_gradients)

    # Verify final accuracy
    assert final_accuracy > 0.95  # MNIST target
```

#### Task 5.2: Attack Scenarios
**File**: `experiments/test_e7_5_attacks.py`

**Test Cases**:

**Test A: Replay Attack**
```python
def test_replay_attack_prevention():
    # Client submits valid proof for round 1
    proof_r1 = client.generate_proof(..., round_number=1)
    assert coordinator.verify_proof(proof_r1)[0]  # First time: OK

    # Attempt to replay same proof
    assert not coordinator.verify_proof(proof_r1)[0]  # Second time: FAIL
```

**Test B: Cross-Round Replay**
```python
def test_cross_round_replay():
    # Valid proof for round 1
    proof_r1 = client.generate_proof(..., round_number=1)
    assert coordinator.verify_proof(proof_r1)[0]

    # Attempt to use round 1 proof in round 2
    # (nonce is same, but round_number different)
    # Should fail because round_number mismatch
    proof_r1_modified = replace(proof_r1, round_number=2)
    assert not coordinator.verify_proof(proof_r1_modified)[0]
```

**Test C: Stolen Proof Attack**
```python
def test_stolen_proof():
    # Honest client generates proof
    proof = honest_client.generate_proof(...)

    # Attacker intercepts and tries to submit
    # (Different client_id, signature won't match)
    stolen_proof = replace(proof, client_id=attacker.client_id)
    assert not coordinator.verify_proof(stolen_proof)[0]
```

**Test D: Sybil Attack**
```python
def test_sybil_prevention():
    # Attacker creates 100 identities
    sybil_clients = [AuthenticatedGradientClient() for _ in range(100)]

    # All must register (coordinator can rate-limit)
    for client in sybil_clients:
        coordinator.register_client(client.keypair.get_public_key())

    # Each Sybil must generate valid zkSTARK proof
    # (Computationally expensive: 100x proof time)
    # Each Sybil must have valid training data
    # (Requires actual participation, not free)
```

---

## 📊 Expected Results

### Performance Targets

| Metric | Phase 2 | Phase 2.5 Target | Max Acceptable |
|--------|---------|------------------|----------------|
| Proof Generation | 4.8ms | 6-8ms | 10ms |
| Proof Verification | 1ms | 2-3ms | 5ms |
| Proof Size | 61.3KB | ~65.9KB | 70KB |
| Coordinator Throughput | - | >100 proofs/sec | >50 proofs/sec |

### Security Targets

| Attack Type | Detection Rate | Response Time |
|-------------|----------------|---------------|
| Replay (same round) | 100% | Immediate |
| Replay (cross-round) | 100% | Immediate |
| Stolen proof | 100% | Immediate |
| Sybil | 100% | Rate-limited |
| Timestamp manipulation | 100% | Immediate |

---

## 🚧 Risk Mitigation

### Risk 1: Database Performance Bottleneck
**Symptom**: Nonce lookup slows verification
**Mitigation**:
- In-memory cache (Redis) for recent nonces
- Batch inserts for nonce recording
- Index optimization (round_number, timestamp)

### Risk 2: Key Management Usability
**Symptom**: Users forget passwords, lose keys
**Mitigation**:
- Mnemonic backup phrase (BIP-39)
- Key recovery process (coordinator holds public key)
- Clear warnings about key loss

### Risk 3: Throughput Below Target
**Symptom**: <50 proofs/second verification
**Mitigation**:
- Parallel verification (multiprocessing)
- GPU acceleration for Dilithium verify
- Coordinator sharding (multiple instances)

---

## 📁 Deliverables

By end of Week 2:

- [ ] `src/zerotrustml/gen7/client_registry.py` - Full implementation
- [ ] `src/zerotrustml/gen7/key_management.py` - Encrypted storage
- [ ] Database schema migrations
- [ ] `experiments/benchmark_phase2_5.py` - Performance suite
- [ ] `experiments/test_e7_5_authenticated.py` - Integration test
- [ ] `experiments/test_e7_5_attacks.py` - Security tests
- [ ] `PHASE_2_5_WEEK2_COMPLETE.md` - Results summary
- [ ] Performance report (CSV + plots)

---

## 🎯 Gate Criteria for Week 3

Week 2 is complete when:

1. ✅ All E7.5 tests pass (5 rounds, 10 clients, 100% success)
2. ✅ Performance within targets (<10ms proof, >50 proofs/sec)
3. ✅ All attack tests fail as expected (100% detection)
4. ✅ Key management secure and usable
5. ✅ Documentation complete

**If all gates pass**: Proceed to Week 3 (Production Deployment)
**If any gate fails**: Fix issues before continuing

---

## 🔗 Dependencies

**From Week 1** (Complete):
- ✅ Dilithium Rust module compiled
- ✅ Python bindings working
- ✅ Basic tests passing (4/4)
- ✅ Empirical sizes verified

**For Week 2** (New):
- SQLite or PostgreSQL (database)
- cryptography library (Python, for key management)
- pytest-benchmark (for performance testing)
- matplotlib (for performance plots)

---

## 📚 References

**Security Standards**:
- NIST FIPS 204 (Dilithium)
- NIST SP 800-63B (Password-based auth)
- OWASP Key Management Cheat Sheet

**Performance Baselines**:
- Phase 2 zkSTARK: 4.8ms generation, 1ms verify, 61KB
- Dilithium5 standalone: ~1ms sign, ~1ms verify

**Testing Frameworks**:
- pytest (unit tests)
- pytest-benchmark (performance)
- locust (load testing, if needed)

---

**Status**: Ready for Week 2 implementation
**Prerequisites**: ✅ Week 1 complete and verified
**Estimated Effort**: 5 person-days (or 3-4 with AI assistance)

---

*"Defense in depth: Every layer verified, every attack detected."*
