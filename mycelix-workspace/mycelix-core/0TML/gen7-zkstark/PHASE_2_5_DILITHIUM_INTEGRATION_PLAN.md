# Phase 2.5: Dilithium Authentication Integration Plan

**Target**: zk-DASTARK (zkSTARK + CRYSTALS-Dilithium) Hybrid System
**Timeline**: 2-3 weeks
**Status**: Planning → Implementation → Testing
**Goal**: Add post-quantum client authentication to zkSTARK gradient proofs

---

## 🎯 Executive Summary

**Problem**: Gen-7 Phase 2 proves gradients are computed correctly (integrity) but doesn't prove WHO submitted them (authentication).

**Solution**: Integrate CRYSTALS-Dilithium digital signatures to create authenticated proofs.

**Benefits**:
- ✅ Client authentication (know who submitted each gradient)
- ✅ Replay attack prevention (signatures include nonce/timestamp)
- ✅ Defense-in-depth (two independent PQC assumptions)
- ✅ Audit trail (regulatory compliance)
- ✅ Minimal overhead (+2.4KB proof size, +1ms signing)

---

## 📊 Architecture Overview

### Current (Phase 2)
```
GradientProof = zkSTARK_Proof {
    gradient_commitment,
    model_params_commitment,
    data_hash,
    proof_bytes[61KB]
}

Weakness: Can't verify WHO generated the proof
```

### New (Phase 2.5)
```
AuthenticatedGradientProof = {
    stark_proof: zkSTARK_Proof,      # 61KB - computation integrity
    signature: Dilithium_Signature,  # 2.4KB - client authentication
    client_id: Hash(public_key),     # 32 bytes
    round_number: u64,               # 8 bytes
    timestamp: u64,                  # 8 bytes
    nonce: [u8; 32]                  # 32 bytes - replay protection
}

Total size: ~64KB (was 61KB)
```

### Verification Flow
```
1. Check signature: Dilithium.verify(message, signature, client_pubkey)
   → Proves: "Client X signed this proof"

2. Check STARK: verify_gradient_zkstark(stark_proof, commitment)
   → Proves: "Gradient was computed correctly"

3. Check freshness: timestamp within ±5 minutes, nonce not reused
   → Proves: "Proof is recent and unique"

4. Check stake: client_id has sufficient stake registered
   → Proves: "Client has skin in the game"

All must pass → Accept gradient
```

---

## 🏗️ Week-by-Week Implementation Plan

### Week 1: Dilithium Library Integration

**Day 1-2: Library Selection & Setup**
```bash
# Add Dilithium to Cargo.toml
cd gen7-zkstark/bindings
cargo add pqcrypto-dilithium --version "0.5"
cargo add pqcrypto-traits --version "0.3"

# Test basic operations
cargo test dilithium_smoke_test
```

**Day 3-4: Rust API Implementation**
```rust
// gen7-zkstark/bindings/src/dilithium.rs
use pqcrypto_dilithium::dilithium5;
use pqcrypto_traits::sign::{PublicKey, SecretKey, SignedMessage};

pub struct DilithiumKeypair {
    pub public_key: dilithium5::PublicKey,
    pub secret_key: dilithium5::SecretKey,
}

impl DilithiumKeypair {
    pub fn generate() -> Self {
        let (pk, sk) = dilithium5::keypair();
        Self { public_key: pk, secret_key: sk }
    }

    pub fn sign(&self, message: &[u8]) -> Vec<u8> {
        dilithium5::sign(message, &self.secret_key).as_bytes().to_vec()
    }

    pub fn verify(message: &[u8], signature: &[u8], public_key: &[u8]) -> bool {
        let pk = dilithium5::PublicKey::from_bytes(public_key).ok()?;
        let signed = dilithium5::SignedMessage::from_bytes(signature).ok()?;
        dilithium5::open(&signed, &pk).is_ok()
    }
}
```

**Day 5: PyO3 Bindings**
```rust
// gen7-zkstark/bindings/src/lib.rs
#[pyfunction]
fn generate_dilithium_keypair() -> PyResult<(Vec<u8>, Vec<u8>)> {
    let keypair = DilithiumKeypair::generate();
    Ok((
        keypair.public_key.as_bytes().to_vec(),
        keypair.secret_key.as_bytes().to_vec(),
    ))
}

#[pyfunction]
fn sign_dilithium(message: Vec<u8>, secret_key: Vec<u8>) -> PyResult<Vec<u8>> {
    let sk = dilithium5::SecretKey::from_bytes(&secret_key)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Invalid secret key: {}", e)
        ))?;

    let signed = dilithium5::sign(&message, &sk);
    Ok(signed.as_bytes().to_vec())
}

#[pyfunction]
fn verify_dilithium(
    message: Vec<u8>,
    signature: Vec<u8>,
    public_key: Vec<u8>
) -> PyResult<bool> {
    Ok(DilithiumKeypair::verify(&message, &signature, &public_key))
}
```

**Deliverable**: Working Dilithium library with Python bindings

---

### Week 2: Authentication Layer Implementation

**Day 1-2: Enhanced Proof Structure**
```rust
// gen7-zkstark/bindings/src/authenticated_proof.rs
#[derive(Serialize, Deserialize)]
pub struct AuthenticatedGradientProof {
    pub stark_proof: Vec<u8>,
    pub signature: Vec<u8>,
    pub client_id: [u8; 32],
    pub round_number: u64,
    pub timestamp: u64,
    pub nonce: [u8; 32],
}

impl AuthenticatedGradientProof {
    pub fn create(
        gradient: &[f32],
        model_params: &[f32],
        data_hash: &[u8],
        client_keypair: &DilithiumKeypair,
        round_number: u64,
    ) -> Result<Self> {
        // 1. Generate zkSTARK proof
        let stark_proof = prove_gradient_zkstark(gradient, model_params, data_hash)?;

        // 2. Create message to sign
        let nonce = generate_nonce();
        let timestamp = current_timestamp();
        let client_id = hash_public_key(&client_keypair.public_key);

        let message = Self::construct_message(
            &stark_proof,
            &client_id,
            round_number,
            timestamp,
            &nonce,
        );

        // 3. Sign with Dilithium
        let signature = client_keypair.sign(&message);

        Ok(Self {
            stark_proof,
            signature,
            client_id,
            round_number,
            timestamp,
            nonce,
        })
    }

    pub fn verify(
        &self,
        client_public_key: &[u8],
        current_round: u64,
        gradient_commitment: &[u8],
    ) -> Result<bool> {
        // 1. Check round number
        if self.round_number != current_round {
            return Err(Error::InvalidRound);
        }

        // 2. Check timestamp (±5 minutes)
        let now = current_timestamp();
        if (now - self.timestamp).abs() > 300 {
            return Err(Error::TimestampExpired);
        }

        // 3. Check nonce (coordinator tracks used nonces)
        if is_nonce_used(&self.nonce) {
            return Err(Error::NonceReused);
        }

        // 4. Verify Dilithium signature
        let message = Self::construct_message(
            &self.stark_proof,
            &self.client_id,
            self.round_number,
            self.timestamp,
            &self.nonce,
        );

        if !DilithiumKeypair::verify(&message, &self.signature, client_public_key) {
            return Err(Error::InvalidSignature);
        }

        // 5. Verify zkSTARK proof
        if !verify_gradient_zkstark(&self.stark_proof, gradient_commitment) {
            return Err(Error::InvalidStarkProof);
        }

        Ok(true)
    }

    fn construct_message(
        stark_proof: &[u8],
        client_id: &[u8; 32],
        round_number: u64,
        timestamp: u64,
        nonce: &[u8; 32],
    ) -> Vec<u8> {
        use sha2::{Sha256, Digest};

        let mut hasher = Sha256::new();
        hasher.update(stark_proof);
        hasher.update(client_id);
        hasher.update(&round_number.to_le_bytes());
        hasher.update(&timestamp.to_le_bytes());
        hasher.update(nonce);

        hasher.finalize().to_vec()
    }
}
```

**Day 3-4: Client Registration System**
```python
# src/zerotrustml/gen7/client_registry.py
from dataclasses import dataclass
from typing import Dict, Set
import gen7_zkstark

@dataclass
class RegisteredClient:
    client_id: bytes
    public_key: bytes
    stake: float
    reputation: float
    registered_at: int
    used_nonces: Set[bytes]

class ClientRegistry:
    """Manages client identities and public keys."""

    def __init__(self):
        self.clients: Dict[bytes, RegisteredClient] = {}

    def register(self, public_key: bytes, stake: float) -> bytes:
        """Register a new client with their Dilithium public key."""
        client_id = hash_public_key(public_key)

        if client_id in self.clients:
            raise ValueError(f"Client {client_id.hex()} already registered")

        self.clients[client_id] = RegisteredClient(
            client_id=client_id,
            public_key=public_key,
            stake=stake,
            reputation=1.0,
            registered_at=current_timestamp(),
            used_nonces=set(),
        )

        return client_id

    def verify_nonce(self, client_id: bytes, nonce: bytes) -> bool:
        """Check if nonce is fresh (not previously used)."""
        client = self.clients.get(client_id)
        if not client:
            return False

        if nonce in client.used_nonces:
            return False

        client.used_nonces.add(nonce)
        return True

    def get_public_key(self, client_id: bytes) -> bytes:
        """Retrieve client's public key for signature verification."""
        client = self.clients.get(client_id)
        if not client:
            raise KeyError(f"Client {client_id.hex()} not registered")

        return client.public_key
```

**Day 5: Update StakingCoordinator**
```python
# src/zerotrustml/gen7/staking_coordinator.py (updated)
class StakingCoordinator:
    def __init__(self):
        self.clients: Dict[str, ClientState] = {}
        self.registry = ClientRegistry()  # NEW
        self.current_round = 0

    def register_client(
        self,
        client_id: str,
        stake: float,
        public_key: bytes  # NEW: Dilithium public key
    ):
        """Register client with stake + public key."""
        # Register in identity registry
        client_id_hash = self.registry.register(public_key, stake)

        # Create staking state
        self.clients[client_id] = ClientState(
            stake=stake,
            reputation=1.0,
            submissions=0,
            slashed_total=0.0,
        )

    def verify_authenticated_proof(
        self,
        client_id: str,
        proof: AuthenticatedGradientProof,
    ) -> Tuple[bool, float]:
        """Verify both signature and STARK proof."""
        client = self.clients[client_id]

        # Get client's public key
        public_key = self.registry.get_public_key(proof.client_id)

        # Verify authenticated proof
        try:
            valid = gen7_zkstark.verify_authenticated_proof(
                proof=proof.serialize(),
                public_key=public_key,
                current_round=self.current_round,
                gradient_commitment=compute_commitment(...),
            )
        except Exception as e:
            logger.warning(f"Proof verification failed for {client_id}: {e}")
            return self._slash_client(client_id, "invalid_proof")

        if valid:
            # Update reputation and weight
            client.reputation *= 1.05
            weight = client.stake * client.reputation
            return True, weight
        else:
            return self._slash_client(client_id, "invalid_proof")
```

**Deliverable**: Full authentication layer with client registry

---

### Week 3: Testing & Integration

**Day 1-2: Unit Tests**
```python
# tests/test_dilithium_integration.py
def test_keypair_generation():
    """Test Dilithium keypair generation."""
    public_key, secret_key = gen7_zkstark.generate_dilithium_keypair()

    assert len(public_key) == 2592  # Dilithium5 public key size
    assert len(secret_key) == 4864  # Dilithium5 secret key size

def test_sign_verify():
    """Test signature creation and verification."""
    public_key, secret_key = gen7_zkstark.generate_dilithium_keypair()

    message = b"test gradient proof"
    signature = gen7_zkstark.sign_dilithium(message, secret_key)

    assert len(signature) == 4595  # Dilithium5 signature size

    valid = gen7_zkstark.verify_dilithium(message, signature, public_key)
    assert valid is True

    # Wrong message
    invalid = gen7_zkstark.verify_dilithium(b"wrong", signature, public_key)
    assert invalid is False

def test_nonce_uniqueness():
    """Test nonce prevents replay attacks."""
    registry = ClientRegistry()
    public_key, _ = gen7_zkstark.generate_dilithium_keypair()
    client_id = registry.register(public_key, stake=50.0)

    nonce = os.urandom(32)

    # First use: valid
    assert registry.verify_nonce(client_id, nonce) is True

    # Second use: rejected (replay)
    assert registry.verify_nonce(client_id, nonce) is False
```

**Day 3-4: E7.5 Integration Test**
```python
# experiments/test_gen7_authenticated.py
def test_authenticated_gradient_proofs():
    """E7.5: End-to-end test with authentication."""

    # Setup: 10 clients with Dilithium keypairs
    n_clients = 10
    clients = []

    for i in range(n_clients):
        public_key, secret_key = gen7_zkstark.generate_dilithium_keypair()
        client_id = f"client_{i:02d}"

        clients.append({
            "id": client_id,
            "public_key": public_key,
            "secret_key": secret_key,
            "honest": i < 7,
        })

    # Register all clients
    coordinator = StakingCoordinator()
    for client in clients:
        coordinator.register_client(
            client_id=client["id"],
            stake=50.0 if client["honest"] else 20.0,
            public_key=client["public_key"],
        )

    # Run 5 training rounds
    for round_num in range(5):
        coordinator.current_round = round_num

        for client in clients:
            if client["honest"]:
                # Generate authentic proof
                proof = gen7_zkstark.create_authenticated_proof(
                    gradient=np.random.randn(784, 10),
                    model_params=np.random.randn(10, 784),
                    data_hash=os.urandom(32),
                    secret_key=client["secret_key"],
                    round_number=round_num,
                )
            else:
                # Attack 1: Try to reuse old proof
                if round_num > 0:
                    proof = old_proof  # Should be rejected (wrong round)
                # Attack 2: Try to steal honest client's proof
                else:
                    stolen_proof = honest_proofs[0]
                    stolen_proof.client_id = client["id"]  # Forge identity
                    proof = stolen_proof  # Should be rejected (signature mismatch)

            valid, weight = coordinator.verify_authenticated_proof(
                client_id=client["id"],
                proof=proof,
            )

            if client["honest"]:
                assert valid is True, f"Honest client {client['id']} rejected!"
            else:
                assert valid is False, f"Attack from {client['id']} accepted!"

    print("✅ E7.5: All authenticated proofs verified correctly!")
```

**E7.5 Acceptance Gates**:
```
E7.5.1: Authentication Rate = 100% (all honest clients authenticated)
E7.5.2: Replay Detection = 100% (all replay attacks rejected)
E7.5.3: Identity Forgery Prevention = 100% (all stolen proofs rejected)
E7.5.4: Performance Overhead < 10% (signing + verification < 2ms)
```

**Day 5: Performance Benchmarking**
```python
# experiments/benchmark_authenticated_proofs.py
def benchmark_auth_overhead():
    """Measure performance impact of Dilithium signatures."""

    # Setup
    public_key, secret_key = gen7_zkstark.generate_dilithium_keypair()
    gradient = np.random.randn(784, 10)
    model_params = np.random.randn(10, 784)

    # Benchmark: Pure STARK (Phase 2)
    start = time.perf_counter()
    for _ in range(100):
        stark_proof = gen7_zkstark.prove_gradient_zkstark(
            gradient, model_params, os.urandom(32)
        )
    stark_time = (time.perf_counter() - start) / 100

    # Benchmark: STARK + Dilithium (Phase 2.5)
    start = time.perf_counter()
    for _ in range(100):
        auth_proof = gen7_zkstark.create_authenticated_proof(
            gradient, model_params, os.urandom(32), secret_key, round_num=0
        )
    auth_time = (time.perf_counter() - start) / 100

    overhead_ms = (auth_time - stark_time) * 1000
    overhead_pct = (overhead_ms / (stark_time * 1000)) * 100

    print(f"STARK proving time: {stark_time*1000:.2f}ms")
    print(f"STARK+Dilithium time: {auth_time*1000:.2f}ms")
    print(f"Overhead: +{overhead_ms:.2f}ms (+{overhead_pct:.1f}%)")

    # Target: <10% overhead (<0.5ms)
    assert overhead_pct < 10, f"Overhead {overhead_pct}% exceeds 10% target!"
```

**Deliverable**: Passing E7.5 tests + performance benchmarks

---

## 🔐 Key Management Strategy

### Client-Side Key Storage

**Option 1: TEE-Protected (Recommended for Production)**
```python
# Client stores private key in TPM/SGX enclave
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend

def store_key_secure(secret_key: bytes, client_id: str):
    """Store key in hardware security module."""
    # Encrypt key with TPM-backed encryption
    encrypted_key = tpm.encrypt(
        secret_key,
        key_id=f"dilithium_key_{client_id}",
    )

    # Store encrypted key
    with open(f".keys/{client_id}.enc", "wb") as f:
        f.write(encrypted_key)

def load_key_secure(client_id: str) -> bytes:
    """Load key from secure storage."""
    with open(f".keys/{client_id}.enc", "rb") as f:
        encrypted_key = f.read()

    return tpm.decrypt(encrypted_key)
```

**Option 2: Password-Protected (Development)**
```python
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.fernet import Fernet

def encrypt_key_with_password(secret_key: bytes, password: str) -> bytes:
    """Encrypt private key with user password."""
    salt = os.urandom(32)

    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
        backend=default_backend()
    )

    key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
    cipher = Fernet(key)

    encrypted = cipher.encrypt(secret_key)
    return salt + encrypted

def decrypt_key_with_password(encrypted_data: bytes, password: str) -> bytes:
    """Decrypt private key with user password."""
    salt = encrypted_data[:32]
    encrypted_key = encrypted_data[32:]

    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
        backend=default_backend()
    )

    key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
    cipher = Fernet(key)

    return cipher.decrypt(encrypted_key)
```

### Coordinator-Side Public Key Storage

**Option 1: On-Chain (Transparent, Auditable)**
```solidity
// Ethereum smart contract for public key registry
contract ClientRegistry {
    struct Client {
        bytes publicKey;  // Dilithium5 public key (2.5KB)
        uint256 stake;
        uint256 reputation;
        uint256 registeredAt;
    }

    mapping(bytes32 => Client) public clients;

    function register(bytes calldata publicKey) external payable {
        bytes32 clientId = keccak256(publicKey);
        require(clients[clientId].registeredAt == 0, "Already registered");

        clients[clientId] = Client({
            publicKey: publicKey,
            stake: msg.value,
            reputation: 1e18,  // 1.0 in fixed-point
            registeredAt: block.timestamp
        });
    }

    function getPublicKey(bytes32 clientId) external view returns (bytes memory) {
        return clients[clientId].publicKey;
    }
}
```

**Option 2: Database (Fast, Centralized)**
```python
# PostgreSQL schema
CREATE TABLE clients (
    client_id BYTEA PRIMARY KEY,
    public_key BYTEA NOT NULL,
    stake NUMERIC(18, 6) NOT NULL,
    reputation NUMERIC(10, 6) NOT NULL DEFAULT 1.0,
    registered_at TIMESTAMP NOT NULL DEFAULT NOW(),
    used_nonces BYTEA[] DEFAULT ARRAY[]::BYTEA[]
);

CREATE INDEX idx_client_pubkey ON clients USING hash(public_key);
CREATE INDEX idx_client_nonces ON clients USING gin(used_nonces);
```

---

## 📋 Migration Plan (Phase 2 → Phase 2.5)

### Backward Compatibility
```python
# Support both authenticated and unauthenticated proofs
class ProofVerifier:
    def verify(self, proof_data: bytes) -> bool:
        # Try to deserialize as authenticated proof
        try:
            auth_proof = AuthenticatedGradientProof.deserialize(proof_data)
            return self.verify_authenticated(auth_proof)
        except DeserializationError:
            # Fall back to unauthenticated STARK-only proof
            logger.warning("Received unauthenticated proof - Phase 2 compatibility mode")
            return self.verify_stark_only(proof_data)
```

### Gradual Rollout
```
Week 1: Deploy Phase 2.5 with backward compatibility (accept both)
Week 2: Monitor adoption rate (% of authenticated proofs)
Week 3: Increase stake requirements for unauthenticated proofs (economic incentive)
Week 4: Deprecate unauthenticated proofs (require authentication)
```

---

## 🎯 Acceptance Criteria

### Functional Requirements
- [ ] Dilithium keypair generation working
- [ ] Sign/verify operations correct
- [ ] Client registration with public keys
- [ ] Nonce tracking prevents replays
- [ ] Timestamp validation (±5 minutes)
- [ ] Full E7.5 test suite passing

### Performance Requirements
- [ ] Signing overhead < 2ms
- [ ] Verification overhead < 1ms
- [ ] Proof size increase < 5% (61KB → 64KB) ✅
- [ ] Memory usage increase < 10%

### Security Requirements
- [ ] Replay attacks rejected (100%)
- [ ] Identity forgery rejected (100%)
- [ ] Stolen proof attacks rejected (100%)
- [ ] Key storage encrypted (TEE or password)

### Integration Requirements
- [ ] Backward compatible with Phase 2 proofs
- [ ] On-chain public key registry deployed
- [ ] Client SDK updated with key management
- [ ] Documentation complete

---

## 📚 Documentation Deliverables

1. **API Reference**: `DILITHIUM_API.md`
2. **Key Management Guide**: `KEY_MANAGEMENT_BEST_PRACTICES.md`
3. **Migration Guide**: `PHASE_2_TO_2_5_MIGRATION.md`
4. **Security Audit Report**: `PHASE_2_5_SECURITY_AUDIT.md`

---

## 🚀 Next Steps

**Immediate Actions**:
1. Create feature branch: `git checkout -b phase-2.5-dilithium`
2. Add Dilithium dependency: `cargo add pqcrypto-dilithium`
3. Implement Week 1 tasks (library integration)
4. Daily standups to track progress

**Success Metrics**:
- E7.5 acceptance gates: 100% pass
- Performance overhead: <10%
- Zero security vulnerabilities in audit

---

**Status**: ✅ APPROVED - Ready to begin Week 1 implementation
**Next Review**: End of Week 1 (Dilithium library integration complete)
**Go-Live Target**: End of Week 3 (full E7.5 validation)

🔐 **zk-DASTARK: The Future of Authenticated Federated Learning** 🔐
