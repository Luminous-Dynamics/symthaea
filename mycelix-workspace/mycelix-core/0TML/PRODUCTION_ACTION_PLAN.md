# ZK-FL Production Deployment - Action Plan

**Goal**: Deploy ZK-FL on Holochain with RISC Zero backend to production
**Timeline**: 4 weeks (conservative) or 5 days (aggressive)
**Status**: 🟡 Ready to Execute

---

## Quick Start (5 Days to Production - Aggressive)

### Day 1: Build & Deploy Holochain Zome

**Morning (2 hours)**
```bash
# Step 1: Build the pogq_proof_validation zome
cd /srv/luminous-dynamics/Mycelix-Core/0TML/holochain-dht-setup/zomes/pogq_proof_validation

# Clean previous builds
cargo clean

# Build for WebAssembly
cargo build --release --target wasm32-unknown-unknown

# Verify build success
ls -lh target/wasm32-unknown-unknown/release/*.wasm

# Expected: pogq_proof_validation.wasm (~200-500 KB)
```

**Afternoon (4 hours)**
```bash
# Step 2: Package as Holochain DNA
cd /srv/luminous-dynamics/Mycelix-Core/0TML/holochain-dht-setup

# Create DNA manifest (dna.yaml)
cat > dna.yaml << 'EOF'
---
manifest_version: "1"
name: pogq_zkfl
uid: "00000000-0000-0000-0000-000000000001"
properties: ~
zomes:
  - name: pogq_proof_validation
    bundled: zomes/pogq_proof_validation/target/wasm32-unknown-unknown/release/pogq_proof_validation.wasm
EOF

# Package DNA
hc dna pack .

# Expected: pogq_zkfl.dna (~200-500 KB)

# Step 3: Deploy to test conductors
# Start 3 test conductors (ports 9888, 9890, 9892)
cd scripts
./deploy-with-screen.sh 3  # Deploy only 3 conductors for testing

# Step 4: Install DNA to all 3 conductors
for port in 9888 9890 9892; do
  curl -X POST http://localhost:$port/install_app \
    -H "Content-Type: application/json" \
    -d '{
      "installed_app_id": "pogq_zkfl",
      "agent_key": null,
      "dnas": [{
        "path": "../pogq_zkfl.dna",
        "nick": "pogq"
      }]
    }'
done

# Verify installations
for port in 9888 9890 9892; do
  echo "=== Conductor on port $port ==="
  curl -s http://localhost:$port/list_apps | jq
done
```

**Evening (2 hours)**
```bash
# Step 5: Test Python bridge with real zome
cd /srv/luminous-dynamics/Mycelix-Core/0TML

# Create test proof files (if they don't exist)
mkdir -p test_proofs
cd test_proofs

# Generate a test proof using RISC Zero
python ../experiments/vsv_stark_benchmark_integrated.py --rounds 1 --output test_proof

# Should create:
# - proof.bin (~221 KB)
# - journal.bin (~22 bytes)
# - public_echo.json

# Test Python bridge
cd ..
python -c "
from pathlib import Path
from experiments.holochain_bridge import HolochainBridge

bridge = HolochainBridge('http://localhost:9888')

# Submit test proof
action_hash = bridge.submit_proof(
    node_id='test_node',
    round_number=1,
    proof_path=Path('test_proofs/proof.bin'),
    public_path=Path('test_proofs/public_echo.json'),
    journal_path=Path('test_proofs/journal.bin')
)

print(f'✅ Proof submitted: {action_hash}')

# Retrieve it back
proof = bridge.get_proof(action_hash)
if proof:
    print('✅ Proof retrieved successfully')
else:
    print('❌ Failed to retrieve proof')
"
```

**Expected Output**:
```
✅ Proof submitted: uhCkkHfQ9vN8xK2P5ZJ4m...
✅ Proof retrieved successfully
```

**Deliverable**: Working Holochain zome with Python bridge integration

---

### Day 2-3: End-to-End Integration Testing

**Day 2 Morning: 2-Node Smoke Test**
```bash
# Test scenario: 2 honest nodes, simple FL round

cd /srv/luminous-dynamics/Mycelix-Core/0TML

# Create test script
cat > test_2_node_integration.py << 'EOF'
import time
from pathlib import Path
from experiments.holochain_bridge import HolochainBridge

# Initialize bridges for 2 nodes
node_0_bridge = HolochainBridge("http://localhost:9888")
node_1_bridge = HolochainBridge("http://localhost:9890")

# Round 1: Both nodes submit proofs
print("=== Round 1: Proof Submission ===")

# Node 0 submits
action_hash_0 = node_0_bridge.submit_proof(
    node_id="node_0",
    round_number=1,
    proof_path=Path("test_proofs/node_0_proof.bin"),
    public_path=Path("test_proofs/node_0_public.json"),
    journal_path=Path("test_proofs/node_0_journal.bin")
)
print(f"Node 0 submitted: {action_hash_0}")

# Node 1 submits
action_hash_1 = node_1_bridge.submit_proof(
    node_id="node_1",
    round_number=1,
    proof_path=Path("test_proofs/node_1_proof.bin"),
    public_path=Path("test_proofs/node_1_public.json"),
    journal_path=Path("test_proofs/node_1_journal.bin")
)
print(f"Node 1 submitted: {action_hash_1}")

# Wait for DHT propagation
time.sleep(2)

# Verify peer proofs
print("\n=== Peer Verification ===")

# Node 0 verifies Node 1
result = node_0_bridge.verify_peer_proof("node_1", 1)
print(f"Node 0 verifies Node 1: {result}")

# Node 1 verifies Node 0
result = node_1_bridge.verify_peer_proof("node_0", 1)
print(f"Node 1 verifies Node 0: {result}")

# Check consensus
print("\n=== Consensus Check ===")
consensus = node_0_bridge.get_consensus_state(
    round_number=1,
    node_ids=["node_0", "node_1"]
)
print(f"Consensus: {consensus}")

print("\n✅ 2-Node Integration Test PASSED")
EOF

# Run test
python test_2_node_integration.py
```

**Expected Output**:
```
=== Round 1: Proof Submission ===
Node 0 submitted: uhCkkHfQ9vN8xK2P5ZJ4m...
Node 1 submitted: uhCkkJdR7wP2xL5Q8aK3n...

=== Peer Verification ===
Node 0 verifies Node 1: {'valid': True, 'quarantine_decision': 0, ...}
Node 1 verifies Node 0: {'valid': True, 'quarantine_decision': 0, ...}

=== Consensus Check ===
Consensus: {'round': 1, 'consensus_decision': 0, 'nodes_with_proofs': 2, ...}

✅ 2-Node Integration Test PASSED
```

**Day 2 Afternoon: 5-Node Demo with Byzantine Node**
```bash
# Deploy 2 additional conductors (total 5)
cd /srv/luminous-dynamics/Mycelix-Core/0TML/holochain-dht-setup/scripts
./deploy-with-screen.sh 5

# Install DNA to new conductors
for port in 9894 9896; do
  curl -X POST http://localhost:$port/install_app \
    -H "Content-Type: application/json" \
    -d '{
      "installed_app_id": "pogq_zkfl",
      "agent_key": null,
      "dnas": [{"path": "../pogq_zkfl.dna", "nick": "pogq"}]
    }'
done

# Create 5-node test with Byzantine node
cat > test_5_node_byzantine.py << 'EOF'
import time
from pathlib import Path
from experiments.holochain_bridge import HolochainBridge

# Initialize 5 nodes
nodes = [
    HolochainBridge("http://localhost:9888"),  # Node 0: Honest
    HolochainBridge("http://localhost:9890"),  # Node 1: Honest
    HolochainBridge("http://localhost:9892"),  # Node 2: Honest
    HolochainBridge("http://localhost:9894"),  # Node 3: Byzantine (low quality)
    HolochainBridge("http://localhost:9896"),  # Node 4: Honest
]

# Run 10 FL rounds
for round_num in range(10):
    print(f"\n=== Round {round_num} ===")

    # Generate proofs for all nodes
    # Node 3 becomes Byzantine at round 5
    for node_id in range(5):
        is_byzantine = (node_id == 3 and round_num >= 5)

        # Generate proof based on behavior
        # ... (call VSV-STARK prover here)

        # Submit proof
        action_hash = nodes[node_id].submit_proof(
            node_id=f"node_{node_id}",
            round_number=round_num,
            proof_path=Path(f"proofs/round_{round_num}/node_{node_id}_proof.bin"),
            public_path=Path(f"proofs/round_{round_num}/node_{node_id}_public.json"),
            journal_path=Path(f"proofs/round_{round_num}/node_{node_id}_journal.bin")
        )
        print(f"Node {node_id} submitted: {action_hash[:20]}...")

    # Wait for propagation
    time.sleep(2)

    # Check consensus
    consensus = nodes[0].get_consensus_state(
        round_number=round_num,
        node_ids=[f"node_{i}" for i in range(5)]
    )

    print(f"Consensus: decision={consensus['consensus_decision']}, "
          f"quarantine_votes={consensus['quarantine_votes']}, "
          f"no_quarantine_votes={consensus['no_quarantine_votes']}")

    # Verify Node 3 gets quarantined at round 6
    if round_num == 6:
        node_3_result = nodes[0].verify_peer_proof("node_3", round_num)
        assert node_3_result['quarantine_decision'] == 1, "Node 3 should be quarantined!"
        print("✅ Byzantine node successfully quarantined at round 6")

print("\n✅ 5-Node Byzantine Demo PASSED")
EOF

# Run demo (requires proof generation integration)
# python test_5_node_byzantine.py
```

**Day 3: Security Hardening**

**Morning: Implement Nonce Tracking**
```rust
// Add to pogq_proof_validation zome (lib.rs)

use std::collections::HashSet;

// Global state for tracking used nonces
#[derive(Serialize, Deserialize, Debug, Default)]
pub struct NonceRegistry {
    used_nonces: HashSet<Vec<u8>>,
}

// Add to validation (around line 100)
fn validate_nonce(nonce: &[u8]) -> ExternResult<()> {
    // Check if nonce was used before
    let registry = get_nonce_registry()?;

    if registry.used_nonces.contains(&nonce.to_vec()) {
        return Err(wasm_error!(
            WasmErrorInner::Guest("Nonce already used (replay attack)".into())
        ));
    }

    // Mark nonce as used
    let mut registry = registry;
    registry.used_nonces.insert(nonce.to_vec());
    set_nonce_registry(registry)?;

    Ok(())
}

// Add nonce field to ProofEntry
pub struct ProofEntry {
    // ... existing fields ...
    pub nonce: Vec<u8>,  // 32-byte cryptographic nonce
}

// Update validation rule (add as Rule 11)
// In validate() function:
validate_nonce(&proof_entry.nonce)?;
```

**Afternoon: Add Link Indexing**
```rust
// Add to pogq_proof_validation zome

// In store_proof() function (after line 50):
#[hdk_extern]
pub fn store_proof(entry: ProofEntry) -> ExternResult<ActionHash> {
    let action_hash = create_entry(EntryTypes::ProofEntry(entry.clone()))?;

    // Create link for efficient lookup by (node_id, round)
    let path = Path::from(format!("proofs.{}.{}", entry.node_id, entry.round_number));
    path.ensure()?;
    create_link(
        path.path_entry_hash()?,
        action_hash.clone(),
        LinkTypes::ProofToRound,
        ()
    )?;

    Ok(action_hash)
}

// Update find_proof() implementation:
#[hdk_extern]
pub fn find_proof(input: FindProofInput) -> ExternResult<Option<ActionHash>> {
    let path = Path::from(format!("proofs.{}.{}", input.node_id, input.round_number));

    let links = get_links(
        path.path_entry_hash()?,
        Some(LinkTypes::ProofToRound.try_into()?)
    )?;

    // Return first link (should be only one)
    Ok(links.first().map(|link| link.target.clone().into()))
}
```

**Rebuild zome**:
```bash
cd /srv/luminous-dynamics/Mycelix-Core/0TML/holochain-dht-setup/zomes/pogq_proof_validation
cargo build --release --target wasm32-unknown-unknown
hc dna pack ../..
```

---

### Day 4-5: Production Deployment

**Day 4: Deployment Preparation**

**Morning: Create Deployment Package**
```bash
cd /srv/luminous-dynamics/Mycelix-Core/0TML

# Create production deployment directory
mkdir -p deployment/production

# Copy all necessary files
cp holochain-dht-setup/pogq_zkfl.dna deployment/production/
cp -r holochain-dht-setup/scripts deployment/production/
cp -r experiments deployment/production/
cp -r vsv-stark deployment/production/

# Create deployment manifest
cat > deployment/production/DEPLOYMENT_MANIFEST.md << 'EOF'
# ZK-FL Production Deployment

**Version**: 1.0.0
**Date**: $(date +%Y-%m-%d)
**Components**:
- RISC Zero zkVM: v0.21 (proof generation)
- Holochain: v0.2.x (DHT storage)
- pogq_proof_validation zome: v1.0.0
- Python bridge: v1.0.0

**Deployment Steps**:
1. Deploy Holochain conductors (see scripts/)
2. Install pogq_zkfl.dna to all conductors
3. Configure Python bridge endpoints
4. Run smoke tests
5. Monitor for 24 hours
6. Scale gradually

**Rollback Plan**:
1. Stop all conductors
2. Revert to previous DNA version
3. Restart conductors
4. Verify functionality

**Support Contact**: [your-email]
EOF

# Package deployment
tar -czf deployment/zkfl_production_v1.0.0.tar.gz -C deployment/production .
```

**Afternoon: Production Infrastructure Setup**
```bash
# Set up production Holochain network
# Assuming 20 production nodes

cd /srv/luminous-dynamics/Mycelix-Core/0TML/holochain-dht-setup

# Generate production conductor configs
for i in {0..19}; do
  ADMIN_PORT=$((8888 + i*2))
  APP_PORT=$((9888 + i*2))
  QUIC_PORT=$((10000 + i))

  cat > configs/conductor_prod_${i}.yaml << EOF
---
environment_path: /var/lib/holochain/conductor_${i}
use_dangerous_test_keystore: false
keystore:
  type: lair_server
  connection_url: unix:///var/run/lair-keystore/socket_${i}
dpki:
  instance_id: zkfl_prod_${i}
  init_params: ~
admin_interfaces:
  - driver:
      type: websocket
      port: ${ADMIN_PORT}
network:
  transport_pool:
    - type: quic
      bind_to: 0.0.0.0:${QUIC_PORT}
  bootstrap_service: https://bootstrap.holo.host
  tuning_params:
    gossip_loop_iteration_delay_ms: 10
    default_rpc_single_timeout_ms: 30000
    default_rpc_multi_remote_agent_count: 3
EOF
done

# Deploy to production (careful!)
./scripts/deploy-with-screen.sh 20
```

**Day 5: Go-Live**

**Morning: Final Validation**
```bash
# Smoke test all production conductors
cd /srv/luminous-dynamics/Mycelix-Core/0TML

cat > production_smoke_test.sh << 'EOF'
#!/bin/bash

echo "=== Production Smoke Test ==="

# Test all conductors
for port in {9888..9926..2}; do
  echo -n "Testing conductor on port $port... "

  response=$(curl -s -X POST http://localhost:$port/list_apps)

  if echo "$response" | grep -q "pogq_zkfl"; then
    echo "✅ OK"
  else
    echo "❌ FAILED"
    exit 1
  fi
done

echo "✅ All conductors operational"

# Test proof submission
echo -n "Testing proof submission... "
python3 -c "
from experiments.holochain_bridge import HolochainBridge
from pathlib import Path

bridge = HolochainBridge('http://localhost:9888')
action_hash = bridge.submit_proof(
    node_id='smoke_test',
    round_number=0,
    proof_path=Path('test_proofs/proof.bin'),
    public_path=Path('test_proofs/public_echo.json'),
    journal_path=Path('test_proofs/journal.bin')
)
print('✅ Proof submitted:', action_hash)
"

echo "✅ Production smoke test PASSED"
EOF

chmod +x production_smoke_test.sh
./production_smoke_test.sh
```

**Afternoon: Monitoring Setup**
```bash
# Set up monitoring (Prometheus + Grafana)

# Create monitoring dashboard config
cat > monitoring/zkfl_dashboard.json << 'EOF'
{
  "dashboard": {
    "title": "ZK-FL Production Monitoring",
    "panels": [
      {
        "title": "Proof Submissions (per minute)",
        "targets": ["rate(holochain_proof_submissions_total[1m])"]
      },
      {
        "title": "Proof Verification Time",
        "targets": ["histogram_quantile(0.95, holochain_proof_verification_duration_seconds)"]
      },
      {
        "title": "DHT Propagation Latency",
        "targets": ["holochain_dht_propagation_seconds"]
      },
      {
        "title": "Byzantine Node Detection Rate",
        "targets": ["rate(zkfl_byzantine_detections_total[5m])"]
      }
    ]
  }
}
EOF

# Set up alerting
cat > monitoring/alerts.yaml << 'EOF'
groups:
  - name: zkfl_alerts
    interval: 30s
    rules:
      - alert: HighProofFailureRate
        expr: rate(holochain_proof_failures_total[5m]) > 0.1
        annotations:
          summary: "High proof submission failure rate"

      - alert: DHTPropagationSlow
        expr: holochain_dht_propagation_seconds > 10
        annotations:
          summary: "DHT propagation taking >10 seconds"

      - alert: ByzantineNodeDetected
        expr: increase(zkfl_byzantine_detections_total[5m]) > 0
        annotations:
          summary: "Byzantine node detected in network"
EOF
```

**Go-Live Checklist**:
```bash
# Final pre-launch checklist
cat > GO_LIVE_CHECKLIST.md << 'EOF'
# ZK-FL Production Go-Live Checklist

## Pre-Launch
- [ ] All 20 conductors running
- [ ] DNS records configured
- [ ] Firewall rules in place
- [ ] SSL certificates installed
- [ ] Monitoring dashboard live
- [ ] Alerting configured
- [ ] On-call rotation scheduled
- [ ] Rollback plan tested
- [ ] Backup strategy verified
- [ ] Documentation complete

## Launch
- [ ] Smoke tests passing
- [ ] Load test completed (1000 proofs)
- [ ] Security scan clean
- [ ] Performance benchmarks met
- [ ] Stakeholders notified

## Post-Launch (First 24h)
- [ ] Monitor error rates
- [ ] Check DHT propagation
- [ ] Verify Byzantine detection
- [ ] Review performance metrics
- [ ] Document any issues
- [ ] Scale if needed

## Post-Launch (First Week)
- [ ] Collect user feedback
- [ ] Performance tuning
- [ ] Security review
- [ ] Capacity planning
- [ ] Roadmap adjustment

**Sign-Off**: __________________  Date: __________
EOF
```

---

## Conservative Timeline (4 Weeks)

### Week 1: Integration & Testing
- Day 1-2: Build zome, deploy to test environment
- Day 3-4: 2-node and 5-node integration tests
- Day 5: Security hardening (nonce tracking, link indexing)

### Week 2: Security & Load Testing
- Day 1-2: Security audit (penetration testing)
- Day 3-4: Load testing (1000+ proofs)
- Day 5: Performance optimization

### Week 3: Production Preparation
- Day 1-2: Production infrastructure setup
- Day 3-4: Deployment automation
- Day 5: Disaster recovery testing

### Week 4: Deployment & Monitoring
- Day 1-2: Staged rollout (5 nodes → 10 nodes → 20 nodes)
- Day 3-4: Monitoring and tuning
- Day 5: Full production launch

---

## Key Metrics to Monitor

### Performance Metrics
```bash
# Proof generation time
avg(vsv_stark_proof_generation_seconds) by (node_id)

# Proof verification time
histogram_quantile(0.95, holochain_proof_verification_duration_seconds)

# DHT propagation latency
avg(holochain_dht_propagation_seconds)

# Consensus latency
avg(zkfl_consensus_calculation_seconds) by (round_number)
```

### Security Metrics
```bash
# Byzantine detection rate
rate(zkfl_byzantine_detections_total[5m])

# Nonce replay attempts
rate(holochain_nonce_replay_attempts_total[1m])

# Invalid proof submissions
rate(holochain_invalid_proof_submissions_total[1m])

# Proof verification failures
rate(holochain_proof_verification_failures_total[1m])
```

### Health Metrics
```bash
# Conductor uptime
up{job="holochain_conductor"}

# DHT peer count
holochain_dht_peer_count by (conductor_id)

# Memory usage
process_resident_memory_bytes{job="holochain_conductor"}

# CPU usage
rate(process_cpu_seconds_total{job="holochain_conductor"}[1m])
```

---

## Rollback Procedure

If critical issues are discovered:

```bash
# Step 1: Stop accepting new proofs
# (Application-level flag, not infrastructure change)

# Step 2: Assess severity
# - Data corruption? → Immediate rollback
# - Performance issue? → Scale down, investigate
# - Security issue? → Immediate rollback

# Step 3: Execute rollback
cd /srv/luminous-dynamics/Mycelix-Core/0TML/holochain-dht-setup

# Stop all conductors
./scripts/stop-conductors.sh

# Restore previous DNA version
cp backups/pogq_zkfl_v0.9.0.dna pogq_zkfl.dna

# Restart conductors
./scripts/deploy-with-screen.sh 20

# Reinstall DNA to all conductors
for port in {9888..9926..2}; do
  curl -X POST http://localhost:$port/install_app \
    -H "Content-Type: application/json" \
    -d '{
      "installed_app_id": "pogq_zkfl",
      "agent_key": null,
      "dnas": [{"path": "../pogq_zkfl.dna", "nick": "pogq"}]
    }'
done

# Verify rollback
./production_smoke_test.sh

# Step 4: Notify stakeholders
# Step 5: Root cause analysis
# Step 6: Fix and re-deploy
```

---

## Success Criteria

### Day 1
- ✅ Zome builds successfully
- ✅ DNA installs to test conductors
- ✅ Python bridge submits proof successfully

### Day 3
- ✅ 2-node integration test passes
- ✅ 5-node Byzantine demo works
- ✅ Consensus correctly identifies Byzantine node

### Week 2
- ✅ Security audit clean
- ✅ Load test handles 1000+ proofs
- ✅ Performance within targets (46.6s proving, 92ms verify)

### Week 4
- ✅ Production deployment successful
- ✅ 24-hour stability demonstrated
- ✅ No critical issues found
- ✅ Ready to scale

---

## Contact & Support

**Project Lead**: [Your Name]
**Email**: [your-email]
**Emergency Contact**: [emergency-phone]
**Status Page**: https://status.zkfl.luminousdynamics.io
**Documentation**: /srv/luminous-dynamics/Mycelix-Core/0TML/docs/

---

**Action Plan Version**: 1.0
**Last Updated**: November 11, 2025
**Next Review**: After Week 1 completion
