# Week 9-10: Production Deployment & Infrastructure Integration

**Version**: 1.0
**Date**: November 11, 2025
**Status**: Planning Phase
**Dependencies**: Week 7-8 Governance Integration (COMPLETE ✅)

---

## Executive Summary

This plan integrates Week 7-8 governance with existing infrastructure:
- **Polygon L2**: For on-chain voting and treasury
- **Filecoin/IPFS**: For decentralized storage
- **Holochain DHT**: For agent-centric identity and reputation

**Goal**: Production-ready, multi-chain governance system with decentralized storage.

---

## Architecture Evolution

### Current State (Week 7-8)

```
Python Governance Coordinator
        ↓
Holochain DHT (mocked)
        ↓
Governance Records (in-memory)
```

### Target State (Week 9-10)

```
┌─────────────────────────────────────────────────────────────┐
│              Python Governance Coordinator                   │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐  │
│  │ Proposal     │  │ Voting       │  │ Guardian Auth   │  │
│  │ Manager      │  │ Engine       │  │ Manager         │  │
│  └──────────────┘  └──────────────┘  └─────────────────┘  │
└───────────┬───────────────────┬─────────────────┬──────────┘
            │                   │                 │
            ▼                   ▼                 ▼
┌─────────────────┐  ┌─────────────────┐  ┌──────────────────┐
│ Holochain DHT   │  │ Polygon L2      │  │ Filecoin/IPFS    │
│ (Identity &     │  │ (Voting &       │  │ (Storage &       │
│  Reputation)    │  │  Treasury)      │  │  Evidence)       │
└─────────────────┘  └─────────────────┘  └──────────────────┘
```

**Key Design Decision**: Use each system for its strengths:
- **Holochain**: Identity, reputation, guardian networks (agent-centric)
- **Polygon**: Voting records, treasury, on-chain execution (global consensus)
- **Filecoin/IPFS**: Large files, evidence, permanent archives (decentralized storage)

---

## Track 1: Production Readiness (15-22 days)

### Phase 1.1: Holochain Integration (5-7 days)

**Objective**: Replace mocked DHT with real Holochain conductor.

#### Tasks

1. **Deploy Governance Record Zome** (2 days)
   ```bash
   # Build zome
   cd zerotrustml-identity-dna/zomes/governance_record
   cargo build --release --target wasm32-unknown-unknown

   # Package DNA
   hc dna pack ../..

   # Deploy to conductor
   hc sandbox create governance-prod
   hc sandbox run -p 8888 governance-prod
   ```

2. **Update Python DHT Client** (1 day)
   ```python
   # Replace mock client with real Holochain client
   from holochain_client import HolochainClient

   dht_client = HolochainClient(
       url="ws://localhost:8888",
       timeout=30
   )

   # Test connection
   await dht_client.call_zome(
       zome_name="governance_record",
       fn_name="get_proposal",
       payload={"proposal_id": "test_id"}
   )
   ```

3. **Integration Testing** (2-3 days)
   - Run all 55+ tests against real DHT
   - Measure actual latency (expect 50-200ms for DHT operations)
   - Fix any issues exposed by real network conditions
   - Test with 10-100 concurrent operations

4. **Performance Baseline** (0.5 days)
   - Measure vote weight calculation: <5ms (should be cached)
   - Measure vote casting: 50-100ms (DHT write)
   - Measure proposal creation: 100-200ms (DHT write + validation)
   - Measure vote tallying (1000 votes): 500-1000ms (DHT read)

**Deliverable**: Governance system working with real Holochain DHT.

---

### Phase 1.2: Filecoin/IPFS Integration (3-5 days)

**Objective**: Use Filecoin for large file storage and evidence.

#### Why Filecoin?

- **Holochain DHT**: Good for small structured data (proposals, votes)
- **Filecoin/IPFS**: Better for large files (evidence documents, logs, attachments)
- **Benefits**:
  - Permanent storage with incentivized persistence
  - Content-addressed (IPFS CID)
  - Verifiable retrieval
  - Censorship resistant

#### Architecture

```python
# Store small data in Holochain
proposal = {
    "proposal_id": "prop_123",
    "title": "Ban participant eve_id",
    "description": "Brief summary...",
    "evidence_cid": "QmX..."  # ← IPFS CID pointing to Filecoin
}

# Store large evidence in Filecoin
evidence = {
    "logs": "50 MB of round logs",
    "screenshots": [multiple images],
    "analysis": "Detailed Byzantine detection report"
}
# → Stored on Filecoin, referenced by CID
```

#### Tasks

1. **Setup Filecoin/IPFS Client** (1 day)
   ```python
   from web3.storage import Web3Storage
   # or
   from lighthouse import Lighthouse  # Lighthouse.storage is excellent

   storage = Lighthouse(api_key=LIGHTHOUSE_API_KEY)

   # Upload evidence
   cid = await storage.upload_json(evidence)
   # Returns: QmX... (IPFS CID)
   ```

2. **Integrate with Proposal Creation** (1 day)
   ```python
   async def create_proposal_with_evidence(
       proposer_id: str,
       title: str,
       description: str,
       evidence_files: List[bytes],
       execution_params: Dict
   ) -> str:
       # Step 1: Upload evidence to Filecoin
       evidence_cids = []
       for file in evidence_files:
           cid = await storage.upload_file(file)
           evidence_cids.append(cid)

       # Step 2: Create proposal with CIDs
       proposal = ProposalData(
           proposal_id=generate_id(),
           title=title,
           description=description,
           execution_params=execution_params,
           evidence_cids=evidence_cids  # ← Reference to Filecoin
       )

       # Step 3: Store proposal metadata in Holochain
       await dht_client.call_zome(
           zome_name="governance_record",
           fn_name="store_proposal",
           payload=proposal.to_dict()
       )
   ```

3. **Evidence Retrieval** (1 day)
   ```python
   async def get_proposal_with_evidence(proposal_id: str) -> Dict:
       # Get proposal from Holochain
       proposal = await dht_client.call_zome(
           zome_name="governance_record",
           fn_name="get_proposal",
           payload={"proposal_id": proposal_id}
       )

       # Fetch evidence from Filecoin
       if proposal.get("evidence_cids"):
           evidence_data = []
           for cid in proposal["evidence_cids"]:
               data = await storage.retrieve(cid)
               evidence_data.append(data)
           proposal["evidence"] = evidence_data

       return proposal
   ```

4. **Archive Old Governance Records** (1 day)
   ```python
   async def archive_completed_proposals():
       """
       Move old proposals to Filecoin for permanent storage
       Keep recent proposals in Holochain for fast access
       """
       # Get proposals older than 90 days
       old_proposals = await get_old_proposals(days=90)

       for proposal in old_proposals:
           # Upload to Filecoin
           cid = await storage.upload_json(proposal.to_dict())

           # Update Holochain record with archive CID
           await dht_client.call_zome(
               zome_name="governance_record",
               fn_name="archive_proposal",
               payload={
                   "proposal_id": proposal.proposal_id,
                   "archive_cid": cid
               }
           )
   ```

5. **Testing** (0.5-1 day)
   - Upload 10+ test evidence files
   - Verify retrieval from IPFS
   - Test with large files (100MB+)
   - Verify Filecoin deal status

**Deliverable**: Governance system with Filecoin storage for evidence.

**Cost Estimate**: ~$5-20/month for typical governance storage (highly dependent on volume).

---

### Phase 1.3: Polygon L2 Integration (4-6 days)

**Objective**: Move voting records to Polygon for on-chain verifiability.

#### Why Polygon?

- **Already deployed**: Leverage existing infrastructure
- **On-chain verifiability**: Anyone can verify votes
- **Gas efficiency**: Polygon has low transaction costs
- **EVM compatibility**: Use Solidity smart contracts
- **Finality**: Votes are final once on-chain

#### Architecture Decision

**Hybrid Approach**:
- **Holochain**: Identity, reputation, guardian networks
- **Polygon**: Vote records, proposal outcomes, treasury
- **Filecoin**: Evidence and large files

**Voting Flow**:
```
1. Participant casts vote (Python Governance Coordinator)
2. Vote signed with participant's private key
3. Vote stored on Polygon (on-chain record)
4. Vote metadata stored on Holochain (fast queries)
5. Proposal outcome finalized on Polygon
```

#### Tasks

1. **Deploy Governance Smart Contracts** (2-3 days)

   ```solidity
   // GovernanceVoting.sol
   pragma solidity ^0.8.20;

   contract GovernanceVoting {
       struct Vote {
           bytes32 proposalId;
           address voter;
           string voterDID;  // W3C DID
           uint8 choice;  // 0=FOR, 1=AGAINST, 2=ABSTAIN
           uint256 creditsSpent;
           uint256 voteWeight;
           uint256 effectiveVotes;
           uint256 timestamp;
           bytes signature;
       }

       struct Proposal {
           bytes32 proposalId;
           string title;
           string descriptionCID;  // IPFS CID
           uint256 votingStart;
           uint256 votingEnd;
           uint256 quorum;
           uint256 approvalThreshold;
           uint256 totalVotesFor;
           uint256 totalVotesAgainst;
           uint256 totalVotesAbstain;
           ProposalStatus status;
       }

       enum ProposalStatus {
           VOTING,
           APPROVED,
           REJECTED,
           EXECUTED
       }

       mapping(bytes32 => Proposal) public proposals;
       mapping(bytes32 => Vote[]) public proposalVotes;

       event ProposalCreated(bytes32 indexed proposalId, string title);
       event VoteCast(bytes32 indexed proposalId, address voter, uint8 choice);
       event ProposalFinalized(bytes32 indexed proposalId, ProposalStatus status);

       function castVote(
           bytes32 proposalId,
           uint8 choice,
           uint256 creditsSpent,
           uint256 voteWeight,
           string calldata voterDID,
           bytes calldata signature
       ) external {
           // Verify signature
           require(verifySignature(proposalId, choice, voterDID, signature), "Invalid signature");

           // Calculate effective votes (quadratic)
           uint256 effectiveVotes = sqrt(creditsSpent) * voteWeight / 1e18;

           // Store vote
           Vote memory vote = Vote({
               proposalId: proposalId,
               voter: msg.sender,
               voterDID: voterDID,
               choice: choice,
               creditsSpent: creditsSpent,
               voteWeight: voteWeight,
               effectiveVotes: effectiveVotes,
               timestamp: block.timestamp,
               signature: signature
           });

           proposalVotes[proposalId].push(vote);

           // Update proposal tallies
           if (choice == 0) {
               proposals[proposalId].totalVotesFor += effectiveVotes;
           } else if (choice == 1) {
               proposals[proposalId].totalVotesAgainst += effectiveVotes;
           } else {
               proposals[proposalId].totalVotesAbstain += effectiveVotes;
           }

           emit VoteCast(proposalId, msg.sender, choice);
       }

       function finalizeProposal(bytes32 proposalId) external {
           Proposal storage proposal = proposals[proposalId];
           require(block.timestamp > proposal.votingEnd, "Voting not ended");
           require(proposal.status == ProposalStatus.VOTING, "Not in voting");

           // Check quorum
           uint256 totalVotes = proposal.totalVotesFor + proposal.totalVotesAgainst + proposal.totalVotesAbstain;
           uint256 participation = totalVotes * 1e18 / getTotalVotingPower();
           require(participation >= proposal.quorum, "Quorum not met");

           // Check approval
           uint256 approvalRate = proposal.totalVotesFor * 1e18 / (proposal.totalVotesFor + proposal.totalVotesAgainst);

           if (approvalRate >= proposal.approvalThreshold) {
               proposal.status = ProposalStatus.APPROVED;
           } else {
               proposal.status = ProposalStatus.REJECTED;
           }

           emit ProposalFinalized(proposalId, proposal.status);
       }
   }
   ```

2. **Python Integration with Polygon** (1-2 days)
   ```python
   from web3 import Web3
   from web3.middleware import geth_poa_middleware

   # Connect to Polygon
   w3 = Web3(Web3.HTTPProvider('https://polygon-rpc.com'))
   w3.middleware_onion.inject(geth_poa_middleware, layer=0)

   # Load contract
   governance_contract = w3.eth.contract(
       address=GOVERNANCE_CONTRACT_ADDRESS,
       abi=GOVERNANCE_ABI
   )

   async def cast_vote_on_chain(
       voter_participant_id: str,
       proposal_id: str,
       choice: VoteChoice,
       credits_spent: int,
       vote_weight: float
   ) -> str:
       # Get voter's wallet
       voter_wallet = await identity_coordinator.get_wallet(voter_participant_id)

       # Create signature
       message = encode_defunct(text=f"{proposal_id}{choice}{credits_spent}")
       signature = w3.eth.account.sign_message(message, private_key=voter_wallet.private_key)

       # Build transaction
       tx = governance_contract.functions.castVote(
           proposalId=bytes.fromhex(proposal_id),
           choice=choice.value,
           creditsSpent=credits_spent,
           voteWeight=int(vote_weight * 1e18),
           voterDID=await identity_coordinator.get_did(voter_participant_id),
           signature=signature.signature
       ).build_transaction({
           'from': voter_wallet.address,
           'gas': 200000,
           'gasPrice': w3.eth.gas_price,
           'nonce': w3.eth.get_transaction_count(voter_wallet.address)
       })

       # Sign and send transaction
       signed_tx = w3.eth.account.sign_transaction(tx, voter_wallet.private_key)
       tx_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)

       # Wait for confirmation
       receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

       return tx_hash.hex()
   ```

3. **Dual-Store Pattern** (1 day)
   ```python
   async def cast_vote(
       voter_participant_id: str,
       proposal_id: str,
       choice: VoteChoice,
       credits_spent: int
   ) -> Tuple[bool, str]:
       # Calculate vote weight
       vote_weight = await gov_coord.gov_extensions.calculate_vote_weight(voter_participant_id)

       # Store on Polygon (authoritative)
       try:
           tx_hash = await cast_vote_on_chain(
               voter_participant_id,
               proposal_id,
               choice,
               credits_spent,
               vote_weight
           )
       except Exception as e:
           return False, f"Polygon transaction failed: {e}"

       # Store on Holochain (fast queries)
       try:
           await dht_client.call_zome(
               zome_name="governance_record",
               fn_name="store_vote",
               payload={
                   "vote_id": generate_id(),
                   "proposal_id": proposal_id,
                   "voter_did": await identity_coordinator.get_did(voter_participant_id),
                   "choice": choice.value,
                   "credits_spent": credits_spent,
                   "vote_weight": vote_weight,
                   "tx_hash": tx_hash  # ← Link to Polygon
               }
           )
       except Exception as e:
           logger.warning(f"Holochain store failed (vote still valid on Polygon): {e}")

       return True, f"Vote cast successfully. Tx: {tx_hash}"
   ```

4. **Testing** (1 day)
   - Deploy contracts to Polygon Mumbai testnet
   - Test vote casting with 10+ participants
   - Verify on-chain data matches Holochain
   - Test gas costs (should be <$0.01 per vote)
   - Test proposal finalization

**Deliverable**: Votes recorded on Polygon, queryable from Holochain.

**Cost Estimate**: ~$0.005-0.01 per vote on Polygon.

---

### Phase 1.4: Security Audit (3-5 days)

**Objective**: Identify and fix security vulnerabilities.

#### Self-Audit Checklist

1. **Sybil Attack Resistance** (1 day)
   - Create 1000 low-reputation identities
   - Attempt to manipulate vote
   - Verify: 1000 Sybils = ~31 honest participants ✅
   - Test: Vote weight calculation prevents Sybil influence

2. **Vote Buying Prevention** (1 day)
   - Simulate vote buying scenario
   - Calculate cost to buy 50% of votes
   - Verify: Quadratic voting makes this expensive
   - Test: Cost increases as O(n²) with vote power

3. **Guardian Collusion** (1 day)
   - Simulate 3/5 guardians colluding
   - Attempt to authorize malicious action
   - Verify: 60% control insufficient for 70-90% thresholds
   - Test: Need 4-5/5 guardians for most actions

4. **Signature Verification** (0.5 day)
   - Test vote signature verification
   - Attempt replay attacks
   - Attempt signature forgery
   - Verify: All invalid signatures rejected

5. **Rate Limiting** (0.5 day)
   - Attempt to exceed rate limits
   - Test multiple accounts from same IP
   - Verify: Rate limits enforced per identity, not IP

6. **Smart Contract Security** (1 day)
   - Run Slither static analysis on Solidity contracts
   - Check for reentrancy vulnerabilities
   - Check for integer overflow/underflow
   - Verify access control (only authorized functions)

#### Professional Audit (Optional)

- **Cost**: $10-50k depending on scope
- **Duration**: 2-4 weeks
- **Firms**: Trail of Bits, OpenZeppelin, ConsenSys Diligence
- **Recommendation**: Do after MVP is battle-tested

**Deliverable**: Security audit report with issues fixed.

---

### Phase 1.5: Monitoring & Observability (2-3 days)

**Objective**: Production monitoring for governance system.

#### Prometheus Metrics

```python
from prometheus_client import Counter, Histogram, Gauge

# Proposal metrics
proposals_created = Counter('governance_proposals_created_total', 'Total proposals created', ['type'])
proposals_approved = Counter('governance_proposals_approved_total', 'Total proposals approved', ['type'])
proposals_rejected = Counter('governance_proposals_rejected_total', 'Total proposals rejected', ['type'])

# Vote metrics
votes_cast = Counter('governance_votes_cast_total', 'Total votes cast', ['proposal_id', 'choice'])
vote_weight_distribution = Histogram('governance_vote_weight_distribution', 'Vote weight distribution')

# Guardian metrics
guardian_approvals = Counter('governance_guardian_approvals_total', 'Total guardian approvals', ['action'])
guardian_response_time = Histogram('governance_guardian_response_time_seconds', 'Guardian response time')

# Performance metrics
vote_cast_latency = Histogram('governance_vote_cast_latency_seconds', 'Vote cast latency')
proposal_creation_latency = Histogram('governance_proposal_creation_latency_seconds', 'Proposal creation latency')

# Engagement metrics
active_voters = Gauge('governance_active_voters', 'Number of active voters in last 7 days')
participation_rate = Gauge('governance_participation_rate', 'Overall participation rate')
```

#### Grafana Dashboards

1. **Governance Overview Dashboard**
   - Total proposals (by type)
   - Approval rate over time
   - Participation rate
   - Active voters

2. **Voting Activity Dashboard**
   - Votes per day
   - Vote weight distribution
   - Choice distribution (FOR/AGAINST/ABSTAIN)
   - Top voters by participation

3. **Guardian Network Dashboard**
   - Guardian response times
   - Approval rates
   - Guardian network graph
   - Authorization requests status

4. **Security Monitoring Dashboard**
   - Sybil detection alerts
   - Unusual voting patterns
   - Rate limit violations
   - Byzantine attack indicators

5. **Performance Dashboard**
   - Latency percentiles (p50, p95, p99)
   - DHT operation times
   - Polygon transaction times
   - Error rates

#### Alert Rules

```yaml
# alerts.yml
groups:
  - name: governance
    rules:
      # High rejection rate
      - alert: HighProposalRejectionRate
        expr: rate(governance_proposals_rejected_total[1h]) / rate(governance_proposals_created_total[1h]) > 0.8
        for: 1h
        annotations:
          summary: "High proposal rejection rate: {{ $value }}"

      # Low participation
      - alert: LowParticipationRate
        expr: governance_participation_rate < 0.3
        for: 24h
        annotations:
          summary: "Participation rate below 30%: {{ $value }}"

      # Slow guardian response
      - alert: SlowGuardianResponse
        expr: histogram_quantile(0.95, governance_guardian_response_time_seconds) > 3600
        for: 1h
        annotations:
          summary: "95th percentile guardian response time > 1 hour"

      # High latency
      - alert: HighVoteCastLatency
        expr: histogram_quantile(0.95, governance_vote_cast_latency_seconds) > 5
        for: 10m
        annotations:
          summary: "95th percentile vote cast latency > 5 seconds"
```

**Deliverable**: Production monitoring with alerts.

---

## Summary: Track 1 (Production Readiness)

### Timeline

| Phase | Duration | Key Deliverable |
|-------|----------|-----------------|
| 1.1 Holochain Integration | 5-7 days | Real DHT, no mocks |
| 1.2 Filecoin Integration | 3-5 days | Evidence storage |
| 1.3 Polygon Integration | 4-6 days | On-chain voting |
| 1.4 Security Audit | 3-5 days | Security fixes |
| 1.5 Monitoring | 2-3 days | Observability |
| **Total** | **17-26 days** | Production-ready |

### Cost Estimate

- **Filecoin Storage**: $5-20/month
- **Polygon Gas**: $0.005-0.01 per vote (~$50-500/month for 10k votes)
- **Holochain Hosting**: $10-50/month (VPS)
- **Monitoring**: $0-20/month (Grafana Cloud free tier)
- **Total**: $65-590/month depending on usage

### Success Criteria

- ✅ All 55+ tests pass with real infrastructure
- ✅ Vote cast latency <5s (p95)
- ✅ Proposal creation latency <10s (p95)
- ✅ Security audit passes with no critical issues
- ✅ Monitoring dashboards operational
- ✅ Tested with 100+ participants
- ✅ Gas costs <$0.02 per vote

---

**Status**: Planning Complete - Ready for Implementation
**Next**: Begin Phase 1.1 (Holochain Integration)
