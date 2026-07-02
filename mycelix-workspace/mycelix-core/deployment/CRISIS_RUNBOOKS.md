# Mycelix Crisis Response Runbooks

> **Purpose**: Operational playbooks for handling system crises

---

## Runbook Index

| ID | Crisis Type | Severity | Response Time |
|----|-------------|----------|---------------|
| CR-001 | Network Partition | Critical | < 15 min |
| CR-002 | Security Breach | Critical | < 5 min |
| CR-003 | Governance Deadlock | High | < 24 hours |
| CR-004 | Bridge Failure | Critical | < 30 min |
| CR-005 | Economic Attack | High | < 1 hour |
| CR-006 | DDoS Attack | High | < 10 min |
| CR-007 | Data Corruption | Critical | < 30 min |
| CR-008 | Community Crisis | Medium | < 4 hours |

---

## CR-001: Network Partition

### Detection
```yaml
alert: NetworkPartition
triggers:
  - mycelix_network_partition_detected == 1
  - mycelix_dht_peer_count < threshold * 0.5
  - mycelix_gossip_failures > 100/min
```

### Severity Assessment
| Condition | Level |
|-----------|-------|
| < 10% nodes affected | Level 2 (Alert) |
| 10-30% nodes affected | Level 3 (Emergency) |
| > 30% nodes affected | Level 4 (Major) |

### Response Steps

#### Phase 1: Triage (0-5 min)
```bash
# 1. Confirm partition
mycelix-ops network status --detailed

# 2. Identify affected regions
mycelix-ops network partitions --list

# 3. Activate incident
mycelix-incident create --type network-partition --severity $LEVEL
```

#### Phase 2: Contain (5-15 min)
```bash
# 4. Enable partition-tolerant mode
mycelix-ops network set-mode partition-tolerant

# 5. Pause cross-partition writes (prevent split-brain)
mycelix-ops governance pause-voting --reason "network-partition"
mycelix-ops economic pause-large-transfers --threshold 1000

# 6. Notify affected communities
mycelix-beacon broadcast --template partition-alert --regions $AFFECTED
```

#### Phase 3: Diagnose (15-60 min)
```bash
# 7. Identify root cause
mycelix-ops network diagnose --partition-id $ID

# Common causes:
# - Internet backbone issues
# - DNS failures
# - Conductor bugs
# - Targeted attack
```

#### Phase 4: Resolve
```bash
# 8. For internet issues: Wait for recovery, monitor

# 9. For DNS: Switch to backup resolvers
mycelix-ops network set-dns --backup

# 10. For conductor bugs: Rolling restart
mycelix-ops conductor restart --rolling --partition $ID

# 11. For attack: Implement mitigations (see CR-006)
```

#### Phase 5: Reconcile
```bash
# 12. Once network heals, reconcile state
mycelix-ops network reconcile --auto

# 13. If conflicts exist, use conflict resolution
mycelix-ops network conflicts --list
mycelix-ops network conflicts resolve --strategy latest-wins

# 14. Resume normal operations
mycelix-ops governance resume-voting
mycelix-ops economic resume-transfers
```

### Post-Incident
- [ ] Complete incident report within 48 hours
- [ ] Update runbook with lessons learned
- [ ] Review network resilience architecture
- [ ] Schedule post-mortem with affected communities

---

## CR-002: Security Breach

### Detection
```yaml
alert: SecurityBreach
triggers:
  - mycelix_security_breach_detected == 1
  - mycelix_unauthorized_access_attempts > 50/min
  - mycelix_data_exfiltration_detected == 1
  - External security report received
```

### Severity Assessment
| Condition | Level |
|-----------|-------|
| Attempted breach (blocked) | Level 2 |
| Limited access breach | Level 3 |
| Significant data access | Level 4 |
| Systemic compromise | Level 5 |

### Response Steps

#### Phase 1: Immediate (0-5 min)
```bash
# 1. IMMEDIATELY escalate to security team
mycelix-incident create --type security-breach --severity critical --page-security

# 2. Preserve evidence
mycelix-security snapshot --all

# 3. DO NOT discuss breach details in public channels
```

#### Phase 2: Contain (5-30 min)
```bash
# 4. Isolate affected systems
mycelix-security isolate --scope $AFFECTED_SYSTEMS

# 5. Revoke suspected compromised credentials
mycelix-security revoke-credentials --agents $SUSPICIOUS_AGENTS

# 6. Enable enhanced logging
mycelix-security logging set-level debug --all

# 7. Block known attack IPs (if applicable)
mycelix-security block-ips --list $ATTACK_IPS
```

#### Phase 3: Investigate
```bash
# 8. Analyze attack vector
mycelix-security analyze --timeframe "last 24h" --detailed

# 9. Identify scope of compromise
mycelix-security scope-assessment

# 10. Check for persistence mechanisms
mycelix-security check-persistence

# 11. Review affected data
mycelix-security data-access-audit --timeframe $BREACH_WINDOW
```

#### Phase 4: Eradicate
```bash
# 12. Remove attacker access
mycelix-security remove-access --agent $ATTACKER_ID

# 13. Patch vulnerability (if identified)
# Follow emergency patch process

# 14. Reset affected credentials
mycelix-security credential-reset --scope $AFFECTED
```

#### Phase 5: Recover
```bash
# 15. Gradually restore services
mycelix-security restore --phased

# 16. Monitor for re-attack
mycelix-security monitor --enhanced --duration 7d

# 17. Notify affected users (as legally required)
mycelix-security notification --template breach-notification
```

### Legal Requirements
- [ ] Determine if breach triggers notification requirements
- [ ] Notify legal counsel within 24 hours
- [ ] Prepare regulatory notifications if required
- [ ] Document all actions taken

---

## CR-003: Governance Deadlock

### Detection
```yaml
alert: GovernanceDeadlock
triggers:
  - mycelix_governance_deadlock_score > 0.8 for 1h
  - mycelix_proposal_cycling > 3 (same proposal resubmitted)
  - mycelix_bicameral_disagreement > 3 consecutive
```

### Response Steps

#### Phase 1: Assessment (0-2 hours)
```bash
# 1. Identify deadlock type
mycelix-gov analyze-deadlock

# Types:
# - Polarization (faction split)
# - Cycling (same proposals returning)
# - Bicameral (chambers disagree)
# - Quorum failure (not enough participation)
```

#### Phase 2: Intervention Selection

**For Polarization:**
```bash
# 2a. Activate mediation protocol
mycelix-gov mediation start --proposal $PROPOSAL_ID

# 2b. Notify Wisdom Council
mycelix-gov notify-wisdom-council --issue deadlock

# 2c. Schedule facilitated dialogue
mycelix-gov schedule-dialogue --facilitator $MEDIATOR
```

**For Cycling:**
```bash
# 2a. Require proposal revision
mycelix-gov require-revision --proposal $PROPOSAL_ID --min-changes 20%

# 2b. Implement cooling-off period
mycelix-gov cooling-off --proposal $PROPOSAL_ID --duration 30d
```

**For Bicameral:**
```bash
# 2a. Activate joint committee
mycelix-gov joint-committee create --proposal $PROPOSAL_ID

# 2b. If constitutional issue, escalate to Wisdom Council
mycelix-gov constitutional-review --request
```

**For Quorum:**
```bash
# 2a. Extend voting period
mycelix-gov extend-voting --proposal $PROPOSAL_ID --days 7

# 2b. Activate voter mobilization
mycelix-beacon broadcast --template quorum-needed

# 2c. Consider adaptive quorum
mycelix-gov enable-adaptive-quorum --threshold 0.8
```

#### Phase 3: Resolution
```bash
# 3. Monitor resolution progress
mycelix-gov deadlock-status

# 4. Document resolution process
mycelix-chronicle record --type governance-resolution
```

### Escalation Path
1. hApp Maintainer Circle (0-24h)
2. Protocol Stewards (24-72h)
3. Wisdom Council (72h+)
4. Community Assembly emergency session (if required)

---

## CR-004: Bridge Failure

### Detection
```yaml
alert: BridgeDown
triggers:
  - mycelix_bridge_health < 0.5 for 5m
  - mycelix_bridge_transactions_failed > 10/min
  - mycelix_bridge_latency_seconds > 60
```

### Response Steps

#### Phase 1: Triage (0-5 min)
```bash
# 1. Identify which bridge component failed
mycelix-bridge status --detailed

# Components: Holochain-side | Blockchain-side | Validator network
```

#### Phase 2: Contain (5-15 min)
```bash
# 2. Pause bridge transactions
mycelix-bridge pause --all

# 3. Queue pending transactions
mycelix-bridge queue-pending

# 4. Notify users
mycelix-beacon broadcast --template bridge-maintenance
```

#### Phase 3: Diagnose and Fix
```bash
# For Holochain-side issues:
mycelix-bridge diagnose holochain
mycelix-conductor restart --bridge-zomes

# For Blockchain-side issues:
mycelix-bridge diagnose blockchain
# May require smart contract interaction

# For Validator issues:
mycelix-bridge validator-status
mycelix-bridge validator-restart --failed
```

#### Phase 4: Recovery
```bash
# 5. Test bridge with small transaction
mycelix-bridge test --amount 1 --type diagnostic

# 6. Process queued transactions
mycelix-bridge process-queue --verify

# 7. Resume normal operations
mycelix-bridge resume

# 8. Monitor closely for 24h
mycelix-bridge monitor --enhanced
```

---

## CR-005: Economic Attack

### Detection
```yaml
alert: EconomicAttack
triggers:
  - mycelix_token_manipulation_score > 0.8
  - mycelix_unusual_transfer_volume > 10x normal
  - mycelix_cgc_concentration_spike detected
  - mycelix_governance_vote_buying detected
```

### Response Steps

#### Phase 1: Identify Attack Type
```bash
mycelix-economic analyze-anomaly

# Types:
# - CIV farming (fake contributions)
# - CGC manipulation (gift rings)
# - FLOW market manipulation
# - Vote buying
# - Sybil economic attack
```

#### Phase 2: Containment
```bash
# For CIV farming:
mycelix-civ freeze --agents $SUSPICIOUS --pending-review

# For CGC manipulation:
mycelix-cgc pause-transfers --agents $SUSPICIOUS

# For FLOW manipulation:
mycelix-flow pause-large-transfers --threshold $LIMIT

# For vote buying:
mycelix-gov pause-voting --proposal $AFFECTED
mycelix-security investigate-voters --proposal $AFFECTED
```

#### Phase 3: Investigation
```bash
# Graph analysis
mycelix-economic graph-analysis --timeframe "7d"

# Identify attack participants
mycelix-economic identify-colluders

# Calculate damage
mycelix-economic damage-assessment
```

#### Phase 4: Remediation
```bash
# Reverse fraudulent transactions (if possible)
mycelix-economic reverse-transactions --batch $FRAUDULENT

# Slash attacker stakes (if staked)
mycelix-economic slash --agents $ATTACKERS --reason "economic-attack"

# Update detection thresholds
mycelix-economic update-thresholds --based-on-attack
```

---

## CR-008: Community Crisis

### Detection
```yaml
triggers:
  - Community health score drops > 20 points in 7 days
  - Member exodus > 10% in 30 days
  - Conflict escalation to Member Redress Council
  - External report of community distress
```

### Response Steps

#### Phase 1: Assessment
```bash
# 1. Gather information
mycelix-community health-report --id $COMMUNITY

# 2. Identify crisis type
# - Leadership failure
# - Internal conflict
# - External threat
# - Economic collapse
# - Governance breakdown
```

#### Phase 2: Support Activation
```bash
# 3. Assign crisis support coordinator
mycelix-crisis assign-coordinator --community $COMMUNITY

# 4. Activate mutual aid network (if economic)
mycelix-mutual-aid activate --for $COMMUNITY

# 5. Provide facilitation support (if conflict)
mycelix-arbiter assign-mediator --community $COMMUNITY
```

#### Phase 3: Intervention
```bash
# For leadership failure:
mycelix-community initiate-leadership-review --id $COMMUNITY

# For internal conflict:
mycelix-arbiter escalate --to "restorative-circle"

# For external threat:
mycelix-crisis coordinate-defense --community $COMMUNITY

# For economic collapse:
mycelix-treasury emergency-grant --community $COMMUNITY
```

#### Phase 4: Recovery Support
- [ ] Assign recovery mentor from healthy community
- [ ] Weekly check-ins for 3 months
- [ ] Document lessons learned
- [ ] Update crisis prevention resources

---

## Communication Templates

### Partition Alert
```markdown
**Network Alert: Temporary Connectivity Issues**

We've detected network connectivity issues affecting [REGION/PERCENTAGE] of the network.

**Current Status:** [STATUS]
**Impact:** [IMPACT DESCRIPTION]
**Estimated Resolution:** [TIME]

**What this means for you:**
- Voting is temporarily paused
- Large transfers are paused
- Local operations continue normally

We're working to resolve this. Updates every [INTERVAL].

Questions? [CONTACT]
```

### Security Incident
```markdown
**Security Notice**

We detected and responded to a security incident on [DATE].

**What happened:** [BRIEF DESCRIPTION]
**What we did:** [RESPONSE ACTIONS]
**Were you affected?** [HOW TO CHECK]
**What should you do?** [USER ACTIONS]

We take security seriously. [LINK TO FULL REPORT]
```

---

## Escalation Contacts

| Role | Response Time | Contact Method |
|------|---------------|----------------|
| On-call Operator | < 5 min | PagerDuty |
| Security Lead | < 15 min | Signal + PagerDuty |
| Protocol Steward | < 1 hour | Email + Signal |
| Wisdom Council | < 24 hours | Formal request |
| Foundation | < 4 hours | Emergency line |

---

*These runbooks should be reviewed quarterly and updated after every incident.*
