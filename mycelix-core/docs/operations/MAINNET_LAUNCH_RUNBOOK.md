# Mainnet Launch Runbook

**Mycelix Protocol**
**Version**: 1.0
**Last Updated**: 2026-01-18
**Classification**: CONFIDENTIAL - Operations Team Only

---

## Executive Summary

This runbook provides step-by-step procedures for launching the Mycelix mainnet. It covers pre-launch verification, genesis block creation, validator coordination, and post-launch monitoring.

**Launch Target**: Q4 2026
**Minimum Validators**: 21
**Recommended Validators**: 50+

---

## Table of Contents

1. [Pre-Launch Checklist](#1-pre-launch-checklist)
2. [T-7 Days: Final Preparation](#2-t-7-days-final-preparation)
3. [T-24 Hours: Pre-Launch](#3-t-24-hours-pre-launch)
4. [T-0: Launch Sequence](#4-t-0-launch-sequence)
5. [Post-Launch: First 72 Hours](#5-post-launch-first-72-hours)
6. [Rollback Procedures](#6-rollback-procedures)
7. [Emergency Contacts](#7-emergency-contacts)
8. [Appendices](#appendices)

---

## 1. Pre-Launch Checklist

### 1.1 Security Requirements

| Requirement | Owner | Status | Verification |
|-------------|-------|--------|--------------|
| Security audit complete | Security Lead | [ ] | Audit report signed |
| All critical findings remediated | Engineering | [ ] | Re-audit confirmation |
| Bug bounty program active | Security Lead | [ ] | Program live 30+ days |
| Penetration test passed | External | [ ] | Pentest report |
| Credentials rotated | DevOps | [ ] | Rotation runbook executed |
| HSM keys generated | Security Lead | [ ] | Key ceremony complete |

### 1.2 Infrastructure Requirements

| Requirement | Owner | Status | Verification |
|-------------|-------|--------|--------------|
| Production infrastructure deployed | DevOps | [ ] | Terraform apply successful |
| DNS configured | DevOps | [ ] | dig mainnet.mycelix.network |
| SSL certificates issued | DevOps | [ ] | cert-manager ready |
| CDN configured | DevOps | [ ] | Cache rules verified |
| DDoS protection active | DevOps | [ ] | Cloudflare/AWS Shield |
| Monitoring operational | DevOps | [ ] | All dashboards green |
| Alerting configured | DevOps | [ ] | Test alerts received |
| Backup procedures tested | DevOps | [ ] | DR drill completed |

### 1.3 Smart Contract Requirements

| Requirement | Owner | Status | Verification |
|-------------|-------|--------|--------------|
| Contracts deployed to mainnet | Smart Contract | [ ] | Etherscan verified |
| Proxy admin secured | Smart Contract | [ ] | Multi-sig controlled |
| Timelock activated | Smart Contract | [ ] | 48hr delay confirmed |
| Initial parameters set | Protocol | [ ] | Parameters document |
| Emergency pause tested | Smart Contract | [ ] | Pause/unpause verified |

### 1.4 Validator Requirements

| Requirement | Owner | Status | Verification |
|-------------|-------|--------|--------------|
| 21+ validators committed | Community | [ ] | Signed agreements |
| Validator hardware verified | Community | [ ] | Benchmark results |
| Genesis stakes deposited | Validators | [ ] | On-chain verification |
| Communication channel active | Community | [ ] | All validators in channel |
| Runbooks distributed | Community | [ ] | Acknowledgment received |

### 1.5 Legal Requirements

| Requirement | Owner | Status | Verification |
|-------------|-------|--------|--------------|
| Foundation established | Legal | [ ] | Registration docs |
| Terms of Service published | Legal | [ ] | Website live |
| Privacy Policy published | Legal | [ ] | Website live |
| Token disclaimers in place | Legal | [ ] | All materials reviewed |
| Geographic restrictions implemented | Engineering | [ ] | Geo-blocking tested |

---

## 2. T-7 Days: Final Preparation

### 2.1 Testnet Freeze (T-7)

```bash
# Stop accepting new testnet validators
kubectl annotate deployment fl-coordinator \
  mycelix.io/validator-registration=disabled

# Announce testnet freeze
./scripts/announce.sh --channel all --template testnet-freeze
```

**Verification**:
- [ ] Testnet running stably for 7+ days
- [ ] No critical issues in last 48 hours
- [ ] All validators acknowledged freeze

### 2.2 Genesis File Preparation (T-6)

```bash
# Collect validator genesis information
./scripts/collect-genesis-info.sh --output genesis-data/

# Verify all validators submitted
./scripts/verify-genesis-submissions.sh --min-validators 21

# Generate preliminary genesis file
./scripts/generate-genesis.sh \
  --network mainnet \
  --validators genesis-data/validators.json \
  --parameters genesis-data/parameters.json \
  --output genesis-preliminary.json

# Distribute for review
./scripts/distribute-genesis.sh --file genesis-preliminary.json
```

**Genesis Parameters**:
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Block Time | 5 seconds | Balance between latency and security |
| FL Round Duration | 5 minutes | Sufficient for aggregation |
| Min Validators | 21 | BFT requirement |
| Initial Stake | 10,000 MYC | Sybil resistance |
| Slashing Rate | 0.5% | Deterrence without destruction |

### 2.3 Infrastructure Verification (T-5)

```bash
# Run full infrastructure test
./scripts/infra-test.sh --environment production --comprehensive

# Verify Ethereum mainnet connectivity
./scripts/eth-connectivity-test.sh --network mainnet

# Test all API endpoints
./scripts/api-health-check.sh --all-endpoints

# Verify monitoring coverage
./scripts/monitoring-coverage.sh
```

**Expected Results**:
- [ ] All nodes reachable
- [ ] Ethereum RPC < 500ms latency
- [ ] All APIs return 200
- [ ] 100% metric coverage

### 2.4 Communication Setup (T-4)

```bash
# Test validator communication
./scripts/validator-ping.sh --all

# Verify PagerDuty integration
./scripts/test-pagerduty.sh --alert-type test

# Setup launch war room
./scripts/setup-war-room.sh --channel #mainnet-launch
```

**Communication Plan**:
| Time | Channel | Message |
|------|---------|---------|
| T-4 | Discord, Twitter | Launch date announcement |
| T-2 | Validators | Final checklist reminder |
| T-1 | All | Genesis file distribution |
| T-0 | All | Launch confirmation |

### 2.5 Rehearsal (T-3)

```bash
# Full launch rehearsal on staging
./scripts/launch-rehearsal.sh --environment staging

# Document issues and resolutions
./scripts/rehearsal-report.sh
```

**Rehearsal Checklist**:
- [ ] Genesis block created successfully
- [ ] All validators came online
- [ ] First FL round completed
- [ ] Payments routed correctly
- [ ] Monitoring captured all events

### 2.6 Final Genesis (T-2)

```bash
# Finalize genesis file
./scripts/finalize-genesis.sh \
  --input genesis-preliminary.json \
  --output genesis.json

# Compute genesis hash
./scripts/genesis-hash.sh --file genesis.json
# Example: 0xabc123...

# Sign genesis (multi-sig)
./scripts/sign-genesis.sh --signers foundation-keys/

# Distribute final genesis
./scripts/distribute-genesis.sh --final --file genesis.json
```

**Verification**:
- [ ] All validators received genesis file
- [ ] Genesis hash matches across all validators
- [ ] Multi-sig threshold reached (3/5)

### 2.7 Go/No-Go Decision (T-1)

**Decision Meeting**: [DATE] [TIME] UTC

**Attendees**:
- [ ] CEO/Executive Sponsor
- [ ] Technical Lead
- [ ] Security Lead
- [ ] DevOps Lead
- [ ] Community Lead
- [ ] Legal Counsel

**Go Criteria**:
- [ ] All security requirements met
- [ ] All infrastructure requirements met
- [ ] 21+ validators ready
- [ ] No blocking issues
- [ ] Legal sign-off

**No-Go Triggers**:
- Critical security vulnerability discovered
- Infrastructure instability
- < 21 validators ready
- Unresolved legal issues
- Team not available

---

## 3. T-24 Hours: Pre-Launch

### 3.1 Final Infrastructure Check

```bash
# Production readiness check
./scripts/production-readiness.sh

# Verify all services healthy
kubectl get pods -n mycelix-mainnet
kubectl top pods -n mycelix-mainnet

# Check certificate validity
./scripts/check-certs.sh --min-days 30
```

### 3.2 Validator Coordination

```bash
# Send T-24 notification
./scripts/notify-validators.sh --template t-24-hours

# Verify validator readiness
./scripts/poll-validator-status.sh

# Expected output:
# Validator 1: READY
# Validator 2: READY
# ...
# Total: 25/25 READY
```

### 3.3 Contract Verification

```bash
# Verify mainnet contracts
./scripts/verify-contracts.sh --network mainnet

# Check contract parameters
./scripts/check-contract-params.sh

# Verify multi-sig controls
./scripts/verify-multisig.sh
```

### 3.4 War Room Setup

**Location**: Virtual (Zoom/Discord)
**Start Time**: T-2 hours

**Roles**:
| Role | Person | Responsibility |
|------|--------|----------------|
| Launch Commander | [NAME] | Overall coordination |
| Technical Lead | [NAME] | Engineering decisions |
| DevOps Lead | [NAME] | Infrastructure operations |
| Community Lead | [NAME] | External communication |
| Security Lead | [NAME] | Security monitoring |
| Scribe | [NAME] | Documentation |

---

## 4. T-0: Launch Sequence

### 4.1 Pre-Launch (T-30 minutes)

```bash
# Final system check
./scripts/final-check.sh

# Notify validators to prepare
./scripts/notify-validators.sh --template t-30-minutes

# Enable enhanced monitoring
kubectl apply -f monitoring/enhanced-launch.yaml
```

### 4.2 Genesis Block Creation (T-0)

**Time**: [SCHEDULED TIME] UTC

```bash
# Bootstrap node creates genesis block
./scripts/create-genesis-block.sh \
  --genesis genesis.json \
  --timestamp $(date +%s)

# Verify genesis block
./scripts/verify-genesis-block.sh

# Announce genesis block hash
./scripts/announce.sh \
  --channel validators \
  --message "Genesis block created: 0x..."
```

### 4.3 Validator Activation (T+1 minute)

```bash
# Signal validators to start
./scripts/notify-validators.sh --template start-now

# Monitor validator joins
./scripts/monitor-validator-joins.sh --min 21 --timeout 300
```

**Expected Timeline**:
| Time | Event | Target |
|------|-------|--------|
| T+0 | Genesis block | 1 block |
| T+1m | First validators | 5+ |
| T+5m | Quorum reached | 21+ |
| T+10m | Network stable | All validators |

### 4.4 Network Stabilization (T+10 minutes)

```bash
# Verify block production
./scripts/verify-blocks.sh --count 10

# Check validator participation
./scripts/check-validator-participation.sh

# Verify FL coordinator online
curl -s https://api.mainnet.mycelix.network/fl/status | jq
```

### 4.5 First FL Round (T+15 minutes)

```bash
# Initiate first FL round
./scripts/initiate-fl-round.sh --round 1

# Monitor round progress
./scripts/monitor-fl-round.sh --round 1

# Verify round completion
./scripts/verify-fl-round.sh --round 1
```

### 4.6 Public Announcement (T+30 minutes)

After verification of stable operation:

```bash
# Send public announcement
./scripts/announce.sh \
  --channel public \
  --template mainnet-live

# Update website
./scripts/update-website.sh --status live

# Enable public API access
kubectl patch ingress mainnet-api -p '{"spec":{"rules":[{"host":"api.mainnet.mycelix.network"}]}}'
```

---

## 5. Post-Launch: First 72 Hours

### 5.1 Hour 1-6: Intensive Monitoring

**Monitoring Checklist** (every 15 minutes):
- [ ] Block production rate: ~12/minute
- [ ] Validator participation: 100%
- [ ] FL round completion: successful
- [ ] Error rate: < 0.1%
- [ ] Latency: < 500ms

```bash
# Continuous monitoring
./scripts/launch-monitor.sh --duration 6h --interval 15m
```

### 5.2 Hour 6-24: Stabilization

**Reduced Monitoring** (every hour):
- [ ] Block production consistent
- [ ] No validator drops
- [ ] FL rounds completing
- [ ] No critical alerts

**Actions**:
- [ ] Review all alerts from first 6 hours
- [ ] Document any incidents
- [ ] Communicate status to community

### 5.3 Hour 24-48: Normal Operations

**Standard Monitoring**:
- Resume normal monitoring schedule
- Team can rotate to on-call

**Verification**:
- [ ] 288+ blocks produced (24h × 12/min)
- [ ] 288+ FL rounds completed
- [ ] No validator slashing events
- [ ] Community feedback positive

### 5.4 Hour 48-72: Post-Launch Review

**Post-Launch Meeting**:
- Review all incidents
- Document lessons learned
- Update runbooks as needed
- Plan any necessary hotfixes

**Deliverables**:
- [ ] Launch report document
- [ ] Incident summary
- [ ] Metrics dashboard snapshot
- [ ] Community feedback summary

### 5.5 Week 1: Stability Report

```markdown
## Week 1 Stability Report

### Network Health
- Blocks Produced: X
- FL Rounds: X
- Uptime: X%

### Validator Performance
- Active Validators: X
- Slashing Events: X
- Average Response Time: Xms

### Incidents
- Critical: X
- High: X
- Medium: X

### Community
- New Participants: X
- Support Tickets: X
- Feedback Summary: ...
```

---

## 6. Rollback Procedures

### 6.1 Decision Criteria

**Rollback if ANY of the following occur**:
- Block production halted > 10 minutes
- < 50% validators online
- Critical security vulnerability exploited
- Smart contract funds at risk
- Consensus failure

### 6.2 Partial Rollback (Soft)

**For recoverable issues**:

```bash
# Pause new FL rounds
kubectl scale deployment fl-coordinator --replicas=0

# Notify validators to stop
./scripts/notify-validators.sh --template pause

# Investigate issue
./scripts/collect-diagnostics.sh

# Fix and resume
kubectl scale deployment fl-coordinator --replicas=3
./scripts/notify-validators.sh --template resume
```

### 6.3 Full Rollback (Hard)

**For unrecoverable issues**:

```bash
# EMERGENCY: Full network halt
./scripts/emergency-halt.sh --confirm

# Notify all stakeholders
./scripts/announce.sh --channel all --template emergency-halt

# Pause smart contracts
# Requires multi-sig execution on Ethereum
./scripts/pause-contracts.sh --multisig

# Document state for forensics
./scripts/snapshot-state.sh --full

# Coordinate restart with new genesis
# ... (requires separate planning)
```

### 6.4 Smart Contract Emergency

```bash
# Pause PaymentRouter (if funds at risk)
cast send $PAYMENT_ROUTER "pause()" --private-key $MULTISIG_KEY

# Verify pause
cast call $PAYMENT_ROUTER "paused()"
# Returns: true

# Communicate to community
./scripts/announce.sh --template contract-paused
```

---

## 7. Emergency Contacts

### 7.1 Internal Team

| Role | Name | Phone | Signal | Escalation Order |
|------|------|-------|--------|------------------|
| Launch Commander | [NAME] | [PHONE] | [SIGNAL] | 1 |
| Technical Lead | [NAME] | [PHONE] | [SIGNAL] | 2 |
| Security Lead | [NAME] | [PHONE] | [SIGNAL] | 2 |
| DevOps Lead | [NAME] | [PHONE] | [SIGNAL] | 3 |
| CEO | [NAME] | [PHONE] | [SIGNAL] | 4 |

### 7.2 External Partners

| Partner | Contact | Purpose |
|---------|---------|---------|
| Cloud Provider | [CONTACT] | Infrastructure emergency |
| RPC Provider | [CONTACT] | Ethereum connectivity |
| Security Firm | [CONTACT] | Incident response |
| Legal Counsel | [CONTACT] | Regulatory issues |

### 7.3 Escalation Path

```
Level 1: On-call engineer (< 15 min response)
    ↓ (if not resolved in 30 min)
Level 2: Technical Lead + DevOps Lead
    ↓ (if critical or > 1 hour)
Level 3: Launch Commander + Security Lead
    ↓ (if funds at risk or public impact)
Level 4: CEO + Legal + External partners
```

---

## Appendices

### A. Checklists Summary

#### A.1 Go-Live Checklist (Day of Launch)

```
PRE-LAUNCH (T-2h to T-0)
[ ] All team members in war room
[ ] Final infrastructure check passed
[ ] All validators reporting ready
[ ] Communication channels verified
[ ] Monitoring dashboards open
[ ] Rollback procedures reviewed

LAUNCH (T-0 to T+30m)
[ ] Genesis block created
[ ] Genesis hash announced
[ ] Validators started
[ ] Quorum reached (21+)
[ ] Block production stable
[ ] First FL round initiated
[ ] First FL round completed
[ ] Public announcement sent

POST-LAUNCH (T+30m to T+6h)
[ ] Enhanced monitoring active
[ ] No critical alerts
[ ] Validator participation stable
[ ] FL rounds completing
[ ] Community feedback monitored
```

### B. Communication Templates

#### B.1 Mainnet Live Announcement

```
🚀 Mycelix Mainnet is LIVE!

After [X] months of development and testing, we are thrilled to announce
that the Mycelix mainnet is now operational!

Genesis Block: 0x[HASH]
Validators: [X]
Network Status: Operational

🔗 Explorer: https://explorer.mycelix.network
📊 Status: https://status.mycelix.network
📚 Docs: https://docs.mycelix.network

Thank you to our incredible community of validators, developers, and
supporters who made this possible.

The future of decentralized AI is here. 🌐🧠
```

#### B.2 Emergency Halt Announcement

```
⚠️ URGENT: Mycelix Network Temporarily Paused

We have identified an issue requiring immediate attention and have
temporarily paused the network as a precautionary measure.

Status: PAUSED
Reason: [BRIEF DESCRIPTION]
Estimated Resolution: [TIME IF KNOWN]

Your funds are safe. We are working to resolve this issue as quickly
as possible and will provide updates every [30 minutes/1 hour].

For real-time updates: https://status.mycelix.network
```

### C. Metrics Thresholds

| Metric | Warning | Critical | Action |
|--------|---------|----------|--------|
| Block Time | > 7s | > 15s | Investigate validators |
| Validator Count | < 25 | < 21 | Emergency coordination |
| FL Success Rate | < 95% | < 80% | Check aggregator |
| Error Rate | > 1% | > 5% | Investigate logs |
| API Latency | > 500ms | > 2s | Scale/investigate |
| Memory Usage | > 80% | > 95% | Scale/restart |

### D. Post-Mortem Template

```markdown
# Post-Mortem: [Incident Title]

**Date**: [DATE]
**Duration**: [START] - [END]
**Severity**: [CRITICAL/HIGH/MEDIUM/LOW]
**Author**: [NAME]

## Summary
[1-2 sentence summary of what happened]

## Impact
- Users affected: [NUMBER]
- Revenue impact: [AMOUNT]
- Reputation impact: [DESCRIPTION]

## Timeline
- [TIME] - [EVENT]
- [TIME] - [EVENT]
- ...

## Root Cause
[Description of the root cause]

## Resolution
[How the issue was resolved]

## Lessons Learned
1. [LESSON]
2. [LESSON]

## Action Items
| Action | Owner | Due Date | Status |
|--------|-------|----------|--------|
| [ACTION] | [NAME] | [DATE] | [ ] |

## Appendix
- Relevant logs
- Dashboards screenshots
- Communication records
```

---

*End of Mainnet Launch Runbook*

**Document Control**:
| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-01-18 | Operations Team | Initial creation |
