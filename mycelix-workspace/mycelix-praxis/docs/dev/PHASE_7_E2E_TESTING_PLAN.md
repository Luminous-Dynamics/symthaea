# 📋 Phase 7: End-to-End Testing Plan

**Phase**: Week 11 of v0.2.0 Implementation
**Status**: In Progress
**Started**: 2025-12-16
**Goal**: Comprehensive end-to-end validation of all system components

---

## 🎯 Objectives

1. **Validate Complete Workflows**: Test entire user journeys across all 4 zomes
2. **Verify Web Client Integration**: Ensure React client correctly interacts with conductor
3. **Test Cross-Zome Interactions**: Validate data flow between zomes
4. **Performance Benchmarking**: Measure and optimize system performance
5. **Documentation & Bug Fixes**: Address any issues discovered during testing

---

## 📊 Test Coverage Matrix

### Current Status
| Component | Unit Tests | Integration Tests | E2E Tests | Status |
|-----------|-----------|-------------------|-----------|---------|
| Learning Zome | 31/31 ✅ | 10 scaffolds | 0 | 🚧 Pending |
| FL Zome | 27/27 ✅ | 10 scaffolds | 0 | 🚧 Pending |
| Credential Zome | 31/31 ✅ | 10 scaffolds | 0 | 🚧 Pending |
| DAO Zome | 27/27 ✅ | 10 scaffolds | 0 | 🚧 Pending |
| Web Client | N/A | N/A | 0 | 🚧 Pending |
| **Total** | **116/116** | **40 scaffolds** | **0** | 🚧 **Pending** |

---

## 🧪 End-to-End Test Scenarios

### Scenario 1: Complete Learning Journey
**Workflow**: Course Creation → Enrollment → Progress → Credential Issuance

**Steps**:
1. **Course Creation**:
   - Instructor creates a new course via web client
   - Verify course appears in DHT
   - Verify course metadata is correct

2. **Learner Enrollment**:
   - Learner enrolls in course
   - Initial progress record created (0% complete)
   - Verify learner appears in course roster

3. **Learning Progress**:
   - Learner completes Module 1
   - Progress updated (25% → 50% → 75%)
   - Activity logs recorded
   - Time tracking updated

4. **Course Completion**:
   - Learner completes final module (100%)
   - Completion timestamp recorded
   - Quiz scores finalized

5. **Credential Issuance**:
   - Instructor issues W3C Verifiable Credential
   - Credential includes course completion data
   - Learner receives credential
   - Credential can be verified by third parties

**Success Criteria**:
- All data correctly stored in DHT
- Web client accurately reflects state changes
- Credential is valid and verifiable
- Total workflow time < 10 seconds

---

### Scenario 2: Federated Learning Round Lifecycle
**Workflow**: Round Creation → Participant Registration → Update Submission → Aggregation

**Steps**:
1. **Round Initialization**:
   - Coordinator creates new FL round
   - Privacy parameters set (gradient clipping, DP)
   - Minimum participants threshold defined

2. **Participant Registration**:
   - 3+ learners register for round
   - Each receives global model hash
   - Participant count updated

3. **Local Training & Submission**:
   - Each learner trains locally
   - Gradients clipped to privacy params
   - Updates submitted to DHT with commitments

4. **Aggregation**:
   - Coordinator waits for minimum participants
   - Retrieves all gradient updates
   - Performs secure aggregation (trimmed mean)
   - Publishes new global model hash

5. **Model Distribution**:
   - Participants download aggregated model
   - Round marked as complete
   - Metrics logged (convergence, participation)

**Success Criteria**:
- Privacy guarantees maintained (no raw data shared)
- Aggregation algorithm correct (matches local test)
- All participants receive updated model
- Round completes within deadline
- DHT load balanced across nodes

---

### Scenario 3: Credential Verification & Revocation
**Workflow**: Credential Issuance → External Verification → Revocation

**Steps**:
1. **Credential Issuance**:
   - Issuer creates W3C VC for learner
   - Credential signed with issuer's key
   - Stored in DHT with indexes

2. **Holder Retrieval**:
   - Learner queries their credentials
   - Receives all credentials they hold
   - Can export to wallet

3. **Verifier Validation**:
   - Third party requests credential verification
   - Signature verified against issuer DID
   - Status checked (not revoked)
   - Expiration date validated

4. **Credential Revocation**:
   - Issuer revokes credential
   - Revocation status updated in DHT
   - Verification now returns "revoked"

**Success Criteria**:
- Signature verification passes for valid credentials
- Revoked credentials fail verification
- Query performance < 500ms
- Supports W3C VC standard fully

---

### Scenario 4: DAO Governance Lifecycle
**Workflow**: Proposal Creation → Voting → Execution

**Steps**:
1. **Proposal Submission**:
   - Member creates curriculum change proposal
   - Category: Curriculum
   - Type: Normal (3-14 day voting period)
   - Actions JSON defined

2. **Voting Period**:
   - Members cast votes (For/Against/Abstain)
   - Voting power calculated
   - Vote counts updated in real-time

3. **Vote Tallying**:
   - Deadline reached
   - Final vote counts tallied
   - Quorum check performed
   - Status updated (Approved/Rejected)

4. **Proposal Execution** (if approved):
   - Actions from JSON executed
   - Curriculum updated
   - Members notified

**Test Cases**:
- Fast proposal (24-72h): Emergency parameter change
- Normal proposal (3-14 days): New course addition
- Slow proposal (14-30 days): Major protocol upgrade
- Failed quorum scenario
- Tied vote scenario
- Veto by guardian

**Success Criteria**:
- All vote types work correctly
- Quorum enforced
- Deadlines respected
- Execution only if approved
- Voting power calculated correctly

---

### Scenario 5: Web Client → Conductor Integration
**Workflow**: Browser → TypeScript Client → WebSocket → Conductor → DHT

**Steps**:
1. **Connection Establishment**:
   - React app loads
   - `useHolochainClient` hook initializes
   - WebSocket connection to conductor
   - App info retrieved
   - Connection status updates UI

2. **Real-time Operations**:
   - User creates course via UI form
   - TypeScript client calls `learning.create_course()`
   - Request sent over WebSocket
   - Zome function executes
   - Response received
   - UI updates with new course

3. **Mock → Real Client Switching**:
   - Dev: VITE_USE_MOCK_CLIENT=true
   - Test with mock data first
   - Switch: VITE_USE_REAL_CLIENT=true
   - Connect to live conductor
   - Verify same functionality

4. **Error Handling**:
   - Conductor unavailable
   - WebSocket disconnect
   - Auto-reconnect logic
   - User-friendly error messages

**Success Criteria**:
- All 29 zome functions callable from web client
- WebSocket connection stable
- Auto-reconnect works
- Mock/real switching seamless
- Response times < 1 second (p99)

---

## 🔧 Integration Test Implementation

### Test Infrastructure

```rust
// tests/e2e_test.rs
#[cfg(test)]
mod e2e_tests {
    use holochain::sweettest::{SweetConductor, SweetDnaFile};

    #[tokio::test]
    async fn test_complete_learning_journey() {
        // 1. Setup conductor
        let conductor = SweetConductor::from_standard_config().await;
        let dna = SweetDnaFile::from_bundle("./dna/praxis.dna").await?;

        // 2. Install app
        let apps = conductor
            .setup_app("mycelix-praxis", &[dna])
            .await?;

        let (alice, bobbo) = apps.into_tuples();

        // 3. Alice creates course
        let course = Course {
            course_id: "course-001".into(),
            title: "Introduction to Holochain".into(),
            description: "Learn the basics".into(),
            instructor: "alice".into(),
            created_at: timestamp(),
            difficulty_level: "Beginner".into(),
            prerequisites: vec![],
            learning_objectives: vec![
                "Understand DHT".into(),
                "Write zomes".into(),
            ],
            estimated_hours: 10,
            tags: vec!["holochain".into(), "rust".into()],
        };

        let course_hash = alice
            .zome(ZomeRef::Learning)
            .call("create_course", course.clone())
            .await?;

        // 4. Bobbo enrolls
        let progress = LearnerProgress {
            learner_id: "bobbo".into(),
            course_id: "course-001".into(),
            enrollment_date: timestamp(),
            completion_percentage: 0,
            last_activity: timestamp(),
            completed_modules: vec![],
            current_module: "Module 1".into(),
            time_spent_minutes: 0,
            quiz_scores: HashMap::new(),
        };

        bobbo
            .zome(ZomeRef::Learning)
            .call("enroll_learner", progress)
            .await?;

        // 5. Bobbo completes course
        // ... (continue test)

        // 6. Alice issues credential
        // ... (continue test)

        // 7. Verify credential
        // ... (continue test)
    }
}
```

---

## 📈 Performance Benchmarks

### Target Performance Metrics

| Operation | Target (p50) | Target (p99) | Acceptable Max |
|-----------|-------------|-------------|----------------|
| **Zome Calls** |
| create_course | 100ms | 500ms | 1s |
| enroll_learner | 100ms | 500ms | 1s |
| update_progress | 50ms | 300ms | 500ms |
| get_all_courses | 200ms | 1s | 2s |
| **FL Operations** |
| create_fl_round | 100ms | 500ms | 1s |
| submit_update | 200ms | 1s | 2s |
| aggregate_round (10 participants) | 2s | 5s | 10s |
| **Credential Operations** |
| issue_credential | 200ms | 1s | 2s |
| verify_credential | 100ms | 500ms | 1s |
| **DAO Operations** |
| create_proposal | 100ms | 500ms | 1s |
| cast_vote | 50ms | 300ms | 500ms |
| **DHT Operations** |
| Entry propagation | 1s | 3s | 5s |
| Link creation | 500ms | 2s | 3s |

### Load Testing Scenarios

1. **High Course Load**:
   - 100 courses created simultaneously
   - 1000 learners enroll
   - Monitor DHT performance

2. **FL Round at Scale**:
   - 50 participants in single round
   - Concurrent gradient submissions
   - Measure aggregation time

3. **DAO Voting Rush**:
   - 200 members vote within 1 minute
   - Verify all votes recorded
   - Check vote tally accuracy

---

## 🐛 Known Issues & Workarounds

### Current Blockers
None identified yet (starting Phase 7)

### Potential Risks
1. **WebSocket Stability**: May need connection retry logic
2. **DHT Propagation Delays**: Links may not be immediately available
3. **WASM Compilation Time**: First zome call may be slow
4. **Conductor Memory**: Large datasets may cause issues

---

## ✅ Success Criteria for Phase 7

### Must Have
- [ ] All 5 E2E scenarios pass
- [ ] Web client successfully connects to conductor
- [ ] All 29 zome functions callable from web client
- [ ] At least 10 integration tests implemented
- [ ] Performance benchmarks documented
- [ ] All discovered bugs fixed or documented

### Should Have
- [ ] Load testing completed for 100+ concurrent users
- [ ] Performance optimizations implemented where needed
- [ ] Comprehensive error handling in web client
- [ ] User documentation updated

### Nice to Have
- [ ] Video demo of complete user journey
- [ ] Performance dashboard for monitoring
- [ ] Automated E2E test suite in CI/CD

---

## 📋 Implementation Checklist

### Week 11 Tasks
- [ ] **Day 1-2**: Set up E2E test infrastructure
  - [ ] Create `tests/e2e_test.rs`
  - [ ] Configure sweettest
  - [ ] Implement Scenario 1: Learning Journey

- [ ] **Day 3**: Implement Scenarios 2-4
  - [ ] FL Round Lifecycle test
  - [ ] Credential Verification test
  - [ ] DAO Governance test

- [ ] **Day 4**: Web Client Integration Testing
  - [ ] Test WebSocket connection
  - [ ] Test all zome function calls from React
  - [ ] Test mock → real client switching

- [ ] **Day 5**: Performance Benchmarking
  - [ ] Run performance tests
  - [ ] Document metrics
  - [ ] Identify optimization opportunities

- [ ] **Day 6**: Bug Fixes & Optimization
  - [ ] Address any failures
  - [ ] Optimize slow operations
  - [ ] Improve error handling

- [ ] **Day 7**: Documentation & Completion
  - [ ] Update documentation
  - [ ] Create completion report
  - [ ] Plan Phase 8 (Release Preparation)

---

## 📚 Reference Documentation

- [Holochain Sweettest Documentation](https://docs.rs/holochain/latest/holochain/sweettest/)
- [Phase 6 Web Client Integration](./PHASE_6_WEB_CLIENT_INTEGRATION_COMPLETE.md)
- [V0.2 Implementation Plan](./V0_2_IMPLEMENTATION_PLAN.md)

---

**Next Steps**: Begin with E2E test infrastructure setup and Scenario 1 implementation.
