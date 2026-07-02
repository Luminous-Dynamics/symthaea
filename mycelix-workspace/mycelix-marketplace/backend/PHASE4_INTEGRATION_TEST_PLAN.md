# 🧪 Phase 4: Integration Testing Plan

**Date**: December 31, 2025
**Status**: ⏭️ **Ready for Execution** (after WASM build completes)
**Prerequisites**: Completed WASM build with hApp package

---

## 🎯 Testing Objectives

### Primary Goals
1. ✅ Verify all 10 zomes load correctly in conductor
2. ✅ Test all zome function calls work
3. ✅ Validate inter-zome communication (remote calls)
4. ✅ Confirm MATL (reputation) system functions
5. ✅ Test Byzantine fault tolerance mechanisms
6. ✅ Ensure data persistence and retrieval
7. ✅ Validate link creation and queries

### Success Criteria
- All zome functions callable without errors
- Inter-zome calls work correctly (e.g., listings → reputation)
- MATL gating prevents spam (score < 0.4 blocked)
- Data survives conductor restart
- No panics or crashes during normal operation

---

## 📋 Test Categories

### Category 1: Conductor Startup ✅
**Purpose**: Verify conductor and hApp installation

**Tests**:
1. **Start Conductor**
   ```bash
   holochain -c conductor-config.yaml
   ```
   - ✅ Conductor starts without errors
   - ✅ Admin interface available on :8888
   - ✅ App interface available on :8889

2. **Install hApp**
   ```bash
   hc app install ./mycelix_marketplace.happ
   ```
   - ✅ hApp installs successfully
   - ✅ All 10 zomes registered
   - ✅ DNA hash generated

3. **Generate Agent**
   - ✅ Agent key generated
   - ✅ Source chain initialized
   - ✅ App enabled

**Expected Output**:
```
Conductor started successfully
hApp installed: mycelix_marketplace
DNA: marketplace (5 integrity + 5 coordinator zomes)
Agent: <pubkey>
Status: Enabled
```

---

### Category 2: Listings Zome Tests 📦
**Purpose**: Test marketplace listing functionality

**Test 2.1: Create Listing**
```javascript
// Via conductor admin interface or test client
{
  "type": "call_zome",
  "data": {
    "cell_id": [...],
    "zome_name": "listings",
    "fn_name": "create_listing",
    "payload": {
      "title": "Test Item",
      "description": "Test description",
      "price_cents": 1000,
      "category": "Electronics",
      "tags": ["test"],
      "condition": "New",
      "shipping_included": true
    }
  }
}
```
**Expected**: ✅ Listing created, ActionHash returned

**Test 2.2: Get Listing**
- ✅ Retrieve listing by hash
- ✅ All fields match input
- ✅ Timestamps populated
- ✅ Epistemic classification set

**Test 2.3: Update Listing**
- ✅ Price update works
- ✅ Description update works
- ✅ New version created
- ✅ History preserved

**Test 2.4: Search Listings**
- ✅ Search by category
- ✅ Search by tags
- ✅ Filter by price range
- ✅ Sort by creation date

**Test 2.5: Delete Listing**
- ✅ Mark as deleted
- ✅ No longer appears in searches
- ✅ Historical data preserved

---

### Category 3: Reputation Zome Tests ⭐
**Purpose**: Test MATL (Mycelix Adaptive Trust Layer)

**Test 3.1: Initialize Agent Reputation**
```javascript
{
  "fn_name": "get_agent_matl_score",
  "payload": <agent_pubkey>
}
```
**Expected**: ✅ New agent score = 0.5 (default)

**Test 3.2: Update MATL Score**
- ✅ Successful transaction increases score
- ✅ Failed transaction decreases score
- ✅ Score bounded between 0.0 and 1.0
- ✅ Updates persist

**Test 3.3: MATL Gating**
```javascript
// Agent with score 0.3 tries to create listing
{
  "fn_name": "create_listing",
  "payload": {...}
}
```
**Expected**: ❌ Error: "Insufficient MATL score (0.30 < 0.40)"

**Test 3.4: Trust Score Fast Cache**
- ✅ First call queries DHT (~100ms)
- ✅ Subsequent calls use cache (<1ms)
- ✅ Cache expires after 5 minutes
- ✅ Cache updates on score change

**Test 3.5: Reputation History**
- ✅ Query agent's full history
- ✅ See all transactions
- ✅ View score progression
- ✅ Filter by date range

---

### Category 4: Transactions Zome Tests 💰
**Purpose**: Test escrow and payment flow

**Test 4.1: Initiate Transaction**
```javascript
{
  "fn_name": "initiate_transaction",
  "payload": {
    "listing_hash": <hash>,
    "buyer": <buyer_pubkey>,
    "seller": <seller_pubkey>
  }
}
```
**Expected**:
- ✅ Transaction created
- ✅ Status: Pending
- ✅ Escrow amount calculated

**Test 4.2: Confirm Transaction**
- ✅ Seller confirms
- ✅ Status → InProgress
- ✅ Timestamps updated

**Test 4.3: Complete Transaction**
- ✅ Buyer receives item
- ✅ Status → Completed
- ✅ MATL scores updated (+0.1 each)
- ✅ Remote call to reputation zome works

**Test 4.4: Cancel Transaction**
- ✅ Either party can cancel before confirmation
- ✅ Funds returned
- ✅ MATL scores unchanged

**Test 4.5: Transaction Dispute**
- ✅ Buyer or seller files dispute
- ✅ Creates arbitration entry
- ✅ Links to arbitration zome
- ✅ Transaction frozen

---

### Category 5: Arbitration Zome Tests ⚖️
**Purpose**: Test dispute resolution via MRC (Mutual Reputation Consensus)

**Test 5.1: File Dispute**
```javascript
{
  "fn_name": "file_dispute",
  "payload": {
    "transaction_hash": <hash>,
    "reason": "Item not as described",
    "evidence_cids": ["Qm..."]
  }
}
```
**Expected**:
- ✅ Dispute created
- ✅ Status: Filed
- ✅ Arbitrators assigned (requires high MATL agents in network)

**Test 5.2: Arbitrator Assignment**
- ✅ Queries network for agents with MATL > 0.7
- ✅ Excludes buyer, seller, filer
- ✅ Selects 3-5 arbitrators
- ✅ Creates links for arbitration opportunities

**Test 5.3: Submit Arbitration Vote**
```javascript
{
  "fn_name": "submit_arbitration_vote",
  "payload": {
    "dispute_hash": <hash>,
    "favor_buyer": true,
    "reasoning": "Evidence supports buyer claim"
  }
}
```
**Expected**:
- ✅ Vote recorded
- ✅ MATL score attached
- ✅ Status updates when all votes in

**Test 5.4: Finalize Arbitration**
- ✅ Weighted vote calculated correctly
- ✅ >66% threshold determines winner
- ✅ Result recorded
- ✅ MATL scores updated (loser -0.15)

**Test 5.5: MRC Algorithm Validation**
```
Given votes:
  Arbitrator A (MATL 0.8): favor_buyer = true
  Arbitrator B (MATL 0.9): favor_buyer = true
  Arbitrator C (MATL 0.7): favor_buyer = false

Weighted calculation:
  (1.0 * 0.8 + 1.0 * 0.9 + 0.0 * 0.7) / (0.8 + 0.9 + 0.7)
  = 1.7 / 2.4 = 0.708 > 0.66
  → Buyer wins
```
**Expected**: ✅ Calculation matches, buyer declared winner

---

### Category 6: Messaging Zome Tests 💬
**Purpose**: Test encrypted P2P messaging

**Test 6.1: Start Conversation**
```javascript
{
  "fn_name": "start_conversation",
  "payload": {
    "recipient": <agent_pubkey>,
    "subject": "About listing XYZ",
    "first_message_content": "Is this still available?",
    "listing_hash": <hash>
  }
}
```
**Expected**:
- ✅ Conversation created
- ✅ First message sent
- ✅ Links to listing
- ✅ MATL gating applied (score > 0.4)

**Test 6.2: Send Message**
- ✅ Message delivered
- ✅ Timestamps accurate
- ✅ Content encrypted (client-side)
- ✅ Conversation updated

**Test 6.3: Mark Message Read**
- ✅ Read receipt created
- ✅ Unread count decremented
- ✅ Timestamp recorded

**Test 6.4: Get Conversations**
- ✅ Lists all conversations
- ✅ Sorted by last activity
- ✅ Shows unread counts
- ✅ Filters by status

**Test 6.5: MATL Spam Prevention**
```javascript
// Agent with MATL 0.3 tries to send message
{
  "fn_name": "send_message",
  "payload": {...}
}
```
**Expected**: ❌ Error: "Insufficient MATL score for messaging"

**Test 6.6: Conversation Search**
- ✅ Search by participant
- ✅ Filter by listing
- ✅ Filter by transaction
- ✅ Subject keyword search

---

### Category 7: Inter-Zome Communication Tests 🔗
**Purpose**: Validate remote zome calls work correctly

**Test 7.1: Listings → Reputation**
```
create_listing() internally calls:
  remote_calls::call_zome("reputation", "get_agent_matl_score", seller)
```
**Expected**: ✅ MATL score retrieved, used for validation

**Test 7.2: Transactions → Reputation**
```
complete_transaction() calls:
  remote_calls::call_zome_void("reputation", "update_matl_score", ...)
```
**Expected**: ✅ Both agents' scores updated

**Test 7.3: Arbitration → Transactions**
```
file_dispute() calls:
  remote_calls::call_zome("transactions", "get_transaction", hash)
```
**Expected**: ✅ Transaction details retrieved

**Test 7.4: Messaging → Reputation**
```
send_message() calls:
  remote_calls::call_zome("reputation", "get_agent_matl_score_fast", sender)
```
**Expected**: ✅ Fast cache used, spam prevention works

**Test 7.5: Remote Call Error Handling**
- ✅ Invalid zome name returns error
- ✅ Invalid function name returns error
- ✅ Type mismatch returns clear error
- ✅ Caller handles errors gracefully

---

### Category 8: Byzantine Fault Tolerance Tests 🛡️
**Purpose**: Verify 45% Byzantine fault tolerance mechanisms

**Test 8.1: Invalid Entry Detection**
- ✅ Submit malformed listing entry
- ✅ Validation rejects entry
- ✅ Entry not added to DHT

**Test 8.2: MATL Score Manipulation Attempt**
- ✅ Try to directly update own MATL score
- ✅ Validation rejects unauthorized update
- ✅ Only transaction completions affect score

**Test 8.3: Fake Transaction Creation**
- ✅ Try to create transaction without listing
- ✅ Validation checks listing exists
- ✅ Transaction rejected

**Test 8.4: Arbitration Vote Fraud**
- ✅ Try to vote when not assigned arbitrator
- ✅ Validation checks arbitrator list
- ✅ Vote rejected

**Test 8.5: Double-Spend Prevention**
- ✅ Try to use listing in multiple active transactions
- ✅ Validation detects conflict
- ✅ Second transaction rejected

---

### Category 9: Data Persistence Tests 💾
**Purpose**: Ensure data survives conductor restarts

**Test 9.1: Create and Restart**
1. Create listings, transactions, messages
2. Stop conductor
3. Restart conductor
4. Query all data

**Expected**: ✅ All data intact, retrievable

**Test 9.2: Update and Restart**
1. Update existing entries
2. Stop conductor
3. Restart
4. Verify latest versions

**Expected**: ✅ Updates preserved, history intact

**Test 9.3: Source Chain Integrity**
- ✅ Source chain unbroken
- ✅ All actions present
- ✅ Hashes validate

---

### Category 10: Performance Tests ⚡
**Purpose**: Validate acceptable performance

**Test 10.1: Listing Creation**
- ✅ Create 100 listings
- ✅ Measure time per listing
- **Target**: <500ms average

**Test 10.2: Search Performance**
- ✅ Search with 1000 listings
- ✅ Measure query time
- **Target**: <2 seconds

**Test 10.3: MATL Score Lookup**
- ✅ First lookup (DHT query)
- **Target**: <100ms
- ✅ Cached lookup
- **Target**: <1ms

**Test 10.4: Transaction Flow**
- ✅ Complete transaction end-to-end
- **Target**: <3 seconds total

**Test 10.5: Concurrent Operations**
- ✅ 10 agents creating listings simultaneously
- ✅ No deadlocks
- ✅ All succeed

---

## 🔧 Testing Tools & Scripts

### Manual Testing via hc CLI
```bash
# Install and enable hApp
hc app install ./mycelix_marketplace.happ
hc app enable mycelix_marketplace

# Call zome functions
hc call listings create_listing '{"title": "Test", ...}'
hc call reputation get_agent_matl_score '<agent_pubkey>'
```

### Automated Testing (Future)
Create `/backend/tests/integration_tests.rs`:
```rust
#[cfg(test)]
mod integration_tests {
    use holochain::conductor::ConductorHandle;

    #[tokio::test]
    async fn test_create_listing() {
        // Setup conductor
        // Call zome
        // Assert results
    }
}
```

### Monitoring Script
Create `/backend/test-conductor.sh`:
```bash
#!/usr/bin/env bash
# Automated conductor testing script

set -e

echo "🧪 Mycelix Marketplace Integration Tests"
echo ""

# Start conductor in background
holochain -c conductor-config.yaml &
CONDUCTOR_PID=$!

sleep 5

# Install hApp
hc app install ./mycelix_marketplace.happ

# Run tests
echo "Running integration tests..."

# Test listings
echo "✓ Testing listings zome..."
# ... test commands

# Test reputation
echo "✓ Testing reputation zome..."
# ... test commands

# Cleanup
kill $CONDUCTOR_PID
```

---

## 📊 Test Results Documentation

### Test Report Template
For each test run, document:

```markdown
## Test Run: YYYY-MM-DD HH:MM

### Environment
- Holochain Version: X.X.X
- hApp Version: X.X.X
- Conductor Config: conductor-config.yaml

### Results Summary
- Tests Run: X
- Tests Passed: X
- Tests Failed: X
- Pass Rate: XX%

### Failed Tests
1. Test Name
   - Error: ...
   - Expected: ...
   - Actual: ...
   - Fix: ...

### Performance Metrics
- Avg listing creation: Xms
- Avg search query: Xms
- MATL lookup (cached): Xms

### Notes
- ...
```

---

## 🚀 Next Steps After Testing

### If All Tests Pass ✅
1. Document final test results
2. Create deployment guide
3. Prepare for mainnet deployment
4. Write user documentation
5. Plan Phase 5 enhancements

### If Tests Fail ❌
1. Document failures clearly
2. Create bug reports with reproduction steps
3. Fix issues systematically
4. Re-test after fixes
5. Update code and documentation

---

## 🎯 Success Metrics

**Phase 4 Complete When**:
- ✅ All 10 zomes load and run
- ✅ Inter-zome communication verified
- ✅ MATL system functioning correctly
- ✅ Data persistence confirmed
- ✅ Byzantine fault tolerance validated
- ✅ Performance targets met
- ✅ No critical bugs found
- ✅ Test results documented

**Quality Gate**:
- Minimum 95% test pass rate
- Zero critical bugs
- All core workflows functional
- Performance within targets

---

## 💡 Testing Best Practices

1. **Test Incrementally**: Start with simple tests, build complexity
2. **Document Everything**: Record all results, errors, observations
3. **Reproduce Failures**: Ensure bugs are reproducible before fixing
4. **Test Edge Cases**: Don't just test happy path
5. **Monitor Resources**: Watch memory, CPU, disk usage
6. **Clean State**: Reset conductor between test runs if needed
7. **Version Control**: Tag tested versions of hApp

---

**Testing Framework**: Manual → Semi-Automated → Fully Automated
**Current Phase**: Manual testing with scripts
**Next Evolution**: Rust integration tests + CI/CD

**Ready to begin testing once WASM build completes!** 🧪✨
