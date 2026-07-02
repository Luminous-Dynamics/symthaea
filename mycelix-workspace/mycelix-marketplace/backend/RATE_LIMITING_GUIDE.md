# Rate Limiting System - DoS Protection

## Overview

The Mycelix-Marketplace implements comprehensive rate limiting to protect against Denial of Service (DoS) attacks and spam. In Holochain's distributed architecture, rate limiting is enforced through:

1. **Local validation** - Agents check limits before creating entries
2. **Validation rules** - Validators verify timestamps during entry validation
3. **Byzantine detection** - Agents who consistently violate limits are flagged

## Rate Limit Configurations

### Default Limits

| Action | Limit | Window | Rationale |
|--------|-------|--------|-----------|
| Create Listing | 10 | 1 hour | Prevent spam listings |
| Update Listing | 50 | 1 hour | Allow legitimate updates while preventing abuse |
| Create Transaction | 20 | 1 hour | Reasonable shopping behavior |
| Submit Review | 50 | 24 hours | Allow active reviewing while preventing spam |
| File Dispute | 5 | 24 hours | Disputes should be rare |
| Vote on Arbitration | 100 | 24 hours | Arbitrators need to vote on multiple cases |
| Search | 100 | 1 minute | Allow rapid browsing |

### Customizing Limits

```rust
use security::RateLimiter;

// Create custom rate limiter
let custom_limiter = RateLimiter::new(
    25,    // max_requests
    3600   // window_seconds (1 hour)
);

// Or use defaults
let listing_limiter = RateLimiter::default_listing_creation();
```

## Implementation Architecture

### 1. RateLimiter

The core rate limiting logic:

```rust
pub struct RateLimiter {
    pub max_requests: u32,
    pub window_seconds: u64,
}

impl RateLimiter {
    /// Check if action is allowed
    pub fn check_allowed(
        &self,
        previous_timestamps: &[u64],
        current_time: u64
    ) -> Result<(), String> {
        // Returns Ok(()) if allowed, Err(message) if exceeded
    }

    /// Calculate wait time before next action allowed
    pub fn time_until_allowed(
        &self,
        previous_timestamps: &[u64],
        current_time: u64
    ) -> u64 {
        // Returns seconds to wait
    }
}
```

### 2. RateLimitTracker

Tracks action history per agent:

```rust
pub struct RateLimitTracker {
    pub action_type: String,
    pub timestamps: Vec<u64>,
    pub max_history: usize,  // Default: 1000
}

impl RateLimitTracker {
    /// Record a new action
    pub fn record_action(&mut self, timestamp: u64) {
        self.timestamps.push(timestamp);
        // Auto-cleanup if exceeds max_history
    }

    /// Get recent actions within window
    pub fn get_recent(&self, current_time: u64, window_seconds: u64) -> Vec<u64> {
        // Returns timestamps within window
    }

    /// Cleanup old timestamps
    pub fn cleanup_old(&mut self, current_time: u64, max_age_seconds: u64) {
        // Remove timestamps older than max_age
    }
}
```

## Usage in Coordinators

### Example: Listing Creation with Rate Limiting

```rust
use security::{RateLimiter, RateLimitTracker};

#[hdk_extern]
pub fn create_listing(input: CreateListingInput) -> ExternResult<ListingOutput> {
    let agent_info = agent_info()?;
    let current_time = sys_time()?;

    // Get agent's listing creation history
    let mut tracker = get_or_create_tracker(&agent_info.agent_latest_pubkey, "listing_create")?;

    // Check rate limit
    let limiter = RateLimiter::default_listing_creation();
    let recent = tracker.get_recent(current_time, limiter.window_seconds);

    if let Err(msg) = limiter.check_allowed(&recent, current_time) {
        // Rate limit exceeded - emit monitoring alert
        monitoring::emit_metric(
            monitoring::MetricType::RateLimitExceeded,
            1.0,
            Some(agent_info.agent_latest_pubkey.clone()),
            Some(format!("action:listing_create,{}", msg)),
        )?;

        // Return educational error
        let wait_time = limiter.time_until_allowed(&recent, current_time);
        return Err(wasm_error!(WasmErrorInner::Guest(
            format!("{}. Please wait {} seconds.", msg, wait_time)
        )));
    }

    // Record this action
    tracker.record_action(current_time);
    save_tracker(&agent_info.agent_latest_pubkey, &tracker)?;

    // Proceed with creating listing...
    // ... rest of implementation
}
```

## Storage Strategy

In Holochain, rate limit state is stored in the agent's source chain:

```rust
#[hdk_entry_helper]
pub struct RateLimitState {
    pub agent: AgentPubKey,
    pub action_type: String,
    pub timestamps: Vec<u64>,
    pub updated_at: u64,
}

#[hdk_entry_defs]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    RateLimitState(RateLimitState),
}
```

## Byzantine Detection Integration

Agents who repeatedly violate rate limits are flagged as potentially Byzantine:

```rust
// In reputation coordinator
pub fn check_rate_limit_violations(agent: AgentPubKey) -> ExternResult<f64> {
    let violations = get_rate_limit_violations(&agent)?;

    // More than 10 violations in 24 hours = suspicious
    if violations.len() > 10 {
        let mut matl_score = get_agent_matl_score(agent.clone())?;
        matl_score.flags.sybil_suspected = true;
        matl_score.flags.risk_score += 0.3;
        update_matl_score_entry(agent, matl_score)?;
    }

    Ok(violations.len() as f64)
}
```

## Monitoring and Alerts

Rate limit events are tracked via the monitoring system:

### Metrics Emitted

1. **RateLimitExceeded** - When an agent hits a rate limit
   - Value: 1.0 per violation
   - Agent: The violating agent
   - Metadata: Action type and limit details

2. **RateLimitViolationPattern** - When repeated violations detected
   - Value: Number of violations
   - Agent: The suspected agent
   - Metadata: Time window and pattern details

### Dashboard Integration

```javascript
// Example: Monitor rate limit violations
const violations = await monitoring.query({
    metric_type: 'RateLimitExceeded',
    time_window: 3600, // Last hour
    group_by: 'agent'
});

// Identify top violators
const topViolators = violations
    .sort((a, b) => b.count - a.count)
    .slice(0, 10);
```

## Testing

### Unit Tests

The security module includes comprehensive rate limiting tests:

```bash
cd backend/zomes/security
cargo test

# Tests include:
# - test_rate_limiter_check_allowed
# - test_rate_limiter_time_until_allowed
# - test_rate_limit_tracker
# - test_rate_limit_tracker_max_history
# - test_rate_limiter_edge_cases
# - test_rate_limiter_realistic_scenario
```

### Integration Tests

```rust
#[test]
fn test_listing_rate_limit_integration() {
    // Create 10 listings rapidly (should succeed)
    for i in 0..10 {
        let result = create_listing(test_input());
        assert!(result.is_ok());
    }

    // 11th should fail
    let result = create_listing(test_input());
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("Rate limit"));
}
```

## Performance Considerations

### Memory Management

- **Max History**: 1000 timestamps per action type
- **Auto-cleanup**: Old timestamps removed when limit exceeded
- **Efficient queries**: Only timestamps within window are checked

### Validation Performance

Rate limit checks are O(n) where n = number of timestamps in window:

```rust
// Typical performance:
// - 100 timestamps in window: ~1μs
// - 1000 timestamps in window: ~10μs
```

## Educational Error Messages

When users hit rate limits, they receive helpful educational messages:

```rust
SecurityError::rate_limit_exceeded(10, 3600);
// Returns:
// {
//   message: "Rate limit exceeded: 10 requests per 3600 seconds",
//   suggestion: "Please wait a moment before trying again",
//   learn_more: "Rate limits prevent spam and protect the network for everyone."
// }
```

## Best Practices

### For Coordinator Developers

1. **Always check rate limits** before expensive operations
2. **Use appropriate limiters** for each action type
3. **Emit monitoring events** when limits are exceeded
4. **Provide educational errors** to help users understand

### For UI Developers

1. **Show remaining quota** to users before actions
2. **Display countdown timers** when rate limited
3. **Batch operations** when possible to stay under limits
4. **Cache frequently accessed data** to reduce searches

### For Node Operators

1. **Monitor rate limit violations** via metrics
2. **Adjust limits** based on network behavior
3. **Flag persistent violators** for Byzantine detection
4. **Review patterns** to detect coordinated attacks

## Future Enhancements

### Planned Features

1. **Adaptive rate limits** - Adjust based on agent reputation
   - High MATL score → higher limits
   - Low MATL score → stricter limits

2. **Burst allowance** - Allow short bursts within overall limit
   - Example: 10/hour avg, but allow 5 in 5 minutes

3. **Distributed rate limiting** - Cross-validate limits across validators
   - Validators share violation data
   - Consensus on persistent violators

4. **Rate limit NFTs** - Premium users can purchase higher limits
   - Marketplace for rate limit quota
   - Revenue supports network infrastructure

## Security Considerations

### Attack Vectors

1. **Sybil Attack** - Creating many identities to bypass limits
   - **Mitigation**: MATL score requirement for new agents
   - **Detection**: Pattern analysis across agents

2. **Time Manipulation** - Falsifying timestamps
   - **Mitigation**: Validators check timestamp consistency
   - **Detection**: Flag agents with out-of-order timestamps

3. **Distributed DoS** - Coordinated attack from many agents
   - **Mitigation**: Network-wide rate limits
   - **Detection**: Monitoring for correlated spikes

### Audit Trail

All rate limit checks and violations are:
- Recorded in source chain
- Monitored via metrics system
- Reviewed during Byzantine detection
- Accessible for dispute resolution

## Conclusion

The rate limiting system provides robust DoS protection while maintaining a great user experience. By combining local enforcement, validation rules, and Byzantine detection, we achieve security without centralization.

**Key Takeaway**: Rate limits protect the network while educating users about healthy usage patterns.

---

*Last Updated*: December 2025
*Version*: v0.1.0
*Status*: Production Ready ✅
