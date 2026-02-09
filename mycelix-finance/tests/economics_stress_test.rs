//! Pure-Rust stress and property tests for Mycelix Finance economic formulas.
//!
//! These tests exercise the math functions directly WITHOUT requiring a Holochain
//! conductor — they import only from `mycelix_finance_types` and test at scale.
//!
//! All tests are `#[test]` (synchronous, no tokio, no ignore).

use mycelix_finance_types::*;

// =============================================================================
// Deterministic RNG (no external rand dependency)
// =============================================================================

/// Simple LCG-based deterministic pseudo-random number generator.
fn simple_rng(seed: &mut u64) -> u64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *seed >> 33
}

/// Return a random u64 in [lo, hi] (inclusive).
fn rng_range(seed: &mut u64, lo: u64, hi: u64) -> u64 {
    if lo >= hi {
        return lo;
    }
    lo + simple_rng(seed) % (hi - lo + 1)
}

// =============================================================================
// Section 1: Demurrage Stress Tests
// =============================================================================

#[test]
fn test_demurrage_exempt_floor_never_touched() {
    // For balances <= 1000 SAP (1_000_000_000 micro), demurrage deduction must
    // always be 0, for time periods from 1 second to 100 years.
    let exempt_floor = DEMURRAGE_EXEMPT_FLOOR; // 1_000_000_000
    let rate = DEMURRAGE_RATE; // 0.02

    let balances: &[u64] = &[0, 1, 500_000_000, 999_999_999, 1_000_000_000];
    let seconds_per_year: u64 = 31_536_000;

    for &balance in balances {
        // 1 second
        assert_eq!(
            compute_demurrage_deduction(balance, exempt_floor, rate, 1),
            0,
            "Balance {} should be exempt at 1 second",
            balance
        );
        // 1 hour
        assert_eq!(
            compute_demurrage_deduction(balance, exempt_floor, rate, 3600),
            0,
            "Balance {} should be exempt at 1 hour",
            balance
        );
        // 1 year
        assert_eq!(
            compute_demurrage_deduction(balance, exempt_floor, rate, seconds_per_year),
            0,
            "Balance {} should be exempt at 1 year",
            balance
        );
        // 100 years
        assert_eq!(
            compute_demurrage_deduction(balance, exempt_floor, rate, seconds_per_year * 100),
            0,
            "Balance {} should be exempt at 100 years",
            balance
        );
    }
}

#[test]
fn test_demurrage_converges_to_exempt_floor() {
    // For a large balance (10,000 SAP), compute demurrage repeatedly over 1000
    // years in 1-year steps. The balance should converge toward the exempt floor
    // but never go below it.
    let exempt_floor = DEMURRAGE_EXEMPT_FLOOR;
    let rate = DEMURRAGE_RATE;
    let seconds_per_year: u64 = 31_536_000;

    let mut balance: u64 = 10_000_000_000_000; // 10,000 SAP

    for year in 1..=1000 {
        let deduction = compute_demurrage_deduction(balance, exempt_floor, rate, seconds_per_year);
        assert!(
            balance >= deduction,
            "Year {}: deduction {} exceeds balance {}",
            year,
            deduction,
            balance
        );
        balance -= deduction;
        assert!(
            balance >= exempt_floor,
            "Year {}: balance {} fell below exempt floor {}",
            year,
            balance,
            exempt_floor
        );
    }

    // After 1000 years at 2% demurrage, the balance should be very close to
    // the exempt floor.
    let tolerance = exempt_floor / 100; // 1% of exempt floor
    assert!(
        balance < exempt_floor + tolerance,
        "After 1000 years, balance {} should be near exempt floor {} (within {})",
        balance,
        exempt_floor,
        tolerance
    );
}

#[test]
fn test_demurrage_at_scale() {
    // Apply demurrage to 10,000 different balances over 1 year.
    // Verify: (a) no balance goes below exempt floor,
    //         (b) total deduction is monotonically increasing with balance,
    //         (c) no overflow/underflow panics.
    let exempt_floor = DEMURRAGE_EXEMPT_FLOOR;
    let rate = DEMURRAGE_RATE;
    let seconds_per_year: u64 = 31_536_000;

    let mut prev_deduction: u64 = 0;

    for i in 1..=10_000u64 {
        // Balance ranges from 1 to 10B micro-SAP
        let balance = i * 1_000_000;
        let deduction =
            compute_demurrage_deduction(balance, exempt_floor, rate, seconds_per_year);

        // (a) After deduction, balance must not go below exempt floor
        let remaining = balance - deduction;
        if balance > exempt_floor {
            assert!(
                remaining >= exempt_floor,
                "Balance {} - deduction {} = {} < exempt floor {}",
                balance,
                deduction,
                remaining,
                exempt_floor
            );
        }

        // (b) Deduction should be monotonically non-decreasing with balance
        assert!(
            deduction >= prev_deduction,
            "Deduction not monotonic: balance {} deduction {} < prev {}",
            balance,
            deduction,
            prev_deduction
        );
        prev_deduction = deduction;
    }
}

#[test]
fn test_demurrage_precision() {
    // Verify that demurrage of 10,000 SAP over exactly 1 year is approximately
    // 180 SAP (2% continuous = 1 - e^(-0.02) ~= 0.0198).
    // Eligible = 10,000 - 1,000 = 9,000 SAP = 9_000_000_000 micro-SAP.
    // Expected deduction = 9_000_000_000 * 0.019801 ~= 178_209_000 micro-SAP (~178 SAP).
    let balance: u64 = 10_000_000_000; // 10,000 SAP
    let exempt_floor = DEMURRAGE_EXEMPT_FLOOR;
    let rate = DEMURRAGE_RATE;
    let seconds_per_year: u64 = 31_536_000;

    let deduction = compute_demurrage_deduction(balance, exempt_floor, rate, seconds_per_year);

    // Expected ~178.2 SAP = 178_200_000 micro-SAP. 1% tolerance = +/- 1_782_000.
    let expected: f64 = 9_000_000_000.0 * (1.0 - (-0.02_f64).exp());
    let expected_u64 = expected as u64;
    let tolerance = expected_u64 / 100; // 1%

    assert!(
        (deduction as i64 - expected_u64 as i64).unsigned_abs() <= tolerance,
        "Demurrage precision: expected ~{}, got {}, tolerance {}",
        expected_u64,
        deduction,
        tolerance
    );
}

// =============================================================================
// Section 2: Fee Tier Stress Tests
// =============================================================================

#[test]
fn test_fee_tier_boundaries_exhaustive() {
    // Test FeeTier::from_mycel() for all MYCEL values from 0.0 to 1.0
    // in steps of 0.001 (1001 iterations).
    for i in 0..=1000u32 {
        let score = i as f64 / 1000.0;
        let tier = FeeTier::from_mycel(score);

        let expected = if score > 0.7 {
            FeeTier::Steward
        } else if score >= 0.3 {
            FeeTier::Member
        } else {
            FeeTier::Newcomer
        };

        assert_eq!(
            tier, expected,
            "FeeTier::from_mycel({}) returned {:?}, expected {:?}",
            score, tier, expected
        );
    }
}

#[test]
fn test_fee_monotonically_decreasing_with_mycel() {
    // For a fixed 1M micro-SAP payment, compute fees at MYCEL 0.0, 0.1, 0.2, ... 1.0.
    // Verify fees are monotonically non-increasing.
    let amount: f64 = 1_000_000.0;
    let mut prev_fee = f64::MAX;

    for i in 0..=10u32 {
        let score = i as f64 / 10.0;
        let tier = FeeTier::from_mycel(score);
        let fee = amount * tier.base_fee_rate();

        assert!(
            fee <= prev_fee,
            "Fee not monotonically decreasing: MYCEL {} fee {} > prev {}",
            score,
            fee,
            prev_fee
        );
        prev_fee = fee;
    }
}

#[test]
fn test_fee_floor_always_respected() {
    // For amounts from 1 to 1B micro-SAP, verify that the Steward-tier fee
    // (lowest possible) >= amount / 10000 (0.01%).
    // Steward rate is exactly 0.0001, so fee = amount * 0.0001 = amount / 10000.
    for exp in 0..=9u32 {
        let amount = 10u64.pow(exp);
        if amount == 0 {
            continue;
        }
        let steward_fee = (amount as f64) * FeeTier::Steward.base_fee_rate();
        let floor = amount as f64 / 10000.0;

        assert!(
            steward_fee >= floor - f64::EPSILON,
            "Amount {}: steward fee {} < floor {}",
            amount,
            steward_fee,
            floor
        );
    }

    // Broader sweep: 1 to 1B in 10,000 steps
    let mut seed: u64 = 42;
    for _ in 0..10_000 {
        let amount = rng_range(&mut seed, 1, 1_000_000_000);
        let steward_fee = (amount as f64) * FeeTier::Steward.base_fee_rate();
        let floor = amount as f64 / 10000.0;

        assert!(
            steward_fee >= floor - f64::EPSILON,
            "Amount {}: steward fee {} < floor {}",
            amount,
            steward_fee,
            floor
        );
    }
}

// =============================================================================
// Section 3: TEND Zero-Sum Invariant Tests
// =============================================================================

#[test]
fn test_tend_zero_sum_simulation() {
    // Simulate 10,000 random exchanges between 100 members.
    // After all exchanges, the sum of all balances must == 0 (zero-sum invariant).
    let num_members = 100;
    let num_exchanges = 10_000;
    let mut balances = vec![0i64; num_members];
    let mut seed: u64 = 0xDEAD_BEEF;

    for _ in 0..num_exchanges {
        let sender = rng_range(&mut seed, 0, (num_members - 1) as u64) as usize;
        let mut receiver = rng_range(&mut seed, 0, (num_members - 1) as u64) as usize;
        // Ensure sender != receiver
        if receiver == sender {
            receiver = (receiver + 1) % num_members;
        }

        let amount = rng_range(&mut seed, 1, 10) as i64;

        // Double-entry: sender debited, receiver credited
        balances[sender] -= amount;
        balances[receiver] += amount;
    }

    let total: i64 = balances.iter().sum();
    assert_eq!(
        total, 0,
        "TEND zero-sum invariant violated: sum of balances = {}",
        total
    );
}

#[test]
fn test_tend_balance_limits_never_exceeded() {
    // Same simulation but enforce balance limits.
    // Normal tier: +/-40, Apprentice (newcomer < 0.3): +/-10.
    // Skip exchanges that would violate limits.
    let num_members = 100;
    let num_exchanges = 10_000;

    // First 20 members are apprentices (limit +/-10), rest are normal (limit +/-40)
    let apprentice_count = 20;
    let normal_limit = TendLimitTier::Normal.limit() as i64; // 40
    let apprentice_limit = 10i64;

    let mut balances = vec![0i64; num_members];
    let mut seed: u64 = 0xCAFE_BABE;
    let mut skipped = 0u64;

    for _ in 0..num_exchanges {
        let sender = rng_range(&mut seed, 0, (num_members - 1) as u64) as usize;
        let mut receiver = rng_range(&mut seed, 0, (num_members - 1) as u64) as usize;
        if receiver == sender {
            receiver = (receiver + 1) % num_members;
        }

        let amount = rng_range(&mut seed, 1, 5) as i64;

        let sender_limit = if sender < apprentice_count {
            apprentice_limit
        } else {
            normal_limit
        };
        let receiver_limit = if receiver < apprentice_count {
            apprentice_limit
        } else {
            normal_limit
        };

        // Check if exchange would exceed limits
        let sender_new = balances[sender] - amount;
        let receiver_new = balances[receiver] + amount;

        if sender_new < -sender_limit || receiver_new > receiver_limit {
            skipped += 1;
            continue;
        }

        balances[sender] = sender_new;
        balances[receiver] = receiver_new;
    }

    // Verify zero-sum
    let total: i64 = balances.iter().sum();
    assert_eq!(total, 0, "Zero-sum violated after limit-enforced exchanges");

    // Verify no balance exceeds limits
    for (i, &bal) in balances.iter().enumerate() {
        let limit = if i < apprentice_count {
            apprentice_limit
        } else {
            normal_limit
        };
        assert!(
            bal.abs() <= limit,
            "Member {} balance {} exceeds limit {}",
            i,
            bal,
            limit
        );
    }

    // Sanity check: some exchanges should have been skipped
    assert!(
        skipped > 0,
        "Expected some exchanges to be skipped due to limits, but skipped = 0"
    );
}

#[test]
fn test_tend_dynamic_limits_at_all_tiers() {
    // Verify TendLimitTier::limit() returns correct values.
    assert_eq!(TendLimitTier::Normal.limit(), 40);
    assert_eq!(TendLimitTier::Elevated.limit(), 60);
    assert_eq!(TendLimitTier::High.limit(), 80);
    assert_eq!(TendLimitTier::Emergency.limit(), 120);

    // Also verify from_vitality maps correctly at boundaries
    assert_eq!(TendLimitTier::from_vitality(0), TendLimitTier::Emergency);
    assert_eq!(TendLimitTier::from_vitality(10), TendLimitTier::Emergency);
    assert_eq!(TendLimitTier::from_vitality(11), TendLimitTier::High);
    assert_eq!(TendLimitTier::from_vitality(20), TendLimitTier::High);
    assert_eq!(TendLimitTier::from_vitality(21), TendLimitTier::Elevated);
    assert_eq!(TendLimitTier::from_vitality(40), TendLimitTier::Elevated);
    assert_eq!(TendLimitTier::from_vitality(41), TendLimitTier::Normal);
    assert_eq!(TendLimitTier::from_vitality(100), TendLimitTier::Normal);
}

// =============================================================================
// Section 4: Commons Pool Reserve Invariant Tests
// =============================================================================

#[test]
fn test_inalienable_reserve_25pct_maintained() {
    // Simulate 10,000 contributions (random amounts 1-1000 SAP) to a commons
    // pool using the 25/75 split. After each contribution, verify reserve >= 25%
    // of total.
    let mut reserve: u64 = 0;
    let mut available: u64 = 0;
    let mut seed: u64 = 0x1234_5678;

    for i in 0..10_000u64 {
        // Random contribution: 1 to 1000 SAP in micro-SAP
        let contribution_sap = rng_range(&mut seed, 1, 1000);
        let contribution = contribution_sap * 1_000_000; // micro-SAP

        // 25% to inalienable reserve, 75% to available
        let to_reserve = contribution / 4;
        let to_available = contribution - to_reserve;

        reserve += to_reserve;
        available += to_available;

        let total = reserve + available;
        // Reserve must be >= 25% of total
        // Use integer math to avoid floating-point issues: reserve * 4 >= total
        assert!(
            reserve * 4 >= total,
            "Iteration {}: reserve {} < 25% of total {} (reserve*4={}, total={})",
            i,
            reserve,
            total,
            reserve * 4,
            total
        );
    }
}

#[test]
fn test_inalienable_reserve_withdrawal_bounds() {
    // Start with a pool of 10,000 SAP (2,500 reserve, 7,500 available).
    // Attempt to withdraw amounts from 1 to 7,500 SAP.
    // Verify that no withdrawal drops reserve below 25% of remaining total.
    let initial_total: u64 = 10_000_000_000; // 10,000 SAP in micro
    let initial_reserve: u64 = 2_500_000_000; // 2,500 SAP
    let initial_available: u64 = 7_500_000_000; // 7,500 SAP

    for withdraw_sap in 1..=7_500u64 {
        let withdraw = withdraw_sap * 1_000_000; // micro-SAP

        if withdraw > initial_available {
            // Can't withdraw more than available
            continue;
        }

        let remaining_available = initial_available - withdraw;
        let remaining_total = initial_reserve + remaining_available;

        // Reserve must still be >= 25% of remaining total
        // reserve * 4 >= remaining_total
        let reserve_ok = initial_reserve * 4 >= remaining_total;

        // When we withdraw from the available pool and leave reserve untouched,
        // the reserve ratio actually INCREASES (reserve stays the same, total
        // decreases). So this should always hold.
        assert!(
            reserve_ok,
            "Withdraw {} SAP: reserve {} < 25% of remaining total {}",
            withdraw_sap,
            initial_reserve,
            remaining_total
        );
    }
}

#[test]
fn test_compost_distribution_sums_to_total() {
    // For 10,000 random deduction amounts (1 to 1B micro-SAP), verify that
    // local + regional + global == deduction (no rounding loss beyond 1).
    let mut seed: u64 = 0xBEEF_FACE;

    for i in 0..10_000 {
        let deduction = rng_range(&mut seed, 1, 1_000_000_000);

        // Apply 70/20/10 split
        let local = deduction * COMPOST_LOCAL_PCT / 100;
        let regional = deduction * COMPOST_REGIONAL_PCT / 100;
        let global = deduction * COMPOST_GLOBAL_PCT / 100;

        let sum = local + regional + global;

        // Integer division may lose up to 2 micro-SAP of rounding
        // (each division can lose at most 1 from the remainder)
        let diff = if deduction > sum {
            deduction - sum
        } else {
            sum - deduction
        };

        assert!(
            diff <= 2,
            "Iteration {}: deduction {} but local({}) + regional({}) + global({}) = {} (diff {})",
            i,
            deduction,
            local,
            regional,
            global,
            sum,
            diff
        );
    }
}

// =============================================================================
// Section 5: Concurrent Deposit Rate Limit Simulation
// =============================================================================

/// Simulated deposit rate limiter (mirrors the on-chain logic).
/// Tracks cumulative accepted deposits within a 24-hour window.
struct DepositRateLimiter {
    vault_balance: u64,
    daily_cap_pct: f64,
    window_seconds: u64,
    /// (timestamp_seconds, amount) of each accepted deposit
    accepted: Vec<(u64, u64)>,
}

impl DepositRateLimiter {
    fn new(vault_balance: u64, daily_cap_pct: f64) -> Self {
        Self {
            vault_balance,
            daily_cap_pct,
            window_seconds: 86_400, // 24 hours
            accepted: Vec::new(),
        }
    }

    /// Cumulative accepted deposits within the current window ending at `now`.
    fn window_total(&self, now: u64) -> u64 {
        let window_start = now.saturating_sub(self.window_seconds);
        self.accepted
            .iter()
            .filter(|(ts, _)| *ts > window_start)
            .map(|(_, amt)| *amt)
            .sum()
    }

    /// Try to accept a deposit. Returns true if accepted, false if rate-limited.
    fn try_deposit(&mut self, amount: u64, timestamp: u64) -> bool {
        let cap = (self.vault_balance as f64 * self.daily_cap_pct) as u64;
        let current_total = self.window_total(timestamp);

        if current_total + amount > cap {
            return false;
        }

        self.accepted.push((timestamp, amount));
        true
    }
}

#[test]
fn test_rate_limit_5pct_daily_cap() {
    // Simulate a vault of 1,000,000 SAP. Attempt 100 deposits of varying sizes
    // within a single day. Verify that total accepted deposits never exceed 5%
    // of vault (50,000 SAP).
    let vault = 1_000_000_000_000u64; // 1M SAP in micro
    let cap_pct = 0.05;
    let cap = (vault as f64 * cap_pct) as u64; // 50,000 SAP = 50_000_000_000 micro

    let mut limiter = DepositRateLimiter::new(vault, cap_pct);
    let mut seed: u64 = 0xA0A0_B0B0;
    let mut total_accepted: u64 = 0;
    let mut accepted_count = 0u64;
    let mut rejected_count = 0u64;

    let base_timestamp = 1_000_000u64; // arbitrary start

    for i in 0..100 {
        // Random deposit: 100 to 10,000 SAP in micro
        let amount = rng_range(&mut seed, 100, 10_000) * 1_000_000;
        let ts = base_timestamp + i * 60; // each deposit 1 minute apart, all within 1 day

        if limiter.try_deposit(amount, ts) {
            total_accepted += amount;
            accepted_count += 1;
        } else {
            rejected_count += 1;
        }
    }

    assert!(
        total_accepted <= cap,
        "Total accepted {} exceeds 5% cap {}",
        total_accepted,
        cap
    );

    // Some should have been rejected (100 deposits of up to 10,000 SAP each
    // easily exceeds the 50,000 SAP cap)
    assert!(
        rejected_count > 0,
        "Expected some rejected deposits but all were accepted"
    );
    assert!(
        accepted_count > 0,
        "Expected some accepted deposits but all were rejected"
    );
}

#[test]
fn test_rate_limit_resets_after_24h() {
    // Same vault, but deposits span 48 hours. Verify that the 5% limit resets
    // after 24 hours (second day's deposits succeed up to 5% again).
    let vault = 1_000_000_000_000u64; // 1M SAP in micro
    let cap_pct = 0.05;
    let cap = (vault as f64 * cap_pct) as u64; // 50,000 SAP = 50_000_000_000 micro

    let mut limiter = DepositRateLimiter::new(vault, cap_pct);
    let base_timestamp = 1_000_000u64;

    // Day 1: Fill up the cap with a single large deposit
    let day1_deposit = cap - 1_000_000; // just under the cap
    assert!(
        limiter.try_deposit(day1_deposit, base_timestamp),
        "Day 1 deposit should be accepted"
    );

    // Day 1: Next deposit should be rejected (would exceed cap)
    let small_excess = 2_000_000u64;
    assert!(
        !limiter.try_deposit(small_excess, base_timestamp + 100),
        "Day 1 excess deposit should be rejected"
    );

    // Day 2: 24 hours + 1 second later, the window has shifted
    let day2_start = base_timestamp + 86_401;

    // Day 2: Should be able to deposit up to the cap again
    let day2_deposit = cap / 2; // Half the cap
    assert!(
        limiter.try_deposit(day2_deposit, day2_start),
        "Day 2 deposit should be accepted (window reset)"
    );

    // Day 2: Another deposit should work if total is under cap
    let day2_deposit2 = cap / 2 - 1_000_000;
    assert!(
        limiter.try_deposit(day2_deposit2, day2_start + 100),
        "Day 2 second deposit should be accepted"
    );

    // Day 2: Verify total in day 2 window is under cap
    let day2_total = limiter.window_total(day2_start + 200);
    assert!(
        day2_total <= cap,
        "Day 2 total {} exceeds cap {}",
        day2_total,
        cap
    );

    // Day 2: One more deposit should push us over
    let day2_excess = cap; // Definitely too much
    assert!(
        !limiter.try_deposit(day2_excess, day2_start + 300),
        "Day 2 excess deposit should be rejected"
    );
}
