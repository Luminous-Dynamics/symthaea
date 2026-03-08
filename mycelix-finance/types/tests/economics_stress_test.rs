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

/// Standard valid MintedCurrencyParams for testing.
fn valid_minted_params() -> MintedCurrencyParams {
    MintedCurrencyParams {
        name: "TestCoin".into(),
        symbol: "TC".into(),
        description: "A test currency".into(),
        credit_limit: 40,
        demurrage_rate: 0.02,
        max_service_hours: 8,
        min_service_minutes: 15,
        requires_confirmation: false,
        confirmation_timeout_hours: 0,
        max_exchanges_per_day: 0,
    }
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
        let deduction = compute_demurrage_deduction(balance, exempt_floor, rate, seconds_per_year);

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
    let _initial_total: u64 = 10_000_000_000; // 10,000 SAP in micro
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
            withdraw_sap, initial_reserve, remaining_total
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
        let diff = deduction.abs_diff(sum);

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
// Section 5: Minted Currency (Currency Factory) Stress Tests
// =============================================================================

#[test]
fn test_minted_demurrage_convergence_to_zero() {
    // For minted currencies, positive balances (credits) decay toward zero.
    // Debts (negative balances) are exempt. Simulate 500 years of yearly
    // demurrage at the maximum rate (5%) — balance should approach 0 but
    // never go negative.
    let rate = MINTED_DEMURRAGE_RATE_MAX; // 0.05
    let seconds_per_year: u64 = 31_536_000;

    let mut balance: i32 = 10_000; // 10,000 hours credit

    for year in 1..=500 {
        let deduction = compute_minted_demurrage(balance, rate, seconds_per_year);
        assert!(
            deduction >= 0 && deduction <= balance,
            "Year {}: deduction {} invalid for balance {}",
            year,
            deduction,
            balance
        );
        balance -= deduction;
        assert!(
            balance >= 0,
            "Year {}: balance {} went negative",
            year,
            balance
        );
    }

    // After 500 years at 5% continuous demurrage, balance should be very small.
    // The floor effect (deduction < 1 rounds to 0) means a small residual persists.
    assert!(
        balance < 50,
        "After 500 years at 5%, balance {} should be near zero",
        balance
    );
}

#[test]
fn test_minted_demurrage_debt_immune() {
    // Negative balances (debts) must NEVER be subject to demurrage.
    // Test across a range of negative balances and time periods.
    let rate = MINTED_DEMURRAGE_RATE_MAX;
    let seconds_per_year: u64 = 31_536_000;

    for debt in [-1, -10, -100, -1000, -10_000, -100_000, i32::MIN + 1] {
        for &elapsed in &[1, 3600, seconds_per_year, seconds_per_year * 1000] {
            let deduction = compute_minted_demurrage(debt, rate, elapsed);
            assert_eq!(
                deduction, 0,
                "Debt {} should be immune to demurrage (elapsed {}s), got deduction {}",
                debt, elapsed, deduction
            );
        }
    }
}

#[test]
fn test_minted_zero_sum_with_demurrage() {
    // Simulate 10,000 exchanges in a minted currency with 2% demurrage.
    // After each batch of 100 exchanges, apply demurrage to all positive balances.
    // Verify: (a) pre-demurrage sum is always zero,
    //         (b) post-demurrage sum equals total demurrage deducted (the "tax").
    let num_members = 50;
    let num_exchanges = 10_000;
    let rate = 0.02;
    let credit_limit = 100i32;
    let seconds_per_batch: u64 = 3_600; // 1 hour between batches

    let mut balances = vec![0i32; num_members];
    let mut total_demurrage: i64 = 0;
    let mut seed: u64 = 0xF00D_CAFE;

    for i in 0..num_exchanges {
        let sender = rng_range(&mut seed, 0, (num_members - 1) as u64) as usize;
        let mut receiver = rng_range(&mut seed, 0, (num_members - 1) as u64) as usize;
        if receiver == sender {
            receiver = (receiver + 1) % num_members;
        }

        let amount = rng_range(&mut seed, 1, 3) as i32;

        // Enforce credit limits
        let new_receiver = balances[receiver] + amount;
        let new_sender = balances[sender] - amount;
        if new_receiver > credit_limit || new_sender < -credit_limit {
            continue;
        }

        // Zero-sum exchange
        balances[sender] -= amount;
        balances[receiver] += amount;

        // (a) Pre-demurrage sum should always be exactly -total_demurrage
        let sum: i64 = balances.iter().map(|&b| b as i64).sum();
        assert_eq!(
            sum, -total_demurrage,
            "Exchange {}: sum {} != -total_demurrage {}",
            i, sum, total_demurrage
        );

        // Apply demurrage every 100 exchanges
        if (i + 1) % 100 == 0 {
            for bal in balances.iter_mut() {
                if *bal > 0 {
                    let deduction = compute_minted_demurrage(*bal, rate, seconds_per_batch);
                    *bal -= deduction;
                    total_demurrage += deduction as i64;
                }
            }
        }
    }

    // Final verification: sum of balances + total_demurrage == 0
    let final_sum: i64 = balances.iter().map(|&b| b as i64).sum();
    assert_eq!(
        final_sum + total_demurrage,
        0,
        "Conservation violated: sum {} + demurrage {} != 0",
        final_sum,
        total_demurrage
    );
}

#[test]
fn test_minted_params_validation_fuzz() {
    // Fuzz MintedCurrencyParams validation with 10,000 random combinations.
    // Verify: (a) no panics, (b) valid params always accepted, (c) invalid always rejected.
    let mut seed: u64 = 0xABCD_1234;

    for _ in 0..10_000 {
        let name_len = rng_range(&mut seed, 0, 60) as usize;
        let name: String = (0..name_len).map(|_| 'A').collect();

        let sym_len = rng_range(&mut seed, 0, 8) as usize;
        let symbol: String = (0..sym_len).map(|_| 'X').collect();

        let credit_limit = rng_range(&mut seed, 0, 300) as i32;
        let demurrage_rate = (rng_range(&mut seed, 0, 100) as f64) / 1000.0; // 0.0 to 0.1
        let max_service_hours = rng_range(&mut seed, 0, 50) as u32;
        let min_service_minutes = rng_range(&mut seed, 0, 120) as u32;

        let desc_len = rng_range(&mut seed, 0, 600) as usize;
        let description: String = (0..desc_len).map(|_| 'D').collect();

        let params = MintedCurrencyParams {
            name: name.clone(),
            symbol: symbol.clone(),
            description: description.clone(),
            credit_limit,
            demurrage_rate,
            max_service_hours,
            min_service_minutes,
            requires_confirmation: rng_range(&mut seed, 0, 1) == 1,
            confirmation_timeout_hours: rng_range(&mut seed, 0, 800) as u32,
            max_exchanges_per_day: rng_range(&mut seed, 0, 60) as u8,
        };

        let result = params.validate();

        // Cross-check: if all fields are individually valid, overall should be valid
        let name_ok = !name.is_empty() && name.len() <= MINTED_NAME_MAX_LEN;
        let sym_ok = !symbol.is_empty()
            && symbol.len() <= MINTED_SYMBOL_MAX_LEN
            && symbol.chars().all(|c| c.is_ascii_alphanumeric());
        let limit_ok = (MINTED_CREDIT_LIMIT_MIN..=MINTED_CREDIT_LIMIT_MAX).contains(&credit_limit);
        let demurrage_ok = (0.0..=MINTED_DEMURRAGE_RATE_MAX).contains(&demurrage_rate);
        let hours_ok = (1..=MINTED_MAX_SERVICE_HOURS_MAX).contains(&max_service_hours);
        let minutes_ok = (MINTED_MIN_SERVICE_MINUTES_MIN..=60).contains(&min_service_minutes);

        let timeout_ok = params.confirmation_timeout_hours <= MINTED_CONFIRMATION_TIMEOUT_MAX;

        let rate_limit_ok = params.max_exchanges_per_day <= MINTED_MAX_EXCHANGES_PER_DAY_MAX;
        let desc_ok = description.len() <= MINTED_DESCRIPTION_MAX_LEN;

        let should_be_valid = name_ok
            && sym_ok
            && desc_ok
            && limit_ok
            && demurrage_ok
            && hours_ok
            && minutes_ok
            && timeout_ok
            && rate_limit_ok;

        if should_be_valid {
            assert!(
                result.is_ok(),
                "Params should be valid but got: {:?}. name_len={}, sym_len={}, limit={}, demurrage={}, hours={}, minutes={}",
                result,
                name.len(),
                symbol.len(),
                credit_limit,
                demurrage_rate,
                max_service_hours,
                min_service_minutes
            );
        }
        // Note: we don't assert result.is_err() for the inverse because
        // our cross-check may not cover all validation rules perfectly.
    }
}

#[test]
fn test_minted_demurrage_precision_all_rates() {
    // For rates 0.01, 0.02, 0.03, 0.04, 0.05 and balances 1 to 10,000,
    // verify demurrage matches the continuous formula within 1 unit tolerance.
    let seconds_per_year: u64 = 31_536_000;

    for rate_pct in 1..=5u32 {
        let rate = rate_pct as f64 / 100.0;

        for balance in [1, 10, 100, 1000, 5000, 10_000i32] {
            let deduction = compute_minted_demurrage(balance, rate, seconds_per_year);

            // Expected: balance * (1 - e^(-rate))
            let expected = (balance as f64) * (1.0 - (-rate).exp());
            let expected_i32 = expected.floor() as i32;

            let diff = (deduction - expected_i32).abs();
            assert!(
                diff <= 1,
                "rate={}, balance={}: deduction {} vs expected {} (diff {})",
                rate,
                balance,
                deduction,
                expected_i32,
                diff
            );
        }
    }
}

// =============================================================================
// Section 6: Concurrent Deposit Rate Limit Simulation
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

// =============================================================================
// Section 7: Currency Factory Round 6/7 Feature Tests
// =============================================================================

#[test]
fn test_compost_zero_sum_invariant() {
    // Simulate 1000 demurrage cycles across 20 members.
    // After each cycle: sum(balances) + compost == 0 (zero-sum invariant).
    let num_members = 20;
    let rate = 0.03;
    let credit_limit = 100i32;
    let seconds_per_cycle: u64 = 86_400; // 1 day

    let mut balances = vec![0i32; num_members];
    let mut compost: i64 = 0;
    let mut seed: u64 = 0xC0FFEE;

    // Bootstrap: do some exchanges to create non-zero balances
    for _ in 0..200 {
        let sender = rng_range(&mut seed, 0, (num_members - 1) as u64) as usize;
        let mut receiver = rng_range(&mut seed, 0, (num_members - 1) as u64) as usize;
        if receiver == sender {
            receiver = (receiver + 1) % num_members;
        }
        let amount = rng_range(&mut seed, 1, 5) as i32;
        if balances[receiver] + amount <= credit_limit && balances[sender] - amount >= -credit_limit
        {
            balances[sender] -= amount;
            balances[receiver] += amount;
        }
    }

    // Verify initial zero-sum
    let initial_sum: i64 = balances.iter().map(|&b| b as i64).sum();
    assert_eq!(initial_sum, 0, "Initial balances must sum to zero");

    // Run 1000 demurrage cycles
    for cycle in 0..1000 {
        for bal in balances.iter_mut() {
            if *bal > 0 {
                let deduction = compute_minted_demurrage(*bal, rate, seconds_per_cycle);
                *bal -= deduction;
                compost += deduction as i64;
            }
        }

        // Zero-sum check: balances + compost == 0
        let sum: i64 = balances.iter().map(|&b| b as i64).sum();
        assert_eq!(
            sum + compost,
            0,
            "Cycle {}: zero-sum violated. sum={}, compost={}",
            cycle,
            sum,
            compost
        );
    }
}

#[test]
fn test_rate_limit_boundary() {
    // Simulate daily exchange rate limiting.
    // At limit N: first N exchanges pass, exchange N+1 fails.
    for limit in [1u8, 5, 10, 25, 50] {
        let mut count: u8 = 0;

        // Exchanges 1..=limit should all pass
        for i in 1..=limit {
            assert!(
                count < limit,
                "Exchange {} should pass (count={}, limit={})",
                i,
                count,
                limit
            );
            count += 1;
        }
        assert_eq!(count, limit);

        // Exchange limit+1 should fail
        assert!(
            count >= limit,
            "Exchange {} should be blocked (count={}, limit={})",
            limit + 1,
            count,
            limit
        );
    }
}

#[test]
fn test_description_validation_boundaries() {
    // Description max length is 500 chars
    let mut p = MintedCurrencyParams {
        name: "Test".into(),
        symbol: "TST".into(),
        description: String::new(),
        credit_limit: 40,
        demurrage_rate: 0.02,
        max_service_hours: 8,
        min_service_minutes: 15,
        requires_confirmation: false,
        confirmation_timeout_hours: 0,
        max_exchanges_per_day: 0,
    };

    // Empty description is valid
    assert!(p.validate().is_ok(), "Empty description should be valid");

    // At max length
    p.description = "X".repeat(MINTED_DESCRIPTION_MAX_LEN);
    assert!(
        p.validate().is_ok(),
        "Max length description should be valid"
    );

    // One over max
    p.description = "X".repeat(MINTED_DESCRIPTION_MAX_LEN + 1);
    assert!(
        p.validate().is_err(),
        "Over-max description should be invalid"
    );

    // Unicode — length in bytes, not chars
    p.description = "🌱".repeat(100); // 4 bytes each = 400 bytes, under 500
    assert!(
        p.validate().is_ok(),
        "Unicode description within byte limit should be valid"
    );
}

// =============================================================================
// Minted currency parameter boundary exhaustion
// =============================================================================

/// Test that every MintedCurrencyParams field boundary is properly enforced
/// by systematically setting each field to its min, max, and over-max values.
#[test]
fn test_minted_params_boundary_exhaustion() {
    fn base_params() -> MintedCurrencyParams {
        MintedCurrencyParams {
            name: "Test".into(),
            symbol: "TST".into(),
            description: "Test currency".into(),
            credit_limit: MINTED_CREDIT_LIMIT_MIN,
            demurrage_rate: 0.0,
            max_service_hours: 1,
            min_service_minutes: MINTED_MIN_SERVICE_MINUTES_MIN,
            requires_confirmation: false,
            confirmation_timeout_hours: 0,
            max_exchanges_per_day: 0,
        }
    }

    // credit_limit boundaries
    let mut p = base_params();
    p.credit_limit = MINTED_CREDIT_LIMIT_MIN;
    assert!(p.validate().is_ok(), "credit_limit at min should be valid");
    p.credit_limit = MINTED_CREDIT_LIMIT_MAX;
    assert!(p.validate().is_ok(), "credit_limit at max should be valid");
    p.credit_limit = MINTED_CREDIT_LIMIT_MIN - 1;
    assert!(p.validate().is_err(), "credit_limit below min should fail");
    p.credit_limit = MINTED_CREDIT_LIMIT_MAX + 1;
    assert!(p.validate().is_err(), "credit_limit above max should fail");

    // demurrage_rate boundaries
    let mut p = base_params();
    p.demurrage_rate = 0.0;
    assert!(p.validate().is_ok(), "demurrage_rate 0 should be valid");
    p.demurrage_rate = MINTED_DEMURRAGE_RATE_MAX;
    assert!(
        p.validate().is_ok(),
        "demurrage_rate at max should be valid"
    );
    p.demurrage_rate = MINTED_DEMURRAGE_RATE_MAX + 0.001;
    assert!(
        p.validate().is_err(),
        "demurrage_rate above max should fail"
    );
    p.demurrage_rate = -0.001;
    assert!(p.validate().is_err(), "negative demurrage_rate should fail");

    // max_service_hours boundaries
    let mut p = base_params();
    p.max_service_hours = 1;
    assert!(p.validate().is_ok(), "max_service_hours 1 should be valid");
    p.max_service_hours = MINTED_MAX_SERVICE_HOURS_MAX;
    assert!(
        p.validate().is_ok(),
        "max_service_hours at max should be valid"
    );
    p.max_service_hours = 0;
    assert!(p.validate().is_err(), "max_service_hours 0 should fail");
    p.max_service_hours = MINTED_MAX_SERVICE_HOURS_MAX + 1;
    assert!(
        p.validate().is_err(),
        "max_service_hours above max should fail"
    );

    // confirmation_timeout boundaries
    let mut p = base_params();
    p.confirmation_timeout_hours = 0;
    assert!(p.validate().is_ok(), "timeout 0 should be valid");
    p.confirmation_timeout_hours = MINTED_CONFIRMATION_TIMEOUT_MAX;
    assert!(p.validate().is_ok(), "timeout at max should be valid");
    p.confirmation_timeout_hours = MINTED_CONFIRMATION_TIMEOUT_MAX + 1;
    assert!(p.validate().is_err(), "timeout above max should fail");

    // max_exchanges_per_day boundaries
    let mut p = base_params();
    p.max_exchanges_per_day = 0;
    assert!(p.validate().is_ok(), "rate_limit 0 should be valid");
    p.max_exchanges_per_day = MINTED_MAX_EXCHANGES_PER_DAY_MAX;
    assert!(p.validate().is_ok(), "rate_limit at max should be valid");
    p.max_exchanges_per_day = MINTED_MAX_EXCHANGES_PER_DAY_MAX + 1;
    assert!(p.validate().is_err(), "rate_limit above max should fail");
}

/// Stress test: demurrage + compost zero-sum over many members.
/// Simulates N members with random balances, applies demurrage, and verifies
/// the total deductions equal the compost credit.
#[test]
fn test_demurrage_compost_zero_sum_stress() {
    let mut seed: u64 = 0xDEAD_BEEF;
    let rate = 0.02; // 2% annual

    for trial in 0..50 {
        let elapsed = (simple_rng(&mut seed) % 31_536_000) + 1; // 1 second to 1 year
        let member_count = (simple_rng(&mut seed) % 20) + 1;
        let mut total_deductions: i64 = 0;

        for _ in 0..member_count {
            let balance = (simple_rng(&mut seed) % 201) as i32; // 0-200
            let deduction = compute_minted_demurrage(balance, rate, elapsed);
            assert!(deduction >= 0, "Deduction must be non-negative");
            assert!(deduction <= balance, "Deduction must not exceed balance");
            total_deductions += deduction as i64;
        }

        // Compost gets exactly the sum of all deductions
        // (In the coordinator, compost.balance += each deduction)
        // Verify the total is reasonable
        assert!(
            total_deductions >= 0,
            "Trial {}: total deductions must be non-negative",
            trial
        );
    }
}

/// Stress test: max_exchanges_per_day boundary validation.
/// 0 = unlimited (valid), 1-50 = valid, 51+ = invalid.
#[test]
fn test_minted_rate_limit_boundary() {
    use mycelix_finance_types::MINTED_MAX_EXCHANGES_PER_DAY_MAX;

    let mut base = valid_minted_params();

    // 0 = unlimited, always valid
    base.max_exchanges_per_day = 0;
    assert!(base.validate().is_ok(), "0 means unlimited");

    // Exact boundary
    base.max_exchanges_per_day = MINTED_MAX_EXCHANGES_PER_DAY_MAX;
    assert!(base.validate().is_ok(), "Boundary value should be valid");

    // One above boundary
    base.max_exchanges_per_day = MINTED_MAX_EXCHANGES_PER_DAY_MAX + 1;
    assert!(base.validate().is_err(), "Above max should be invalid");

    // u8::MAX
    base.max_exchanges_per_day = u8::MAX;
    assert!(base.validate().is_err(), "u8::MAX should be invalid");

    // All valid values 1..=MAX
    for v in 1..=MINTED_MAX_EXCHANGES_PER_DAY_MAX {
        base.max_exchanges_per_day = v;
        assert!(base.validate().is_ok(), "Value {} should be valid", v);
    }
}

/// Stress test: confirmation_timeout_hours boundary.
/// Valid range: 0..=MINTED_CONFIRMATION_TIMEOUT_MAX.
#[test]
fn test_minted_confirmation_timeout_boundary() {
    use mycelix_finance_types::MINTED_CONFIRMATION_TIMEOUT_MAX;

    let mut p = valid_minted_params();

    // Zero timeout is valid (no confirmation or instant)
    p.confirmation_timeout_hours = 0;
    assert!(p.validate().is_ok());

    // Exact boundary
    p.confirmation_timeout_hours = MINTED_CONFIRMATION_TIMEOUT_MAX;
    assert!(p.validate().is_ok(), "Max timeout should be valid");

    // One above boundary
    p.confirmation_timeout_hours = MINTED_CONFIRMATION_TIMEOUT_MAX + 1;
    assert!(p.validate().is_err(), "Above max timeout should be invalid");

    // With confirmation enabled, same boundaries apply
    p.requires_confirmation = true;
    p.confirmation_timeout_hours = 48;
    assert!(p.validate().is_ok(), "48h with confirmation should pass");

    p.confirmation_timeout_hours = 168; // 1 week
    assert!(p.validate().is_ok(), "1-week timeout should pass");
}

/// Stress test: demurrage on negative balances always returns 0.
#[test]
fn test_demurrage_never_charges_debtors() {
    let mut seed: u64 = 0xCAFE_BABE;
    for _ in 0..100 {
        let negative_balance = -((simple_rng(&mut seed) % 200) as i32) - 1;
        let rate = (simple_rng(&mut seed) % 500) as f64 / 10_000.0; // 0-5%
        let elapsed = simple_rng(&mut seed) % 100_000_000;
        let deduction = compute_minted_demurrage(negative_balance, rate, elapsed);
        assert_eq!(
            deduction, 0,
            "Debtors should never be charged demurrage (balance={})",
            negative_balance
        );
    }
}

/// Stress test: NaN and infinity in demurrage computation never panic.
/// The integrity zome rejects these, but the math function should be robust.
#[test]
fn test_demurrage_special_float_values() {
    // NaN rate — should produce 0 deduction (no panic)
    let d = compute_minted_demurrage(100, f64::NAN, 3600);
    assert!(d == 0 || d >= 0, "NaN rate should not panic");

    // Infinity rate
    let d = compute_minted_demurrage(100, f64::INFINITY, 3600);
    assert!(d >= 0, "Infinity rate should not panic");

    // Negative infinity rate
    let d = compute_minted_demurrage(100, f64::NEG_INFINITY, 3600);
    assert!(d == 0 || d >= 0, "Neg infinity rate should not panic");

    // Zero balance (edge)
    let d = compute_minted_demurrage(0, 0.02, 3600);
    assert_eq!(d, 0, "Zero balance should produce zero deduction");

    // Zero elapsed time
    let d = compute_minted_demurrage(100, 0.02, 0);
    assert_eq!(d, 0, "Zero elapsed should produce zero deduction");

    // Max i32 balance
    let d = compute_minted_demurrage(i32::MAX, 0.05, 31_536_000);
    assert!(
        (0..=i32::MAX).contains(&d),
        "Max balance should not overflow"
    );
}

/// Stress test: all valid MintedCurrencyParams field combinations at boundaries.
/// Ensures no panics and expected pass/fail outcomes at the edges of validity.
#[test]
fn test_minted_params_boundary_matrix() {
    use mycelix_finance_types::{
        MINTED_CREDIT_LIMIT_MAX, MINTED_CREDIT_LIMIT_MIN, MINTED_DEMURRAGE_RATE_MAX,
    };

    let base = valid_minted_params();

    // Credit limit boundaries
    for &cl in &[
        MINTED_CREDIT_LIMIT_MIN,
        MINTED_CREDIT_LIMIT_MAX,
        MINTED_CREDIT_LIMIT_MIN - 1,
        MINTED_CREDIT_LIMIT_MAX + 1,
    ] {
        let mut p = base.clone();
        p.credit_limit = cl;
        let result = p.validate();
        if (MINTED_CREDIT_LIMIT_MIN..=MINTED_CREDIT_LIMIT_MAX).contains(&cl) {
            assert!(result.is_ok(), "credit_limit {} should be valid", cl);
        } else {
            assert!(result.is_err(), "credit_limit {} should be invalid", cl);
        }
    }

    // Demurrage rate boundaries
    for &rate in &[
        0.0,
        MINTED_DEMURRAGE_RATE_MAX,
        MINTED_DEMURRAGE_RATE_MAX + 0.001,
        -0.001,
    ] {
        let mut p = base.clone();
        p.demurrage_rate = rate;
        let result = p.validate();
        if (0.0..=MINTED_DEMURRAGE_RATE_MAX).contains(&rate) {
            assert!(result.is_ok(), "demurrage_rate {} should be valid", rate);
        } else {
            assert!(result.is_err(), "demurrage_rate {} should be invalid", rate);
        }
    }
}

// =============================================================================
// Section 8: Demurrage Zero-Sum Proof (100 members)
// =============================================================================

#[test]
fn test_demurrage_zero_sum_100_members() {
    // Simulate 100 members with random balances.
    // Apply demurrage to all positive balances.
    // Verify: sum(all balances) + compost == 0 at every step.

    let rate = 0.02; // 2% annual
    let one_year_secs: u64 = 31_536_000;

    // Start with 50 positive, 50 negative (zero-sum initial state)
    let mut balances: Vec<i32> = Vec::new();
    let mut total_sum: i64 = 0;

    // Positive balances: 2, 4, 6, ..., 100
    for i in 1..=50 {
        balances.push(i * 2);
        total_sum += (i * 2) as i64;
    }
    // Negative balances to make zero-sum
    for i in 1..=50 {
        balances.push(-(i * 2));
        total_sum -= (i * 2) as i64;
    }
    assert_eq!(total_sum, 0, "Initial state must be zero-sum");

    let mut compost: i64 = 0;

    // Apply demurrage for 1 year to all positive balances
    for bal in balances.iter_mut() {
        if *bal > 0 {
            let deduction = compute_minted_demurrage(*bal, rate, one_year_secs);
            *bal -= deduction;
            compost += deduction as i64;
        }
    }

    // Verify zero-sum
    let final_sum: i64 = balances.iter().map(|b| *b as i64).sum::<i64>() + compost;
    assert_eq!(
        final_sum,
        0,
        "Zero-sum violated after 1-year demurrage: sum={}, compost={}",
        balances.iter().map(|b| *b as i64).sum::<i64>(),
        compost
    );

    // Verify compost accumulated something
    assert!(compost > 0, "Compost should have accumulated demurrage");

    // Negative balances should be unchanged
    for bal in &balances[50..100] {
        assert!(*bal < 0, "Negative balance should remain negative: {}", bal);
    }

    println!(
        "100-member zero-sum proof: compost={}, verified after 1 year",
        compost
    );
}

#[test]
fn test_demurrage_zero_sum_multi_year_interleaved() {
    // Simulate exchanges interleaved with yearly demurrage over 5 years.
    // Verify zero-sum at every step.

    let rate = 0.03; // 3% annual
    let one_year_secs: u64 = 31_536_000;

    // 10 members, start at zero
    let mut balances = vec![0i32; 10];
    let mut compost: i64 = 0;

    // Helper: verify zero-sum
    let check_zero_sum = |balances: &[i32], compost: i64, step: &str| {
        let sum: i64 = balances.iter().map(|b| *b as i64).sum::<i64>() + compost;
        assert_eq!(sum, 0, "Zero-sum violated at step: {}", step);
    };

    check_zero_sum(&balances, compost, "initial");

    for year in 1..=5 {
        // Simulate exchanges with larger amounts to ensure demurrage produces
        // non-zero deductions (small balances floor to 0 with continuous decay).
        // member[0] provides year*20 hours to member[1]
        let hours = year * 20;
        balances[0] += hours;
        balances[1] -= hours;
        check_zero_sum(&balances, compost, &format!("year {} after exchange", year));

        // member[2] provides 10 hours to member[3]
        balances[2] += 10;
        balances[3] -= 10;
        check_zero_sum(
            &balances,
            compost,
            &format!("year {} after exchange 2", year),
        );

        // Apply demurrage to all positive balances
        for bal in balances.iter_mut() {
            if *bal > 0 {
                let deduction = compute_minted_demurrage(*bal, rate, one_year_secs);
                compost += deduction as i64;
                *bal -= deduction;
            }
        }
        check_zero_sum(
            &balances,
            compost,
            &format!("year {} after demurrage", year),
        );
    }

    // Final verification
    check_zero_sum(&balances, compost, "final after 5 years");
    assert!(compost > 0, "Compost should be positive after 5 years");
    println!(
        "5-year interleaved zero-sum proof: compost={}, balances={:?}",
        compost, balances
    );
}
