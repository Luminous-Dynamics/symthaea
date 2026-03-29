// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! End-to-end DKG ceremony integration tests
//!
//! Tests the full distributed key generation flow from participant
//! registration through secret reconstruction.

use feldman_dkg::{
    ceremony::{CeremonyPhase, DkgCeremony, DkgConfig},
    dealer::Dealer,
    participant::ParticipantId,
    scalar::Scalar,
};

/// Test a complete 3-of-5 DKG ceremony
#[test]
fn test_full_dkg_ceremony_3_of_5() {
    // Configuration: 3-of-5 threshold scheme
    let threshold = 3;
    let num_participants = 5;

    let config = DkgConfig::new(threshold, num_participants).expect("Valid config");
    let mut ceremony = DkgCeremony::new(config);

    // Phase 1: Registration
    assert_eq!(ceremony.phase(), CeremonyPhase::Registration);

    for i in 1..=num_participants {
        ceremony
            .add_participant(ParticipantId(i as u32))
            .expect("Registration should succeed");
    }

    // Should auto-transition to Dealing phase
    assert_eq!(ceremony.phase(), CeremonyPhase::Dealing);

    // Phase 2: Each participant generates and submits a deal
    let mut dealers = Vec::new();
    for i in 1..=num_participants {
        let dealer =
            Dealer::new(ParticipantId(i as u32), threshold, num_participants).expect("Valid dealer");
        dealers.push(dealer);
    }

    for dealer in &dealers {
        let deal = dealer.generate_deal();
        ceremony
            .submit_deal(dealer.id(), deal)
            .expect("Deal submission should succeed");
    }

    // Should auto-transition to Verification phase
    assert_eq!(ceremony.phase(), CeremonyPhase::Verification);

    // Phase 3: Finalize (no complaints in this test)
    let result = ceremony.finalize().expect("Finalization should succeed");

    assert_eq!(ceremony.phase(), CeremonyPhase::Complete);
    assert_eq!(result.qualified_count, num_participants);
    assert!(result.disqualified.is_empty());

    // Phase 4: Verify secret reconstruction
    // Get combined shares from any 3 participants
    let shares: Vec<_> = [1, 2, 3]
        .iter()
        .map(|&i| {
            ceremony
                .get_combined_share(ParticipantId(i))
                .expect("Should get share")
        })
        .collect();

    let reconstructed = ceremony
        .reconstruct_secret(&shares)
        .expect("Reconstruction should succeed");

    // Verify with different subset
    let shares_alt: Vec<_> = [3, 4, 5]
        .iter()
        .map(|&i| {
            ceremony
                .get_combined_share(ParticipantId(i))
                .expect("Should get share")
        })
        .collect();

    let reconstructed_alt = ceremony
        .reconstruct_secret(&shares_alt)
        .expect("Reconstruction should succeed");

    // Both reconstructions should yield the same secret
    assert_eq!(reconstructed, reconstructed_alt);
}

/// Test DKG with known secrets to verify correctness
#[test]
fn test_dkg_with_known_secrets() {
    let threshold = 2;
    let num_participants = 3;

    let config = DkgConfig::new(threshold, num_participants).expect("Valid config");
    let mut ceremony = DkgCeremony::new(config);

    // Register participants
    for i in 1..=num_participants {
        ceremony
            .add_participant(ParticipantId(i as u32))
            .expect("Registration should succeed");
    }

    // Create dealers with known secrets
    let secrets = [
        Scalar::from_u64(1000),
        Scalar::from_u64(2000),
        Scalar::from_u64(3000),
    ];
    let expected_combined = Scalar::from_u64(6000); // Sum of secrets

    for (i, secret) in secrets.iter().enumerate() {
        let dealer = Dealer::with_secret(
            ParticipantId((i + 1) as u32),
            secret.clone(),
            threshold,
            num_participants,
        )
        .expect("Valid dealer");

        let deal = dealer.generate_deal();
        ceremony
            .submit_deal(dealer.id(), deal)
            .expect("Deal submission should succeed");
    }

    ceremony.finalize().expect("Finalization should succeed");

    // Reconstruct and verify
    let shares: Vec<_> = [1, 2]
        .iter()
        .map(|&i| {
            ceremony
                .get_combined_share(ParticipantId(i))
                .expect("Should get share")
        })
        .collect();

    let reconstructed = ceremony
        .reconstruct_secret(&shares)
        .expect("Reconstruction should succeed");

    assert_eq!(reconstructed, expected_combined);
}

/// Test that insufficient shares fail reconstruction
#[test]
fn test_insufficient_shares_fail() {
    let threshold = 3;
    let num_participants = 5;

    let config = DkgConfig::new(threshold, num_participants).expect("Valid config");
    let mut ceremony = DkgCeremony::new(config);

    for i in 1..=num_participants {
        ceremony
            .add_participant(ParticipantId(i as u32))
            .expect("Registration should succeed");
    }

    for i in 1..=num_participants {
        let dealer =
            Dealer::new(ParticipantId(i as u32), threshold, num_participants).expect("Valid dealer");
        let deal = dealer.generate_deal();
        ceremony
            .submit_deal(dealer.id(), deal)
            .expect("Deal submission should succeed");
    }

    ceremony.finalize().expect("Finalization should succeed");

    // Try to reconstruct with only 2 shares (threshold is 3)
    let shares: Vec<_> = [1, 2]
        .iter()
        .map(|&i| {
            ceremony
                .get_combined_share(ParticipantId(i))
                .expect("Should get share")
        })
        .collect();

    let result = ceremony.reconstruct_secret(&shares);
    assert!(result.is_err(), "Should fail with insufficient shares");
}

/// Test DKG ceremony with participant disqualification
#[test]
fn test_dkg_with_disqualification() {
    let threshold = 2;
    let num_participants = 4;

    let config = DkgConfig::new(threshold, num_participants).expect("Valid config");
    let mut ceremony = DkgCeremony::new(config);

    for i in 1..=num_participants {
        ceremony
            .add_participant(ParticipantId(i as u32))
            .expect("Registration should succeed");
    }

    // Only 3 participants submit deals (simulating one offline)
    for i in 1..=3 {
        let dealer =
            Dealer::new(ParticipantId(i as u32), threshold, num_participants).expect("Valid dealer");
        let deal = dealer.generate_deal();
        ceremony
            .submit_deal(dealer.id(), deal)
            .expect("Deal submission should succeed");
    }

    // Should still succeed with 3 qualified (>= threshold of 2)
    let result = ceremony.finalize().expect("Finalization should succeed");

    assert_eq!(result.qualified_count, 3);
    assert_eq!(ceremony.phase(), CeremonyPhase::Complete);
}

/// Test multiple sequential DKG ceremonies
#[test]
fn test_sequential_ceremonies() {
    for round in 1..=3 {
        let threshold = 2;
        let num_participants = 3;

        let config = DkgConfig::new(threshold, num_participants).expect("Valid config");
        let mut ceremony = DkgCeremony::new(config);

        for i in 1..=num_participants {
            ceremony
                .add_participant(ParticipantId(i as u32))
                .expect("Registration should succeed");
        }

        // Use round number as part of secret for deterministic testing
        for i in 1..=num_participants {
            let secret = Scalar::from_u64((round * 100 + i) as u64);
            let dealer = Dealer::with_secret(
                ParticipantId(i as u32),
                secret,
                threshold,
                num_participants,
            )
            .expect("Valid dealer");

            let deal = dealer.generate_deal();
            ceremony
                .submit_deal(dealer.id(), deal)
                .expect("Deal submission should succeed");
        }

        let result = ceremony.finalize().expect("Finalization should succeed");
        assert_eq!(result.qualified_count, num_participants);
    }
}

#[cfg(test)]
mod benchmarks {
    use super::*;
    use std::time::Instant;

    /// Benchmark DKG ceremony performance
    #[test]
    fn benchmark_dkg_ceremony() {
        let configs = [(2, 3), (3, 5), (5, 10), (7, 15)];

        for (threshold, num_participants) in configs {
            let start = Instant::now();

            let config = DkgConfig::new(threshold, num_participants).expect("Valid config");
            let mut ceremony = DkgCeremony::new(config);

            for i in 1..=num_participants {
                ceremony
                    .add_participant(ParticipantId(i as u32))
                    .unwrap();
            }

            for i in 1..=num_participants {
                let dealer =
                    Dealer::new(ParticipantId(i as u32), threshold, num_participants).unwrap();
                let deal = dealer.generate_deal();
                ceremony.submit_deal(dealer.id(), deal).unwrap();
            }

            ceremony.finalize().unwrap();

            let duration = start.elapsed();
            println!(
                "DKG {}-of-{}: {:?}",
                threshold, num_participants, duration
            );
        }
    }
}
