// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Opt-in live PostgreSQL campaign.
//!
//! These tests require a dedicated database named by
//! `SYMTHAEA_WISDOM_TEST_POSTGRES_DSN`. They are ignored by default because
//! they run migrations and mutate the crate's fixed production tables.

#![cfg(feature = "postgres-sync-driver")]

use std::env;

use symthaea_wisdom::{
    AtomicLedgerBackend, BackendCompareExchange, EvidenceLedger, LedgerStorageFrame,
    RuntimeSourceIdentity, RuntimeSourceProvisioningManifest, StartupAttemptClaimOutcome,
    StartupAttemptStore, StartupIdentityBundle, SyncPostgresDeployment, SyncPostgresExecutor,
    TrustRegistry, TrustRole, TrustedKey, TrustedKeyStatus,
};

fn dsn() -> String {
    env::var("SYMTHAEA_WISDOM_TEST_POSTGRES_DSN")
        .expect("set SYMTHAEA_WISDOM_TEST_POSTGRES_DSN to a dedicated PostgreSQL database")
}

fn startup_identity() -> StartupIdentityBundle {
    let mut registry = TrustRegistry::new();
    registry
        .register_initial(TrustedKey {
            role: TrustRole::RuntimeSource,
            algorithm: "test-signature".into(),
            key_id: "scheduler-live-test".into(),
            valid_from_millis: 0,
            valid_until_millis: None,
            status: TrustedKeyStatus::Active,
        })
        .unwrap();
    let source = RuntimeSourceIdentity::new(
        "scheduler-live-test",
        1,
        "test-signature",
        "scheduler-live-test",
    )
    .unwrap();
    let provisioning =
        RuntimeSourceProvisioningManifest::new(1, 1, &registry, vec![source], vec![]).unwrap();
    StartupIdentityBundle::new(
        "symthaea-wisdom-postgres-live-test",
        1,
        registry,
        provisioning,
    )
    .unwrap()
}

#[test]
#[ignore = "requires SYMTHAEA_WISDOM_TEST_POSTGRES_DSN and mutates a dedicated database"]
fn live_migration_initialization_and_global_fencing() {
    let executor_a = SyncPostgresExecutor::connect(&dsn(), "wisdom-live-cluster").unwrap();
    let executor_b = SyncPostgresExecutor::connect(&dsn(), "wisdom-live-cluster").unwrap();
    let deployment_a = SyncPostgresDeployment::new(executor_a);
    let deployment_b = SyncPostgresDeployment::new(executor_b);

    deployment_a.migrate().unwrap();
    let frame = LedgerStorageFrame::new(EvidenceLedger::default());
    deployment_a
        .initialize(&frame, &startup_identity())
        .unwrap();

    let backend_a = deployment_a.backend();
    let backend_b = deployment_b.backend();
    let epoch_a = backend_a.acquire_fence("live-writer-a").unwrap();
    let epoch_b = backend_b.acquire_fence("live-writer-b").unwrap();
    assert!(epoch_b > epoch_a);

    let bytes = backend_a.load_frame().unwrap();
    let stored = LedgerStorageFrame::from_bytes(&bytes).unwrap();
    assert_eq!(
        backend_a
            .compare_exchange_frame(
                "live-writer-a",
                epoch_a,
                stored.revision,
                stored.revision,
                &bytes,
            )
            .unwrap(),
        BackendCompareExchange::LeaseFenced {
            current_epoch: epoch_b,
        }
    );
}

#[test]
#[ignore = "requires SYMTHAEA_WISDOM_TEST_POSTGRES_DSN and mutates a dedicated database"]
fn live_startup_attempt_claim_is_durable_and_idempotent() {
    let executor = SyncPostgresExecutor::connect(&dsn(), "wisdom-live-cluster").unwrap();
    let deployment = SyncPostgresDeployment::new(executor);
    deployment.migrate().unwrap();
    let frame = LedgerStorageFrame::new(EvidenceLedger::default());
    deployment.initialize(&frame, &startup_identity()).unwrap();

    let store = deployment.startup_attempt_store();
    let attempt_id = format!("live-attempt-{}", std::process::id());
    let first = store.claim(&attempt_id, 41, 100).unwrap();
    let second = store.claim(&attempt_id, 41, 101).unwrap();
    assert!(matches!(
        (first, second),
        (
            StartupAttemptClaimOutcome::Claimed,
            StartupAttemptClaimOutcome::AlreadyClaimed
        ) | (
            StartupAttemptClaimOutcome::AlreadyClaimed,
            StartupAttemptClaimOutcome::AlreadyClaimed
        )
    ));
}
