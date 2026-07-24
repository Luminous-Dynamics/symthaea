// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::{
    convert::Infallible,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use symthaea_wisdom::{
    ArchiveSignatureVerifier, AtomicLedgerBackend, BackendCompareExchange, CoordinationError,
    EvidenceEvent, EvidenceLedger, EvidenceSigner, EvidenceVerifier,
    InMemoryAtomicProductionBackend, LedgerStorageFrame, OperationalStartupRequirements,
    ProductionStartupError, RuntimeSignerFactory, RuntimeSourceIdentity,
    RuntimeSourceProvisioningManifest, ServiceBootstrapError, StartupIdentityBundle,
    StructuralEthicsPolicy, TrustRegistry, TrustRole, TrustedKey, TrustedKeyStatus, WisdomConfig,
    fingerprint_bytes, run_production_startup,
};

#[derive(Debug, Clone)]
struct ToyKey {
    algorithm: String,
    key_id: String,
}

impl EvidenceSigner for ToyKey {
    type Error = Infallible;
    fn algorithm(&self) -> &str {
        &self.algorithm
    }
    fn key_id(&self) -> &str {
        &self.key_id
    }
    fn sign(&self, message: &[u8]) -> Result<Vec<u8>, Self::Error> {
        Ok(fingerprint_bytes(message).to_le_bytes().to_vec())
    }
}

impl EvidenceVerifier for ToyKey {
    type Error = Infallible;
    fn algorithm(&self) -> &str {
        &self.algorithm
    }
    fn key_id(&self) -> &str {
        &self.key_id
    }
    fn verify(&self, message: &[u8], signature: &[u8]) -> Result<bool, Self::Error> {
        Ok(signature == fingerprint_bytes(message).to_le_bytes())
    }
}

impl ArchiveSignatureVerifier for ToyKey {
    fn verify_signature(
        &self,
        algorithm: &str,
        key_id: &str,
        message: &[u8],
        signature: &[u8],
    ) -> Result<bool, String> {
        Ok(algorithm == self.algorithm
            && key_id == self.key_id
            && signature == fingerprint_bytes(message).to_le_bytes())
    }
}

fn backend() -> InMemoryAtomicProductionBackend {
    let mut registry = TrustRegistry::new();
    registry
        .register_initial(TrustedKey {
            role: TrustRole::RuntimeSource,
            algorithm: "toy".into(),
            key_id: "scheduler".into(),
            valid_from_millis: 0,
            valid_until_millis: None,
            status: TrustedKeyStatus::Active,
        })
        .unwrap();
    let provisioning = RuntimeSourceProvisioningManifest::new(
        1,
        1,
        &registry,
        vec![RuntimeSourceIdentity::new("scheduler", 1, "toy", "scheduler").unwrap()],
        vec![],
    )
    .unwrap();
    let identity = StartupIdentityBundle::new("deployment", 1, registry, provisioning).unwrap();
    InMemoryAtomicProductionBackend::new(EvidenceLedger::default(), identity, vec![]).unwrap()
}

struct RacingFactory {
    backend: InMemoryAtomicProductionBackend,
    raced: Arc<AtomicBool>,
}

impl RuntimeSignerFactory for RacingFactory {
    type Signer = ToyKey;
    type Error = Infallible;

    fn load_signer(&self, identity: &RuntimeSourceIdentity) -> Result<Self::Signer, Self::Error> {
        if !self.raced.swap(true, Ordering::SeqCst) {
            let current =
                LedgerStorageFrame::from_bytes(&self.backend.load_frame().unwrap()).unwrap();
            let mut ledger = current.ledger;
            ledger.append(
                2,
                "race-injection",
                0,
                EvidenceEvent::PredictionObserved {
                    ticket_id: 1,
                    actual_error: 0.2,
                },
            );
            let replacement = LedgerStorageFrame::new(ledger);
            let epoch = self.backend.acquire_fence("racing-writer").unwrap();
            assert_eq!(
                self.backend
                    .compare_exchange_frame(
                        "racing-writer",
                        epoch,
                        current.revision,
                        replacement.revision,
                        &replacement.to_bytes(),
                    )
                    .unwrap(),
                BackendCompareExchange::Committed
            );
        }
        Ok(ToyKey {
            algorithm: identity.algorithm.clone(),
            key_id: identity.key_id.clone(),
        })
    }
}

#[test]
fn durable_head_change_after_atomic_snapshot_fails_before_activation() {
    let backend = backend();
    let key = ToyKey {
        algorithm: "toy".into(),
        key_id: "scheduler".into(),
    };
    let result = run_production_startup(
        backend.clone(),
        &RacingFactory {
            backend,
            raced: Arc::new(AtomicBool::new(false)),
        },
        "deployment",
        "production-startup",
        WisdomConfig::default(),
        StructuralEthicsPolicy::default(),
        &key,
        &key,
        &key,
        OperationalStartupRequirements::default(),
        10,
    );
    assert!(matches!(
        result,
        Err(ProductionStartupError::Service(
            ServiceBootstrapError::Coordination(CoordinationError::RevisionConflict { .. })
        ))
    ));
}

#[test]
fn scheduler_readiness_occurs_before_service_activation() {
    let production_backend = backend();
    let key = ToyKey {
        algorithm: "toy".into(),
        key_id: "scheduler".into(),
    };
    let outcome = run_production_startup(
        production_backend,
        &RacingFactory {
            backend: backend(),
            raced: Arc::new(AtomicBool::new(true)),
        },
        "deployment",
        "production-startup",
        WisdomConfig::default(),
        StructuralEthicsPolicy::default(),
        &key,
        &key,
        &key,
        OperationalStartupRequirements::default(),
        10,
    )
    .unwrap();
    let readiness = outcome.runtime_signers[0].readiness();
    assert_eq!(readiness.observed_at_millis, 10);
    assert!(readiness.signature_length > 0);
    assert!(outcome.runtime_service().is_active());
}
