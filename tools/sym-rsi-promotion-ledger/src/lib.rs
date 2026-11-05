#![forbid(unsafe_code)]

// Reuse the exact production statistical implementation; this service crate
// is deliberately outside the self-modifying Symthaea core.
#[path = "../../../crates/domains/symthaea-fabrication-kernel/src/crypto_digest.rs"]
pub mod crypto_digest;
#[path = "../../../src/consciousness/recursive_improvement/paired_sign_test.rs"]
pub mod paired_sign_test;
#[path = "../../../src/consciousness/recursive_improvement/reuse_confidence_certificate.rs"]
pub mod reuse_confidence_certificate;
#[path = "../../../src/consciousness/recursive_improvement/reuse_error_budget.rs"]
pub mod reuse_error_budget;

pub mod ledger;

pub use ledger::{
    CandidateIdentity, CandidateSubmission, LedgerConfig, LedgerError, LedgerSigner,
    LedgerVerifier, PromotionLedger, PromotionLedgerReport, PromotionRoundDecision,
    MAX_LEDGER_ROUNDS,
};
