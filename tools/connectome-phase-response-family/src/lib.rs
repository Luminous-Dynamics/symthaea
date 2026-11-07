//! Exact finite-family anytime-FWER ledger for phase-response claims.
//!
//! This crate validates precommitted claim identity, fixed risk weights,
//! common-filtration commitments, e-process path replay and persistent
//! weighted-Ville threshold decisions. It does not prove that opaque upstream
//! e-process receipts are authentic or valid; those guarantees must be
//! independently established in the shared global filtration.

use connectome_asof_filtration::{
    AdmissionStatus, FiltrationAdmission, GlobalLedger, audit_admission, audit_ledger,
};
use std::cmp::Ordering;
use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;

/// Exact non-negative rational number. All arithmetic is checked.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Rational {
    numerator: u128,
    denominator: u128,
}

impl Rational {
    pub fn new(numerator: u128, denominator: u128) -> Result<Self, FamilyError> {
        if denominator == 0 {
            return Err(FamilyError("rational denominator is zero".into()));
        }
        let divisor = gcd(numerator, denominator);
        Ok(Self {
            numerator: numerator / divisor,
            denominator: denominator / divisor,
        })
    }

    pub const fn zero() -> Self {
        Self {
            numerator: 0,
            denominator: 1,
        }
    }

    pub const fn one() -> Self {
        Self {
            numerator: 1,
            denominator: 1,
        }
    }

    pub fn numerator(self) -> u128 {
        self.numerator
    }

    pub fn denominator(self) -> u128 {
        self.denominator
    }

    pub fn is_zero(self) -> bool {
        self.numerator == 0
    }

    pub fn checked_cmp(self, other: Self) -> Result<Ordering, FamilyError> {
        let left = self
            .numerator
            .checked_mul(other.denominator)
            .ok_or_else(|| FamilyError("rational comparison overflow".into()))?;
        let right = other
            .numerator
            .checked_mul(self.denominator)
            .ok_or_else(|| FamilyError("rational comparison overflow".into()))?;
        Ok(left.cmp(&right))
    }

    pub fn checked_add(self, other: Self) -> Result<Self, FamilyError> {
        let common = gcd(self.denominator, other.denominator);
        let left_factor = other.denominator / common;
        let right_factor = self.denominator / common;
        let left = self
            .numerator
            .checked_mul(left_factor)
            .ok_or_else(|| FamilyError("rational addition overflow".into()))?;
        let right = other
            .numerator
            .checked_mul(right_factor)
            .ok_or_else(|| FamilyError("rational addition overflow".into()))?;
        let numerator = left
            .checked_add(right)
            .ok_or_else(|| FamilyError("rational addition overflow".into()))?;
        let denominator = self
            .denominator
            .checked_mul(left_factor)
            .ok_or_else(|| FamilyError("rational denominator overflow".into()))?;
        Self::new(numerator, denominator)
    }

    pub fn checked_sub(self, other: Self) -> Result<Self, FamilyError> {
        if self.checked_cmp(other)? == Ordering::Less {
            return Err(FamilyError("rational subtraction underflow".into()));
        }
        let common = gcd(self.denominator, other.denominator);
        let left_factor = other.denominator / common;
        let right_factor = self.denominator / common;
        let left = self
            .numerator
            .checked_mul(left_factor)
            .ok_or_else(|| FamilyError("rational subtraction overflow".into()))?;
        let right = other
            .numerator
            .checked_mul(right_factor)
            .ok_or_else(|| FamilyError("rational subtraction overflow".into()))?;
        let denominator = self
            .denominator
            .checked_mul(left_factor)
            .ok_or_else(|| FamilyError("rational denominator overflow".into()))?;
        Self::new(left - right, denominator)
    }

    pub fn checked_mul(self, other: Self) -> Result<Self, FamilyError> {
        let cross_left = gcd(self.numerator, other.denominator);
        let cross_right = gcd(other.numerator, self.denominator);
        let numerator = (self.numerator / cross_left)
            .checked_mul(other.numerator / cross_right)
            .ok_or_else(|| FamilyError("rational multiplication overflow".into()))?;
        let denominator = (self.denominator / cross_right)
            .checked_mul(other.denominator / cross_left)
            .ok_or_else(|| FamilyError("rational denominator overflow".into()))?;
        Self::new(numerator, denominator)
    }

    pub fn checked_div(self, other: Self) -> Result<Self, FamilyError> {
        if other.numerator == 0 {
            return Err(FamilyError("division by zero rational".into()));
        }
        self.checked_mul(Self::new(other.denominator, other.numerator)?)
    }
}

impl fmt::Display for Rational {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.denominator == 1 {
            write!(formatter, "{}", self.numerator)
        } else {
            write!(formatter, "{}/{}", self.numerator, self.denominator)
        }
    }
}

fn gcd(mut left: u128, mut right: u128) -> u128 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left.max(1)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FamilyError(pub String);

impl fmt::Display for FamilyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for FamilyError {}

/// Directional biological measurement identity. Sender and recipient are
/// deliberately separate; reversing them creates a different claim.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct PhaseResponseClaimKey {
    pub sender_leg: String,
    pub recipient_leg: String,
    pub probe_class: String,
    pub descending_neuron: Option<String>,
    pub context: String,
    pub phase_bin: u8,
    pub magnitude_bin: u8,
    pub delay_index: u32,
    pub endpoint: String,
}

impl PhaseResponseClaimKey {
    fn validate(&self) -> Result<(), FamilyError> {
        const LEGS: [&str; 6] = [
            "left_fore",
            "left_middle",
            "left_hind",
            "right_fore",
            "right_middle",
            "right_hind",
        ];
        if !LEGS.contains(&self.sender_leg.as_str()) || !LEGS.contains(&self.recipient_leg.as_str())
        {
            return Err(FamilyError("claim key contains an unknown leg axis".into()));
        }
        for (label, value) in [
            ("probe_class", self.probe_class.as_str()),
            ("context", self.context.as_str()),
            ("endpoint", self.endpoint.as_str()),
        ] {
            if value.is_empty() || value.contains('|') {
                return Err(FamilyError(format!("invalid {label} in claim key")));
            }
        }
        if let Some(neuron) = &self.descending_neuron {
            if !matches!(neuron.as_str(), "DNg100" | "DNg97" | "DNb08") {
                return Err(FamilyError(
                    "unregistered descending-neuron identity".into(),
                ));
            }
        }
        if self.phase_bin >= 8 {
            return Err(FamilyError("phase bin must be in 0..8 for V1".into()));
        }
        if self.magnitude_bin >= 16 {
            return Err(FamilyError("magnitude bin must be in 0..16 for V1".into()));
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct ClaimEvidence {
    pub claim_id: String,
    pub key: PhaseResponseClaimKey,
    pub query_spec_sha256: String,
    pub null_envelope_sha256: String,
    pub component_e_process_receipt_sha256: String,
    pub component_validity_receipt_sha256: String,
    /// Exact digest for the raw input process bound by the as-of receipt.
    pub component_input_process_sha256: String,
    /// SHA-256 of the canonical, stage-ordered rational e-process path below.
    pub e_process_path_sha256: String,
    pub global_filtration_sha256: String,
    pub confirmation_split_sha256: String,
    /// Typed receipt consumed by the same as-of auditor used for ledger replay.
    pub filtration_admission: FiltrationAdmission,
    pub weight: Rational,
    /// E-process values E_0,...,E_T. Values may fall; the verifier computes
    /// their running supremum and never treats that supremum as an e-value.
    pub e_process_values: Vec<Rational>,
}

#[derive(Clone, Debug)]
pub struct FamilyManifest {
    pub family_id: String,
    pub campaign_id: String,
    pub requested_guarantee: String,
    pub registered_before_confirmatory_outcomes: bool,
    pub confirmation_lockbox_opened_before_freeze: bool,
    pub family_commitment_sha256: String,
    pub alpha: Rational,
    pub global_filtration_sha256: String,
    pub confirmation_split_sha256: String,
    /// Shared append-only ledger replayed by the as-of filtration auditor.
    pub global_ledger: GlobalLedger,
    pub claims: Vec<ClaimEvidence>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ClaimAudit {
    pub claim_id: String,
    pub allocated_alpha: Rational,
    pub threshold: Rational,
    pub running_maxima: Vec<Rational>,
    pub rejection_path: Vec<bool>,
    pub first_crossing_stage: Option<usize>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FamilyAudit {
    pub family_id: String,
    pub campaign_id: String,
    pub alpha: Rational,
    pub allocated_weight: Rational,
    pub unused_weight: Rational,
    pub family_fwer_bound: Rational,
    pub horizon: usize,
    pub claims: Vec<ClaimAudit>,
    pub guarantee: &'static str,
    pub upstream_validity: &'static str,
}

fn valid_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

// Small dependency-free SHA-256 implementation. The canonical path encoding
// is versioned and exact, so the as-of admission output is bound to the actual
// sequence of reduced rational values consumed by the statistical engine.
fn sha256_hex(input: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
        0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
        0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
        0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
        0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
        0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
        0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
        0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    ];
    let mut state = [
        0x6a09e667u32, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    ];
    let mut message = input.to_vec();
    let bit_len = (message.len() as u64).wrapping_mul(8);
    message.push(0x80);
    while message.len() % 64 != 56 {
        message.push(0);
    }
    message.extend_from_slice(&bit_len.to_be_bytes());

    for block in message.chunks_exact(64) {
        let mut words = [0u32; 64];
        for (index, chunk) in block.chunks_exact(4).take(16).enumerate() {
            words[index] = u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        for index in 16..64 {
            let s0 = words[index - 15].rotate_right(7)
                ^ words[index - 15].rotate_right(18)
                ^ (words[index - 15] >> 3);
            let s1 = words[index - 2].rotate_right(17)
                ^ words[index - 2].rotate_right(19)
                ^ (words[index - 2] >> 10);
            words[index] = words[index - 16]
                .wrapping_add(s0)
                .wrapping_add(words[index - 7])
                .wrapping_add(s1);
        }

        let [mut a, mut b, mut cc, mut d, mut e, mut f, mut g, mut h] = state;
        for index in 0..64 {
            let big1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let choose = (e & f) ^ ((!e) & g);
            let temp1 = h
                .wrapping_add(big1)
                .wrapping_add(choose)
                .wrapping_add(K[index])
                .wrapping_add(words[index]);
            let big0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let majority = (a & b) ^ (a & cc) ^ (b & cc);
            let temp2 = big0.wrapping_add(majority);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = cc;
            cc = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }
        for (slot, value) in state.iter_mut().zip([a, b, cc, d, e, f, g, h]) {
            *slot = slot.wrapping_add(value);
        }
    }
    state.iter().map(|word| format!("{word:08x}")).collect()
}

pub fn canonical_e_process_path_sha256(values: &[Rational]) -> String {
    let mut encoded = b"symthaea-eprocess-path-v1\n".to_vec();
    for value in values {
        encoded.extend_from_slice(value.to_string().as_bytes());
        encoded.push(b'\n');
    }
    sha256_hex(&encoded)
}

pub fn audit_family(manifest: &FamilyManifest) -> Result<FamilyAudit, FamilyError> {
    let ledger_audit = audit_ledger(&manifest.global_ledger)
        .map_err(|error| FamilyError(format!("global as-of ledger rejected: {error}")))?;
    if ledger_audit.campaign_id != manifest.campaign_id
        || ledger_audit.global_filtration_sha256 != manifest.global_filtration_sha256
    {
        return Err(FamilyError(
            "as-of ledger campaign or global filtration does not match family".into(),
        ));
    }
    if manifest.family_id.is_empty() || manifest.campaign_id.is_empty() {
        return Err(FamilyError(
            "family_id and campaign_id must be non-empty".into(),
        ));
    }
    if manifest.requested_guarantee != "FWER" {
        return Err(FamilyError(
            "V1 supports FWER only; FDR-sup requires an adjusted running-maximum procedure".into(),
        ));
    }
    if !manifest.registered_before_confirmatory_outcomes
        || manifest.confirmation_lockbox_opened_before_freeze
    {
        return Err(FamilyError(
            "confirmatory family was not frozen before lockbox outcomes".into(),
        ));
    }
    if !valid_digest(&manifest.family_commitment_sha256)
        || !valid_digest(&manifest.global_filtration_sha256)
        || !valid_digest(&manifest.confirmation_split_sha256)
    {
        return Err(FamilyError(
            "malformed family, filtration or split digest".into(),
        ));
    }
    if manifest.alpha.is_zero() || manifest.alpha.checked_cmp(Rational::one())? == Ordering::Greater
    {
        return Err(FamilyError("alpha must be in (0,1]".into()));
    }
    if manifest.claims.is_empty() {
        return Err(FamilyError("empty confirmatory claim family".into()));
    }

    let mut ids = BTreeSet::new();
    let mut keys = BTreeSet::new();
    let mut allocated_weight = Rational::zero();
    let horizon = manifest.claims[0]
        .e_process_values
        .len()
        .checked_sub(1)
        .ok_or_else(|| FamilyError("empty e-process path".into()))?;
    if horizon == 0 {
        return Err(FamilyError(
            "V1 requires at least one evidence update".into(),
        ));
    }

    let mut audits = Vec::with_capacity(manifest.claims.len());
    for claim in &manifest.claims {
        if claim.claim_id.is_empty() || !ids.insert(claim.claim_id.as_str()) {
            return Err(FamilyError("empty or duplicate claim_id".into()));
        }
        claim.key.validate()?;
        if !keys.insert(claim.key.clone()) {
            return Err(FamilyError("duplicate phase-response claim key".into()));
        }
        for (label, digest) in [
            ("query specification", claim.query_spec_sha256.as_str()),
            ("null envelope", claim.null_envelope_sha256.as_str()),
            (
                "component e-process receipt",
                claim.component_e_process_receipt_sha256.as_str(),
            ),
            (
                "component validity receipt",
                claim.component_validity_receipt_sha256.as_str(),
            ),
            (
                "component input process",
                claim.component_input_process_sha256.as_str(),
            ),
            (
                "canonical e-process path",
                claim.e_process_path_sha256.as_str(),
            ),
            ("claim filtration", claim.global_filtration_sha256.as_str()),
            (
                "claim confirmation split",
                claim.confirmation_split_sha256.as_str(),
            ),
        ] {
            if !valid_digest(digest) {
                return Err(FamilyError(format!(
                    "malformed {label} digest for {}",
                    claim.claim_id
                )));
            }
        }
        if claim.global_filtration_sha256 != manifest.global_filtration_sha256 {
            return Err(FamilyError(
                "claim e-process receipt is not bound to the common global filtration".into(),
            ));
        }
        if claim.confirmation_split_sha256 != manifest.confirmation_split_sha256 {
            return Err(FamilyError(
                "claim is bound to a different confirmation split".into(),
            ));
        }
        let admission = &claim.filtration_admission;
        let admission_audit = audit_admission(&manifest.global_ledger, admission);
        if !matches!(
            admission_audit.status,
            AdmissionStatus::AdmittedDirectGlobal | AdmissionStatus::AdmittedAfterLift
        ) {
            return Err(FamilyError(format!(
                "{} is not admitted to the family filtration: {:?}",
                claim.claim_id, admission_audit.status
            )));
        }
        if admission.campaign_id != manifest.campaign_id
            || admission.process_id != claim.claim_id
            || admission.null_spec_sha256 != claim.null_envelope_sha256
            || admission.input_process_sha256 != claim.component_input_process_sha256
            || admission.output_process_sha256 != claim.e_process_path_sha256
            || admission.target_filtration_sha256 != manifest.global_filtration_sha256
            || admission.proof_receipt_sha256.as_deref()
                != Some(claim.component_validity_receipt_sha256.as_str())
        {
            return Err(FamilyError(format!(
                "{} filtration admission is not bound to this exact claim/null/process/path/proof",
                claim.claim_id
            )));
        }
        if canonical_e_process_path_sha256(&claim.e_process_values)
            != claim.e_process_path_sha256
        {
            return Err(FamilyError(format!(
                "{} e-process path digest does not match its exact rational path",
                claim.claim_id
            )));
        }
        if claim.weight.is_zero() {
            return Err(FamilyError(
                "confirmatory claim has zero allocated weight".into(),
            ));
        }
        if claim.e_process_values.len() != horizon + 1 {
            return Err(FamilyError(
                "e-process paths do not share one horizon".into(),
            ));
        }
        if claim.e_process_values[0] != Rational::one() {
            return Err(FamilyError(
                "every component e-process must start at one".into(),
            ));
        }

        allocated_weight = allocated_weight.checked_add(claim.weight)?;
        let allocated_alpha = manifest.alpha.checked_mul(claim.weight)?;
        let threshold = Rational::one().checked_div(allocated_alpha)?;
        let mut running_max = Rational::zero();
        let mut running_maxima = Vec::with_capacity(horizon + 1);
        let mut rejection_path = Vec::with_capacity(horizon + 1);
        let mut first_crossing_stage = None;
        for (stage, current_value) in claim.e_process_values.iter().copied().enumerate() {
            if current_value.checked_cmp(running_max)? == Ordering::Greater {
                running_max = current_value;
            }
            running_maxima.push(running_max);
            let rejected = running_max.checked_cmp(threshold)? != Ordering::Less;
            if rejected && first_crossing_stage.is_none() {
                first_crossing_stage = Some(stage);
            }
            rejection_path.push(rejected);
        }
        audits.push(ClaimAudit {
            claim_id: claim.claim_id.clone(),
            allocated_alpha,
            threshold,
            running_maxima,
            rejection_path,
            first_crossing_stage,
        });
    }

    if allocated_weight.checked_cmp(Rational::one())? == Ordering::Greater {
        return Err(FamilyError("precommitted family weights exceed one".into()));
    }
    let unused_weight = Rational::one().checked_sub(allocated_weight)?;
    let family_fwer_bound = manifest.alpha.checked_mul(allocated_weight)?;

    Ok(FamilyAudit {
        family_id: manifest.family_id.clone(),
        campaign_id: manifest.campaign_id.clone(),
        alpha: manifest.alpha,
        allocated_weight,
        unused_weight,
        family_fwer_bound,
        horizon,
        claims: audits,
        guarantee: "ANYTIME_FWER_WEIGHTED_VILLE_UNION_BOUND",
        upstream_validity: "COMPONENT_GLOBAL_FILTRATION_VALIDITY_ASSUMED_FROM_BOUND_RECEIPTS",
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(numerator: u128, denominator: u128) -> Rational {
        Rational::new(numerator, denominator).expect("valid rational")
    }

    fn digest(byte: char) -> String {
        std::iter::repeat_n(byte, 64).collect()
    }

    fn claim(
        id: &str,
        sender: &str,
        recipient: &str,
        context: &str,
        weight: Rational,
        path: &[Rational],
    ) -> ClaimEvidence {
        ClaimEvidence {
            claim_id: id.into(),
            key: PhaseResponseClaimKey {
                sender_leg: sender.into(),
                recipient_leg: recipient.into(),
                probe_class: "LOCAL_PHASE_KICK".into(),
                descending_neuron: None,
                context: context.into(),
                phase_bin: 2,
                magnitude_bin: 1,
                delay_index: 3,
                endpoint: "PHASE_RESET".into(),
            },
            query_spec_sha256: digest('a'),
            null_envelope_sha256: digest('b'),
            component_e_process_receipt_sha256: digest('c'),
            component_validity_receipt_sha256: digest('d'),
            global_filtration_sha256: digest('e'),
            confirmation_split_sha256: digest('f'),
            component_status: "VALIDATED_GLOBAL_FILTRATION_EPROCESS".into(),
            weight,
            e_process_values: path.into(),
        }
    }

    fn fixture() -> FamilyManifest {
        FamilyManifest {
            family_id: "phase-response-confirmatory-family-v1".into(),
            campaign_id: "synthetic-phase-response-campaign".into(),
            requested_guarantee: "FWER".into(),
            registered_before_confirmatory_outcomes: true,
            confirmation_lockbox_opened_before_freeze: false,
            family_commitment_sha256: digest('1'),
            alpha: r(1, 10),
            global_filtration_sha256: digest('e'),
            confirmation_split_sha256: digest('f'),
            claims: vec![
                claim(
                    "C1-local-reset",
                    "left_fore",
                    "left_fore",
                    "BALL_CONTACT",
                    r(1, 2),
                    &[r(1, 1), r(5, 1), r(25, 1)],
                ),
                claim(
                    "C2-off-diagonal-transfer",
                    "left_fore",
                    "right_fore",
                    "BALL_CONTACT",
                    r(1, 4),
                    &[r(1, 1), r(10, 1), r(50, 1)],
                ),
                claim(
                    "C3-contact-recovery",
                    "left_fore",
                    "right_fore",
                    "SLIPPERY_SURFACE",
                    r(1, 4),
                    &[r(1, 1), r(2, 1), r(12, 1)],
                ),
            ],
        }
    }

    #[test]
    fn exact_weighted_fwer_known_answer() {
        let report = audit_family(&fixture()).expect("fixture valid");
        assert_eq!(report.allocated_weight, r(1, 1));
        assert_eq!(report.unused_weight, r(0, 1));
        assert_eq!(report.family_fwer_bound, r(1, 10));
        assert_eq!(report.horizon, 2);
        assert_eq!(report.claims[0].threshold, r(20, 1));
        assert_eq!(report.claims[1].threshold, r(40, 1));
        assert_eq!(report.claims[2].threshold, r(40, 1));
        assert_eq!(report.claims[0].first_crossing_stage, Some(2));
        assert_eq!(report.claims[1].first_crossing_stage, Some(2));
        assert_eq!(report.claims[2].first_crossing_stage, None);
        assert_eq!(report.claims[0].rejection_path, vec![false, false, true]);
        assert_eq!(report.claims[1].rejection_path, vec![false, false, true]);
        assert_eq!(report.claims[2].rejection_path, vec![false, false, false]);
    }

    #[test]
    fn rejection_is_persistent_when_raw_e_process_falls() {
        let mut manifest = fixture();
        manifest.claims[0].e_process_values = vec![r(1, 1), r(25, 1), r(3, 1)];
        let report = audit_family(&manifest).expect("nonmonotone e-process path is valid");
        assert_eq!(
            report.claims[0].running_maxima,
            vec![r(1, 1), r(25, 1), r(25, 1)]
        );
        assert_eq!(report.claims[0].rejection_path, vec![false, true, true]);
    }

    #[test]
    fn unused_weight_is_not_reallocated() {
        let mut manifest = fixture();
        manifest.claims.pop();
        let report = audit_family(&manifest).expect("partial weight allocation is valid");
        assert_eq!(report.allocated_weight, r(3, 4));
        assert_eq!(report.unused_weight, r(1, 4));
        assert_eq!(report.family_fwer_bound, r(3, 40));
    }

    #[test]
    fn rejects_weight_overallocation() {
        let mut manifest = fixture();
        manifest.claims[2].weight = r(1, 2);
        assert!(audit_family(&manifest).is_err());
    }

    #[test]
    fn rejects_duplicate_claim_ids() {
        let mut manifest = fixture();
        manifest.claims[1].claim_id = manifest.claims[0].claim_id.clone();
        assert!(audit_family(&manifest).is_err());
    }

    #[test]
    fn rejects_inconsistent_path_horizon() {
        let mut manifest = fixture();
        manifest.claims[1].e_process_values.pop();
        assert!(audit_family(&manifest).is_err());
    }

    #[test]
    fn rejects_e_process_not_started_at_one() {
        let mut manifest = fixture();
        manifest.claims[1].e_process_values[0] = r(2, 1);
        assert!(audit_family(&manifest).is_err());
    }

    #[test]
    fn rejects_zero_weight_claim() {
        let mut manifest = fixture();
        manifest.claims[2].weight = Rational::zero();
        assert!(audit_family(&manifest).is_err());
    }

    #[test]
    fn rejects_malformed_receipt_digest() {
        let mut manifest = fixture();
        manifest.claims[0].component_validity_receipt_sha256 = "not-a-digest".into();
        assert!(audit_family(&manifest).is_err());
    }

    #[test]
    fn rejects_mixed_global_filtrations() {
        let mut manifest = fixture();
        manifest.claims[1].global_filtration_sha256 = digest('9');
        assert!(audit_family(&manifest).is_err());
    }

    #[test]
    fn rejects_family_registered_after_lockbox_open() {
        let mut manifest = fixture();
        manifest.confirmation_lockbox_opened_before_freeze = true;
        assert!(audit_family(&manifest).is_err());
    }

    #[test]
    fn rejects_fdr_sup_request_on_fwer_only_verifier() {
        let mut manifest = fixture();
        manifest.requested_guarantee = "FDR_SUP".into();
        assert!(audit_family(&manifest).is_err());
    }

    #[test]
    fn rejects_directional_key_collision() {
        let mut manifest = fixture();
        manifest.claims[1].key.sender_leg = "right_fore".into();
        manifest.claims[1].key.recipient_leg = "left_fore".into();
        manifest.claims[2].key = manifest.claims[1].key.clone();
        assert!(audit_family(&manifest).is_err());
    }

    #[test]
    fn rejects_non_global_component_status() {
        let mut manifest = fixture();
        manifest.claims[0].component_status = "LOCAL_FILTRATION_ONLY".into();
        assert!(audit_family(&manifest).is_err());
    }

    #[test]
    fn rational_arithmetic_is_exact() {
        assert_eq!(r(1, 2).checked_add(r(1, 4)).unwrap(), r(3, 4));
        assert_eq!(r(1, 10).checked_mul(r(1, 2)).unwrap(), r(1, 20));
        assert_eq!(r(1, 1).checked_div(r(1, 20)).unwrap(), r(20, 1));
        assert_eq!(r(1, 1).checked_sub(r(3, 4)).unwrap(), r(1, 4));
    }
}
