//! Exact closed-testing procedure for finite phase-response e-process families.
//!
//! For each nonempty intersection I of elementary null hypotheses, V1 forms
//! a fixed convex mixture of the elementary e-processes in I. Under H_I all
//! constituent nulls are true, so the mixture is a test supermartingale in
//! the common global filtration. Closed testing rejects H_i only after every
//! intersection containing i has crossed its Ville threshold.
//!
//! The crate assumes upstream elementary e-process validity. It checks shared
//! filtration/split commitments and exact family semantics; it does not
//! authenticate source receipts or prove those upstream processes valid.

use connectome_asof_filtration::{
    AdmissionMode, AdmissionStatus, AdjusterReceipt, AvailabilityState, DecisionAuditStatus,
    EvidenceEvent, EvidenceSplit, FiltrationAdmission, GlobalLedger, LedgerEntry, LedgerPayload,
    audit_admission, audit_ledger,
};
use std::cmp::Ordering;
use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;

/// Exact non-negative rational number with checked u128 arithmetic.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Rational {
    numerator: u128,
    denominator: u128,
}

impl Rational {
    pub fn new(numerator: u128, denominator: u128) -> Result<Self, ClosedError> {
        if denominator == 0 {
            return Err(ClosedError("rational denominator is zero".into()));
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

    pub fn checked_cmp(self, other: Self) -> Result<Ordering, ClosedError> {
        let left = self
            .numerator
            .checked_mul(other.denominator)
            .ok_or_else(|| ClosedError("rational comparison overflow".into()))?;
        let right = other
            .numerator
            .checked_mul(self.denominator)
            .ok_or_else(|| ClosedError("rational comparison overflow".into()))?;
        Ok(left.cmp(&right))
    }

    pub fn checked_add(self, other: Self) -> Result<Self, ClosedError> {
        let common = gcd(self.denominator, other.denominator);
        let left_factor = other.denominator / common;
        let right_factor = self.denominator / common;
        let left = self
            .numerator
            .checked_mul(left_factor)
            .ok_or_else(|| ClosedError("rational addition overflow".into()))?;
        let right = other
            .numerator
            .checked_mul(right_factor)
            .ok_or_else(|| ClosedError("rational addition overflow".into()))?;
        let numerator = left
            .checked_add(right)
            .ok_or_else(|| ClosedError("rational addition overflow".into()))?;
        let denominator = self
            .denominator
            .checked_mul(left_factor)
            .ok_or_else(|| ClosedError("rational denominator overflow".into()))?;
        Self::new(numerator, denominator)
    }

    pub fn checked_mul(self, other: Self) -> Result<Self, ClosedError> {
        let cross_left = gcd(self.numerator, other.denominator);
        let cross_right = gcd(other.numerator, self.denominator);
        let numerator = (self.numerator / cross_left)
            .checked_mul(other.numerator / cross_right)
            .ok_or_else(|| ClosedError("rational multiplication overflow".into()))?;
        let denominator = (self.denominator / cross_right)
            .checked_mul(other.denominator / cross_left)
            .ok_or_else(|| ClosedError("rational denominator overflow".into()))?;
        Self::new(numerator, denominator)
    }

    pub fn checked_div(self, other: Self) -> Result<Self, ClosedError> {
        if other.numerator == 0 {
            return Err(ClosedError("division by zero rational".into()));
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
pub struct ClosedError(pub String);

impl fmt::Display for ClosedError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for ClosedError {}

#[derive(Clone, Debug)]
pub struct ElementaryClaim {
    pub claim_id: String,
    pub null_spec_sha256: String,
    pub global_filtration_sha256: String,
    pub confirmation_split_sha256: String,
    pub component_validity_receipt_sha256: String,
    pub component_input_process_sha256: String,
    pub e_process_path_sha256: String,
    pub filtration_admission: FiltrationAdmission,
    /// E_0,...,E_T. Every process must start at one.
    pub e_process_values: Vec<Rational>,
}

#[derive(Clone, Debug)]
pub struct ClosedFamily {
    pub family_id: String,
    pub campaign_id: String,
    pub global_ledger: GlobalLedger,
    pub requested_guarantee: String,
    pub registered_before_confirmatory_outcomes: bool,
    pub confirmation_lockbox_opened_before_freeze: bool,
    pub family_commitment_sha256: String,
    pub global_filtration_sha256: String,
    pub confirmation_split_sha256: String,
    pub alpha: Rational,
    pub claims: Vec<ElementaryClaim>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IntersectionAudit {
    /// Canonical sorted elementary claim IDs in this intersection.
    pub claim_ids: Vec<String>,
    /// E_I,t, the fixed equal-weight mixture of elementary e-processes.
    pub mixture_e_process_values: Vec<Rational>,
    /// Running supremum of E_I,t; this is not itself described as an e-value.
    pub running_suprema: Vec<Rational>,
    pub rejection_path: Vec<bool>,
    pub first_crossing_stage: Option<usize>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ElementaryAudit {
    pub claim_id: String,
    /// Number of complete intersection hypotheses that contain this claim.
    pub required_intersection_count: usize,
    /// First stage at which every required intersection has crossed.
    pub first_rejection_stage: Option<usize>,
    pub rejection_path: Vec<bool>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ClosedAudit {
    pub family_id: String,
    pub alpha: Rational,
    pub threshold: Rational,
    pub horizon: usize,
    pub elementary_claim_count: usize,
    pub intersection_count: usize,
    pub intersections: Vec<IntersectionAudit>,
    pub elementary: Vec<ElementaryAudit>,
    pub family_fwer_bound: Rational,
    pub guarantee: &'static str,
    pub upstream_validity: &'static str,
}

fn valid_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

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
    let mut encoded = b"symthaea-eprocess-path-v1\\n".to_vec();
    for value in values {
        encoded.extend_from_slice(value.to_string().as_bytes());
        encoded.push(b'\\n');
    }
    sha256_hex(&encoded)
}

fn validate_confirmatory_admission_scope(
    ledger: &GlobalLedger,
    observed_event_ids: &[String],
    family_id: &str,
    family_commitment_sha256: &str,
) -> Result<(), ClosedError> {
    let observed: BTreeSet<&str> = observed_event_ids.iter().map(String::as_str).collect();
    let mut matching_freeze_seen = false;
    for entry in &ledger.entries {
        match &entry.payload {
            LedgerPayload::FamilyFreeze(freeze) if freeze.family_id == family_id => {
                if freeze.family_commitment_sha256 != family_commitment_sha256 {
                    return Err(ClosedError(
                        "confirmatory family freeze commitment does not match manifest".into(),
                    ));
                }
                matching_freeze_seen = true;
            }
            LedgerPayload::Evidence(event)
                if observed.contains(event.event_id.as_str())
                    && event.split == EvidenceSplit::Confirmation =>
            {
                if !matching_freeze_seen
                    || event.confirmatory_family_id.as_deref() != Some(family_id)
                {
                    return Err(ClosedError(
                        "claim admission consumes confirmation evidence outside its frozen family"
                            .into(),
                    ));
                }
            }
            _ => {}
        }
    }
    Ok(())
}

/// Audit complete closed testing over all nonempty subsets (at most 8 claims).
pub fn audit_closed_family(family: &ClosedFamily) -> Result<ClosedAudit, ClosedError> {
    let ledger_audit = audit_ledger(&family.global_ledger)
        .map_err(|error| ClosedError(format!("global as-of ledger rejected: {error}")))?;
    if ledger_audit.campaign_id != family.campaign_id
        || ledger_audit.global_filtration_sha256 != family.global_filtration_sha256
    {
        return Err(ClosedError(
            "as-of ledger campaign or global filtration does not match closed family".into(),
        ));
    }
    if ledger_audit
        .decisions
        .iter()
        .any(|decision| decision.status != DecisionAuditStatus::Accepted)
    {
        return Err(ClosedError(
            "global as-of ledger contains a rejected adaptive decision prefix".into(),
        ));
    }
    if family.family_id.is_empty() || family.campaign_id.is_empty() {
        return Err(ClosedError("family_id must be non-empty".into()));
    }
    if family.requested_guarantee != "FWER" {
        return Err(ClosedError(
            "V1 supports closed FWER only; FDR/FDR-sup is a different guarantee".into(),
        ));
    }
    if !family.registered_before_confirmatory_outcomes
        || family.confirmation_lockbox_opened_before_freeze
    {
        return Err(ClosedError(
            "confirmatory family was not frozen before lockbox outcomes".into(),
        ));
    }
    for (label, digest) in [
        (
            "family commitment",
            family.family_commitment_sha256.as_str(),
        ),
        (
            "global filtration",
            family.global_filtration_sha256.as_str(),
        ),
        (
            "confirmation split",
            family.confirmation_split_sha256.as_str(),
        ),
    ] {
        if !valid_digest(digest) {
            return Err(ClosedError(format!("malformed {label} digest")));
        }
    }
    if family.alpha.is_zero() || family.alpha.checked_cmp(Rational::one())? == Ordering::Greater {
        return Err(ClosedError("alpha must lie in (0,1]".into()));
    }
    let claim_count = family.claims.len();
    if claim_count == 0 {
        return Err(ClosedError("empty elementary claim family".into()));
    }
    if claim_count > 8 {
        return Err(ClosedError(
            "NOT_SUPPORTED: V1 complete closure is capped at 8 elementary claims".into(),
        ));
    }

    let horizon = family.claims[0]
        .e_process_values
        .len()
        .checked_sub(1)
        .ok_or_else(|| ClosedError("empty e-process path".into()))?;
    if horizon == 0 {
        return Err(ClosedError(
            "V1 requires at least one evidence update".into(),
        ));
    }

    let mut ids = BTreeSet::new();
    for claim in &family.claims {
        if claim.claim_id.is_empty() || !ids.insert(claim.claim_id.as_str()) {
            return Err(ClosedError("empty or duplicate elementary claim ID".into()));
        }
        for (label, digest) in [
            ("null specification", claim.null_spec_sha256.as_str()),
            (
                "claim global filtration",
                claim.global_filtration_sha256.as_str(),
            ),
            (
                "claim confirmation split",
                claim.confirmation_split_sha256.as_str(),
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
        ] {
            if !valid_digest(digest) {
                return Err(ClosedError(format!(
                    "malformed {label} digest for {}",
                    claim.claim_id
                )));
            }
        }
        if claim.global_filtration_sha256 != family.global_filtration_sha256 {
            return Err(ClosedError(format!(
                "{} is not bound to the common global filtration",
                claim.claim_id
            )));
        }
        if claim.confirmation_split_sha256 != family.confirmation_split_sha256 {
            return Err(ClosedError(format!(
                "{} is not bound to the common confirmation split",
                claim.claim_id
            )));
        }
        let admission = &claim.filtration_admission;
        validate_confirmatory_admission_scope(
            &family.global_ledger,
            &admission.observed_event_ids,
            &family.family_id,
            &family.family_commitment_sha256,
        )?;
        let admission_audit = audit_admission(&family.global_ledger, admission);
        if !matches!(
            admission_audit.status,
            AdmissionStatus::AdmittedDirectGlobal | AdmissionStatus::AdmittedAfterLift
        ) {
            return Err(ClosedError(format!(
                "{} is not admitted to the closed family filtration: {:?}",
                claim.claim_id, admission_audit.status
            )));
        }
        if admission.campaign_id != family.campaign_id
            || admission.process_id != claim.claim_id
            || admission.null_spec_sha256 != claim.null_spec_sha256
            || admission.input_process_sha256 != claim.component_input_process_sha256
            || admission.output_process_sha256 != claim.e_process_path_sha256
            || admission.target_filtration_sha256 != family.global_filtration_sha256
            || admission.proof_receipt_sha256.as_deref()
                != Some(claim.component_validity_receipt_sha256.as_str())
        {
            return Err(ClosedError(format!(
                "{} filtration admission is not bound to this exact null/process/path/proof",
                claim.claim_id
            )));
        }
        if canonical_e_process_path_sha256(&claim.e_process_values)
            != claim.e_process_path_sha256
        {
            return Err(ClosedError(format!(
                "{} e-process path digest does not match its exact rational path",
                claim.claim_id
            )));
        }
        if claim.e_process_values.len() != horizon + 1 {
            return Err(ClosedError(
                "elementary e-processes do not share one horizon".into(),
            ));
        }
        if claim.e_process_values[0] != Rational::one() {
            return Err(ClosedError(format!(
                "{}: elementary e-process must start at one",
                claim.claim_id
            )));
        }
    }

    let threshold = Rational::one().checked_div(family.alpha)?;
    let intersection_count = (1usize << claim_count) - 1;
    let mut intersections = Vec::with_capacity(intersection_count);
    let mut rejection_by_mask = vec![Vec::new(); intersection_count + 1];

    // Enumerate all nonempty subsets. For each I, E_I is the fixed equal
    // mixture of {E_i : i in I}; the weights do not depend on observed data.
    for mask in 1..=intersection_count {
        let member_indices: Vec<usize> = (0..claim_count)
            .filter(|index| (mask & (1usize << index)) != 0)
            .collect();
        let mixture_weight =
            Rational::one().checked_div(Rational::new(member_indices.len() as u128, 1)?)?;
        let mut mixture_path = Vec::with_capacity(horizon + 1);
        let mut running_suprema = Vec::with_capacity(horizon + 1);
        let mut rejection_path = Vec::with_capacity(horizon + 1);
        let mut running_supremum = Rational::zero();
        let mut first_crossing_stage = None;

        for stage in 0..=horizon {
            let mut mixture_value = Rational::zero();
            for index in &member_indices {
                mixture_value = mixture_value.checked_add(
                    family.claims[*index].e_process_values[stage].checked_mul(mixture_weight)?,
                )?;
            }
            mixture_path.push(mixture_value);
            if mixture_value.checked_cmp(running_supremum)? == Ordering::Greater {
                running_supremum = mixture_value;
            }
            running_suprema.push(running_supremum);
            let crossed = running_supremum.checked_cmp(threshold)? != Ordering::Less;
            if crossed && first_crossing_stage.is_none() {
                first_crossing_stage = Some(stage);
            }
            rejection_path.push(crossed);
        }

        rejection_by_mask[mask] = rejection_path.clone();
        let claim_ids = member_indices
            .iter()
            .map(|index| family.claims[*index].claim_id.clone())
            .collect();
        intersections.push(IntersectionAudit {
            claim_ids,
            mixture_e_process_values: mixture_path,
            running_suprema,
            rejection_path,
            first_crossing_stage,
        });
    }

    // Closed testing: H_i is rejected only if every nonempty intersection
    // containing i has crossed. This yields a persistent elementary decision.
    let mut elementary = Vec::with_capacity(claim_count);
    for index in 0..claim_count {
        let required_masks: Vec<usize> = (1..=intersection_count)
            .filter(|mask| (mask & (1usize << index)) != 0)
            .collect();
        let mut rejection_path = vec![false; horizon + 1];
        let mut first_rejection_stage = None;
        for stage in 0..=horizon {
            let all_crossed = required_masks
                .iter()
                .all(|mask| rejection_by_mask[*mask][stage]);
            rejection_path[stage] = all_crossed;
            if all_crossed && first_rejection_stage.is_none() {
                first_rejection_stage = Some(stage);
            }
        }
        elementary.push(ElementaryAudit {
            claim_id: family.claims[index].claim_id.clone(),
            required_intersection_count: required_masks.len(),
            first_rejection_stage,
            rejection_path,
        });
    }

    Ok(ClosedAudit {
        family_id: family.family_id.clone(),
        alpha: family.alpha,
        threshold,
        horizon,
        elementary_claim_count: claim_count,
        intersection_count,
        intersections,
        elementary,
        family_fwer_bound: family.alpha,
        guarantee: "CLOSED_ANYTIME_FWER_INTERSECTION_EPROCESS",
        upstream_validity: "ELEMENTARY_GLOBAL_FILTRATION_EPROCESS_VALIDITY_ASSUMED",
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(n: u128, d: u128) -> Rational {
        Rational::new(n, d).unwrap()
    }

    fn digest(byte: char) -> String {
        std::iter::repeat_n(byte, 64).collect()
    }

    fn global_ledger() -> GlobalLedger {
        GlobalLedger {
            campaign_id: "synthetic-closed-phase-response-campaign".into(),
            global_filtration_id: "connectome-global-v1".into(),
            global_filtration_sha256: digest('2'),
            entries: vec![
                LedgerEntry {
                    sequence: 1,
                    entry_commitment_sha256: digest('9'),
                    payload: LedgerPayload::Evidence(EvidenceEvent {
                        event_id: "S1".into(),
                        evidence_family_id: "structural-revision-s1".into(),
                        source_stream: "structural-em".into(),
                        source_artifact_sha256: digest('a'),
                        revision: 0,
                        supersedes_event_id: None,
                        event_time_ms: Some(100),
                        acquired_at_ms: Some(101),
                        annotated_at_ms: Some(102),
                        qc_at_ms: Some(103),
                        availability_state: AvailabilityState::Approved,
                        split: EvidenceSplit::Fit,
                        confirmatory_family_id: None,
                    }),
                },
                LedgerEntry {
                    sequence: 2,
                    entry_commitment_sha256: digest('8'),
                    payload: LedgerPayload::Evidence(EvidenceEvent {
                        event_id: "T7".into(),
                        evidence_family_id: "behaviour-trial-7".into(),
                        source_stream: "behaviour-kinematics".into(),
                        source_artifact_sha256: digest('b'),
                        revision: 0,
                        supersedes_event_id: None,
                        event_time_ms: Some(90),
                        acquired_at_ms: Some(91),
                        annotated_at_ms: Some(110),
                        qc_at_ms: Some(111),
                        availability_state: AvailabilityState::Approved,
                        split: EvidenceSplit::Exploration,
                        confirmatory_family_id: None,
                    }),
                },
            ],
        }
    }

    fn claim(id: &str, null_byte: char, path: &[Rational]) -> ElementaryClaim {
        let e_process_values = path.to_vec();
        let e_process_path_sha256 = canonical_e_process_path_sha256(&e_process_values);
        ElementaryClaim {
            claim_id: id.into(),
            null_spec_sha256: digest(null_byte),
            global_filtration_sha256: digest('2'),
            confirmation_split_sha256: digest('3'),
            component_validity_receipt_sha256: digest('d'),
            component_input_process_sha256: digest('1'),
            e_process_path_sha256: e_process_path_sha256.clone(),
            filtration_admission: FiltrationAdmission {
                campaign_id: "synthetic-closed-phase-response-campaign".into(),
                process_id: id.into(),
                null_spec_sha256: digest(null_byte),
                input_process_sha256: digest('1'),
                output_process_sha256: e_process_path_sha256,
                source_filtration_id: "connectome-global-v1".into(),
                source_filtration_sha256: digest('2'),
                target_filtration_id: "connectome-global-v1".into(),
                target_filtration_sha256: digest('2'),
                observed_event_ids: vec!["S1".into()],
                process_output_sequence: 2,
                mode: AdmissionMode::DirectGlobalSupermartingale,
                proof_receipt_sha256: Some(digest('d')),
                assumption_sha256: Some(digest('4')),
                adjuster: None,
            },
            e_process_values,
        }
    }

    fn fixture() -> ClosedFamily {
        ClosedFamily {
            family_id: "phase-response-closed-family-v1".into(),
            campaign_id: "synthetic-closed-phase-response-campaign".into(),
            global_ledger: global_ledger(),
            requested_guarantee: "FWER".into(),
            registered_before_confirmatory_outcomes: true,
            confirmation_lockbox_opened_before_freeze: false,
            family_commitment_sha256: digest('1'),
            global_filtration_sha256: digest('2'),
            confirmation_split_sha256: digest('3'),
            alpha: r(1, 5),
            claims: vec![
                claim("H1-local-reset", 'a', &[r(1, 1), r(5, 1), r(20, 1)]),
                claim(
                    "H2-off-diagonal-transfer",
                    'b',
                    &[r(1, 1), r(4, 1), r(15, 1)],
                ),
                claim("H3-contact-modulation", 'c', &[r(1, 1), r(3, 2), r(2, 1)]),
            ],
        }
    }

    #[test]
    fn exact_closed_testing_known_answer() {
        let report = audit_closed_family(&fixture()).unwrap();
        assert_eq!(report.threshold, r(5, 1));
        assert_eq!(report.intersection_count, 7);
        let values: Vec<Rational> = report
            .intersections
            .iter()
            .map(|row| *row.mixture_e_process_values.last().unwrap())
            .collect();
        assert_eq!(
            values,
            vec![
                r(20, 1),
                r(15, 1),
                r(35, 2),
                r(2, 1),
                r(11, 1),
                r(17, 2),
                r(37, 3)
            ]
        );
        assert_eq!(report.elementary[0].first_rejection_stage, Some(2));
        assert_eq!(report.elementary[1].first_rejection_stage, Some(2));
        assert_eq!(report.elementary[2].first_rejection_stage, None);
        assert_eq!(
            report.elementary[0].rejection_path,
            vec![false, false, true]
        );
        assert_eq!(
            report.elementary[1].rejection_path,
            vec![false, false, true]
        );
        assert_eq!(
            report.elementary[2].rejection_path,
            vec![false, false, false]
        );
        assert_eq!(report.family_fwer_bound, r(1, 5));
    }

    #[test]
    fn every_intersection_is_enumerated_once() {
        let report = audit_closed_family(&fixture()).unwrap();
        let ids: BTreeSet<Vec<String>> = report
            .intersections
            .iter()
            .map(|row| row.claim_ids.clone())
            .collect();
        assert_eq!(ids.len(), 7);
        assert!(ids.contains(&vec!["H1-local-reset".into()]));
        assert!(ids.contains(&vec![
            "H1-local-reset".into(),
            "H2-off-diagonal-transfer".into()
        ]));
    }

    #[test]
    fn decisions_are_persistent_after_raw_processes_fall() {
        let mut family = fixture();
        family.claims[0].e_process_values = vec![r(1, 1), r(6, 1), r(20, 1), r(2, 1)];
        family.claims[1].e_process_values = vec![r(1, 1), r(3, 1), r(15, 1), r(1, 1)];
        family.claims[2].e_process_values = vec![r(1, 1), r(3, 2), r(2, 1), r(1, 1)];
        let report = audit_closed_family(&family).unwrap();
        assert_eq!(
            report.elementary[0].rejection_path,
            vec![false, false, true, true]
        );
        assert_eq!(
            report.elementary[1].rejection_path,
            vec![false, false, true, true]
        );
        assert_eq!(
            report.elementary[2].rejection_path,
            vec![false, false, false, false]
        );
    }

    #[test]
    fn rejects_more_than_eight_elementary_claims() {
        let mut family = fixture();
        for index in 3..9 {
            family.claims.push(claim(
                &format!("H{index}"),
                'd',
                &[r(1, 1), r(1, 1), r(1, 1)],
            ));
        }
        assert!(audit_closed_family(&family).is_err());
    }

    #[test]
    fn rejects_duplicate_elementary_claim_ids() {
        let mut family = fixture();
        family.claims[1].claim_id = family.claims[0].claim_id.clone();
        assert!(audit_closed_family(&family).is_err());
    }

    #[test]
    fn rejects_mixed_global_filtrations() {
        let mut family = fixture();
        family.global_filtration_sha256 = digest('9');
        assert!(audit_closed_family(&family).is_err());
    }

    #[test]
    fn rejects_claim_with_mixed_global_filtration_receipt() {
        let mut family = fixture();
        family.claims[1].global_filtration_sha256 = digest('9');
        assert!(audit_closed_family(&family).is_err());
    }

    #[test]
    fn rejects_claim_with_mixed_confirmation_split() {
        let mut family = fixture();
        family.claims[1].confirmation_split_sha256 = digest('9');
        assert!(audit_closed_family(&family).is_err());
    }

    #[test]
    fn rejects_claim_without_component_validity_receipt() {
        let mut family = fixture();
        family.claims[1].component_status = "LOCAL_FILTRATION_ONLY".into();
        assert!(audit_closed_family(&family).is_err());
    }

    #[test]
    fn rejects_lockbox_open_before_family_freeze() {
        let mut family = fixture();
        family.confirmation_lockbox_opened_before_freeze = true;
        assert!(audit_closed_family(&family).is_err());
    }

    #[test]
    fn rejects_fdr_sup_request() {
        let mut family = fixture();
        family.requested_guarantee = "FDR_SUP".into();
        assert!(audit_closed_family(&family).is_err());
    }

    #[test]
    fn rejects_incomplete_or_inconsistent_horizon() {
        let mut family = fixture();
        family.claims[2].e_process_values.pop();
        assert!(audit_closed_family(&family).is_err());
    }

    #[test]
    fn rejects_e_process_not_started_at_one() {
        let mut family = fixture();
        family.claims[1].e_process_values[0] = r(2, 1);
        assert!(audit_closed_family(&family).is_err());
    }

    #[test]
    fn rejects_malformed_null_digest() {
        let mut family = fixture();
        family.claims[0].null_spec_sha256 = "bad".into();
        assert!(audit_closed_family(&family).is_err());
    }

    #[test]
    fn canonical_path_sha256_matches_standard_vector() {
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn admits_a_theorem_matched_lifted_elementary_process() {
        let mut family = fixture();
        let claim = &mut family.claims[1];
        let admission = &mut claim.filtration_admission;
        admission.mode = AdmissionMode::AdjusterLifted;
        admission.source_filtration_id = "neural-local-stream-v1".into();
        admission.source_filtration_sha256 = digest('7');
        admission.adjuster = Some(AdjusterReceipt {
            adjuster_id: "chore-ramdas-filtration-lift".into(),
            adjuster_version: "reviewed-v1".into(),
            source_process_sha256: admission.input_process_sha256.clone(),
            source_filtration_sha256: admission.source_filtration_sha256.clone(),
            target_filtration_sha256: admission.target_filtration_sha256.clone(),
            parameter_sha256: digest('8'),
            theorem_assumptions_sha256: admission.assumption_sha256.clone().unwrap(),
            lifted_output_sha256: admission.output_process_sha256.clone(),
        });
        assert!(audit_closed_family(&family).is_ok());
    }

    #[test]
    fn rejects_local_only_and_incomparable_elementary_streams() {
        let mut local = fixture();
        local.claims[0].filtration_admission.mode = AdmissionMode::LocalOnly;
        assert!(audit_closed_family(&local).is_err());

        let mut incomparable = fixture();
        incomparable.claims[0].filtration_admission.mode = AdmissionMode::IncomparableOrUnknown;
        assert!(audit_closed_family(&incomparable).is_err());
    }

    #[test]
    fn rejects_e_process_path_mutation_without_matching_admission() {
        let mut family = fixture();
        family.claims[0].e_process_values[2] = r(21, 1);
        assert!(audit_closed_family(&family).is_err());
    }

    #[test]
    fn rejects_wrong_null_process_or_proof_binding() {
        for mutation in 0..3 {
            let mut family = fixture();
            match mutation {
                0 => family.claims[0].filtration_admission.null_spec_sha256 = digest('9'),
                1 => family.claims[0].filtration_admission.process_id = "other-process".into(),
                _ => {
                    family.claims[0].filtration_admission.proof_receipt_sha256 =
                        Some(digest('9'));
                }
            }
            assert!(audit_closed_family(&family).is_err(), "mutation {mutation}");
        }
    }

    #[test]
    fn rejects_event_observed_at_or_after_process_output() {
        let mut family = fixture();
        family.claims[0]
            .filtration_admission
            .observed_event_ids
            .push("T7".into());
        assert!(audit_closed_family(&family).is_err());
    }

    #[test]
    fn exact_rational_mixture_is_normalized() {
        let report = audit_closed_family(&fixture()).unwrap();
        let intersection = report
            .intersections
            .iter()
            .find(|row| row.claim_ids.len() == 3)
            .unwrap();
        assert_eq!(intersection.mixture_e_process_values[2], r(37, 3));
    }
}
