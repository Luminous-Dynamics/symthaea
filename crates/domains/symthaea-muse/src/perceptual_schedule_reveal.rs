// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Post-collection randomization-key reveal verification for MEL-003P1D.
//!
//! The secret is intentionally absent from the public schedule and private
//! audit during blinded collection. Once reveal is authorized, this verifier
//! rebuilds the entire schedule from the committed key and requires exact
//! equality with both frozen artifacts. That makes participant ranking,
//! factorial-cell rotation, task order, item order, opaque identifiers, and
//! private arm mappings reproducible rather than trusted metadata.

use crate::evidence_digest::{
    sha256_hex,
    perceptual_participant_schedule::{
        build_perceptual_participant_schedule, PerceptualCohortSlotsV1,
        PerceptualParticipantScheduleAuditV1, PerceptualParticipantScheduleBookV1,
    },
    perceptual_stimulus_pack::{
        FrozenC6fRenderSubjectBindingV1, FrozenPerceptualStimulusPackV1,
    },
    perceptual_study_protocol::FrozenPerceptualStudyProtocolV1,
};
use serde::{Deserialize, Serialize};

pub const PERCEPTUAL_SCHEDULE_REVEAL_VERSION: &str =
    "mel003-perceptual-schedule-reveal-v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PerceptualScheduleRevealIssueV1 {
    WrongRevealVersion,
    SecretCommitmentMismatch,
    RebuildRejected,
    PublicScheduleMismatch,
    PrivateAuditMismatch,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PerceptualScheduleRevealResultV1 {
    pub reveal_version: String,
    pub secret_commitment_matches: bool,
    pub public_schedule_reproduced_exactly: bool,
    pub private_audit_reproduced_exactly: bool,
    pub issues: Vec<PerceptualScheduleRevealIssueV1>,
}

impl PerceptualScheduleRevealResultV1 {
    pub fn success(&self) -> bool {
        self.issues.is_empty()
            && self.secret_commitment_matches
            && self.public_schedule_reproduced_exactly
            && self.private_audit_reproduced_exactly
    }
}

pub fn verify_perceptual_schedule_key_reveal(
    protocol: &FrozenPerceptualStudyProtocolV1,
    stimulus_pack: &FrozenPerceptualStimulusPackV1,
    render_binding: &FrozenC6fRenderSubjectBindingV1,
    cohort: &PerceptualCohortSlotsV1,
    frozen_book: &PerceptualParticipantScheduleBookV1,
    frozen_audit: &PerceptualParticipantScheduleAuditV1,
    secret_key: [u8; 32],
) -> PerceptualScheduleRevealResultV1 {
    let secret_commitment_matches =
        sha256_hex(&secret_key) == protocol.blinding.randomization_commitment_sha256;
    let mut issues = Vec::new();
    if !secret_commitment_matches {
        issues.push(PerceptualScheduleRevealIssueV1::SecretCommitmentMismatch);
    }

    let rebuilt = build_perceptual_participant_schedule(
        protocol,
        stimulus_pack,
        render_binding,
        cohort,
        secret_key,
    );
    let (public_schedule_reproduced_exactly, private_audit_reproduced_exactly) = match rebuilt {
        Ok((rebuilt_book, rebuilt_audit)) => {
            let public_matches = rebuilt_book == *frozen_book;
            let private_matches = rebuilt_audit == *frozen_audit;
            if !public_matches {
                issues.push(PerceptualScheduleRevealIssueV1::PublicScheduleMismatch);
            }
            if !private_matches {
                issues.push(PerceptualScheduleRevealIssueV1::PrivateAuditMismatch);
            }
            (public_matches, private_matches)
        }
        Err(_) => {
            issues.push(PerceptualScheduleRevealIssueV1::RebuildRejected);
            (false, false)
        }
    };

    PerceptualScheduleRevealResultV1 {
        reveal_version: PERCEPTUAL_SCHEDULE_REVEAL_VERSION.into(),
        secret_commitment_matches,
        public_schedule_reproduced_exactly,
        private_audit_reproduced_exactly,
        issues,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn success_requires_empty_issue_registry() {
        let result = PerceptualScheduleRevealResultV1 {
            reveal_version: PERCEPTUAL_SCHEDULE_REVEAL_VERSION.into(),
            secret_commitment_matches: true,
            public_schedule_reproduced_exactly: true,
            private_audit_reproduced_exactly: true,
            issues: Vec::new(),
        };
        assert!(result.success());
    }

    #[test]
    fn any_schedule_mismatch_blocks_success() {
        let result = PerceptualScheduleRevealResultV1 {
            reveal_version: PERCEPTUAL_SCHEDULE_REVEAL_VERSION.into(),
            secret_commitment_matches: true,
            public_schedule_reproduced_exactly: false,
            private_audit_reproduced_exactly: true,
            issues: vec![PerceptualScheduleRevealIssueV1::PublicScheduleMismatch],
        };
        assert!(!result.success());
    }
}
