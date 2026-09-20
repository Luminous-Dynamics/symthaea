// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Public gate for cadence-conformant institutional currentness.
//!
//! V1 deliberately supports only same-root cadence continuity. Every authorized
//! cadence policy in the lineage must be rooted in the exact same institutional
//! root as the original positive currentness decision. Cross-root cadence
//! continuity requires a future explicit root-transition theorem; merely naming
//! a predecessor policy digest is not sufficient authority.

use crate::{
    qualification_cadence_conformant_currentness as raw,
    CadenceConformantInstitutionalCurrentnessAtEvaluation,
    QualificationCadenceCurrentnessError, QualificationCadenceCurrentnessInputs,
};

pub fn establish_cadence_conformant_currentness_at_evaluation(
    inputs: QualificationCadenceCurrentnessInputs<'_>,
) -> Result<CadenceConformantInstitutionalCurrentnessAtEvaluation, QualificationCadenceCurrentnessError> {
    let expected_root = inputs.current.root_authority_sha256();
    if inputs
        .cadence_policies
        .iter()
        .any(|policy| policy.root_authority_sha256() != expected_root)
    {
        return Err(QualificationCadenceCurrentnessError::RootAuthorityMismatch);
    }

    raw::establish_cadence_conformant_currentness_at_evaluation(inputs)
}
