// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Prediction-only authority narrowed to the comparator selected on Calibration.

use super::analysis_plan::EUREKA_002_ANALYSIS_PLAN_V1;
use super::baselines::ShortcutBaselineKind;
use super::consequence::ConsequencePrediction;
use super::hidden_world::PublicAction;
use super::v2_comparator_custody::{
    V2ComparatorCustodyError, V2ComparatorCustodyReceipt, V2DevelopmentFitCorpus,
};
use super::v2_frozen_comparator::V2FrozenComparatorSubject;
use super::v2_public_schema::{V2PublicFamily, V2PublicState};
use super::v2_selection_authorization::{
    V2ComparatorSelectionAuthorization, V2SelectionAuthorizationError,
    freeze_comparator_custody_from_authorization,
};

pub(super) const V2_SELECTED_COMPARATOR_REVISION: &str =
    "EUREKA.002.V2.SELECTED_COMPARATOR.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum V2SelectedComparatorError {
    FitCorpusMismatch,
    SchemaMismatch,
    FrozenSubjectMismatch,
    CustodySelectedMismatch,
    CustodyFitMismatch,
    CustodySchemaMismatch,
    AnalysisPlanMismatch,
    Selection(V2SelectionAuthorizationError),
    Custody(V2ComparatorCustodyError),
}

impl From<V2SelectionAuthorizationError> for V2SelectedComparatorError {
    fn from(value: V2SelectionAuthorizationError) -> Self {
        Self::Selection(value)
    }
}

impl From<V2ComparatorCustodyError> for V2SelectedComparatorError {
    fn from(value: V2ComparatorCustodyError) -> Self {
        Self::Custody(value)
    }
}

/// Non-Clone by design: downstream campaign code receives one narrowed
/// comparator authority rather than the general four-kind prediction surface.
#[derive(Debug)]
pub(super) struct V2SelectedComparatorSubject {
    subject: V2FrozenComparatorSubject,
    custody: V2ComparatorCustodyReceipt,
    selected: ShortcutBaselineKind,
    selection_authorization_commitment: [u8; 32],
    calibration_corpus_commitment: [u8; 32],
    implementation_commitment: [u8; 32],
    commitment: [u8; 32],
}

impl V2SelectedComparatorSubject {
    pub(super) fn freeze(
        development: &V2DevelopmentFitCorpus,
        subject: V2FrozenComparatorSubject,
        authorization: &V2ComparatorSelectionAuthorization,
    ) -> Result<Self, V2SelectedComparatorError> {
        if development.commitment() != authorization.fit_corpus_commitment()
            || subject.fit_corpus_commitment() != development.commitment()
        {
            return Err(V2SelectedComparatorError::FitCorpusMismatch);
        }
        if development.schema_commitment() != authorization.schema_commitment()
            || subject.schema_commitment() != development.schema_commitment()
        {
            return Err(V2SelectedComparatorError::SchemaMismatch);
        }
        if subject.commitment() != authorization.comparator_subject_commitment() {
            return Err(V2SelectedComparatorError::FrozenSubjectMismatch);
        }

        let custody = freeze_comparator_custody_from_authorization(development, authorization)?;
        if custody.selected() != authorization.selected() {
            return Err(V2SelectedComparatorError::CustodySelectedMismatch);
        }
        if custody.fit_corpus_commitment() != development.commitment() {
            return Err(V2SelectedComparatorError::CustodyFitMismatch);
        }
        if custody.schema_commitment() != development.schema_commitment() {
            return Err(V2SelectedComparatorError::CustodySchemaMismatch);
        }
        if custody.analysis_plan_commitment()
            != EUREKA_002_ANALYSIS_PLAN_V1.cryptographic_commitment()
        {
            return Err(V2SelectedComparatorError::AnalysisPlanMismatch);
        }

        let selected = authorization.selected();
        let selection_authorization_commitment = authorization.commitment();
        let calibration_corpus_commitment = authorization.calibration_corpus_commitment();
        let implementation_commitment = subject.implementation_commitment();
        let commitment = selected_comparator_commitment(
            development.commitment(),
            subject.commitment(),
            custody.commitment(),
            selected,
            selection_authorization_commitment,
            calibration_corpus_commitment,
            implementation_commitment,
            custody.analysis_plan_commitment(),
        );
        Ok(Self {
            subject,
            custody,
            selected,
            selection_authorization_commitment,
            calibration_corpus_commitment,
            implementation_commitment,
            commitment,
        })
    }

    pub(super) fn predict(
        &self,
        family: V2PublicFamily,
        pre: V2PublicState,
        action: PublicAction,
    ) -> ConsequencePrediction {
        self.subject.predict(self.selected, family, pre, action)
    }

    pub(super) const fn selected(&self) -> ShortcutBaselineKind {
        self.selected
    }

    pub(super) const fn schema_commitment(&self) -> [u8; 32] {
        self.custody.schema_commitment()
    }

    pub(super) const fn fit_corpus_commitment(&self) -> [u8; 32] {
        self.custody.fit_corpus_commitment()
    }

    pub(super) const fn frozen_subject_commitment(&self) -> [u8; 32] {
        self.subject.commitment()
    }

    pub(super) const fn custody_commitment(&self) -> [u8; 32] {
        self.custody.commitment()
    }

    pub(super) const fn selection_authorization_commitment(&self) -> [u8; 32] {
        self.selection_authorization_commitment
    }

    pub(super) const fn calibration_corpus_commitment(&self) -> [u8; 32] {
        self.calibration_corpus_commitment
    }

    pub(super) const fn implementation_commitment(&self) -> [u8; 32] {
        self.implementation_commitment
    }

    pub(super) const fn analysis_plan_commitment(&self) -> [u8; 32] {
        self.custody.analysis_plan_commitment()
    }

    pub(super) const fn commitment(&self) -> [u8; 32] {
        self.commitment
    }
}

#[allow(clippy::too_many_arguments)]
fn selected_comparator_commitment(
    fit_corpus_commitment: [u8; 32],
    frozen_subject_commitment: [u8; 32],
    custody_commitment: [u8; 32],
    selected: ShortcutBaselineKind,
    selection_authorization_commitment: [u8; 32],
    calibration_corpus_commitment: [u8; 32],
    implementation_commitment: [u8; 32],
    analysis_plan_commitment: [u8; 32],
) -> [u8; 32] {
    let mut bytes = Vec::new();
    encode_bytes(&mut bytes, V2_SELECTED_COMPARATOR_REVISION.as_bytes());
    bytes.extend_from_slice(&fit_corpus_commitment);
    bytes.extend_from_slice(&frozen_subject_commitment);
    bytes.extend_from_slice(&custody_commitment);
    encode_bytes(&mut bytes, selected.stable_id().as_bytes());
    bytes.extend_from_slice(&selection_authorization_commitment);
    bytes.extend_from_slice(&calibration_corpus_commitment);
    bytes.extend_from_slice(&implementation_commitment);
    bytes.extend_from_slice(&analysis_plan_commitment);
    *blake3::hash(&bytes).as_bytes()
}

fn encode_bytes(bytes: &mut Vec<u8>, value: &[u8]) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value);
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::v2_corpus_schedule::materialize_canonical_corpora;
    use super::super::v2_selection_authorization::{
        V2ComparatorSelectionOutcome, execute_calibration_selection,
    };

    fn selected_subject() -> V2SelectedComparatorSubject {
        let corpora = materialize_canonical_corpora().unwrap();
        let subject = V2FrozenComparatorSubject::freeze(corpora.development());
        let outcome = execute_calibration_selection(
            corpora.development(),
            corpora.calibration(),
            &subject,
        )
        .unwrap();
        let V2ComparatorSelectionOutcome::Selected(authorization) = outcome else {
            panic!("canonical V2 construct must select a comparator");
        };
        V2SelectedComparatorSubject::freeze(corpora.development(), subject, &authorization).unwrap()
    }

    #[test]
    fn selected_subject_binds_real_calibration_authorization_and_custody() {
        let selected = selected_subject();
        assert_ne!(selected.commitment(), [0_u8; 32]);
        assert_ne!(selected.custody_commitment(), [0_u8; 32]);
        assert_ne!(selected.selection_authorization_commitment(), [0_u8; 32]);
        assert_ne!(selected.calibration_corpus_commitment(), [0_u8; 32]);
        assert_ne!(selected.implementation_commitment(), [0_u8; 32]);
        assert_eq!(
            selected.analysis_plan_commitment(),
            EUREKA_002_ANALYSIS_PLAN_V1.cryptographic_commitment()
        );
    }

    #[test]
    fn selected_prediction_surface_has_no_kind_parameter() {
        let selected = selected_subject();
        let row = super::super::v2_heldout_plan::materialize_heldout_plan()
            .unwrap()
            .ordered_rows()[0];
        let prediction = selected.predict(row.family(), row.pre(), row.action());
        assert_eq!(prediction.action, row.action());
    }
}
