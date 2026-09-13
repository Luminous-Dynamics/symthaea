// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Static authority checks for V2 pre-HeldOut custody.

#[test]
fn preheldout_custody_does_not_execute_or_score_heldout() {
    let whole_source = include_str!("v2_preheldout_custody.rs");
    let production_source = whole_source
        .split("#[cfg(test)]")
        .next()
        .expect("pre-HeldOut custody has a production section");

    for forbidden in [
        ".trial(",
        "predict_once(",
        "score_consequence(",
        "learn_from_actual(",
        "FepPredictionSession",
        "FepEvaluationSnapshot",
        "ActiveInferenceAgent",
    ] {
        assert!(
            !production_source.contains(forbidden),
            "pre-HeldOut custody must not execute or train subjects: {forbidden}"
        );
    }
}

#[test]
fn campaign_capability_contains_only_sealed_or_narrowed_authorities() {
    let whole_source = include_str!("v2_preheldout_custody.rs");
    let production_source = whole_source
        .split("#[cfg(test)]")
        .next()
        .expect("pre-HeldOut custody has a production section");
    let start = production_source
        .find("struct V2PreHeldOutCampaignCapability")
        .expect("campaign capability exists");
    let tail = &production_source[start..];
    let end = tail
        .find("}\n\nimpl V2PreHeldOutCampaignCapability")
        .expect("campaign capability body terminates");
    let body = &tail[..end];

    for required in [
        "V2FepDevelopmentArtifact",
        "V2SelectedComparatorSubject",
        "V2HeldOutPlan",
        "V2PreHeldOutManifest",
    ] {
        assert!(body.contains(required), "missing sealed authority: {required}");
    }
    for forbidden in [
        "FepPredictionSession",
        "FepEvaluationSnapshot",
        "V2FrozenComparatorSubject",
        "V2ComparatorSelectionAuthorization",
    ] {
        assert!(
            !body.contains(forbidden),
            "campaign capability must not expose broader authority: {forbidden}"
        );
    }
}

#[test]
fn selected_comparator_public_prediction_surface_has_no_kind_argument() {
    let whole_source = include_str!("v2_selected_comparator.rs");
    let production_source = whole_source
        .split("#[cfg(test)]")
        .next()
        .expect("selected comparator has a production section");
    let predict_start = production_source
        .find("pub(super) fn predict(")
        .expect("narrow prediction method exists");
    let tail = &production_source[predict_start..];
    let predict_end = tail
        .find("    }\n\n    pub(super) const fn selected")
        .expect("narrow prediction method terminates");
    let signature_and_body = &tail[..predict_end];
    assert!(!signature_and_body.contains("kind:"));
    assert!(signature_and_body.contains("self.selected"));
}
