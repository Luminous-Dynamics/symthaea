//! Strict paired target/control assessment for OPEN-OBS-002.
//!
//! This module only evaluates whether a detector candidate is confined to a
//! complete target panel, shared with a control panel, or unresolved because
//! the paired capture is incomplete. It does not establish astrophysical,
//! extraterrestrial, extra-dimensional, or intelligent origin.

use crate::{digest_hex, valid_sha256, Capture};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum PanelRole {
    Control,
    Target,
}

impl PanelRole {
    fn as_str(self) -> &'static str {
        match self {
            Self::Control => "control",
            Self::Target => "target",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PairedPanel {
    pub role: PanelRole,
    pub capture: Capture,
    pub panel_sha256: String,
    pub candidate_detected: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PairedObservationBundle {
    pub background_group: String,
    pub instrument_profile_sha256: String,
    pub detector_profile_sha256: String,
    pub panels: Vec<PairedPanel>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PairedStatus {
    TargetAssociatedCandidate,
    CandidateAlsoInControl,
    ControlOnlyCandidate,
    NoCandidate,
    UnknownIncompleteCapture,
}

impl PairedStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::TargetAssociatedCandidate => "TARGET_ASSOCIATED_CANDIDATE",
            Self::CandidateAlsoInControl => "CANDIDATE_ALSO_IN_CONTROL",
            Self::ControlOnlyCandidate => "CONTROL_ONLY_CANDIDATE",
            Self::NoCandidate => "NO_CANDIDATE",
            Self::UnknownIncompleteCapture => "UNKNOWN_INCOMPLETE_CAPTURE",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PairedAssessment {
    pub bundle_sha256: String,
    pub assessment_sha256: String,
    pub status: PairedStatus,
}

fn canonical_panel_list(
    panels: &[PairedPanel],
) -> Result<(&PairedPanel, &PairedPanel), String> {
    if panels.len() != 2 {
        return Err(format!(
            "paired v1 bundle requires exactly two panels (target and control); found {}",
            panels.len()
        ));
    }

    let mut target: Option<&PairedPanel> = None;
    let mut control: Option<&PairedPanel> = None;
    for panel in panels {
        match panel.role {
            PanelRole::Target => {
                if target.replace(panel).is_some() {
                    return Err("duplicate target panel role".to_owned());
                }
            }
            PanelRole::Control => {
                if control.replace(panel).is_some() {
                    return Err("duplicate control panel role".to_owned());
                }
            }
        }
        if !valid_sha256(&panel.panel_sha256) {
            return Err(format!(
                "{} panel digest must be 64 lowercase hexadecimal SHA-256 characters",
                panel.role.as_str()
            ));
        }
    }

    let target = target.ok_or_else(|| "required target panel is missing".to_owned())?;
    let control = control.ok_or_else(|| "required control panel is missing".to_owned())?;
    if target.panel_sha256 == control.panel_sha256 {
        return Err("target and control panels share the same content digest".to_owned());
    }
    Ok((target, control))
}

pub fn assess_paired_bundle(
    bundle: &PairedObservationBundle,
) -> Result<PairedAssessment, String> {
    if bundle.background_group.trim().is_empty()
        || bundle.background_group.trim() != bundle.background_group
        || bundle.background_group.chars().any(char::is_control)
    {
        return Err("background_group must be non-empty canonical text".to_owned());
    }
    if !valid_sha256(&bundle.instrument_profile_sha256) {
        return Err(
            "instrument profile digest must be 64 lowercase hexadecimal SHA-256 characters"
                .to_owned(),
        );
    }
    if !valid_sha256(&bundle.detector_profile_sha256) {
        return Err(
            "detector profile digest must be 64 lowercase hexadecimal SHA-256 characters"
                .to_owned(),
        );
    }

    let (target, control) = canonical_panel_list(&bundle.panels)?;
    let mut ordered = vec![control, target];
    ordered.sort_by_key(|panel| panel.role);

    // Bundle identity is derived only from declared acquisition/profile facts
    // and panel content hashes. Detector outcomes are included in a separate
    // assessment digest so model-output changes cannot rewrite source identity.
    let mut bundle_identity = format!(
        "OPEN-OBS-PAIRED-BUNDLE-V1\nbackground_group={}\ninstrument_profile_sha256={}\n",
        bundle.background_group, bundle.instrument_profile_sha256
    );
    for panel in &ordered {
        bundle_identity.push_str(&format!(
            "panel_role={}\ncapture={}\npanel_sha256={}\n",
            panel.role.as_str(),
            panel.capture.as_str(),
            panel.panel_sha256
        ));
    }
    let bundle_sha256 = digest_hex(bundle_identity.as_bytes());

    let status = if target.capture != Capture::Complete || control.capture != Capture::Complete {
        PairedStatus::UnknownIncompleteCapture
    } else {
        match (target.candidate_detected, control.candidate_detected) {
            (true, false) => PairedStatus::TargetAssociatedCandidate,
            (true, true) => PairedStatus::CandidateAlsoInControl,
            (false, true) => PairedStatus::ControlOnlyCandidate,
            (false, false) => PairedStatus::NoCandidate,
        }
    };

    let assessment_identity = format!(
        "OPEN-OBS-PAIRED-ASSESSMENT-V1\nbundle_sha256={}\ndetector_profile_sha256={}\ntarget_candidate={}\ncontrol_candidate={}\nstatus={}\n",
        bundle_sha256,
        bundle.detector_profile_sha256,
        target.candidate_detected,
        control.candidate_detected,
        status.as_str()
    );
    let assessment_sha256 = digest_hex(assessment_identity.as_bytes());

    Ok(PairedAssessment {
        bundle_sha256,
        assessment_sha256,
        status,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sha(value: &str) -> String {
        digest_hex(value.as_bytes())
    }

    fn bundle(target: bool, control: bool) -> PairedObservationBundle {
        PairedObservationBundle {
            background_group: "acquisition-001".to_owned(),
            instrument_profile_sha256: sha("instrument-v1"),
            detector_profile_sha256: sha("detector-v1"),
            panels: vec![
                PairedPanel {
                    role: PanelRole::Target,
                    capture: Capture::Complete,
                    panel_sha256: sha("target-panel"),
                    candidate_detected: target,
                },
                PairedPanel {
                    role: PanelRole::Control,
                    capture: Capture::Complete,
                    panel_sha256: sha("control-panel"),
                    candidate_detected: control,
                },
            ],
        }
    }

    #[test]
    fn target_only_candidate_is_only_panel_association() {
        let result = assess_paired_bundle(&bundle(true, false)).unwrap();
        assert_eq!(result.status, PairedStatus::TargetAssociatedCandidate);
        assert!(valid_sha256(&result.bundle_sha256));
        assert!(valid_sha256(&result.assessment_sha256));
    }

    #[test]
    fn candidate_shared_with_control_is_not_target_specific() {
        let result = assess_paired_bundle(&bundle(true, true)).unwrap();
        assert_eq!(result.status, PairedStatus::CandidateAlsoInControl);
    }

    #[test]
    fn control_only_candidate_is_not_target_associated() {
        let result = assess_paired_bundle(&bundle(false, true)).unwrap();
        assert_eq!(result.status, PairedStatus::ControlOnlyCandidate);
    }

    #[test]
    fn missing_candidate_in_both_panels_is_explicit() {
        let result = assess_paired_bundle(&bundle(false, false)).unwrap();
        assert_eq!(result.status, PairedStatus::NoCandidate);
    }

    #[test]
    fn incomplete_control_blocks_target_attribution() {
        let mut input = bundle(true, false);
        input.panels[1].capture = Capture::Incomplete;
        let result = assess_paired_bundle(&input).unwrap();
        assert_eq!(result.status, PairedStatus::UnknownIncompleteCapture);
    }

    #[test]
    fn panel_order_does_not_change_canonical_bundle_identity() {
        let original = bundle(true, false);
        let mut reversed = original.clone();
        reversed.panels.reverse();
        let a = assess_paired_bundle(&original).unwrap();
        let b = assess_paired_bundle(&reversed).unwrap();
        assert_eq!(a.bundle_sha256, b.bundle_sha256);
        assert_eq!(a.assessment_sha256, b.assessment_sha256);
    }

    #[test]
    fn rejects_missing_or_duplicate_roles() {
        let mut missing = bundle(true, false);
        missing.panels.pop();
        assert!(assess_paired_bundle(&missing)
            .unwrap_err()
            .contains("exactly two panels"));

        let mut duplicate = bundle(true, false);
        duplicate.panels[1].role = PanelRole::Target;
        assert!(assess_paired_bundle(&duplicate)
            .unwrap_err()
            .contains("duplicate target"));
    }

    #[test]
    fn rejects_same_panel_digest_and_malformed_profiles() {
        let mut same_panel = bundle(true, false);
        same_panel.panels[1].panel_sha256 = same_panel.panels[0].panel_sha256.clone();
        assert!(assess_paired_bundle(&same_panel)
            .unwrap_err()
            .contains("share the same content digest"));

        let mut malformed_profile = bundle(true, false);
        malformed_profile.instrument_profile_sha256 = "bad".to_owned();
        assert!(assess_paired_bundle(&malformed_profile)
            .unwrap_err()
            .contains("instrument profile digest"));
    }

    #[test]
    fn source_identity_is_separate_from_detector_outcome() {
        let first = assess_paired_bundle(&bundle(true, false)).unwrap();
        let second = assess_paired_bundle(&bundle(false, false)).unwrap();
        assert_eq!(first.bundle_sha256, second.bundle_sha256);
        assert_ne!(first.assessment_sha256, second.assessment_sha256);
    }
}
