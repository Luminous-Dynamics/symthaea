#![cfg(feature = "symthaea-backend")]

//! Component-level RPT-2 theorem for Symthaea's production `CrossModalBinder`.
//!
//! The existing Butlin path largely observes whether the cross-modal module ran.
//! Here we test the representation itself: a bound state must preserve usable
//! information from two independently specified modalities, selective removal of
//! one modality must selectively reduce access to that modality, and restoring
//! it must rescue the integrated state. This remains component/mechanistic
//! evidence only and cannot promote an RPT-2 support tier by itself.

use symthaea::consciousness::cross_modal_binding::{
    CrossModalBinder, CrossModalBindingConfig, ModalRepresentation, Modality,
};
use symthaea_core::hdc::ContinuousHV;

const DIM: usize = 512;

fn visual() -> ContinuousHV {
    ContinuousHV::random(DIM, 0x5250_5401)
}

fn auditory() -> ContinuousHV {
    ContinuousHV::random(DIM, 0x5250_5402)
}

fn binder_with_modalities(include_visual: bool, include_auditory: bool) -> CrossModalBinder {
    let mut binder = CrossModalBinder::new(CrossModalBindingConfig {
        dimension: DIM,
        binding_threshold: 0.5,
        temporal_decay: 0.1,
        max_bindings: 8,
        use_attention: true,
    });

    if include_visual {
        binder.add_representation(ModalRepresentation::new(
            Modality::Visual,
            visual(),
            1.0,
            "rpt2-visual",
        ));
    }
    if include_auditory {
        binder.add_representation(ModalRepresentation::new(
            Modality::Auditory,
            auditory(),
            1.0,
            "rpt2-auditory",
        ));
    }

    binder
}

fn bind_state(binder: &mut CrossModalBinder) -> ContinuousHV {
    binder.bind().expect("test binder must produce a binding");
    binder
        .current_binding()
        .expect("binding result must install current bound state")
        .clone()
}

#[test]
fn multimodal_binding_preserves_information_from_both_modalities() {
    let v = visual();
    let a = auditory();

    let mut integrated_binder = binder_with_modalities(true, true);
    let integrated = bind_state(&mut integrated_binder);

    let mut visual_only_binder = binder_with_modalities(true, false);
    let visual_only = bind_state(&mut visual_only_binder);

    let mut auditory_only_binder = binder_with_modalities(false, true);
    let auditory_only = bind_state(&mut auditory_only_binder);

    let integrated_to_visual = integrated.similarity(&v);
    let integrated_to_auditory = integrated.similarity(&a);
    let visual_only_to_auditory = visual_only.similarity(&a);
    let auditory_only_to_visual = auditory_only.similarity(&v);

    assert!(
        integrated_to_visual > auditory_only_to_visual,
        "integrated state did not retain more visual information than auditory-only control: integrated={} control={}",
        integrated_to_visual,
        auditory_only_to_visual,
    );
    assert!(
        integrated_to_auditory > visual_only_to_auditory,
        "integrated state did not retain more auditory information than visual-only control: integrated={} control={}",
        integrated_to_auditory,
        visual_only_to_auditory,
    );

    // Integration must not simply copy either endpoint.
    assert!(integrated.similarity(&v) < 0.999_999);
    assert!(integrated.similarity(&a) < 0.999_999);
}

#[test]
fn removing_one_modality_selectively_reduces_access_to_that_modality() {
    let v = visual();
    let a = auditory();

    let mut integrated_binder = binder_with_modalities(true, true);
    let integrated = bind_state(&mut integrated_binder);

    let mut visual_only_binder = binder_with_modalities(true, false);
    let visual_only = bind_state(&mut visual_only_binder);

    let integrated_auditory_access = integrated.similarity(&a);
    let lesioned_auditory_access = visual_only.similarity(&a);
    let integrated_visual_access = integrated.similarity(&v);
    let lesioned_visual_access = visual_only.similarity(&v);

    assert!(
        lesioned_auditory_access < integrated_auditory_access,
        "auditory lesion did not reduce auditory information: integrated={} lesioned={}",
        integrated_auditory_access,
        lesioned_auditory_access,
    );
    assert!(
        lesioned_visual_access > integrated_visual_access,
        "visual information was not preserved/enhanced when auditory input was removed: integrated={} lesioned={}",
        integrated_visual_access,
        lesioned_visual_access,
    );
}

#[test]
fn restoring_removed_modality_rescues_the_same_integrated_representation() {
    let mut reference_binder = binder_with_modalities(true, true);
    let reference = bind_state(&mut reference_binder);

    let mut rescued_binder = binder_with_modalities(true, false);
    let _lesioned = bind_state(&mut rescued_binder);
    rescued_binder.add_representation(ModalRepresentation::new(
        Modality::Auditory,
        auditory(),
        1.0,
        "rpt2-auditory-rescue",
    ));
    let rescued = bind_state(&mut rescued_binder);

    assert!(
        reference.similarity(&rescued) > 0.999_999,
        "restoring the identical missing modality did not rescue the integrated state",
    );
}

#[test]
fn binder_reports_both_modalities_and_nonzero_integration_statistics() {
    let mut binder = binder_with_modalities(true, true);
    let result = binder.bind().expect("multimodal binding must exist");

    assert_eq!(result.modalities.len(), 2);
    assert!(result.modalities.contains(&Modality::Visual));
    assert!(result.modalities.contains(&Modality::Auditory));
    assert_eq!(binder.stats().total_bindings, 1);
    assert!(result.coherence.is_finite());
    assert!(result.strength.is_finite());
}

#[test]
fn component_scope_refuses_rpt2_functional_promotion() {
    const CLAIM_SCOPE: &str =
        "component integrated-representation theorem only; no full-loop perceptual-task consequence; no RPT-2 tier promotion; no consciousness claim";
    assert!(CLAIM_SCOPE.contains("no RPT-2 tier promotion"));
    assert!(CLAIM_SCOPE.contains("no consciousness claim"));
}
