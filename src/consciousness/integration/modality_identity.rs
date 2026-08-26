// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Stable identity and enumeration contracts for cross-modal modality variants.
//!
//! Some existing deterministic paths use the numeric discriminant of
//! [`Modality`] as a noise seed or ordering key. That is safe only while the
//! discriminants of existing variants never move. This module makes that
//! compatibility requirement executable and provides explicit stable IDs for
//! new code so future modalities can be added without depending on declaration
//! order.
//!
//! Enumeration is a separate contract from identity. The historical
//! `Modality::all()` helper does **not** enumerate every defined enum variant; it
//! selects the legacy root channel set. Likewise `Modality::sensory()` is a
//! legacy runtime subset rather than an exhaustive scientific classification.
//! Those behaviors are captured explicitly below so adding olfaction,
//! gustation, or chemesthesis cannot silently alter root topology merely by
//! changing an enum helper.
//!
//! Existing variants intentionally retain their historical IDs `0..=12`.
//! Chemical modalities have reserved IDs so the upcoming olfactory/gustatory
//! integration cannot accidentally renumber an existing sense.

use super::cross_modal_binding::Modality;

/// Exhaustive list of every modality variant currently defined by the enum.
///
/// Use this for identity/invariant checks that genuinely mean "every defined
/// variant". Do not substitute `Modality::all()`: that helper is a legacy root
/// runtime selection and intentionally has a different contract.
pub const DEFINED_MODALITIES: [Modality; 13] = [
    Modality::Visual,
    Modality::Auditory,
    Modality::Textual,
    Modality::Linguistic,
    Modality::Proprioceptive,
    Modality::Somatosensory,
    Modality::Motor,
    Modality::Temporal,
    Modality::Spatial,
    Modality::Affective,
    Modality::Emotional,
    Modality::Interoceptive,
    Modality::Abstract,
];

/// Historical set instantiated by `MultiModalIntegrator` through
/// `Modality::all()`.
///
/// This is named explicitly because it is not exhaustive. Chemical integration
/// must decide deliberately whether a new modality joins this default topology;
/// appending an enum variant alone must not make that decision.
pub const LEGACY_ROOT_MODALITIES: [Modality; 7] = [
    Modality::Visual,
    Modality::Auditory,
    Modality::Linguistic,
    Modality::Somatosensory,
    Modality::Motor,
    Modality::Emotional,
    Modality::Interoceptive,
];

/// Historical set returned by `Modality::sensory()` and used by the legacy
/// amodal convergence zone.
pub const LEGACY_SENSORY_MODALITIES: [Modality; 4] = [
    Modality::Visual,
    Modality::Auditory,
    Modality::Linguistic,
    Modality::Somatosensory,
];

/// First IDs reserved for the chemical-sensing tranche.
pub const OLFACTORY_STABLE_ID: u16 = 13;
pub const GUSTATORY_STABLE_ID: u16 = 14;
pub const CHEMESTHETIC_STABLE_ID: u16 = 15;

/// Stable numeric identity for every currently-defined modality.
///
/// New code should use this instead of `modality as u64` / `modality as u8`.
/// Existing ordinal-based code is protected by the unit tests below until it is
/// migrated to this helper.
pub const fn stable_modality_id(modality: Modality) -> u16 {
    match modality {
        Modality::Visual => 0,
        Modality::Auditory => 1,
        Modality::Textual => 2,
        Modality::Linguistic => 3,
        Modality::Proprioceptive => 4,
        Modality::Somatosensory => 5,
        Modality::Motor => 6,
        Modality::Temporal => 7,
        Modality::Spatial => 8,
        Modality::Affective => 9,
        Modality::Emotional => 10,
        Modality::Interoceptive => 11,
        Modality::Abstract => 12,
    }
}

/// Stable deterministic seed domain for modality-specific stochastic transforms.
pub const fn modality_seed(modality: Modality) -> u64 {
    stable_modality_id(modality) as u64
}

/// Stable ordering key for deterministic modality iteration.
pub const fn modality_sort_key(modality: Modality) -> u16 {
    stable_modality_id(modality)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn exhaustive_defined_set_has_every_historical_variant_once() {
        let ids: BTreeSet<u16> = DEFINED_MODALITIES
            .into_iter()
            .map(stable_modality_id)
            .collect();
        assert_eq!(DEFINED_MODALITIES.len(), 13);
        assert_eq!(ids.len(), DEFINED_MODALITIES.len());
    }

    #[test]
    fn historical_discriminants_match_explicit_stable_ids() {
        for modality in DEFINED_MODALITIES {
            assert_eq!(
                modality as u16,
                stable_modality_id(modality),
                "reordering or inserting before an existing modality changes deterministic seeds"
            );
        }
    }

    #[test]
    fn legacy_root_enumeration_is_explicit_not_exhaustive() {
        assert_eq!(Modality::all(), LEGACY_ROOT_MODALITIES.to_vec());
        assert!(LEGACY_ROOT_MODALITIES.len() < DEFINED_MODALITIES.len());
        assert!(!LEGACY_ROOT_MODALITIES.contains(&Modality::Proprioceptive));
        assert!(!LEGACY_ROOT_MODALITIES.contains(&Modality::Temporal));
        assert!(!LEGACY_ROOT_MODALITIES.contains(&Modality::Spatial));
        assert!(!LEGACY_ROOT_MODALITIES.contains(&Modality::Abstract));
    }

    #[test]
    fn legacy_sensory_enumeration_is_pinned_separately() {
        assert_eq!(Modality::sensory(), LEGACY_SENSORY_MODALITIES.to_vec());
        assert!(!LEGACY_SENSORY_MODALITIES.contains(&Modality::Proprioceptive));
        assert!(!LEGACY_SENSORY_MODALITIES.contains(&Modality::Interoceptive));
    }

    #[test]
    fn stable_ids_are_unique() {
        let ids: BTreeSet<u16> = DEFINED_MODALITIES
            .into_iter()
            .map(stable_modality_id)
            .collect();
        assert_eq!(ids.len(), DEFINED_MODALITIES.len());
    }

    #[test]
    fn chemical_ids_are_reserved_after_existing_modalities() {
        let existing: BTreeSet<u16> = DEFINED_MODALITIES
            .into_iter()
            .map(stable_modality_id)
            .collect();
        assert!(!existing.contains(&OLFACTORY_STABLE_ID));
        assert!(!existing.contains(&GUSTATORY_STABLE_ID));
        assert!(!existing.contains(&CHEMESTHETIC_STABLE_ID));
        assert_eq!(OLFACTORY_STABLE_ID, 13);
        assert_eq!(GUSTATORY_STABLE_ID, 14);
        assert_eq!(CHEMESTHETIC_STABLE_ID, 15);
    }
}
