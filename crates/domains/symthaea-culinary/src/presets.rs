//! Named style presets — externalized `CulinarySpec`s as data, per
//! `data/presets/PROVENANCE.md`. This is what makes "edit the JSON to make a
//! new culinary world" real: the validator (`crate::validate`) never changes;
//! only which physically-valid point within its fixed safe ranges a style
//! prefers does. Every preset here is asserted to validate cleanly
//! (`tests::every_preset_validates_cleanly`) — a preset that failed its own
//! validator would be a self-inconsistency bug, not a stylistic choice.

use crate::spec::CulinarySpec;

const FRENCH_CLASSICAL: &str = include_str!("../data/presets/french_classical.json");
const MOLECULAR_GASTRONOMY: &str = include_str!("../data/presets/molecular_gastronomy.json");
const RUSTIC_FERMENTATION: &str = include_str!("../data/presets/rustic_fermentation.json");

/// Names of every built-in preset, in the order [`all`] returns them.
pub const PRESET_NAMES: [&str; 3] = [
    "french_classical",
    "molecular_gastronomy",
    "rustic_fermentation",
];

/// Load a built-in preset by name (see [`PRESET_NAMES`]). `None` for an
/// unknown name — parse failure on a *known* name is a bug and panics, since
/// these are compiled-in constants, not user input.
pub fn preset(name: &str) -> Option<CulinarySpec> {
    let json = match name {
        "french_classical" => FRENCH_CLASSICAL,
        "molecular_gastronomy" => MOLECULAR_GASTRONOMY,
        "rustic_fermentation" => RUSTIC_FERMENTATION,
        _ => return None,
    };
    Some(
        serde_json::from_str(json)
            .unwrap_or_else(|e| panic!("built-in preset {name:?} failed to parse: {e}")),
    )
}

/// Every built-in preset, parsed.
pub fn all() -> Vec<CulinarySpec> {
    PRESET_NAMES
        .iter()
        .map(|name| preset(name).expect("name drawn from PRESET_NAMES"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::validate::validate;

    #[test]
    fn every_named_preset_loads() {
        for name in PRESET_NAMES {
            assert!(preset(name).is_some(), "preset {name:?} failed to load");
        }
    }

    #[test]
    fn unknown_preset_name_is_none() {
        assert!(preset("nonexistent_style").is_none());
    }

    #[test]
    fn every_preset_validates_cleanly() {
        // The actual point of externalizing these as data: a preset that
        // fails its own validator would be a self-inconsistency bug, not a
        // legitimate stylistic difference — the validator never bends per
        // style, only the spec's chosen numbers do.
        for name in PRESET_NAMES {
            let spec = preset(name).unwrap();
            let violations = validate(&spec);
            assert!(
                violations.is_empty(),
                "preset {name:?} failed its own validator: {violations:#?}"
            );
        }
    }

    #[test]
    fn round_trips_through_json() {
        for name in PRESET_NAMES {
            let spec = preset(name).unwrap();
            let json = serde_json::to_string(&spec).unwrap();
            let reparsed: CulinarySpec = serde_json::from_str(&json).unwrap();
            assert_eq!(spec, reparsed, "preset {name:?} did not round-trip");
        }
    }

    #[test]
    fn editing_json_to_an_illegal_state_is_rejected_with_the_physics_reason() {
        // The other half of the pitch: hand-edit a preset's JSON to something
        // physically impossible and confirm the SAME validator rejects it,
        // through the deserialized path, not just the Rust-constructed one
        // already covered in tests/invariants.rs.
        let mut broken: serde_json::Value = serde_json::from_str(RUSTIC_FERMENTATION).unwrap();
        broken["hydration"]["water_g"] = serde_json::json!(300.0); // 30% hydration: badly under-hydrated
        let spec: CulinarySpec = serde_json::from_value(broken).unwrap();
        let violations = validate(&spec);
        assert!(
            !violations.is_empty(),
            "expected the under-hydrated edit to be rejected"
        );
        let msg = violations[0].to_string();
        assert!(
            msg.contains("hydration") || msg.contains("Bread"),
            "violation should name the physics: {msg}"
        );
    }
}
