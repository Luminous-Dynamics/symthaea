use symthaea_quantum_comp::{
    ReleaseChannel, catalog_has_no_deprecated_surfaces, current_api_inventory,
    current_release_manifest, known_schema_labels, stability_catalog,
};

#[test]
fn alpha10_schema_labels_are_current() {
    assert!(
        known_schema_labels()
            .iter()
            .all(|label| label.ends_with("alpha10"))
    );
}

#[test]
fn api_inventory_contains_stability_records() {
    let inventory = current_api_inventory();
    assert!(!inventory.stability_records.is_empty());
    assert!(inventory.to_text().contains("schemas="));
    assert!(inventory.to_markdown().contains("Stability catalog"));
}

#[test]
fn stability_catalog_has_no_deprecated_surfaces() {
    let catalog = stability_catalog();
    assert!(catalog_has_no_deprecated_surfaces(&catalog));
}

#[test]
fn release_manifest_blocks_overclaims() {
    let manifest = current_release_manifest();
    assert_eq!(manifest.channel, ReleaseChannel::Alpha);
    assert!(manifest.blocked_claims.contains(&"quantum consciousness"));
    assert!(manifest.to_text().contains("quantum advantage"));
}
