use anyhow::{ensure, Context, Result};
use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::fs;
use std::path::Path;

const FIXTURE_PATH: &str = "docs/research/fixtures/env-mobility-spm0-source-register-v1.json";

#[derive(Debug, Serialize)]
pub struct SourceRegisterAudit {
    pub report_schema: &'static str,
    pub registry_id: String,
    pub schema_version: String,
    pub fixture_path: String,
    pub raw_sha256: String,
    pub catalog_listing_count: usize,
    pub formal_quote_count: usize,
    pub unpriced_dependency_count: usize,
    pub derived_profile_count: usize,
    pub all_item_ids_unique: bool,
    pub all_public_listings_have_source_links: bool,
    pub no_unknown_prices_defaulted_to_zero: bool,
    pub catalog_arithmetic_matches: bool,
    pub profile_measurands_are_consistent: bool,
    pub claim_ceiling: &'static str,
}

fn object<'a>(value: &'a Value, context: &str) -> Result<&'a serde_json::Map<String, Value>> {
    value.as_object().with_context(|| format!("{context}: expected object"))
}

fn exact_keys(value: &Value, required: &[&str], context: &str) -> Result<()> {
    let actual: HashSet<&str> = object(value, context)?.keys().map(String::as_str).collect();
    let expected: HashSet<&str> = required.iter().copied().collect();
    ensure!(actual == expected, "{context}: field set differs from fixed schema");
    Ok(())
}

fn string_field<'a>(value: &'a Value, key: &str, context: &str) -> Result<&'a str> {
    value.get(key).and_then(Value::as_str)
        .with_context(|| format!("{context}: required string {key:?} is missing or invalid"))
}

fn required_string_set(value: &Value, key: &str, context: &str) -> Result<HashSet<String>> {
    let array = value.get(key).and_then(Value::as_array)
        .with_context(|| format!("{context}: {key} must be an array"))?;
    let mut values = HashSet::with_capacity(array.len());
    for (index, item) in array.iter().enumerate() {
        let string = item.as_str()
            .with_context(|| format!("{context}: {key}[{index}] must be a string"))?;
        ensure!(!string.trim().is_empty(), "{context}: {key}[{index}] is empty");
        ensure!(values.insert(string.to_owned()), "{context}: duplicate {key} value {string:?}");
    }
    Ok(values)
}

fn optional_positive_price(value: &Value, key: &str, context: &str) -> Result<Option<f64>> {
    let raw = value.get(key).with_context(|| format!("{context}: {key} missing"))?;
    if raw.is_null() {
        return Ok(None);
    }
    let price = raw.as_f64().with_context(|| format!("{context}: {key} must be a number or null"))?;
    ensure!(price.is_finite() && price > 0.0, "{context}: {key} must be finite and positive");
    Ok(Some(price))
}

fn expect_null(value: &Value, key: &str, context: &str) -> Result<()> {
    ensure!(value.get(key).is_some_and(Value::is_null), "{context}: {key} must remain null until sourced");
    Ok(())
}

fn parse_price_state(item: &Value, context: &str) -> Result<(String, Option<f64>)> {
    exact_keys(item, &[
        "item_id", "role", "vendor", "vendor_sku", "description", "product_url",
        "displayed_unit_price", "currency", "vat_included", "availability_statement",
        "price_evidence_class", "form_factor", "measurands", "substitution_profile",
        "confirmed_quote", "delivered_to_gauteng_price_known", "qualification_status",
        "calibration_status", "notes",
    ], context)?;

    let item_id = string_field(item, "item_id", context)?.to_owned();
    ensure!(!item_id.trim().is_empty(), "{context}: empty item ID");
    for key in ["role", "description", "price_evidence_class", "form_factor",
        "substitution_profile", "qualification_status", "calibration_status", "notes"] {
        ensure!(!string_field(item, key, context)?.trim().is_empty(), "{context}: {key} is empty");
    }
    ensure!(!required_string_set(item, "measurands", context)?.is_empty()
        || string_field(item, "role", context)?.contains("controller")
        || string_field(item, "role", context)?.contains("harness")
        || string_field(item, "role", context)?.contains("enclosure")
        || string_field(item, "role", context)?.contains("mounting")
        || string_field(item, "role", context)?.contains("reference")
        || string_field(item, "role", context)?.contains("cable"),
        "{context}: empty measurands need an explicit non-sensor role");
    let confirmed_quote = item.get("confirmed_quote").and_then(Value::as_bool)
        .context("confirmed_quote must be boolean")?;
    let delivered_price_known = item.get("delivered_to_gauteng_price_known").and_then(Value::as_bool)
        .context("delivered_to_gauteng_price_known must be boolean")?;
    let price = optional_positive_price(item, "displayed_unit_price", context)?;
    let evidence = string_field(item, "price_evidence_class", context)?.to_owned();

    match evidence.as_str() {
        "public_catalog_listing" => {
            ensure!(price.is_some(), "{context}: public listing must have its displayed price");
            ensure!(!confirmed_quote, "{context}: catalog listing must not be claimed as a written quote");
            ensure!(!delivered_price_known, "{context}: listing must not imply delivered price without evidence");
            ensure!(
                item.get("vendor").and_then(Value::as_str).is_some_and(|s| !s.trim().is_empty()),
                "{context}: public listing vendor required"
            );
            ensure!(
                item.get("vendor_sku").and_then(Value::as_str).is_some_and(|s| !s.trim().is_empty()),
                "{context}: public listing SKU required"
            );
            ensure!(
                item.get("product_url").and_then(Value::as_str).is_some_and(|s| s.starts_with("https://")),
                "{context}: public listing URL required"
            );
            ensure!(
                item.get("vat_included").and_then(Value::as_bool).is_some(),
                "{context}: VAT status must be explicit for a public listing"
            );
            ensure!(!string_field(item, "availability_statement", context)?.trim().is_empty(),
                "{context}: source-specific availability statement required");
        }
        "unknown_unpriced_dependency" => {
            ensure!(price.is_none(), "{context}: unknown price must be null, not zero or a guessed amount");
            expect_null(item, "vendor", context)?;
            expect_null(item, "vendor_sku", context)?;
            expect_null(item, "product_url", context)?;
            expect_null(item, "vat_included", context)?;
            ensure!(!confirmed_quote, "{context}: unknown dependency cannot have a confirmed quote");
            ensure!(!delivered_price_known, "{context}: unpriced dependency cannot claim delivery cost");
            ensure!(
                string_field(item, "availability_statement", context)?.to_lowercase().contains("not sourced")
                    || string_field(item, "availability_statement", context)?.to_lowercase().contains("not selected"),
                "{context}: unknown dependency must explain why it is unpriced"
            );
        }
        _ => anyhow::bail!("{context}: unsupported price evidence class {evidence:?}"),
    }

    Ok((item_id, price))
}

fn validate_register(value: &Value, raw_sha256: String, context: &str) -> Result<SourceRegisterAudit> {
    exact_keys(value, &[
        "registry_id", "schema_version", "status", "observed_on", "region", "currency",
        "price_basis", "claim_ceiling", "items", "derived_arithmetic_only", "next_evidence_gates",
    ], context)?;
    ensure!(string_field(value, "schema_version", context)? == "env-mobility-source-register-v1",
        "{context}: unsupported source-register schema");
    ensure!(string_field(value, "status", context)? == "catalog_snapshot_not_procurement_qualified",
        "{context}: register must remain explicitly unqualified for procurement");
    ensure!(string_field(value, "currency", context)? == "ZAR",
        "{context}: starter register currently standardizes on ZAR only");
    for key in ["registry_id", "observed_on", "region", "currency", "price_basis", "claim_ceiling"] {
        ensure!(!string_field(value, key, context)?.trim().is_empty(), "{context}: {key} is empty");
    }

    let items = value.get("items").and_then(Value::as_array).context("items must be an array")?;
    ensure!(items.len() == 8, "{context}: expected eight source/dependency records");
    let mut ids = HashSet::with_capacity(items.len());
    let mut catalog_listing_count = 0usize;
    let mut formal_quote_count = 0usize;
    let mut unpriced_dependency_count = 0usize;
    let mut item_prices = std::collections::BTreeMap::new();
    let mut item_measurands = std::collections::BTreeMap::new();
    let no_prices_defaulted = true;
    let mut all_public_links = true;

    for (index, item) in items.iter().enumerate() {
        let at = format!("{context} item[{index}]");
        let (item_id, price) = parse_price_state(item, &at)?;
        ensure!(ids.insert(item_id.clone()), "{at}: duplicate item ID {item_id:?}");
        let evidence = string_field(item, "price_evidence_class", &at)?;
        match evidence {
            "public_catalog_listing" => catalog_listing_count += 1,
            "unknown_unpriced_dependency" => unpriced_dependency_count += 1,
            _ => {}
        }
        if item.get("confirmed_quote").and_then(Value::as_bool) == Some(true) {
            formal_quote_count += 1;
        }
        if evidence == "public_catalog_listing"
            && item.get("product_url").and_then(Value::as_str).is_none()
        {
            all_public_links = false;
        }
        if evidence == "unknown_unpriced_dependency" && price == Some(0.0) {
            no_prices_defaulted = false;
        }
        item_prices.insert(item_id.clone(), price);
        item_measurands.insert(item_id, required_string_set(item, "measurands", &at)?);
    }

    let profiles = value.get("derived_arithmetic_only").and_then(Value::as_array)
        .context("derived_arithmetic_only must be an array")?;
    ensure!(profiles.len() == 2, "{context}: expected two explicit arithmetic-only profiles");
    let mut profile_ids = HashSet::new();
    let catalog_arithmetic_matches = true;
    let profile_measurands_consistent = true;

    for (index, profile) in profiles.iter().enumerate() {
        let at = format!("{context} profile[{index}]");
        exact_keys(profile, &[
            "option_id", "items", "displayed_electronics_subtotal", "currency",
            "vat_included", "status", "caveat",
        ], &[], &at)?;
        let option_id = string_field(profile, "option_id", &at)?;
        ensure!(profile_ids.insert(option_id.to_owned()), "{at}: duplicate profile ID");
        ensure!(string_field(profile, "currency", &at)? == "ZAR", "{at}: currency mismatch");
        ensure!(string_field(profile, "status", &at)? == "catalog arithmetic only",
            "{at}: derived subtotal cannot imply a quote or qualified BOM");
        ensure!(profile.get("vat_included").and_then(Value::as_bool) == Some(true),
            "{at}: subtotal VAT basis must be explicit");
        let listed_items = required_string_set(profile, "items", &at)?;
        ensure!(!listed_items.is_empty(), "{at}: profile has no items");
        let mut sum = 0.0;
        for id in &listed_items {
            let price = item_prices.get(id).with_context(|| format!("{at}: unknown item reference {id:?}"))?
                .with_context(|| format!("{at}: unpriced dependency {id:?} cannot enter a priced subtotal"))?;
            let item = items.iter().find(|item| item.get("item_id").and_then(Value::as_str) == Some(id.as_str()))
                .context("known item must exist")?;
            ensure!(item.get("currency").and_then(Value::as_str) == Some("ZAR"),
                "{at}: item currency mismatch");
            ensure!(item.get("vat_included").and_then(Value::as_bool) == Some(true),
                "{at}: all listed items must have the same VAT basis");
            sum += price;
        }
        let declared = profile.get("displayed_electronics_subtotal").and_then(Value::as_f64)
            .context("displayed_electronics_subtotal must be numeric")?;
        ensure!(declared.is_finite() && declared > 0.0, "{at}: subtotal must be positive");
        ensure!(
            (sum - declared).abs() <= 1e-6,
            "{at}: displayed subtotal does not match the sum of sourced catalog prices"
        );
        let expected_items: HashSet<String> = if option_id == "SPM0-PRESSURE-TEMPERATURE-BENCH-PAIR" {
            ["SPM0-MCU-ESP32C3-SUPERMINI", "SPM0-SENSOR-PRESSURE-TEMP-BMP280"]
                .into_iter().map(str::to_owned).collect()
        } else if option_id == "SPM0-TEMP-HUMIDITY-PRESSURE-BENCH-PAIR" {
            ["SPM0-MCU-ESP32C3-SUPERMINI", "SPM0-SENSOR-ENV3-BME280"]
                .into_iter().map(str::to_owned).collect()
        } else {
            anyhow::bail!("{at}: unexpected profile ID {option_id:?}")
        };
        ensure!(listed_items == expected_items, "{at}: profile items differ from frozen variant contract");

        let bme = item_measurands.get("SPM0-SENSOR-ENV3-BME280").context("BME280 profile item missing")?;
        let bmp = item_measurands.get("SPM0-SENSOR-PRESSURE-TEMP-BMP280").context("BMP280 profile item missing")?;
        ensure!(
            bme.contains("temperature")
                && bme.contains("relative_humidity")
                && bme.contains("barometric_pressure"),
            "{at}: the BME280 profile must declare all three measurands"
        );
        ensure!(
            bmp.contains("temperature")
                && bmp.contains("barometric_pressure")
                && !bmp.contains("relative_humidity"),
            "{at}: the BMP280 profile must not claim a humidity measurand"
        );
    }

    let evidence_gates = value.get("next_evidence_gates").and_then(Value::as_array)
        .context("next_evidence_gates must be an array")?;
    ensure!(evidence_gates.len() >= 5, "{context}: required evidence gates are missing");
    for (index, gate) in evidence_gates.iter().enumerate() {
        ensure!(gate.as_str().is_some_and(|s| !s.trim().is_empty()),
            "{context}: evidence gate {index} must be a non-empty string");
    }

    Ok(SourceRegisterAudit {
        report_schema: "env-mobility-source-register-audit-v1",
        registry_id: string_field(value, "registry_id", context)?.to_owned(),
        schema_version: string_field(value, "schema_version", context)?.to_owned(),
        fixture_path: FIXTURE_PATH.to_owned(),
        raw_sha256,
        catalog_listing_count,
        formal_quote_count,
        unpriced_dependency_count,
        derived_profile_count: profiles.len(),
        all_item_ids_unique: ids.len() == items.len(),
        all_public_listings_have_source_links: all_public_links,
        no_unknown_prices_defaulted_to_zero: no_prices_defaulted,
        catalog_arithmetic_matches,
        profile_measurands_are_consistent: profile_measurands_consistent,
        claim_ceiling: "Catalog-source integrity and arithmetic only; no written quote, actual delivered cost, SPM-0 qualification, calibration, production capability or purchase recommendation is established.",
    })
}

pub fn run() -> Result<SourceRegisterAudit> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent().context("xtask must be one level below workspace root")?;
    let path = root.join(FIXTURE_PATH);
    let bytes = fs::read(&path).with_context(|| format!("read source register {}", path.display()))?;
    let digest = format!("{:x}", Sha256::digest(&bytes));
    let value: Value = serde_json::from_slice(&bytes)
        .with_context(|| format!("parse source register {}", path.display()))?;
    validate_register(&value, digest, FIXTURE_PATH)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Value {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().expect("workspace root");
        let bytes = fs::read(root.join(FIXTURE_PATH)).expect("read source register");
        serde_json::from_slice(&bytes).expect("parse source register")
    }

    #[test]
    fn source_register_preserves_prices_evidence_classes_and_profile_arithmetic() {
        let report = run().expect("audit source register");
        assert_eq!(report.catalog_listing_count, 5);
        assert_eq!(report.formal_quote_count, 0);
        assert_eq!(report.unpriced_dependency_count, 3);
        assert_eq!(report.derived_profile_count, 2);
        assert!(report.all_item_ids_unique);
        assert!(report.all_public_listings_have_source_links);
        assert!(report.no_unknown_prices_defaulted_to_zero);
        assert!(report.catalog_arithmetic_matches);
        assert!(report.profile_measurands_are_consistent);
    }

    #[test]
    fn unknown_price_cannot_be_converted_to_zero_or_a_quote() {
        let mut value = fixture();
        value["items"][5]["displayed_unit_price"] = serde_json::json!(0);
        assert!(validate_register(&value, "mutation".into(), "test guessed zero price").is_err());

        let mut value = fixture();
        value["items"][0]["confirmed_quote"] = serde_json::json!(true);
        assert!(validate_register(&value, "mutation".into(), "test catalog promoted to quote").is_err());

        let mut value = fixture();
        value["derived_arithmetic_only"][0]["displayed_electronics_subtotal"] =
            serde_json::json!(999);
        assert!(validate_register(&value, "mutation".into(), "test subtotal drift").is_err());
    }

    #[test]
    fn sensor_profiles_must_not_conflate_humidity_capability() {
        let mut value = fixture();
        value["items"][3]["measurands"] =
            serde_json::json!(["temperature", "relative_humidity", "barometric_pressure"]);
        assert!(validate_register(&value, "mutation".into(), "test false humidity claim").is_err());
    }
}
