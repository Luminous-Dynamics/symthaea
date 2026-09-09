mod reference {
    include!("hak_canonical_json_rust.rs");

    const PROFILE_MANIFEST: &str = include_str!("../docs/architecture/hak/canonical-json-v1.profile.json");

    fn exact_object_keys(value: &Value, expected: &[&str], label: &str) -> Result<(), String> {
        let fields = match value {
            Value::Object(fields) => fields,
            _ => return Err(format!("{label} must be an object")),
        };
        if fields.len() != expected.len() {
            return Err(format!("{label} key count differs: got {}, expected {}", fields.len(), expected.len()));
        }
        for expected_key in expected {
            if !fields.iter().any(|(key, _)| key == expected_key) {
                return Err(format!("{label} missing key {expected_key}"));
            }
        }
        for (key, _) in fields {
            if !expected.iter().any(|expected_key| key == expected_key) {
                return Err(format!("{label} contains unknown key {key}"));
            }
        }
        Ok(())
    }

    fn expect_str(value: &Value, key: &str, expected: &str, label: &str) -> Result<(), String> {
        match object_get(value, key) {
            Value::Str(actual) if actual == expected => Ok(()),
            Value::Str(actual) => Err(format!("{label}.{key}={actual:?}, expected {expected:?}")),
            _ => Err(format!("{label}.{key} must be a string")),
        }
    }

    fn expect_bool(value: &Value, key: &str, expected: bool, label: &str) -> Result<(), String> {
        match object_get(value, key) {
            Value::Bool(actual) if *actual == expected => Ok(()),
            Value::Bool(actual) => Err(format!("{label}.{key}={actual}, expected {expected}")),
            _ => Err(format!("{label}.{key} must be a boolean")),
        }
    }

    fn expect_int(value: &Value, key: &str, expected: i64, label: &str) -> Result<(), String> {
        match object_get(value, key) {
            Value::Int(actual) if *actual == expected => Ok(()),
            Value::Int(actual) => Err(format!("{label}.{key}={actual}, expected {expected}")),
            _ => Err(format!("{label}.{key} must be an integer")),
        }
    }

    fn validate_profile_contract(profile: &Value) -> Result<(), String> {
        exact_object_keys(
            profile,
            &["schema_version", "profile_id", "base_standard", "input_contract", "serialization", "digest_contract", "migration"],
            "profile",
        )?;
        expect_str(profile, "schema_version", "hak.canonical-json-profile.v1", "profile")?;
        expect_str(profile, "profile_id", PROFILE_ID, "profile")?;

        let base = object_get(profile, "base_standard");
        exact_object_keys(base, &["name", "relationship"], "base_standard")?;
        expect_str(base, "name", "RFC 8785", "base_standard")?;
        expect_str(base, "relationship", "CompatibleSubset", "base_standard")?;

        let input = object_get(profile, "input_contract");
        exact_object_keys(input, &["encoding", "duplicate_object_names", "unicode_strings", "numbers"], "input_contract")?;
        expect_str(input, "encoding", "UTF-8", "input_contract")?;
        expect_str(input, "duplicate_object_names", "Reject", "input_contract")?;
        expect_str(input, "unicode_strings", "UnicodeScalarValuesOnly", "input_contract")?;

        let numbers = object_get(input, "numbers");
        exact_object_keys(numbers, &["profile", "minimum", "maximum", "floating_point_allowed", "negative_zero_canonicalizes_to"], "input_contract.numbers")?;
        expect_str(numbers, "profile", "SafeIntegerOnly", "input_contract.numbers")?;
        expect_int(numbers, "minimum", -(MAX_SAFE_INTEGER as i64), "input_contract.numbers")?;
        expect_int(numbers, "maximum", MAX_SAFE_INTEGER as i64, "input_contract.numbers")?;
        expect_bool(numbers, "floating_point_allowed", false, "input_contract.numbers")?;
        expect_int(numbers, "negative_zero_canonicalizes_to", 0, "input_contract.numbers")?;

        let serialization = object_get(profile, "serialization");
        exact_object_keys(
            serialization,
            &["object_key_order", "object_sorting_recursive", "array_order", "whitespace", "string_escaping", "unicode_normalization", "output_encoding"],
            "serialization",
        )?;
        expect_str(serialization, "object_key_order", "RFC8785Utf16CodeUnits", "serialization")?;
        expect_bool(serialization, "object_sorting_recursive", true, "serialization")?;
        expect_str(serialization, "array_order", "PreserveExactly", "serialization")?;
        expect_str(serialization, "whitespace", "None", "serialization")?;
        expect_str(serialization, "string_escaping", "RFC8785Compatible", "serialization")?;
        expect_str(serialization, "unicode_normalization", "None", "serialization")?;
        expect_str(serialization, "output_encoding", "UTF-8", "serialization")?;

        let digest = object_get(profile, "digest_contract");
        exact_object_keys(digest, &["algorithm", "preimage", "domain_must_be_nonempty", "domain_must_not_contain_nul"], "digest_contract")?;
        expect_str(digest, "algorithm", "SHA-256", "digest_contract")?;
        expect_str(digest, "preimage", "UTF8(profile_id) || 0x00 || UTF8(domain) || 0x00 || canonical_bytes", "digest_contract")?;
        expect_bool(digest, "domain_must_be_nonempty", true, "digest_contract")?;
        expect_bool(digest, "domain_must_not_contain_nul", true, "digest_contract")?;

        let migration = object_get(profile, "migration");
        exact_object_keys(migration, &["reinterpret_existing_digests", "historical_python_sort_keys_digests_remain_historical"], "migration")?;
        expect_bool(migration, "reinterpret_existing_digests", false, "migration")?;
        expect_bool(migration, "historical_python_sort_keys_digests_remain_historical", true, "migration")?;
        Ok(())
    }

    pub fn run_profile_contract() -> Result<(), String> {
        let profile = parse_strict_json(PROFILE_MANIFEST)?;
        validate_profile_contract(&profile)?;

        let mut unknown_root = profile.clone();
        match &mut unknown_root {
            Value::Object(fields) => fields.push(("unexpected_semantic_field".to_string(), Value::Bool(true))),
            _ => return Err("profile root unexpectedly not object".to_string()),
        }
        if validate_profile_contract(&unknown_root).is_ok() {
            return Err("Rust exact-shape validator accepted unknown root profile field".to_string());
        }

        let mut unknown_nested = profile.clone();
        match &mut unknown_nested {
            Value::Object(fields) => {
                let serialization = fields.iter_mut().find(|(key, _)| key == "serialization")
                    .ok_or_else(|| "missing serialization object".to_string())?;
                match &mut serialization.1 {
                    Value::Object(nested) => nested.push(("unexpected_nested_semantics".to_string(), Value::Bool(true))),
                    _ => return Err("serialization unexpectedly not object".to_string()),
                }
            }
            _ => return Err("profile root unexpectedly not object".to_string()),
        }
        if validate_profile_contract(&unknown_nested).is_ok() {
            return Err("Rust exact-shape validator accepted unknown nested profile field".to_string());
        }
        Ok(())
    }
}

fn main() {
    if let Err(error) = reference::run_profile_contract() {
        eprintln!("HAK-015 Rust profile contract FAIL: {error}");
        std::process::exit(1);
    }
    println!("HAK-015 Rust profile contract OK");
}
