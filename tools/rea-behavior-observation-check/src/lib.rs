//! Structural and capture-integrity checks for REA behavior-observation sidecars.
//!
//! This validates synthetic development data. It does not verify the referenced
//! REA Evidence bytes or authenticate the runner/provider.

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD;
use jsonschema::draft202012;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

const SCHEMA: &str =
    include_str!("../../../docs/research/rea-behavior-observation-profile-v0.schema.json");
const FIXTURE: &str =
    include_str!("../../../docs/research/rea-behavior-observation-profile-v0.synthetic.json");
const MAX_PROFILE_BYTES: usize = 4 * 1024 * 1024;
const MAX_CAPTURE_BYTES: usize = 4 * 1024 * 1024;
const MAX_PROFILE_BYTES: usize = 4 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObservationError(pub String);

impl std::fmt::Display for ObservationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}
impl std::error::Error for ObservationError {}

fn get_str<'a>(value: &'a Value, key: &str, context: &str) -> Result<&'a str, ObservationError> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| ObservationError(format!("{context}.{key} must be a string")))
}

fn digest_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        write!(&mut out, "{byte:02x}").expect("String formatting cannot fail");
    }
    out
}

fn validate_byte_capture(
    value: &Value,
    context: &str,
    require_complete: bool,
) -> Result<(), ObservationError> {
    let state = get_str(value, "state", context)?;
    if require_complete && state != "complete" {
        return Err(ObservationError(format!(
            "{context} must be complete for this assertion"
        )));
    }
    match state {
        "complete" | "partial" => {
            let encoded = get_str(value, "bytes_base64", context)?;
            let bytes = STANDARD
                .decode(encoded)
                .map_err(|error| ObservationError(format!("{context}.bytes_base64 invalid: {error}")))?;
            if bytes.len() > MAX_CAPTURE_BYTES {
                return Err(ObservationError(format!(
                    "{context} capture exceeds {MAX_CAPTURE_BYTES} bytes"
                )));
            }
            let declared_length = value
                .get("byte_length")
                .and_then(Value::as_u64)
                .ok_or_else(|| ObservationError(format!("{context}.byte_length must be an integer")))?;
            if declared_length != bytes.len() as u64 {
                return Err(ObservationError(format!(
                    "{context}.byte_length does not match decoded bytes"
                )));
            }
            let declared_digest = get_str(value, "sha256", context)?;
            if declared_digest != digest_hex(&bytes) {
                return Err(ObservationError(format!(
                    "{context}.sha256 does not match decoded bytes"
                )));
            }
        }
        "unavailable" => {
            if value.get("bytes_base64").is_some()
                || value.get("sha256").is_some()
                || value.get("byte_length").is_some()
            {
                return Err(ObservationError(format!(
                    "{context} unavailable state must not carry captured bytes/digest/length"
                )));
            }
            let _reason = get_str(value, "reason", context)?;
        }
        "not_applicable" => {
            if value.get("bytes_base64").is_some()
                || value.get("sha256").is_some()
                || value.get("byte_length").is_some()
            {
                return Err(ObservationError(format!(
                    "{context} not_applicable state must not carry captured bytes/digest/length"
                )));
            }
        }
        other => {
            return Err(ObservationError(format!(
                "{context} has unsupported capture state {other}"
            )));
        }
    }
    Ok(())
}

fn safe_relative_path(path: &str) -> bool {
    if path.is_empty() || path.starts_with('/') || path.starts_with('\\') || path.contains('\\') {
        return false;
    }
    let parts: Vec<&str> = path.split('/').collect();
    !parts.iter().any(|part| part.is_empty() || *part == "." || *part == "..")
}

fn validate_filesystem(value: &Value, context: &str) -> Result<(), ObservationError> {
    let state = get_str(value, "state", context)?;
    let entries = value
        .get("entries")
        .and_then(Value::as_array)
        .ok_or_else(|| ObservationError(format!("{context}.entries must be an array")))?;
    if state == "unavailable" {
        if !entries.is_empty() {
            return Err(ObservationError(format!(
                "{context} unavailable state must not contain entries"
            )));
        }
        let _reason = get_str(value, "reason", context)?;
        return Ok(());
    }
    if state == "partial" {
        let _reason = get_str(value, "reason", context)?;
    } else if state != "complete" {
        return Err(ObservationError(format!(
            "{context} unsupported filesystem state {state}"
        )));
    }

    let mut seen = BTreeSet::new();
    for (index, entry) in entries.iter().enumerate() {
        let entry_context = format!("{context}.entries[{index}]");
        let path = get_str(entry, "path", &entry_context)?;
        if !safe_relative_path(path) {
            return Err(ObservationError(format!(
                "{entry_context}.path must be a normalized relative path within the sandbox"
            )));
        }
        if !seen.insert(path.to_owned()) {
            return Err(ObservationError(format!(
                "{context} contains duplicate filesystem path {path}"
            )));
        }
        let change = get_str(entry, "change", &entry_context)?;
        let before = entry.get("before_sha256").and_then(Value::as_str);
        let after = entry.get("after_sha256").and_then(Value::as_str);
        let valid = match change {
            "created" => before.is_none() && after.is_some(),
            "modified" => before.is_some() && after.is_some(),
            "deleted" => before.is_some() && after.is_none(),
            _ => false,
        };
        if !valid {
            return Err(ObservationError(format!(
                "{entry_context} has inconsistent change/before/after digest fields"
            )));
        }
    }
    Ok(())
}

fn validate_observation(value: &Value) -> Result<(), ObservationError> {
    let id = get_str(value, "observation_id", "observation")?;
    if get_str(value, "claim_scope", id)? != "exact_invocation_and_environment_only" {
        return Err(ObservationError(format!(
            "{id} must remain exact-invocation scoped"
        )));
    }
    let binding = value.get("evidence_binding").ok_or_else(|| {
        ObservationError(format!("{id}.evidence_binding missing"))
    })?;
    if get_str(binding, "admission_state", id)? != "unverified_reference" {
        return Err(ObservationError(format!(
            "{id}: v0 cannot claim REA Evidence admission or authenticity"
        )));
    }
    if get_str(binding, "subject_sha256", id)?
        != get_str(value.get("target").ok_or_else(|| ObservationError("target missing".into()))?, "subject_sha256", id)?
    {
        return Err(ObservationError(format!(
            "{id}: Evidence binding subject digest differs from target subject digest"
        )));
    }

    let invocation = value.get("invocation").ok_or_else(|| {
        ObservationError(format!("{id}.invocation missing"))
    })?;
    let argv = invocation.get("argv").and_then(Value::as_array).ok_or_else(|| {
        ObservationError(format!("{id}.invocation.argv must be an array"))
    })?;
    if argv.is_empty() {
        return Err(ObservationError(format!("{id} argv must not be empty")));
    }
    let mut total_argv_bytes = 0usize;
    for (index, argument) in argv.iter().enumerate() {
        let context = format!("{id}.argv[{index}]");
        let encoded = get_str(argument, "bytes_base64", &context)?;
        let bytes = STANDARD
            .decode(encoded)
            .map_err(|error| ObservationError(format!("{context} invalid base64: {error}")))?;
        total_argv_bytes = total_argv_bytes.saturating_add(bytes.len());
        if bytes.is_empty() {
            return Err(ObservationError(format!("{context} argument must not be empty")));
        }
        if bytes.len() > 16 * 1024 {
            return Err(ObservationError(format!("{context} argument exceeds 16 KiB")));
        }
        if argument.get("byte_length").and_then(Value::as_u64) != Some(bytes.len() as u64)
            || get_str(argument, "sha256", &context)? != digest_hex(&bytes)
        {
            return Err(ObservationError(format!(
                "{context} length/digest does not match exact argument bytes"
            )));
        }
    }
    if total_argv_bytes > 64 * 1024 {
        return Err(ObservationError(format!("{id} total argv exceeds 64 KiB")));
    }

    let stdin = invocation.get("stdin").ok_or_else(|| ObservationError(format!("{id} stdin missing")))?;
    validate_byte_capture(stdin, &format!("{id}.stdin"), false)?;

    let cwd = get_str(invocation, "working_directory", id)?;
    if !safe_relative_path(cwd) && cwd != "." {
        return Err(ObservationError(format!(
            "{id} working_directory must be normalized and sandbox-relative"
        )));
    }

    let mut env_names = BTreeSet::new();
    for variable in invocation
        .get("environment")
        .and_then(Value::as_array)
        .ok_or_else(|| ObservationError(format!("{id} environment must be an array")))? {
        let name = get_str(variable, "name", id)?;
        if !env_names.insert(name.to_owned()) {
            return Err(ObservationError(format!("{id} has duplicate environment variable {name}")));
        }
        let state = get_str(variable, "state", id)?;
        match state {
            "captured" => {
                let capture = variable.get("value").ok_or_else(|| {
                    ObservationError(format!("{id} captured environment value missing"))
                })?;
                validate_byte_capture(capture, &format!("{id}.environment.{name}"), true)?;
            }
            "redacted" => {
                let _digest = get_str(variable, "value_sha256", id)?;
                let _reason = get_str(variable, "reason", id)?;
                if variable.get("value").is_some() {
                    return Err(ObservationError(format!(
                        "{id} redacted environment variable must not expose value bytes"
                    )));
                }
            }
            "absent" => {
                if variable.get("value").is_some() || variable.get("value_sha256").is_some() {
                    return Err(ObservationError(format!(
                        "{id} absent environment variable must not carry a value"
                    )));
                }
            }
            other => {
                return Err(ObservationError(format!(
                    "{id} unsupported environment state {other}"
                )));
            }
        }
    }

    let result = value.get("result").ok_or_else(|| ObservationError(format!("{id}.result missing")))?;
    let termination = result.get("termination").ok_or_else(|| ObservationError(format!("{id}.termination missing")))?;
    let termination_kind = get_str(termination, "kind", id)?;
    match termination_kind {
        "exit" if termination.get("exit_code").and_then(Value::as_i64).is_none() || termination.get("signal").is_some() => {
            return Err(ObservationError(format!("{id} exit termination needs only exit_code")));
        }
        "signal" if termination.get("signal").and_then(Value::as_u64).is_none() || termination.get("exit_code").is_some() => {
            return Err(ObservationError(format!("{id} signal termination needs only signal")));
        }
        "timeout" | "launch_error" if termination.get("reason").and_then(Value::as_str).is_none() || termination.get("exit_code").is_some() || termination.get("signal").is_some() => {
            return Err(ObservationError(format!("{id} {termination_kind} termination requires a reason and no exit/signal")));
        }
        "exit" | "signal" | "timeout" | "launch_error" => {}
        other => return Err(ObservationError(format!("{id} unsupported termination kind {other}"))),
    }

    validate_byte_capture(result.get("stdout").ok_or_else(|| ObservationError(format!("{id} stdout missing")))?, &format!("{id}.stdout"), false)?;
    validate_byte_capture(result.get("stderr").ok_or_else(|| ObservationError(format!("{id} stderr missing")))?, &format!("{id}.stderr"), false)?;
    validate_filesystem(result.get("filesystem").ok_or_else(|| ObservationError(format!("{id} filesystem missing")))?, &format!("{id}.filesystem"))?;

    let assertions = value.get("assertions").and_then(Value::as_array).ok_or_else(|| ObservationError(format!("{id}.assertions must be an array")))?;
    let mut assertion_ids = BTreeSet::new();
    for assertion in assertions {
        let assertion_id = get_str(assertion, "assertion_id", id)?;
        if !assertion_ids.insert(assertion_id.to_owned()) {
            return Err(ObservationError(format!("{id} duplicate assertion ID {assertion_id}")));
        }
        let predicate = get_str(assertion, "predicate", id)?;
        let capture_ref = get_str(assertion, "capture_ref", id)?;
        match (predicate, capture_ref) {
            ("stdout_sha256", "stdout") => {
                validate_byte_capture(&result["stdout"], &format!("{id}.stdout assertion"), true)?;
                if assertion["observed_value"].as_str() != result["stdout"]["sha256"].as_str() {
                    return Err(ObservationError(format!("{id} stdout assertion does not match captured digest")));
                }
            }
            ("stderr_sha256", "stderr") => {
                validate_byte_capture(&result["stderr"], &format!("{id}.stderr assertion"), true)?;
                if assertion["observed_value"].as_str() != result["stderr"]["sha256"].as_str() {
                    return Err(ObservationError(format!("{id} stderr assertion does not match captured digest")));
                }
            }
            ("exit_code", "termination") if termination_kind == "exit" => {
                if assertion["observed_value"].as_i64() != termination["exit_code"].as_i64() {
                    return Err(ObservationError(format!("{id} exit-code assertion does not match termination")));
                }
            }
            ("signal", "termination") if termination_kind == "signal" => {
                if assertion["observed_value"].as_u64() != termination["signal"].as_u64() {
                    return Err(ObservationError(format!("{id} signal assertion does not match termination")));
                }
            }
            ("filesystem_diff_empty", "filesystem") => {
                let filesystem = &result["filesystem"];
                if get_str(filesystem, "state", id)? != "complete" {
                    return Err(ObservationError(format!("{id} filesystem assertion requires a complete diff")));
                }
                let empty = filesystem["entries"].as_array().is_some_and(Vec::is_empty);
                if assertion["observed_value"].as_bool() != Some(empty) {
                    return Err(ObservationError(format!("{id} filesystem-empty assertion does not match captured diff")));
                }
            }
            ("filesystem_path_present", "filesystem") => {
                let filesystem = &result["filesystem"];
                if get_str(filesystem, "state", id)? != "complete" {
                    return Err(ObservationError(format!("{id} filesystem path assertion requires a complete diff")));
                }
                let path = assertion["observed_value"].as_str().ok_or_else(|| {
                    ObservationError(format!("{id} filesystem_path_present observed_value must be a path string"))
                })?;
                let found = filesystem["entries"].as_array().is_some_and(|entries| {
                    entries.iter().any(|entry| entry["path"].as_str() == Some(path))
                });
                if !found {
                    return Err(ObservationError(format!("{id} asserted filesystem path is absent from captured diff")));
                }
            }
            _ => return Err(ObservationError(format!("{id} assertion predicate/capture reference is incompatible or incomplete"))),
        }
    }
    Ok(())
}

/// Parse JSON while rejecting duplicate object keys at every nesting level.
pub fn parse_unique_json(bytes: &[u8]) -> Result<Value, ObservationError> {
    use serde::de::{self, Deserialize, MapAccess, SeqAccess, Visitor};
    use serde_json::{Map, Number};
    use std::collections::HashSet;
    use std::fmt;

    struct UniqueValue(Value);
    struct UniqueVisitor;

    impl<'de> Deserialize<'de> for UniqueValue {
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: serde::Deserializer<'de>,
        {
            deserializer.deserialize_any(UniqueVisitor)
        }
    }

    impl<'de> Visitor<'de> for UniqueVisitor {
        type Value = UniqueValue;

        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("JSON with unique object keys")
        }

        fn visit_bool<E>(self, value: bool) -> Result<Self::Value, E> {
            Ok(UniqueValue(Value::Bool(value)))
        }
        fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E> {
            Ok(UniqueValue(Value::Number(Number::from(value))))
        }
        fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E> {
            Ok(UniqueValue(Value::Number(Number::from(value))))
        }
        fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Number::from_f64(value)
                .map(|number| UniqueValue(Value::Number(number)))
                .ok_or_else(|| E::custom("JSON number must be finite"))
        }
        fn visit_str<E>(self, value: &str) -> Result<Self::Value, E> {
            Ok(UniqueValue(Value::String(value.to_owned())))
        }
        fn visit_string<E>(self, value: String) -> Result<Self::Value, E> {
            Ok(UniqueValue(Value::String(value)))
        }
        fn visit_unit<E>(self) -> Result<Self::Value, E> {
            Ok(UniqueValue(Value::Null))
        }
        fn visit_none<E>(self) -> Result<Self::Value, E> {
            Ok(UniqueValue(Value::Null))
        }
        fn visit_some<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
        where
            D: serde::Deserializer<'de>,
        {
            UniqueValue::deserialize(deserializer)
        }
        fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
        where
            A: SeqAccess<'de>,
        {
            let mut values = Vec::new();
            while let Some(item) = sequence.next_element::<UniqueValue>()? {
                values.push(item.0);
            }
            Ok(UniqueValue(Value::Array(values)))
        }
        fn visit_map<A>(self, mut object: A) -> Result<Self::Value, A::Error>
        where
            A: MapAccess<'de>,
        {
            let mut values = Map::new();
            let mut seen = HashSet::new();
            while let Some((key, item)) = object.next_entry::<String, UniqueValue>()? {
                if !seen.insert(key.clone()) {
                    return Err(de::Error::custom(format!("duplicate JSON object key '{key}'")));
                }
                values.insert(key, item.0);
            }
            Ok(UniqueValue(Value::Object(values)))
        }
    }

    if bytes.len() > MAX_PROFILE_BYTES {
        return Err(ObservationError(format!(
            "profile exceeds {MAX_PROFILE_BYTES} bytes"
        )));
    }
    let mut deserializer = serde_json::Deserializer::from_slice(bytes);
    let parsed = UniqueValue::deserialize(&mut deserializer)
        .map_err(|error| ObservationError(format!("profile JSON rejected: {error}")))?;
    deserializer
        .end()
        .map_err(|error| ObservationError(format!("profile has trailing data: {error}")))?;
    Ok(parsed.0)
}

/// Validate a caller-supplied raw profile against the embedded schema and v0 semantics.
pub fn validate_profile_bytes(bytes: &[u8]) -> Result<Value, ObservationError> {
    let schema: Value = serde_json::from_str(SCHEMA)
        .map_err(|error| ObservationError(format!("schema JSON parse failed: {error}")))?;
    draft202012::meta::validate(&schema)
        .map_err(|error| ObservationError(format!("Draft 2020-12 schema invalid: {error}")))?;
    let profile = parse_unique_json(bytes)?;
    let validator = draft202012::options()
        .build(&schema)
        .map_err(|error| ObservationError(format!("schema compilation failed: {error}")))?;
    let errors: Vec<String> = validator
        .iter_errors(&profile)
        .map(|error| error.to_string())
        .collect();
    if !errors.is_empty() {
        return Err(ObservationError(format!(
            "profile fails Draft 2020-12: {}",
            errors.join("; ")
        )));
    }
    validate_profile(&profile)?;
    let count = profile["observations"].as_array().map_or(0, Vec::len);
    Ok(serde_json::json!({
        "schema":"luminous.rea.behavior-observation-validation-report.v0",
        "status":"capture_valid_evidence_admission_unverified",
        "observation_count":count,
        "evidence_admission":"unverified_reference",
        "claim_ceiling":"Captured byte integrity and sidecar consistency only; referenced REA Evidence, producer, runner and real execution are not authenticated."
    }))
}

pub fn validate_profile(value: &Value) -> Result<(), ObservationError> {
    if get_str(value, "status", "profile")? != "design_fixture_only"
        || get_str(value, "source_policy", "profile")? != "original_synthetic_capture_data_only"
        || get_str(value, "claim_scope", "profile")? != "exact_invocation_and_environment_only"
    {
        return Err(ObservationError(
            "profile must preserve synthetic-only, design-only, exact-invocation status".into(),
        ));
    }
    let observations = value.get("observations").and_then(Value::as_array).ok_or_else(|| {
        ObservationError("observations must be an array".into())
    })?;
    let mut ids = BTreeSet::new();
    for observation in observations {
        let id = get_str(observation, "observation_id", "observation")?;
        if !ids.insert(id.to_owned()) {
            return Err(ObservationError(format!("duplicate observation_id {id}")));
        }
        validate_observation(observation)?;
    }
    Ok(())
}

pub fn validate_embedded_profile() -> Result<(), ObservationError> {
    let schema: Value = serde_json::from_str(SCHEMA)
        .map_err(|error| ObservationError(format!("schema JSON parse failed: {error}")))?;
    draft202012::meta::validate(&schema)
        .map_err(|error| ObservationError(format!("Draft 2020-12 schema invalid: {error}")))?;
    let fixture: Value = serde_json::from_str(FIXTURE)
        .map_err(|error| ObservationError(format!("fixture JSON parse failed: {error}")))?;
    let validator = draft202012::options()
        .build(&schema)
        .map_err(|error| ObservationError(format!("schema compilation failed: {error}")))?;
    let errors: Vec<String> = validator
        .iter_errors(&fixture)
        .map(|error| error.to_string())
        .collect();
    if !errors.is_empty() {
        return Err(ObservationError(format!(
            "fixture fails Draft 2020-12: {}",
            errors.join("; ")
        )));
    }
    validate_profile(&fixture)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Value {
        serde_json::from_str(FIXTURE).unwrap()
    }

    #[test]
    fn schema_is_meta_valid_and_synthetic_profile_conforms() {
        validate_embedded_profile().unwrap();
    }

    #[test]
    fn all_complete_capture_digests_and_lengths_match() {
        let value = fixture();
        for observation in value["observations"].as_array().unwrap() {
            let result = &observation["result"];
            validate_byte_capture(&observation["invocation"]["stdin"], "stdin", false).unwrap();
            validate_byte_capture(&result["stdout"], "stdout", false).unwrap();
            validate_byte_capture(&result["stderr"], "stderr", false).unwrap();
        }
    }

    #[test]
    fn digest_tampering_is_rejected() {
        let mut value = fixture();
        value["observations"][0]["result"]["stdout"]["sha256"] = Value::String("0".repeat(64));
        assert!(validate_profile(&value).is_err());
    }

    #[test]
    fn byte_length_tampering_is_rejected() {
        let mut value = fixture();
        value["observations"][0]["result"]["stdout"]["byte_length"] = Value::from(999);
        assert!(validate_profile(&value).is_err());
    }

    #[test]
    fn duplicate_observation_id_is_rejected() {
        let mut value = fixture();
        value["observations"][1]["observation_id"] = value["observations"][0]["observation_id"].clone();
        assert!(validate_profile(&value).is_err());
    }

    #[test]
    fn evidence_binding_is_never_upgraded_by_the_sidecar() {
        let mut value = fixture();
        value["observations"][0]["evidence_binding"]["admission_state"] = Value::String("verified".into());
        assert!(validate_profile(&value).is_err());
    }

    #[test]
    fn subject_digest_must_match_sidecar_target() {
        let mut value = fixture();
        value["observations"][0]["evidence_binding"]["subject_sha256"] = Value::String("0".repeat(64));
        assert!(validate_profile(&value).is_err());
    }

    #[test]
    fn incomplete_stream_cannot_back_an_exact_output_assertion() {
        let mut value = fixture();
        value["observations"][0]["result"]["stdout"]["state"] = Value::String("partial".into());
        value["observations"][0]["result"]["stdout"]["reason"] = Value::String("truncated".into());
        assert!(validate_profile(&value).is_err());
    }

    #[test]
    fn path_traversal_is_rejected() {
        let mut value = fixture();
        value["observations"][0]["invocation"]["working_directory"] = Value::String("../outside".into());
        assert!(validate_profile(&value).is_err());
    }

    #[test]
    fn duplicate_environment_names_are_rejected() {
        let mut value = fixture();
        value["observations"][0]["invocation"]["environment"] = serde_json::json!([
            {"name":"LANG","state":"absent"},
            {"name":"LANG","state":"absent"}
        ]);
        assert!(validate_profile(&value).is_err());
    }

    #[test]
    fn duplicate_filesystem_paths_are_rejected() {
        let mut value = fixture();
        value["observations"][0]["result"]["filesystem"]["entries"] = serde_json::json!([
            {"path":"out.txt","change":"created","before_sha256":null,"after_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},
            {"path":"out.txt","change":"modified","before_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","after_sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"}
        ]);
        assert!(validate_profile(&value).is_err());
    }

    #[test]
    fn non_exit_termination_cannot_claim_exit_code() {
        let mut value = fixture();
        value["observations"][0]["result"]["termination"] = serde_json::json!({
            "kind":"timeout","reason":"timed out","exit_code":0
        });
        assert!(validate_profile(&value).is_err());
    }

    #[test]
    fn complete_filesystem_capture_can_explicitly_record_no_changes() {
        let value = fixture();
        assert_eq!(value["observations"][0]["result"]["filesystem"]["state"], "complete");
        assert!(value["observations"][0]["result"]["filesystem"]["entries"].as_array().unwrap().is_empty());
    }

    #[test]
    fn redacted_environment_is_not_replayable_but_digest_bound() {
        let mut value = fixture();
        value["observations"][0]["invocation"]["environment"] = serde_json::json!([
            {"name":"TOKEN","state":"redacted","value_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","reason":"secret material omitted"}
        ]);
        assert!(validate_profile(&value).is_ok());
    }
    #[test]
    fn assertion_value_must_match_the_captured_observation() {
        let mut value = fixture();
        value["observations"][0]["assertions"][0]["observed_value"] =
            Value::String("0".repeat(64));
        assert!(validate_profile(&value).is_err());
    }

    #[test]
    fn filesystem_empty_assertion_must_match_complete_diff() {
        let mut value = fixture();
        value["observations"][0]["assertions"][2]["observed_value"] = Value::Bool(false);
        assert!(validate_profile(&value).is_err());
    }

    #[test]
    fn duplicate_json_keys_and_trailing_data_are_rejected() {
        assert!(parse_unique_json(br#"{"x":1,"x":2}"#).is_err());
        assert!(parse_unique_json(br#"{"x":1} {}"#).is_err());
    }

    #[test]
    fn partial_and_unavailable_streams_remain_non_assertable() {
        let mut value = fixture();
        value["observations"][2]["assertions"] = serde_json::json!([
            {"assertion_id":"stdout_exact","predicate":"stdout_sha256","capture_ref":"stdout","observed_value":"2725aea2d2dea736fbe38e41ecb518f6098cb68ac77362c5e34faafb356567c5"}
        ]);
        assert!(validate_profile(&value).is_err());
    }

}
