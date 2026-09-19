use crate::cargo_context::{
    CargoBuildContextDocument, CargoBuildContextSpec, CargoOperation, FeatureSelection,
    PackageSelection, TargetSelection, ToolchainIdentity, build_document,
};
use anyhow::{Context, bail};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

const PARSE_SCHEMA: &str = "symthaea.cargo-invocation-parse.v1";
const CONTEXT_SCHEMA: &str = "symthaea.cargo-build-context.v1";

/// Inputs the argv parser cannot infer from Cargo's command line alone.
///
/// `raw_argv` must already be tokenized exactly as Cargo receives it. Shell,
/// GitHub Actions, Just, Make, and other command-language parsing are separate
/// adapters; this module only interprets Cargo semantics.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CargoInvocationParseSpec {
    pub schema: String,
    #[serde(default = "default_manifest_path")]
    pub default_manifest_path: String,
    pub toolchain: ToolchainIdentity,
    #[serde(default)]
    pub cargo_config_sha256: Option<String>,
    #[serde(default)]
    pub rustflags_sha256: Option<String>,
    #[serde(default)]
    pub rustdocflags_sha256: Option<String>,
    #[serde(default)]
    pub environment_fingerprints: BTreeMap<String, String>,
    pub raw_argv: Vec<String>,
}

fn default_manifest_path() -> String {
    "Cargo.toml".into()
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum InvocationParseState {
    Parsed,
    UnknownCommandSurface,
}

/// Machine-readable result. `UnknownCommandSurface` intentionally has no
/// context document: consumers must broaden qualification instead of treating
/// an incomplete parse as an equivalent build context.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct InvocationParseReport {
    pub schema: &'static str,
    pub state: InvocationParseState,
    pub context_document: Option<CargoBuildContextDocument>,
    pub invocation_only_args: Vec<String>,
    pub execution_policy_args: Vec<String>,
    pub unknown_tokens: Vec<String>,
    pub reasons: Vec<String>,
}

#[derive(Debug)]
struct ParsedFields {
    operation: CargoOperation,
    manifest_path: String,
    package_selection: PackageSelection,
    target_selection: TargetSelection,
    feature_selection: FeatureSelection,
    target_triples: Vec<String>,
    profile: Option<String>,
    toolchain: ToolchainIdentity,
    invocation_only_args: Vec<String>,
    execution_policy_args: Vec<String>,
    unknown_tokens: Vec<String>,
    reasons: Vec<String>,
}

impl ParsedFields {
    fn unknown(&mut self, token: impl Into<String>, reason: impl Into<String>) {
        self.unknown_tokens.push(token.into());
        self.reasons.push(reason.into());
    }

    fn set_profile(&mut self, profile: String, token: &str) {
        if let Some(existing) = &self.profile {
            if existing != &profile {
                self.unknown(
                    token,
                    format!("conflicting Cargo profiles requested: {existing} vs {profile}"),
                );
            }
        } else {
            self.profile = Some(profile);
        }
    }
}

pub fn run(spec_path: &Path, output: Option<PathBuf>) -> anyhow::Result<()> {
    let bytes = fs::read(spec_path)
        .with_context(|| format!("read Cargo invocation parse spec {}", spec_path.display()))?;
    let spec: CargoInvocationParseSpec = serde_json::from_slice(&bytes)
        .with_context(|| format!("parse Cargo invocation spec {}", spec_path.display()))?;
    let report = parse_spec(spec)?;

    let mut rendered = serde_json::to_string_pretty(&report)?;
    rendered.push('\n');
    if let Some(path) = output {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("create invocation output directory {}", parent.display()))?;
        }
        fs::write(&path, rendered)
            .with_context(|| format!("write Cargo invocation report {}", path.display()))?;
        println!("Cargo invocation parse report written to {}", path.display());
    } else {
        print!("{rendered}");
    }
    Ok(())
}

pub fn parse_spec(spec: CargoInvocationParseSpec) -> anyhow::Result<InvocationParseReport> {
    if spec.schema != PARSE_SCHEMA {
        bail!("unsupported Cargo invocation parse schema: {}", spec.schema);
    }
    if spec.raw_argv.is_empty() || spec.raw_argv.iter().any(|arg| arg.is_empty()) {
        bail!("raw_argv must contain exact non-empty Cargo argv tokens");
    }

    let raw_argv = spec.raw_argv.clone();
    let mut parsed = match parse_argv(&spec) {
        Ok(parsed) => parsed,
        Err(reason) => {
            return Ok(InvocationParseReport {
                schema: PARSE_SCHEMA,
                state: InvocationParseState::UnknownCommandSurface,
                context_document: None,
                invocation_only_args: Vec::new(),
                execution_policy_args: Vec::new(),
                unknown_tokens: raw_argv,
                reasons: vec![reason],
            });
        }
    };

    if parsed.package_selection.workspace && !parsed.package_selection.packages.is_empty() {
        parsed.unknown(
            "--workspace/+package",
            "workspace selection combined with explicit packages is not normalized by parser v1",
        );
    }

    if !parsed.unknown_tokens.is_empty() {
        canonicalize(&mut parsed.unknown_tokens);
        canonicalize(&mut parsed.reasons);
        return Ok(InvocationParseReport {
            schema: PARSE_SCHEMA,
            state: InvocationParseState::UnknownCommandSurface,
            context_document: None,
            invocation_only_args: parsed.invocation_only_args,
            execution_policy_args: parsed.execution_policy_args,
            unknown_tokens: parsed.unknown_tokens,
            reasons: parsed.reasons,
        });
    }

    let profile = parsed
        .profile
        .unwrap_or_else(|| default_profile(parsed.operation).to_string());
    let context_spec = CargoBuildContextSpec {
        schema: CONTEXT_SCHEMA.into(),
        operation: parsed.operation,
        manifest_path: parsed.manifest_path,
        package_selection: parsed.package_selection,
        target_selection: parsed.target_selection,
        feature_selection: parsed.feature_selection,
        target_triples: parsed.target_triples,
        profile,
        toolchain: parsed.toolchain,
        cargo_config_sha256: spec.cargo_config_sha256,
        rustflags_sha256: spec.rustflags_sha256,
        rustdocflags_sha256: spec.rustdocflags_sha256,
        environment_fingerprints: spec.environment_fingerprints,
        raw_argv,
    };

    match build_document(context_spec) {
        Ok(document) => Ok(InvocationParseReport {
            schema: PARSE_SCHEMA,
            state: InvocationParseState::Parsed,
            context_document: Some(document),
            invocation_only_args: parsed.invocation_only_args,
            execution_policy_args: parsed.execution_policy_args,
            unknown_tokens: Vec::new(),
            reasons: Vec::new(),
        }),
        Err(error) => Ok(InvocationParseReport {
            schema: PARSE_SCHEMA,
            state: InvocationParseState::UnknownCommandSurface,
            context_document: None,
            invocation_only_args: parsed.invocation_only_args,
            execution_policy_args: parsed.execution_policy_args,
            unknown_tokens: Vec::new(),
            reasons: vec![format!("normalized context rejected: {error:#}")],
        }),
    }
}

fn parse_argv(spec: &CargoInvocationParseSpec) -> Result<ParsedFields, String> {
    let argv = &spec.raw_argv;
    if !is_cargo_executable(&argv[0]) {
        return Err(format!(
            "argv[0] is not a Cargo executable: {}",
            argv[0]
        ));
    }

    let mut index = 1;
    let mut toolchain = spec.toolchain.clone();
    if let Some(token) = argv.get(index) {
        if let Some(requested) = token.strip_prefix('+') {
            if requested.is_empty() {
                return Err("empty +toolchain selector".into());
            }
            match &toolchain.toolchain_name {
                Some(declared) if declared != requested => {
                    return Err(format!(
                        "argv toolchain +{requested} disagrees with declared toolchain {declared}"
                    ));
                }
                None => toolchain.toolchain_name = Some(requested.to_string()),
                Some(_) => {}
            }
            index += 1;
        }
    }

    let operation_token = argv
        .get(index)
        .ok_or_else(|| "Cargo invocation is missing a subcommand".to_string())?;
    let operation = parse_operation(operation_token)
        .ok_or_else(|| format!("unsupported Cargo subcommand: {operation_token}"))?;
    index += 1;

    let mut fields = ParsedFields {
        operation,
        manifest_path: spec.default_manifest_path.clone(),
        package_selection: PackageSelection {
            workspace: false,
            packages: Vec::new(),
            exclude: Vec::new(),
        },
        target_selection: TargetSelection {
            lib: false,
            bins: Vec::new(),
            examples: Vec::new(),
            tests: Vec::new(),
            benches: Vec::new(),
            all_targets: false,
        },
        feature_selection: FeatureSelection {
            requested: Vec::new(),
            no_default_features: false,
            all_features: false,
        },
        target_triples: Vec::new(),
        profile: None,
        toolchain,
        invocation_only_args: Vec::new(),
        execution_policy_args: Vec::new(),
        unknown_tokens: Vec::new(),
        reasons: Vec::new(),
    };

    while index < argv.len() {
        let arg = &argv[index];
        if arg == "--" {
            fields
                .invocation_only_args
                .extend(argv[index + 1..].iter().cloned());
            break;
        }

        match arg.as_str() {
            "--workspace" | "--all" => fields.package_selection.workspace = true,
            "--lib" => fields.target_selection.lib = true,
            "--all-targets" => fields.target_selection.all_targets = true,
            "--all-features" => fields.feature_selection.all_features = true,
            "--no-default-features" => fields.feature_selection.no_default_features = true,
            "--release" => fields.set_profile("release".into(), arg),
            "--locked" | "--frozen" | "--offline" | "--keep-going" | "--quiet" | "-q"
            | "--verbose" | "-v" | "--future-incompat-report" | "--ignore-rust-version"
            | "--timings" => fields.execution_policy_args.push(arg.clone()),
            "--bins" | "--examples" | "--tests" | "--benches" | "--doc" => fields.unknown(
                arg,
                format!("{arg} changes Cargo target semantics not representable by build-context v1"),
            ),
            "--config" => {
                let value = take_value(argv, &mut index, arg, &mut fields)?;
                fields.execution_policy_args.push(arg.clone());
                fields.execution_policy_args.push(value);
                fields.unknown(
                    arg,
                    "inline --config needs an effective-config parser/fingerprint adapter before it can be normalized",
                );
            }
            "--manifest-path" => {
                fields.manifest_path = take_value(argv, &mut index, arg, &mut fields)?;
            }
            "--package" | "-p" => {
                let value = take_value(argv, &mut index, arg, &mut fields)?;
                fields.package_selection.packages.push(value);
            }
            "--exclude" => {
                let value = take_value(argv, &mut index, arg, &mut fields)?;
                fields.package_selection.exclude.push(value);
            }
            "--bin" => {
                let value = take_value(argv, &mut index, arg, &mut fields)?;
                fields.target_selection.bins.push(value);
            }
            "--example" => {
                let value = take_value(argv, &mut index, arg, &mut fields)?;
                fields.target_selection.examples.push(value);
            }
            "--test" => {
                let value = take_value(argv, &mut index, arg, &mut fields)?;
                fields.target_selection.tests.push(value);
            }
            "--bench" => {
                let value = take_value(argv, &mut index, arg, &mut fields)?;
                fields.target_selection.benches.push(value);
            }
            "--features" | "-F" => {
                let value = take_value(argv, &mut index, arg, &mut fields)?;
                append_features(&mut fields.feature_selection.requested, &value);
            }
            "--target" => {
                let value = take_value(argv, &mut index, arg, &mut fields)?;
                fields.target_triples.push(value);
            }
            "--profile" => {
                let value = take_value(argv, &mut index, arg, &mut fields)?;
                fields.set_profile(value, arg);
            }
            "--message-format" | "--color" | "--jobs" | "-j" | "--target-dir" => {
                let value = take_value(argv, &mut index, arg, &mut fields)?;
                fields.execution_policy_args.push(arg.clone());
                fields.execution_policy_args.push(value);
            }
            _ => {
                if let Some(value) = arg.strip_prefix("--manifest-path=") {
                    fields.manifest_path = value.to_string();
                } else if let Some(value) = arg.strip_prefix("--package=") {
                    fields.package_selection.packages.push(value.to_string());
                } else if let Some(value) = arg.strip_prefix("--exclude=") {
                    fields.package_selection.exclude.push(value.to_string());
                } else if let Some(value) = arg.strip_prefix("--bin=") {
                    fields.target_selection.bins.push(value.to_string());
                } else if let Some(value) = arg.strip_prefix("--example=") {
                    fields.target_selection.examples.push(value.to_string());
                } else if let Some(value) = arg.strip_prefix("--test=") {
                    fields.target_selection.tests.push(value.to_string());
                } else if let Some(value) = arg.strip_prefix("--bench=") {
                    fields.target_selection.benches.push(value.to_string());
                } else if let Some(value) = arg.strip_prefix("--features=") {
                    append_features(&mut fields.feature_selection.requested, value);
                } else if let Some(value) = arg.strip_prefix("--target=") {
                    fields.target_triples.push(value.to_string());
                } else if let Some(value) = arg.strip_prefix("--profile=") {
                    fields.set_profile(value.to_string(), arg);
                } else if arg.starts_with("--message-format=")
                    || arg.starts_with("--color=")
                    || arg.starts_with("--jobs=")
                    || arg.starts_with("--target-dir=")
                    || arg.starts_with("--timings=")
                {
                    fields.execution_policy_args.push(arg.clone());
                } else if arg.starts_with("--config=") {
                    fields.execution_policy_args.push(arg.clone());
                    fields.unknown(
                        arg,
                        "inline --config needs an effective-config parser/fingerprint adapter before it can be normalized",
                    );
                } else if arg.starts_with("-p") && arg.len() > 2 {
                    fields.package_selection.packages.push(arg[2..].to_string());
                } else if arg.starts_with("-F") && arg.len() > 2 {
                    append_features(&mut fields.feature_selection.requested, &arg[2..]);
                } else if arg.starts_with("-j") && arg.len() > 2 {
                    fields.execution_policy_args.push(arg.clone());
                } else if arg.starts_with('-') {
                    fields.unknown(arg, format!("unrecognized Cargo semantic flag: {arg}"));
                } else if matches!(operation, CargoOperation::Test | CargoOperation::Bench) {
                    // Test/bench filters affect which cases execute, not the build graph.
                    // They remain bound by invocation_id and are surfaced explicitly.
                    fields.invocation_only_args.push(arg.clone());
                } else {
                    fields.unknown(
                        arg,
                        format!("unexpected positional argument for Cargo {operation_token}: {arg}"),
                    );
                }
            }
        }

        index += 1;
    }

    Ok(fields)
}

fn take_value(
    argv: &[String],
    index: &mut usize,
    flag: &str,
    fields: &mut ParsedFields,
) -> Result<String, String> {
    let next = *index + 1;
    match argv.get(next) {
        Some(value) if !value.starts_with('-') => {
            *index = next;
            Ok(value.clone())
        }
        Some(value) => {
            fields.unknown(
                flag,
                format!("{flag} is missing a value before token {value}"),
            );
            Err(format!("{flag} is missing a value"))
        }
        None => Err(format!("{flag} is missing a value at end of argv")),
    }
}

fn parse_operation(value: &str) -> Option<CargoOperation> {
    match value {
        "check" => Some(CargoOperation::Check),
        "build" => Some(CargoOperation::Build),
        "test" => Some(CargoOperation::Test),
        "clippy" => Some(CargoOperation::Clippy),
        "doc" => Some(CargoOperation::Doc),
        "run" => Some(CargoOperation::Run),
        "bench" => Some(CargoOperation::Bench),
        "metadata" => Some(CargoOperation::Metadata),
        _ => None,
    }
}

fn default_profile(operation: CargoOperation) -> &'static str {
    match operation {
        CargoOperation::Test => "test",
        CargoOperation::Bench => "bench",
        CargoOperation::Metadata => "none",
        CargoOperation::Check
        | CargoOperation::Build
        | CargoOperation::Clippy
        | CargoOperation::Doc
        | CargoOperation::Run => "dev",
    }
}

fn is_cargo_executable(value: &str) -> bool {
    Path::new(value)
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name == "cargo" || name == "cargo.exe")
}

fn append_features(output: &mut Vec<String>, value: &str) {
    output.extend(
        value
            .split(|ch: char| ch == ',' || ch.is_whitespace())
            .filter(|part| !part.is_empty())
            .map(str::to_string),
    );
}

fn canonicalize(values: &mut Vec<String>) {
    values.sort();
    values.dedup();
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(raw: &[&str]) -> InvocationParseReport {
        parse_spec(CargoInvocationParseSpec {
            schema: PARSE_SCHEMA.into(),
            default_manifest_path: "Cargo.toml".into(),
            toolchain: ToolchainIdentity {
                cargo_version: "cargo 1.96.0".into(),
                rustc_version: "rustc 1.96.0".into(),
                host_triple: "x86_64-unknown-linux-gnu".into(),
                toolchain_name: Some("1.96.0".into()),
            },
            cargo_config_sha256: None,
            rustflags_sha256: None,
            rustdocflags_sha256: None,
            environment_fingerprints: BTreeMap::new(),
            raw_argv: raw.iter().map(|value| (*value).to_string()).collect(),
        })
        .unwrap()
    }

    #[test]
    fn parses_real_coding_agent_quality_check() {
        let report = parse(&[
            "cargo",
            "check",
            "--lib",
            "--features",
            "code_generation",
        ]);
        assert_eq!(report.state, InvocationParseState::Parsed);
        let context = &report.context_document.unwrap().context;
        assert_eq!(context.operation, CargoOperation::Check);
        assert!(context.target_selection.lib);
        assert_eq!(context.feature_selection.requested, vec!["code_generation"]);
        assert_eq!(context.profile, "dev");
    }

    #[test]
    fn parses_real_coding_backend_example_check() {
        let report = parse(&[
            "cargo",
            "check",
            "--example",
            "benchmark_coding_backends",
            "--features",
            "code_generation,geodesic_synthesis",
        ]);
        assert_eq!(report.state, InvocationParseState::Parsed);
        let context = &report.context_document.unwrap().context;
        assert_eq!(
            context.target_selection.examples,
            vec!["benchmark_coding_backends"]
        );
        assert_eq!(
            context.feature_selection.requested,
            vec!["code_generation", "geodesic_synthesis"]
        );
    }

    #[test]
    fn parses_real_no_default_all_targets_clippy_lane() {
        let report = parse(&[
            "cargo",
            "clippy",
            "-p",
            "symthaea-therapeutic",
            "--no-default-features",
            "--all-targets",
            "--",
            "-D",
            "warnings",
        ]);
        assert_eq!(report.state, InvocationParseState::Parsed);
        let context = &report.context_document.as_ref().unwrap().context;
        assert_eq!(
            context.package_selection.packages,
            vec!["symthaea-therapeutic"]
        );
        assert!(context.feature_selection.no_default_features);
        assert!(context.target_selection.all_targets);
        assert_eq!(report.invocation_only_args, vec!["-D", "warnings"]);
    }

    #[test]
    fn test_filter_is_invocation_only_not_silently_discarded() {
        let report = parse(&[
            "cargo",
            "test",
            "proof_memory",
            "--lib",
            "--features",
            "code_generation",
        ]);
        assert_eq!(report.state, InvocationParseState::Parsed);
        assert_eq!(report.invocation_only_args, vec!["proof_memory"]);
        assert_eq!(
            report.context_document.unwrap().context.profile,
            "test"
        );
    }

    #[test]
    fn unrepresentable_target_selector_fails_toward_unknown() {
        let report = parse(&["cargo", "test", "--tests"]);
        assert_eq!(report.state, InvocationParseState::UnknownCommandSurface);
        assert!(report.context_document.is_none());
        assert!(report.unknown_tokens.contains(&"--tests".to_string()));
    }

    #[test]
    fn inline_config_is_not_assumed_equivalent_to_external_digest() {
        let report = parse(&["cargo", "check", "--config", "net.git-fetch-with-cli=true"]);
        assert_eq!(report.state, InvocationParseState::UnknownCommandSurface);
        assert!(report
            .reasons
            .iter()
            .any(|reason| reason.contains("effective-config")));
    }

    #[test]
    fn message_format_is_retained_as_execution_policy_not_context_semantics() {
        let report = parse(&["cargo", "check", "--message-format=json", "--lib"]);
        assert_eq!(report.state, InvocationParseState::Parsed);
        assert_eq!(report.execution_policy_args, vec!["--message-format=json"]);
    }

    #[test]
    fn argv_toolchain_must_match_declared_identity() {
        let report = parse(&["cargo", "+nightly", "check"]);
        assert_eq!(report.state, InvocationParseState::UnknownCommandSurface);
        assert!(report
            .reasons
            .iter()
            .any(|reason| reason.contains("disagrees")));
    }

    #[test]
    fn unstable_unit_graph_stays_outside_stable_parser_v1() {
        let mut spec = CargoInvocationParseSpec {
            schema: PARSE_SCHEMA.into(),
            default_manifest_path: "Cargo.toml".into(),
            toolchain: ToolchainIdentity {
                cargo_version: "cargo 1.97.0-nightly".into(),
                rustc_version: "rustc 1.97.0-nightly".into(),
                host_triple: "x86_64-unknown-linux-gnu".into(),
                toolchain_name: Some("nightly".into()),
            },
            cargo_config_sha256: None,
            rustflags_sha256: None,
            rustdocflags_sha256: None,
            environment_fingerprints: BTreeMap::new(),
            raw_argv: vec![
                "cargo".into(),
                "+nightly".into(),
                "build".into(),
                "--unit-graph".into(),
                "-Z".into(),
                "unstable-options".into(),
            ],
        };
        let report = parse_spec(spec.clone()).unwrap();
        assert_eq!(report.state, InvocationParseState::UnknownCommandSurface);

        spec.raw_argv = vec!["cargo".into(), "+nightly".into(), "build".into()];
        assert_eq!(
            parse_spec(spec).unwrap().state,
            InvocationParseState::Parsed
        );
    }
}
