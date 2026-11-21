// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Static source-contract auditor for the pinned public DFA-CfNs repository.
//!
//! This tool checks whether the public DFA/PRDFA Cfc.forward wrappers replace
//! caller-supplied elapsed intervals with all-ones values, while the PhysioNet
//! trainer computes and passes elapsed gaps. It is a source audit only: it does
//! not import Python, execute training, test model output, or evaluate the paper.
//!
//! Usage:
//!   cargo run --locked -p symthaea-silicon --example nialim_source_contract_audit -- /path/to/DFA-CfNs
//!
//! The expected source revision is intentionally pinned. A different checkout
//! returns a revision-mismatch status rather than silently auditing new code.

use std::env;
use std::error::Error;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

const EXPECTED_REPOSITORY: &str = "nuist-xf/DFA-CfNs";
const EXPECTED_COMMIT: &str = "f0e77101e6a22f1c87eae9b105b8cc4e1b3175b7";

const DFA_PATH: &str = "IEs/torch_cfc_DFA.py";
const PRDFA_PATH: &str = "IEs/torch_cfc_PRDFA.py";
const PHYSIONET_PATH: &str =
    "basic continuous time series modelling tasks/train_physio.py";

const FORWARD_SIGNATURE: &str =
    "def forward(self, x, timespans=None, mask=None):";
const OVERWRITE_STATEMENT: &str =
    "timespans = torch.ones((batch_size, seq_len)).to(device)";

#[derive(Debug)]
struct AuditError(String);

impl std::fmt::Display for AuditError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for AuditError {}

fn git_text(root: &Path, args: &[&str]) -> Result<String, AuditError> {
    let output = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(args)
        .output()
        .map_err(|error| AuditError(format!("cannot execute git: {error}")))?;

    if !output.status.success() {
        return Err(AuditError(format!(
            "git {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }

    String::from_utf8(output.stdout)
        .map(|value| value.trim().to_owned())
        .map_err(|error| AuditError(format!("git output is not UTF-8: {error}")))
}

fn read_source(root: &Path, relative: &str) -> Result<String, AuditError> {
    let path = root.join(relative);
    std::fs::read_to_string(&path).map_err(|error| {
        AuditError(format!(
            "cannot read {}: {error}",
            path.to_string_lossy()
        ))
    })
}

fn blob_sha(root: &Path, relative: &str) -> Result<String, AuditError> {
    git_text(root, &["rev-parse", &format!("HEAD:{relative}")])
}

fn leading_spaces(line: &str) -> usize {
    line.chars().take_while(|character| *character == ' ').count()
}

/// Extract a Python method by its exact signature, stopping at the next
/// non-blank line dedented to the method's enclosing class/module level.
fn extract_python_method(source: &str, signature: &str) -> Option<String> {
    let lines: Vec<&str> = source.lines().collect();
    let start = lines.iter().position(|line| line.contains(signature))?;
    let method_indent = leading_spaces(lines[start]);

    let mut result = String::new();
    for (index, line) in lines.iter().enumerate().skip(start) {
        if index > start && !line.trim().is_empty() {
            let indent = leading_spaces(line);
            let trimmed = line.trim_start();
            let sibling_definition = indent == method_indent
                && (trimmed.starts_with("def ")
                    || trimmed.starts_with("async def ")
                    || trimmed.starts_with("class ")
                    || trimmed.starts_with('@'));
            if indent < method_indent || sibling_definition {
                break;
            }
        }
        result.push_str(line);
        result.push('\n');
    }
    Some(result)
}

fn normalize_whitespace(source: &str) -> String {
    source.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn line_number(source: &str, needle: &str) -> Option<usize> {
    source.lines().position(|line| line.contains(needle)).map(|index| index + 1)
}

/// Match only a direct statement in the method body. A guarded fallback such
/// as "if timespans is None: timespans = ones(...)" must not be called an
/// unconditional overwrite.
fn has_direct_method_statement(method: &str, statement: &str) -> bool {
    let lines: Vec<&str> = method.lines().collect();
    let Some(first) = lines.first() else {
        return false;
    };
    let method_indent = leading_spaces(first);
    let expected = normalize_whitespace(statement);

    lines.iter().skip(1).any(|line| {
        !line.trim().is_empty()
            && leading_spaces(line) == method_indent + 4
            && normalize_whitespace(line) == expected
    })
}

fn json_string(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len() + 2);
    escaped.push('"');
    for character in value.chars() {
        match character {
            '"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            character if character.is_control() => {
                let _ = write!(escaped, "\\u{:04x}", character as u32);
            }
            character => escaped.push(character),
        }
    }
    escaped.push('"');
    escaped
}

fn json_bool(value: bool) -> &'static str {
    if value { "true" } else { "false" }
}

fn json_optional_number(value: Option<usize>) -> String {
    value.map_or_else(|| "null".to_owned(), |number| number.to_string())
}

fn emit_revision_mismatch(commit: &str, tree: &str) {
    println!(
        concat!(
            "{{\n",
            "  \"schema\": \"symthaea.nialim-source-contract-audit\",\n",
            "  \"schema_version\": \"0.1\",\n",
            "  \"status\": \"source_revision_mismatch\",\n",
            "  \"repository\": {},\n",
            "  \"observed_commit\": {},\n",
            "  \"observed_tree\": {},\n",
            "  \"expected_commit\": {},\n",
            "  \"claim_ceiling\": \"static_source_audit_only\"\n",
            "}}"
        ),
        json_string(EXPECTED_REPOSITORY),
        json_string(commit),
        json_string(tree),
        json_string(EXPECTED_COMMIT)
    );
}

fn run(root: PathBuf) -> Result<bool, AuditError> {
    if !root.is_dir() {
        return Err(AuditError(format!(
            "reference repository path is not a directory: {}",
            root.display()
        )));
    }

    let commit = git_text(&root, &["rev-parse", "HEAD"])?;
    let tree = git_text(&root, &["rev-parse", "HEAD^{tree}"])?;
    if commit != EXPECTED_COMMIT {
        emit_revision_mismatch(&commit, &tree);
        return Ok(false);
    }

    let dfa_source = read_source(&root, DFA_PATH)?;
    let prdfa_source = read_source(&root, PRDFA_PATH)?;
    let physionet_source = read_source(&root, PHYSIONET_PATH)?;

    let dfa_forward = extract_python_method(&dfa_source, FORWARD_SIGNATURE)
        .ok_or_else(|| AuditError(format!("DFA forward method not found in {DFA_PATH}")))?;
    let prdfa_forward = extract_python_method(&prdfa_source, FORWARD_SIGNATURE)
        .ok_or_else(|| AuditError(format!("PRDFA forward method not found in {PRDFA_PATH}")))?;
    let physionet_prepare = extract_python_method(
        &physionet_source,
        "def _prepare_batch(self, batch):",
    )
    .ok_or_else(|| AuditError(format!("PhysioNet _prepare_batch not found in {PHYSIONET_PATH}")))?;

    let dfa_overwrites = has_direct_method_statement(&dfa_forward, OVERWRITE_STATEMENT);
    let prdfa_overwrites = has_direct_method_statement(&prdfa_forward, OVERWRITE_STATEMENT);

    let trainer_computes_elapsed = physionet_prepare
        .contains("t_elapsed = tt[:, 1:] - tt[:, :-1]");
    let trainer_returns_elapsed = physionet_prepare
        .contains("return x, t, mask, y");
    let trainer_passes_elapsed = physionet_source
        .contains("self.model.forward(x, tt, mask)");
    let trainer_uses_dfa = physionet_source
        .contains("from torch_cfc_DFA import Cfc");

    let dfa_line = line_number(&dfa_forward, "timespans = torch.ones");
    let prdfa_line = line_number(&prdfa_forward, "timespans = torch.ones");

    // The supplied PhysioNet script imports DFA, not PRDFA. Keep the PRDFA
    // finding visible, but do not infer that it is on this specific call path.
    let static_mismatch = trainer_computes_elapsed
        && trainer_returns_elapsed
        && trainer_passes_elapsed
        && trainer_uses_dfa
        && dfa_overwrites;

    let verdict = if static_mismatch {
        "dfa_elapsed_time_argument_discarded_in_public_physionet_path"
    } else {
        "no_matching_static_overwrite_detected_dynamic_test_still_required"
    };

    let dfa_blob = blob_sha(&root, DFA_PATH)?;
    let prdfa_blob = blob_sha(&root, PRDFA_PATH)?;
    let phys_blob = blob_sha(&root, PHYSIONET_PATH)?;

    println!(
        concat!(
            "{{\n",
            "  \"schema\": \"symthaea.nialim-source-contract-audit\",\n",
            "  \"schema_version\": \"0.1\",\n",
            "  \"status\": {},\n",
            "  \"repository\": {},\n",
            "  \"source\": {{\n",
            "    \"commit_sha\": {},\n",
            "    \"tree_sha\": {},\n",
            "    \"expected_commit_sha\": {},\n",
            "    \"dfa_blob_sha\": {},\n",
            "    \"prdfa_blob_sha\": {},\n",
            "    \"physionet_trainer_blob_sha\": {}\n",
            "  }},\n",
            "  \"observations\": {{\n",
            "    \"dfa_overwrites_passed_timespans\": {},\n",
            "    \"prdfa_overwrites_passed_timespans\": {},\n",
            "    \"physionet_computes_elapsed_gaps\": {},\n",
            "    \"physionet_returns_elapsed_gaps\": {},\n",
            "    \"physionet_passes_time_argument\": {},\n",
            "    \"physionet_imports_dfa\": {},\n",
            "    \"dfa_overwrite_line_in_method\": {},\n",
            "    \"prdfa_overwrite_line_in_method\": {}\n",
            "  }},\n",
            "  \"verdict\": {},\n",
            "  \"claim_ceiling\": \"static_source_audit_only\"\n",
            "}}"
        ),
        json_string(if static_mismatch { "source_contract_violation" } else { "review_required" }),
        json_string(EXPECTED_REPOSITORY),
        json_string(&commit),
        json_string(&tree),
        json_string(EXPECTED_COMMIT),
        json_string(&dfa_blob),
        json_string(&prdfa_blob),
        json_string(&phys_blob),
        json_bool(dfa_overwrites),
        json_bool(prdfa_overwrites),
        json_bool(trainer_computes_elapsed),
        json_bool(trainer_returns_elapsed),
        json_bool(trainer_passes_elapsed),
        json_bool(trainer_uses_dfa),
        json_optional_number(dfa_line),
        json_optional_number(prdfa_line),
        json_string(verdict)
    );

    Ok(!static_mismatch)
}

fn main() -> ExitCode {
    let mut arguments = env::args_os().skip(1);
    let Some(root) = arguments.next() else {
        eprintln!(
            "usage: nialim_source_contract_audit <pinned-DFA-CfNs-checkout>"
        );
        return ExitCode::from(2);
    };
    if arguments.next().is_some() {
        eprintln!("expected exactly one reference-repository path");
        return ExitCode::from(2);
    }

    match run(PathBuf::from(root)) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(1),
        Err(error) => {
            eprintln!("source audit blocked: {error}");
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn method_extraction_stops_at_dedented_class_method_boundary() {
        let source = r#"class Model:
    def forward(self, x, timespans=None, mask=None):
        timespans = torch.ones((batch_size, seq_len)).to(device)
        return timespans

    def next_method(self):
        return 1
"#;
        let method = extract_python_method(source, FORWARD_SIGNATURE).unwrap();
        assert!(method.contains("timespans = torch.ones"));
        assert!(!method.contains("def next_method"));
    }

    #[test]
    fn method_extraction_keeps_nested_definitions_inside_the_method() {
        let source = r#"class Model:
    def forward(self, x, timespans=None, mask=None):
        def helper(value):
            return value
        return helper(x)

    def next_method(self):
        return 1
"#;
        let method = extract_python_method(source, FORWARD_SIGNATURE).unwrap();
        assert!(method.contains("def helper"));
        assert!(!method.contains("def next_method"));
    }

    #[test]
    fn whitespace_normalization_handles_line_wrapped_python_statements() {
        let source = r#"timespans = torch.ones((batch_size,
 seq_len)).to(device)"#;
        assert_eq!(
            normalize_whitespace(source),
            normalize_whitespace(OVERWRITE_STATEMENT)
        );
    }

    #[test]
    fn absent_signature_is_not_mistaken_for_a_clean_method() {
        assert!(extract_python_method(r#"class Empty:
    pass
"#, FORWARD_SIGNATURE).is_none());
    }

    #[test]
    fn guarded_default_does_not_match_unconditional_overwrite_rule() {
        let method = r#"    def forward(self, x, timespans=None, mask=None):
        if timespans is None:
            timespans = torch.ones((batch_size, seq_len)).to(device)
"#;
        assert!(!has_direct_method_statement(method, OVERWRITE_STATEMENT));
    }

    #[test]
    fn direct_unconditional_overwrite_is_detected() {
        let method = r#"    def forward(self, x, timespans=None, mask=None):
        timespans = torch.ones((batch_size, seq_len)).to(device)
        for t in range(seq_len):
"#;
        assert!(has_direct_method_statement(method, OVERWRITE_STATEMENT));
    }
}
