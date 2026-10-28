//! Rust AST discovery of likely external-effect call sites.
//!
//! This is a candidate-discovery tool, not a sound effect analysis. It reports
//! selected direct calls and receiver-chain method candidates with source
//! locations and digests. Receiver checks are syntactic heuristics, not type
//! resolution. The scanner does not resolve aliases/types/control flow, analyze
//! macro token bodies, FFI, generated code, scripts, dynamic loading, trait-object
//! dispatch, or arbitrary indirect calls. It may over- and under-report.
//! A scan cannot prove reachability, effect occurrence, or authorization.

use anyhow::{bail, Context, Result};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};
use syn::{spanned::Spanned, visit::Visit, Expr, ExprCall, ExprMethodCall};

const SCAN_SCHEMA: &str = "symthaea.effect_primitive_scan.v1";

#[derive(Debug, Serialize)]
struct ScanReport {
    schema: &'static str,
    repository_head: String,
    scan_roots: Vec<String>,
    source_files_seen: usize,
    source_files_parsed: usize,
    parse_failures: Vec<ParseFailure>,
    candidate_count: usize,
    candidate_sites: Vec<CandidateSite>,
    scan_scope: &'static str,
    claim_boundary: &'static str,
}

#[derive(Debug, Serialize)]
struct ParseFailure {
    source_path: String,
    error: String,
}

#[derive(Debug, Serialize)]
struct CandidateSite {
    site_id_sha256: String,
    source_path: String,
    source_sha256: String,
    line: usize,
    column: usize,
    class: &'static str,
    callee: String,
    confidence: &'static str,
}

struct CandidateVisitor<'a> {
    source_path: &'a str,
    source_sha256: &'a str,
    candidates: Vec<CandidateSite>,
}

pub fn scan(root: &Path, output: Option<&Path>) -> Result<()> {
    let root = root.canonicalize().with_context(|| {
        format!("effect primitive scan root does not exist: {}", root.display())
    })?;
    let scan_roots = vec![
        "repository_root (recursive Rust source discovery)".to_string(),
        "excluded generated/cache directories: .git, target, node_modules, .direnv, .cache, .venv, .devenv".to_string(),
    ];
    let mut files = Vec::new();
    collect_rust_files(&root, &mut files)?;
    files.sort();
    files.dedup();
    if files.is_empty() {
        bail!("effect primitive scan found no Rust source files; refusing an empty/inconclusive report");
    }

    let mut candidates = Vec::new();
    let mut parse_failures = Vec::new();
    let source_files_seen = files.len();
    let mut source_files_parsed = 0usize;

    for path in files {
        let relative_path = path
            .strip_prefix(&root)
            .context("discovered source path escaped the scan root")?
            .to_string_lossy()
            .replace('\\', "/");
        let bytes = fs::read(&path)
            .with_context(|| format!("read Rust source {}", path.display()))?;
        let source_sha256 = format!("{:x}", Sha256::digest(&bytes));
        let source = match std::str::from_utf8(&bytes) {
            Ok(source) => source,
            Err(error) => {
                parse_failures.push(ParseFailure {
                    source_path: relative_path,
                    error: format!("source is not UTF-8: {error}"),
                });
                continue;
            }
        };

        match syn::parse_file(source) {
            Ok(file) => {
                source_files_parsed += 1;
                let mut visitor = CandidateVisitor {
                    source_path: &relative_path,
                    source_sha256: &source_sha256,
                    candidates: Vec::new(),
                };
                visitor.visit_file(&file);
                candidates.extend(visitor.candidates);
            }
            Err(error) => parse_failures.push(ParseFailure {
                source_path: relative_path,
                error: error.to_string(),
            }),
        }
    }

    candidates.sort_by(|left, right| {
        (
            &left.source_path,
            left.line,
            left.column,
            left.class,
            &left.callee,
        )
            .cmp(&(
                &right.source_path,
                right.line,
                right.column,
                right.class,
                &right.callee,
            ))
    });
    parse_failures.sort_by(|left, right| left.source_path.cmp(&right.source_path));

    let report = ScanReport {
        schema: SCAN_SCHEMA,
        repository_head: git_head(&root).unwrap_or_else(|| "UNKNOWN".to_string()),
        scan_roots,
        source_files_seen,
        source_files_parsed,
        parse_failures,
        candidate_count: candidates.len(),
        candidate_sites: candidates,
        scan_scope: "candidate_discovery_direct_calls_and_receiver_chain_methods",
        claim_boundary: "not_complete_not_security_qualification_not_proof_of_reachability_or_effect",
    };
    let had_parse_failures = !report.parse_failures.is_empty();
    let rendered = serde_json::to_string_pretty(&report)
        .context("serialize effect primitive scan report")?;

    if let Some(output) = output {
        if let Some(parent) = output.parent().filter(|path| !path.as_os_str().is_empty()) {
            fs::create_dir_all(parent)
                .with_context(|| format!("create report directory {}", parent.display()))?;
        }
        fs::write(output, format!("{rendered}\n"))
            .with_context(|| format!("write scan report {}", output.display()))?;
        println!(
            "report_path={}",
            output.to_string_lossy()
        );
        println!("source_files_seen={source_files_seen}");
        println!("source_files_parsed={source_files_parsed}");
        println!("candidate_count={}", report.candidate_count);
        println!("parse_failures={}", report.parse_failures.len());
        println!("claim_boundary={}", report.claim_boundary);
    } else {
        println!("{rendered}");
    }

    if had_parse_failures {
        bail!(
            "effect primitive scan incomplete: {} Rust files could not be parsed; inspect report parse_failures",
            report.parse_failures.len()
        );
    }
    Ok(())
}

impl<'source> CandidateVisitor<'source> {
    fn push_candidate(
        &mut self,
        line: usize,
        column: usize,
        class: &'static str,
        callee: String,
        confidence: &'static str,
    ) {
        let identity = format!(
            "{}:{}:{}:{}:{}",
            self.source_path, line, column, class, callee
        );
        let site_id_sha256 = format!("{:x}", Sha256::digest(identity.as_bytes()));
        self.candidates.push(CandidateSite {
            site_id_sha256,
            source_path: self.source_path.to_string(),
            source_sha256: self.source_sha256.to_string(),
            line,
            column,
            class,
            callee,
            confidence,
        });
    }
}

impl<'ast, 'source> Visit<'ast> for CandidateVisitor<'source> {
    fn visit_expr_call(&mut self, node: &'ast ExprCall) {
        if let Expr::Path(function) = node.func.as_ref() {
            if let Some((class, confidence)) = classify_call(&function.path) {
                let callee = function
                    .path
                    .segments
                    .iter()
                    .map(|segment| segment.ident.to_string())
                    .collect::<Vec<_>>()
                    .join("::");
                let start = node.span().start();
                self.push_candidate(start.line, start.column, class, callee, confidence);
            }
        }
        syn::visit::visit_expr_call(self, node);
    }

    fn visit_expr_method_call(&mut self, node: &'ast ExprMethodCall) {
        let method = node.method.to_string();
        if let Some((class, confidence)) = classify_method_call(&method, &node.receiver) {
            let start = node.span().start();
            self.push_candidate(
                start.line,
                start.column,
                class,
                format!("<method>::{method}"),
                confidence,
            );
        }
        syn::visit::visit_expr_method_call(self, node);
    }

fn classify_call(path: &syn::Path) -> Option<(&'static str, &'static str)> {
    let segments = path
        .segments
        .iter()
        .map(|segment| segment.ident.to_string())
        .collect::<Vec<_>>();
    let last = segments.last()?.as_str();
    let has = |name: &str| segments.iter().any(|segment| segment.as_str() == name);
    let previous = segments
        .len()
        .checked_sub(2)
        .and_then(|index| segments.get(index))
        .map(String::as_str);

    if last == "new" && previous == Some("Command") {
        return Some(("process_command_construction", "heuristic"));
    }

    if has("fs") {
        match last {
            "read" | "read_to_string" | "read_link" | "read_dir" | "metadata"
            | "symlink_metadata" | "canonicalize" => {
                return Some(("filesystem_read_or_inspect", "heuristic"));
            }
            "write" | "remove_file" | "remove_dir" | "remove_dir_all" | "create_dir"
            | "create_dir_all" | "rename" | "copy" | "hard_link" | "soft_link"
            | "set_permissions" => {
                return Some(("filesystem_mutation", "heuristic"));
            }
            _ => {}
        }
    }

    if matches!(previous, Some("File" | "OpenOptions")) && matches!(last, "open" | "create") {
        return Some(("filesystem_handle_open_or_create", "heuristic"));
    }

    if matches!(last, "connect" | "bind")
        && ["TcpStream", "UnixStream", "UdpSocket", "TcpListener"]
            .iter()
            .any(|name| has(name))
    {
        return Some(("network_socket_connect_or_bind", "heuristic"));
    }

    if last == "system" && has("libc") {
        return Some(("ffi_process_execution", "heuristic"));
    }

    None
}

fn classify_method_call(method: &str, receiver: &Expr) -> Option<(&'static str, &'static str)> {
    if is_command_receiver(receiver) && matches!(method, "spawn" | "output" | "status" | "kill" | "start_kill" | "wait") {
        return Some(("process_dispatch_method", "receiver_chain_heuristic"));
    }
    if is_file_receiver(receiver)
        && matches!(method, "read" | "read_to_end" | "read_to_string" | "write" | "write_all" | "flush" | "set_len" | "sync_all" | "sync_data")
    {
        return Some(("filesystem_handle_io_method", "receiver_chain_heuristic"));
    }
    if is_socket_receiver(receiver) && matches!(method, "connect" | "bind" | "listen" | "accept" | "send" | "send_to" | "recv" | "recv_from")
    {
        return Some(("network_socket_method", "receiver_chain_heuristic"));
    }
    None
}

fn is_command_receiver(expr: &Expr) -> bool {
    match expr {
        Expr::Call(call) => matches!(call.func.as_ref(), Expr::Path(path)
            if path.path.segments.iter().any(|segment| segment.ident.to_string() == "Command")
                && path.path.segments.last().is_some_and(|segment| segment.ident.to_string() == "new")),
        Expr::MethodCall(call) => is_command_receiver(&call.receiver),
        Expr::Paren(expr) => is_command_receiver(&expr.expr),
        Expr::Reference(expr) => is_command_receiver(&expr.expr),
        Expr::Path(path) => path.path.segments.last().is_some_and(|segment| {
            let name = segment.ident.to_string().to_ascii_lowercase();
            name.contains("command") || name == "cmd" || name.contains("process")
        }),
        _ => false,
    }
}

fn is_file_receiver(expr: &Expr) -> bool {
    match expr {
        Expr::Call(call) => matches!(call.func.as_ref(), Expr::Path(path)
            if path.path.segments.iter().any(|segment| matches!(segment.ident.to_string().as_str(), "File" | "OpenOptions"))
                && path.path.segments.iter().any(|segment| matches!(segment.ident.to_string().as_str(), "open" | "create" | "new"))),
        Expr::MethodCall(call) => is_file_receiver(&call.receiver),
        Expr::Paren(expr) => is_file_receiver(&expr.expr),
        Expr::Reference(expr) => is_file_receiver(&expr.expr),
        Expr::Path(path) => path.path.segments.last().is_some_and(|segment| {
            let name = segment.ident.to_string().to_ascii_lowercase();
            name.contains("file") || name == "reader" || name == "writer" || name == "handle"
        }),
        _ => false,
    }
}

fn is_socket_receiver(expr: &Expr) -> bool {
    match expr {
        Expr::Call(call) => matches!(call.func.as_ref(), Expr::Path(path)
            if path.path.segments.iter().any(|segment| {
                matches!(segment.ident.to_string().as_str(), "TcpStream" | "TcpListener" | "UdpSocket" | "UnixStream" | "UnixListener")
            })),
        Expr::MethodCall(call) => is_socket_receiver(&call.receiver),
        Expr::Paren(expr) => is_socket_receiver(&expr.expr),
        Expr::Reference(expr) => is_socket_receiver(&expr.expr),
        Expr::Path(path) => path.path.segments.last().is_some_and(|segment| {
            let name = segment.ident.to_string().to_ascii_lowercase();
            name.contains("socket") || name.contains("stream") || name.contains("listener")
        }),
        _ => false,
    }
}

fn collect_rust_files(directory: &Path, output: &mut Vec<PathBuf>) -> Result<()> {
    if !directory.exists() {
        return Ok(());
    }
    for entry in fs::read_dir(directory)
        .with_context(|| format!("read source directory {}", directory.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        let file_type = entry.file_type()?;
        if file_type.is_symlink() {
            continue;
        }
        if file_type.is_dir() {
            const EXCLUDED_DIRS: &[&str] = &[
                ".git", "target", "node_modules", ".direnv", ".cache", ".venv", ".devenv",
            ];
            if !path
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| EXCLUDED_DIRS.contains(&name))
            {
                collect_rust_files(&path, output)?;
            }
        } else if file_type.is_file() && path.extension().and_then(|extension| extension.to_str()) == Some("rs") {
            output.push(path);
        }
    }
    Ok(())
}

fn git_head(root: &Path) -> Option<String> {
    let output = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(root)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8(output.stdout)
        .ok()
        .map(|value| value.trim().to_string())
}

#[cfg(test)]
mod tests {
    use sha2::Digest;
    use syn::parse_quote;

    use super::{CandidateVisitor, classify_call};

    #[test]
    fn classifies_direct_process_construction() {
        let path: syn::Path = parse_quote!(tokio::process::Command::new);
        assert_eq!(
            classify_call(&path).map(|result| result.0),
            Some("process_command_construction")
        );
    }

    #[test]
    fn classifies_filesystem_mutation_but_not_unrelated_names() {
        let write: syn::Path = parse_quote!(std::fs::write);
        let remove: syn::Path = parse_quote!(std::fs::remove_file);
        let unrelated: syn::Path = parse_quote!(my_application::write);
        assert_eq!(
            classify_call(&write).map(|result| result.0),
            Some("filesystem_mutation")
        );
        assert_eq!(
            classify_call(&remove).map(|result| result.0),
            Some("filesystem_mutation")
        );
        assert_eq!(classify_call(&unrelated), None);
    }

    #[test]
    fn classifies_socket_and_ffi_execution_candidates() {
        let socket: syn::Path = parse_quote!(std::net::TcpStream::connect);
        let ffi: syn::Path = parse_quote!(libc::system);
        assert_eq!(
            classify_call(&socket).map(|result| result.0),
            Some("network_socket_connect_or_bind")
        );
        assert_eq!(
            classify_call(&ffi).map(|result| result.0),
            Some("ffi_process_execution")
        );
    }

    #[test]
    fn classifies_command_dispatch_methods_but_not_unrelated_output_methods() {
        let source = r#"
            fn main() {
                std::process::Command::new("sh").arg("-c").arg("true").output();
                report.output();
            }
        "#;
        let file = syn::parse_file(source).unwrap();
        let source_sha = format!("{:x}", sha2::Sha256::digest(source.as_bytes()));
        let mut visitor = CandidateVisitor {
            source_path: "fixtures/process_chain.rs",
            source_sha256: &source_sha,
            candidates: Vec::new(),
        };
        use syn::visit::Visit;
        visitor.visit_file(&file);

        assert!(visitor.candidates.iter().any(|site| {
            site.class == "process_dispatch_method" && site.callee == "<method>::output"
        }));
        assert_eq!(
            visitor.candidates
                .iter()
                .filter(|site| site.class == "process_dispatch_method")
                .count(),
            1,
            "unrelated report.output() must not be treated as a command dispatch"
        );
    }

    #[test]
    fn classifies_file_handle_writes_through_unwrap_chains() {
        let source = r#"
            fn main() {
                std::fs::File::create("sentinel").unwrap().write_all(b"effect");
                report.write_all(b"not a file");
            }
        "#;
        let file = syn::parse_file(source).unwrap();
        let source_sha = format!("{:x}", sha2::Sha256::digest(source.as_bytes()));
        let mut visitor = CandidateVisitor {
            source_path: "fixtures/file_chain.rs",
            source_sha256: &source_sha,
            candidates: Vec::new(),
        };
        use syn::visit::Visit;
        visitor.visit_file(&file);

        assert!(visitor.candidates.iter().any(|site| {
            site.class == "filesystem_handle_io_method" && site.callee == "<method>::write_all"
        }));
        assert_eq!(
            visitor.candidates
                .iter()
                .filter(|site| site.class == "filesystem_handle_io_method")
                .count(),
            1,
            "unrelated report.write_all() must not be classified as a file-handle effect"
        );
    }

    #[test]
    fn scans_source_location_and_hashes_finding_identity() {
        let source = "fn main() { std::fs::write(\"sentinel\", b\"x\").unwrap(); }";
        let file = syn::parse_file(source).unwrap();
        let path = "fixtures/side_effect.rs";
        let source_sha = format!("{:x}", sha2::Sha256::digest(source.as_bytes()));
        let mut visitor = CandidateVisitor {
            source_path: path,
            source_sha256: &source_sha,
            candidates: Vec::new(),
        };
        use syn::visit::Visit;
        visitor.visit_file(&file);
        assert_eq!(visitor.candidates.len(), 1);
        assert_eq!(visitor.candidates[0].source_path, path);
        assert_eq!(visitor.candidates[0].class, "filesystem_mutation");
        assert!(visitor.candidates[0].site_id_sha256.len() == 64);
    }
}
