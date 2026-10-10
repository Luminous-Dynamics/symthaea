// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Closed-world ngspice include/library dependency bundle.
//!
//! This module resolves a deliberately narrow static subset of ngspice input
//! references against caller-supplied bytes. It never reads the filesystem.
//! The resulting manifest proves byte identity and closure for recognized
//! .include, .incpslt, and external .lib directives only. It does NOT discover
//! arbitrary data files or scripts invoked from ngspice's control language and
//! is not permission to execute the bundle outside a sandbox.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

use symthaea_sim_bridge::SimulationRequest;

use crate::input::{NetlistArtifact, NetlistArtifactError};

/// Maximum number of files (primary plus dependencies) in one bundle.
pub const MAX_BUNDLE_FILES: usize = 256;
/// Maximum combined bytes across the primary netlist and all dependencies.
pub const MAX_BUNDLE_BYTES: usize = 32 * 1024 * 1024;
/// Maximum static include directives examined across the bundle.
pub const MAX_INCLUDE_DIRECTIVES: usize = 4096;

/// One closed-world input bundle, bound to one request and deterministic over
/// exact file bytes plus canonical relative paths.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelInputBundle {
    request_id: String,
    primary_path: String,
    primary: NetlistArtifact,
    dependencies: BTreeMap<String, NetlistArtifact>,
    manifest_digest: String,
}

/// Failure to construct or verify a closed-world dependency bundle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelInputBundleError {
    Artifact(NetlistArtifactError),
    InvalidPath { path: String, reason: String },
    DuplicatePath(String),
    TooManyFiles { actual: usize, maximum: usize },
    BundleTooLarge { actual: usize, maximum: usize },
    TooManyIncludeDirectives { actual: usize, maximum: usize },
    UnsupportedIncludeSyntax { path: String, line: usize, reason: String },
    MissingDependency { from: String, target: String },
    UnreferencedDependency(String),
    IncludeCycle(String),
    RequestIdMismatch { bundle: String, request: String },
    DigestMismatch,
}

impl fmt::Display for ModelInputBundleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Artifact(error) => write!(f, "input artifact rejected: {error}"),
            Self::InvalidPath { path, reason } => {
                write!(f, "invalid bundle path {path:?}: {reason}")
            }
            Self::DuplicatePath(path) => write!(f, "duplicate bundle path {path:?}"),
            Self::TooManyFiles { actual, maximum } => {
                write!(f, "bundle has {actual} files; maximum is {maximum}")
            }
            Self::BundleTooLarge { actual, maximum } => {
                write!(f, "bundle has {actual} bytes; maximum is {maximum}")
            }
            Self::TooManyIncludeDirectives { actual, maximum } => write!(
                f,
                "bundle has {actual} include directives; maximum is {maximum}"
            ),
            Self::UnsupportedIncludeSyntax { path, line, reason } => {
                write!(f, "unsupported include syntax in {path:?} at line {line}: {reason}")
            }
            Self::MissingDependency { from, target } => {
                write!(f, "{from:?} references missing dependency {target:?}")
            }
            Self::UnreferencedDependency(path) => {
                write!(f, "dependency {path:?} is not reachable from the primary netlist")
            }
            Self::IncludeCycle(path) => {
                write!(f, "include dependency cycle reaches {path:?}")
            }
            Self::RequestIdMismatch { bundle, request } => write!(
                f,
                "input bundle belongs to request {bundle:?}, not {request:?}"
            ),
            Self::DigestMismatch => f.write_str("input bundle manifest digest mismatch"),
        }
    }
}

impl Error for ModelInputBundleError {}

impl From<NetlistArtifactError> for ModelInputBundleError {
    fn from(value: NetlistArtifactError) -> Self {
        Self::Artifact(value)
    }
}

impl ModelInputBundle {
    /// Construct a bundle from a primary netlist and explicit dependency bytes.
    ///
    /// Every supplied dependency must be reachable through a recognized static
    /// include/library directive. Every recognized include must resolve to a
    /// supplied file. Paths are canonical relative slash paths; '..', '.', empty
    /// components, absolute paths, backslashes, and environment expansion are
    /// rejected. No filesystem access occurs.
    pub fn new(
        request_id: impl Into<String>,
        primary_path: impl Into<String>,
        primary_bytes: impl Into<Vec<u8>>,
        dependencies: impl IntoIterator<Item = (String, Vec<u8>)>,
    ) -> Result<Self, ModelInputBundleError> {
        let request_id = request_id.into();
        let primary_path = validate_relative_path(primary_path.into())?;
        let primary = NetlistArtifact::new(request_id.clone(), primary_bytes)?;

        let mut dependency_artifacts = BTreeMap::new();
        let mut total_bytes = primary.bytes().len();
        for (path, bytes) in dependencies {
            let path = validate_relative_path(path)?;
            if path == primary_path || dependency_artifacts.contains_key(&path) {
                return Err(ModelInputBundleError::DuplicatePath(path));
            }
            total_bytes = total_bytes.checked_add(bytes.len()).ok_or(
                ModelInputBundleError::BundleTooLarge {
                    actual: usize::MAX,
                    maximum: MAX_BUNDLE_BYTES,
                },
            )?;
            if total_bytes > MAX_BUNDLE_BYTES {
                return Err(ModelInputBundleError::BundleTooLarge {
                    actual: total_bytes,
                    maximum: MAX_BUNDLE_BYTES,
                });
            }
            let artifact = NetlistArtifact::new(request_id.clone(), bytes)?;
            dependency_artifacts.insert(path, artifact);
            if dependency_artifacts.len() + 1 > MAX_BUNDLE_FILES {
                return Err(ModelInputBundleError::TooManyFiles {
                    actual: dependency_artifacts.len() + 1,
                    maximum: MAX_BUNDLE_FILES,
                });
            }
        }

        let mut bundle = Self {
            request_id,
            primary_path,
            primary,
            dependencies: dependency_artifacts,
            manifest_digest: String::new(),
        };
        bundle.validate_closure()?;
        bundle.manifest_digest = bundle.compute_manifest_digest();
        Ok(bundle)
    }

    /// Request identity bound when this bundle was created.
    pub fn request_id(&self) -> &str {
        &self.request_id
    }

    /// Canonical relative path of the primary netlist.
    pub fn primary_path(&self) -> &str {
        &self.primary_path
    }

    /// Exact primary netlist bytes.
    pub fn primary_bytes(&self) -> &[u8] {
        self.primary.bytes()
    }

    /// Digest of the exact primary netlist bytes.
    pub fn primary_digest(&self) -> &str {
        self.primary.blake3_digest()
    }

    /// Sorted dependency paths (the primary file is excluded).
    pub fn dependency_paths(&self) -> impl Iterator<Item = &str> {
        self.dependencies.keys().map(String::as_str)
    }

    /// Exact bytes of a dependency by canonical path.
    pub fn dependency_bytes(&self, path: &str) -> Option<&[u8]> {
        self.dependencies.get(path).map(NetlistArtifact::bytes)
    }

    /// Digest of a dependency by canonical path.
    pub fn dependency_digest(&self, path: &str) -> Option<&str> {
        self.dependencies
            .get(path)
            .map(NetlistArtifact::blake3_digest)
    }

    /// Digest of the canonical manifest, including request ID, primary path,
    /// primary bytes digest, and sorted dependency paths/content digests.
    pub fn manifest_digest(&self) -> &str {
        &self.manifest_digest
    }

    /// Re-verify per-file identities, request binding, include closure, and
    /// manifest digest before any future executor consumes this bundle.
    pub fn verify_for_request(
        &self,
        request: &SimulationRequest,
    ) -> Result<(), ModelInputBundleError> {
        if self.request_id != request.id {
            return Err(ModelInputBundleError::RequestIdMismatch {
                bundle: self.request_id.clone(),
                request: request.id.clone(),
            });
        }
        self.primary.verify_for_request(request)?;
        for artifact in self.dependencies.values() {
            artifact.verify_for_request(request)?;
        }
        self.validate_closure()?;
        if self.compute_manifest_digest() != self.manifest_digest {
            return Err(ModelInputBundleError::DigestMismatch);
        }
        Ok(())
    }

    fn validate_closure(&self) -> Result<(), ModelInputBundleError> {
        let total_files = self.dependencies.len() + 1;
        if total_files > MAX_BUNDLE_FILES {
            return Err(ModelInputBundleError::TooManyFiles {
                actual: total_files,
                maximum: MAX_BUNDLE_FILES,
            });
        }

        let total_bytes = self.primary.bytes().len()
            + self.dependencies.values().map(|artifact| artifact.bytes().len()).sum::<usize>();
        if total_bytes > MAX_BUNDLE_BYTES {
            return Err(ModelInputBundleError::BundleTooLarge {
                actual: total_bytes,
                maximum: MAX_BUNDLE_BYTES,
            });
        }

        let mut visiting = BTreeSet::new();
        let mut visited = BTreeSet::new();
        let mut directive_count = 0usize;
        self.visit(
            &self.primary_path,
            self.primary.bytes(),
            &mut visiting,
            &mut visited,
            &mut directive_count,
        )?;

        if visited.len() != total_files {
            for path in self.dependencies.keys() {
                if !visited.contains(path) {
                    return Err(ModelInputBundleError::UnreferencedDependency(path.clone()));
                }
            }
        }
        Ok(())
    }

    fn visit(
        &self,
        path: &str,
        bytes: &[u8],
        visiting: &mut BTreeSet<String>,
        visited: &mut BTreeSet<String>,
        directive_count: &mut usize,
    ) -> Result<(), ModelInputBundleError> {
        if visiting.contains(path) {
            return Err(ModelInputBundleError::IncludeCycle(path.to_string()));
        }
        if visited.contains(path) {
            return Ok(());
        }
        visiting.insert(path.to_string());

        let source = std::str::from_utf8(bytes).map_err(|_| {
            ModelInputBundleError::Artifact(NetlistArtifactError::InvalidUtf8)
        })?;
        for (line_index, line) in source.lines().enumerate() {
            let target = parse_include_target(path, line_index + 1, line)?;
            let Some(target) = target else {
                continue;
            };
            *directive_count = directive_count.saturating_add(1);
            if *directive_count > MAX_INCLUDE_DIRECTIVES {
                return Err(ModelInputBundleError::TooManyIncludeDirectives {
                    actual: *directive_count,
                    maximum: MAX_INCLUDE_DIRECTIVES,
                });
            }
            let resolved = resolve_include_path(path, &target)?;
            let dependency = self.dependencies.get(&resolved).ok_or_else(|| {
                ModelInputBundleError::MissingDependency {
                    from: path.to_string(),
                    target: resolved.clone(),
                }
            })?;
            self.visit(
                &resolved,
                dependency.bytes(),
                visiting,
                visited,
                directive_count,
            )?;
        }

        visiting.remove(path);
        visited.insert(path.to_string());
        Ok(())
    }

    fn compute_manifest_digest(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hash_field(&mut hasher, self.request_id.as_bytes());
        hash_field(&mut hasher, self.primary_path.as_bytes());
        hash_field(&mut hasher, self.primary.blake3_digest().as_bytes());
        for (path, artifact) in &self.dependencies {
            hash_field(&mut hasher, path.as_bytes());
            hash_field(&mut hasher, artifact.blake3_digest().as_bytes());
        }
        hasher.finalize().to_hex().to_string()
    }
}

fn hash_field(hasher: &mut blake3::Hasher, bytes: &[u8]) {
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

fn validate_relative_path(path: String) -> Result<String, ModelInputBundleError> {
    let invalid = |reason: &str| ModelInputBundleError::InvalidPath {
        path: path.clone(),
        reason: reason.to_string(),
    };
    if path.is_empty() {
        return Err(invalid("path cannot be empty"));
    }
    if path.len() > 512 {
        return Err(invalid("path exceeds 512 bytes"));
    }
    if path.starts_with('/') || path.starts_with('\\') || path.contains('\\') {
        return Err(invalid("only relative paths using '/' separators are accepted"));
    }
    if path.contains(':') || path.contains('$') || path.contains('%') || path.starts_with('~') {
        return Err(invalid("absolute, drive-qualified, or expanded paths are not accepted"));
    }
    if path.chars().any(char::is_control) {
        return Err(invalid("control characters are not accepted"));
    }
    if path
        .split('/')
        .any(|part| part.is_empty() || part == "." || part == "..")
    {
        return Err(invalid("empty, '.' and '..' path components are not accepted"));
    }
    Ok(path)
}

fn resolve_include_path(from: &str, target: &str) -> Result<String, ModelInputBundleError> {
    let target = validate_relative_path(target.to_string())?;
    let parent = from.rsplit_once('/').map(|(parent, _)| parent);
    let candidate = match parent {
        Some(parent) => format!("{parent}/{target}"),
        None => target,
    };
    validate_relative_path(candidate)
}

fn parse_include_target(
    path: &str,
    line_number: usize,
    line: &str,
) -> Result<Option<String>, ModelInputBundleError> {
    let trimmed = line.trim_start();
    if trimmed.is_empty() || trimmed.starts_with('*') {
        return Ok(None);
    }

    let (directive, tail) = match trimmed.split_once(char::is_whitespace) {
        Some((directive, tail)) => (directive.to_ascii_lowercase(), tail.trim()),
        None => (trimmed.to_ascii_lowercase(), ""),
    };

    let args = match directive.as_str() {
        ".include" | ".incpslt" => {
            let args = parse_tokens(tail).map_err(|reason| {
                ModelInputBundleError::UnsupportedIncludeSyntax {
                    path: path.to_string(),
                    line: line_number,
                    reason,
                }
            })?;
            if args.len() != 1 {
                return Err(ModelInputBundleError::UnsupportedIncludeSyntax {
                    path: path.to_string(),
                    line: line_number,
                    reason: "expected exactly one path token".to_string(),
                });
            }
            return Ok(Some(args[0].clone()));
        }
        ".lib" => {
            let args = parse_tokens(tail).map_err(|reason| {
                ModelInputBundleError::UnsupportedIncludeSyntax {
                    path: path.to_string(),
                    line: line_number,
                    reason,
                }
            })?;
            match args.len() {
                0 => {
                    return Err(ModelInputBundleError::UnsupportedIncludeSyntax {
                        path: path.to_string(),
                        line: line_number,
                        reason: "expected a library section or file/section pair".to_string(),
                    });
                }
                1 => return Ok(None), // In-file library section opener.
                2 => args,
                _ => {
                    return Err(ModelInputBundleError::UnsupportedIncludeSyntax {
                        path: path.to_string(),
                        line: line_number,
                        reason: "external .lib accepts exactly a file path and section name".to_string(),
                    });
                }
            }
        }
        _ => return Ok(None),
    };

    // External .lib has two operands: path and section name.
    Ok(Some(args[0].clone()))
}

/// Tokenizer for the intentionally supported include syntax: whitespace
/// separators and one quoted or unquoted token per operand. No shell expansion,
/// escapes, or command substitution are performed.
fn parse_tokens(input: &str) -> Result<Vec<String>, String> {
    let chars: Vec<char> = input.chars().collect();
    let mut tokens = Vec::new();
    let mut i = 0usize;
    while i < chars.len() {
        while i < chars.len() && chars[i].is_whitespace() {
            i += 1;
        }
        if i >= chars.len() {
            break;
        }
        let quote = if chars[i] == '"' || chars[i] == '\'' {
            let quote = chars[i];
            i += 1;
            Some(quote)
        } else {
            None
        };
        let mut token = String::new();
        let mut closed_quote = quote.is_none();
        while i < chars.len() {
            let ch = chars[i];
            if let Some(quote_char) = quote {
                if ch == quote_char {
                    i += 1;
                    closed_quote = true;
                    break;
                }
                token.push(ch);
                i += 1;
            } else if ch.is_whitespace() {
                break;
            } else if ch == '"' || ch == '\'' {
                return Err("quote character in unquoted operand".into());
            } else {
                token.push(ch);
                i += 1;
            }
        }
        if !closed_quote {
            return Err("unterminated quoted operand".into());
        }
        if token.is_empty() {
            return Err("empty operands are not accepted".into());
        }
        tokens.push(token);
        if quote.is_some() && i < chars.len() && !chars[i].is_whitespace() {
            return Err("quoted operand must be followed by whitespace".into());
        }
    }
    Ok(tokens)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bundle(
        primary_path: &str,
        primary: &str,
        dependencies: Vec<(&str, &str)>,
    ) -> Result<ModelInputBundle, ModelInputBundleError> {
        ModelInputBundle::new(
            "bundle-test-1",
            primary_path,
            primary.as_bytes().to_vec(),
            dependencies
                .into_iter()
                .map(|(path, bytes)| (path.to_string(), bytes.as_bytes().to_vec())),
        )
    }

    #[test]
    fn resolves_transitive_includes_and_verifies_request() {
        let bundle = bundle(
            "main.cir",
            ".include \"models/device.lib\"\n",
            vec![
                ("models/device.lib", ".include nested/params.inc\n"),
                ("models/nested/params.inc", ".model Dtest D(Is=1e-14)\n"),
            ],
        )
        .unwrap();
        assert_eq!(bundle.dependency_paths().collect::<Vec<_>>(), vec!["models/device.lib", "models/nested/params.inc"]);
        assert_eq!(bundle.primary_path(), "main.cir");
        assert_eq!(bundle.manifest_digest().len(), 64);
        let request = SimulationRequest::new(
            "bundle-test-1",
            symthaea_sim_bridge::EngineeringDomain::Electrical,
            symthaea_sim_bridge::SolverKind::Circuit,
            "verify bundle",
        );
        bundle.verify_for_request(&request).unwrap();
    }

    #[test]
    fn manifest_digest_is_independent_of_dependency_input_order() {
        let deps_a = vec![("b.inc", ".param b=2\n"), ("a.inc", ".param a=1\n")];
        let deps_b = vec![("a.inc", ".param a=1\n"), ("b.inc", ".param b=2\n")];
        let a = bundle("main.cir", ".include a.inc\n.include b.inc\n", deps_a).unwrap();
        let b = bundle("main.cir", ".include a.inc\n.include b.inc\n", deps_b).unwrap();
        assert_eq!(a.manifest_digest(), b.manifest_digest());
    }

    #[test]
    fn resolves_external_lib_but_does_not_confuse_section_markers() {
        let bundle = bundle(
            "main.cir",
            ".lib models/device.lib tt\n",
            vec![("models/device.lib", ".lib tt\n.model D1 D(Is=1e-14)\n.endl tt\n")],
        )
        .unwrap();
        assert!(bundle.dependency_bytes("models/device.lib").is_some());
    }

    #[test]
    fn rejects_missing_dependency() {
        assert!(matches!(
            bundle("main.cir", ".include absent.lib\n", vec![]),
            Err(ModelInputBundleError::MissingDependency { .. })
        ));
    }

    #[test]
    fn rejects_parent_directory_traversal() {
        assert!(matches!(
            bundle("main.cir", ".include ../outside.lib\n", vec![]),
            Err(ModelInputBundleError::InvalidPath { .. })
        ));
    }

    #[test]
    fn rejects_absolute_include_paths() {
        assert!(matches!(
            bundle("main.cir", ".include /etc/passwd\n", vec![]),
            Err(ModelInputBundleError::InvalidPath { .. })
        ));
    }

    #[test]
    fn rejects_unreferenced_dependencies() {
        assert!(matches!(
            bundle("main.cir", "R1 a b 1k\n", vec![("extra.lib", ".model D1 D\n")]),
            Err(ModelInputBundleError::UnreferencedDependency(_))
        ));
    }

    #[test]
    fn rejects_include_cycles() {
        assert!(matches!(
            bundle(
                "main.cir",
                ".include a.inc\n",
                vec![("a.inc", ".include main.cir\n")],
            ),
            Err(ModelInputBundleError::IncludeCycle(_))
        ));
    }

    #[test]
    fn rejects_duplicate_paths_and_unsupported_quoting() {
        assert!(matches!(
            ModelInputBundle::new(
                "bundle-test-1",
                "main.cir",
                b"R1 a b 1k\n".to_vec(),
                vec![
                    ("a.inc".to_string(), b".param a=1\n".to_vec()),
                    ("a.inc".to_string(), b".param a=2\n".to_vec()),
                ],
            ),
            Err(ModelInputBundleError::DuplicatePath(_))
        ));
        assert!(matches!(
            bundle("main.cir", ".include \"unterminated.lib\n", vec![]),
            Err(ModelInputBundleError::UnsupportedIncludeSyntax { .. })
        ));
    }

    #[test]
    fn resolves_quoted_dependency_paths_with_spaces() {
        let bundle = bundle(
            "main.cir",
            ".include \"models/power device.lib\"\n",
            vec![("models/power device.lib", ".model D1 D(Is=1e-14)\n")],
        )
        .unwrap();
        assert!(bundle.dependency_bytes("models/power device.lib").is_some());
    }

    #[test]
    fn rejects_noncanonical_dependency_paths() {
        assert!(matches!(
            bundle(
                "main.cir",
                ".include a.inc\n",
                vec![("../a.inc", ".param a=1\n")],
            ),
            Err(ModelInputBundleError::InvalidPath { .. })
        ));
        assert!(matches!(
            bundle(
                "main.cir",
                ".include a.inc\n",
                vec![("models//a.inc", ".param a=1\n")],
            ),
            Err(ModelInputBundleError::InvalidPath { .. })
        ));
    }

    #[test]
    fn detects_manifest_tampering_and_request_mismatch() {
        let mut artifact = bundle("main.cir", "R1 a b 1k\n", vec![]).unwrap();
        artifact.manifest_digest.push('0');
        let request = SimulationRequest::new(
            "bundle-test-1",
            symthaea_sim_bridge::EngineeringDomain::Electrical,
            symthaea_sim_bridge::SolverKind::Circuit,
            "verify bundle",
        );
        assert_eq!(
            artifact.verify_for_request(&request),
            Err(ModelInputBundleError::DigestMismatch)
        );
        let artifact = bundle("main.cir", "R1 a b 1k\n", vec![]).unwrap();
        let wrong_request = SimulationRequest::new(
            "different-request",
            symthaea_sim_bridge::EngineeringDomain::Electrical,
            symthaea_sim_bridge::SolverKind::Circuit,
            "verify bundle",
        );
        assert!(matches!(
            artifact.verify_for_request(&wrong_request),
            Err(ModelInputBundleError::RequestIdMismatch { .. })
        ));
    }
}
