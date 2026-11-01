// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Verifies PITB's versioned, structural Lean expression wire format.
//!
//! This computes an environment-scoped structural identity. It does not
//! normalize definitional equality and is not a substitute for Lean's kernel.

use std::{
    collections::BTreeSet,
    env,
    fs,
    path::PathBuf,
    process::ExitCode,
};

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};

const SCHEMA_VERSION: &str = "pitb.lean-expr-wire.v1";
const SERIALIZER_ID: &str = "pitb.lean-expr-wire.v1/binary-transcript-1";
const MAX_EXPR_DEPTH: usize = 512;
const MAX_EXPR_NODES: usize = 250_000;
const MAX_CONTEXT_DECLARATIONS: usize = 65_536;
const MAX_DECIMAL_DIGITS: usize = 4_096;
const MAX_EVIDENCE_BYTES: usize = 64 * 1024 * 1024;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Evidence {
    schema_version: String,
    evidence_class: String,
    environment: Environment,
    target: Expr,
    local_context: LocalContext,
    digests: Digests,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Environment {
    lean_toolchain: String,
    lean_version: String,
    lean_binary_sha256: String,
    lean_runtime_manifest_sha256: String,
    imported_olean_manifest_sha256: String,
    lean_runtime_artifacts: Vec<RuntimeArtifact>,
    imported_olean_artifacts: Vec<ImportedOleanArtifact>,
    task_id: String,
    source_repository: String,
    source_path: String,
    source_file_sha256: String,
    source_commit: String,
    lean_toolchain_file_sha256: String,
    lakefile_sha256: String,
    lake_manifest_sha256: String,
    root_module_sha256: String,
    context_exporter_source_sha256: String,
    clean_checkout: bool,
    serializer_id: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RuntimeArtifact {
    path: String,
    kind: String,
    sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ImportedOleanArtifact {
    module_name: String,
    path: String,
    sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Digests {
    environment_sha256: String,
    target_expr_sha256: String,
    context_sha256: String,
    proof_state_sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct LocalContext {
    declarations: Vec<LocalDecl>,
    local_instances: Vec<LocalInstance>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct LocalDecl {
    local_index: u64,
    user_name: String,
    binder_info: BinderInfo,
    is_aux_declaration: bool,
    is_implementation_detail: bool,
    is_let_declaration: bool,
    is_nondep: bool,
    #[serde(rename = "type")]
    ty: Expr,
    transparent_value: Option<Expr>,
    nondep_value_provenance: Option<Expr>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct LocalInstance {
    class_name: LeanName,
    local_index: u64,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
enum BinderInfo {
    Explicit,
    Implicit,
    StrictImplicit,
    InstanceImplicit,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "tag", deny_unknown_fields)]
enum LeanName {
    #[serde(rename = "anonymous")]
    Anonymous,
    #[serde(rename = "str")]
    Str { prefix: Box<LeanName>, value: String },
    #[serde(rename = "num")]
    Num { prefix: Box<LeanName>, value_decimal: String },
}

#[derive(Debug, Deserialize)]
#[serde(tag = "tag", deny_unknown_fields)]
enum Level {
    #[serde(rename = "zero")]
    Zero,
    #[serde(rename = "succ")]
    Succ { of: Box<Level> },
    #[serde(rename = "max")]
    Max { left: Box<Level>, right: Box<Level> },
    #[serde(rename = "imax")]
    IMax { left: Box<Level>, right: Box<Level> },
    #[serde(rename = "param")]
    Param { name: LeanName },
}

#[derive(Debug, Deserialize)]
#[serde(tag = "tag", deny_unknown_fields)]
enum Expr {
    #[serde(rename = "bvar")]
    BVar { index: u64 },
    #[serde(rename = "fvar")]
    FVar { local_index: u64 },
    #[serde(rename = "sort")]
    Sort { level: Level },
    #[serde(rename = "const")]
    Const { name: LeanName, levels: Vec<Level> },
    #[serde(rename = "app")]
    App { function: Box<Expr>, argument: Box<Expr> },
    #[serde(rename = "lam")]
    Lam {
        binder_info: BinderInfo,
        domain: Box<Expr>,
        body: Box<Expr>,
    },
    #[serde(rename = "forall_e")]
    ForallE {
        binder_info: BinderInfo,
        domain: Box<Expr>,
        body: Box<Expr>,
    },
    #[serde(rename = "let_e")]
    LetE {
        domain: Box<Expr>,
        value: Box<Expr>,
        body: Box<Expr>,
        nondep: bool,
    },
    #[serde(rename = "literal_nat")]
    LiteralNat { value_decimal: String },
    #[serde(rename = "literal_string")]
    LiteralString { value: String },
    #[serde(rename = "proj")]
    Proj {
        type_name: LeanName,
        field_index: u64,
        structure: Box<Expr>,
    },
}

#[derive(Default)]
struct Encoder(Vec<u8>);

impl Encoder {
    fn raw(&mut self, bytes: &[u8]) {
        self.0.extend_from_slice(bytes);
    }

    fn tag(&mut self, value: u8) {
        self.0.push(value);
    }

    fn u64(&mut self, value: u64) {
        self.0.extend_from_slice(&value.to_be_bytes());
    }

    fn bytes(&mut self, value: &[u8]) {
        self.u64(value.len() as u64);
        self.raw(value);
    }

    fn string(&mut self, value: &str) {
        self.bytes(value.as_bytes());
    }

    fn boolean(&mut self, value: bool) {
        self.tag(u8::from(value));
    }
}

fn valid_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn validate_decimal(value: &str, label: &str) -> Result<()> {
    if value.is_empty()
        || value.len() > MAX_DECIMAL_DIGITS
        || !value.bytes().all(|byte| byte.is_ascii_digit())
        || (value.len() > 1 && value.starts_with('0'))
    {
        bail!("{label} is not a canonical non-negative decimal string");
    }
    Ok(())
}

fn validate_name(name: &LeanName, depth: usize) -> Result<()> {
    if depth > MAX_EXPR_DEPTH {
        bail!("Lean Name exceeds the maximum nesting depth");
    }
    match name {
        LeanName::Anonymous => Ok(()),
        LeanName::Str { prefix, .. } => validate_name(prefix, depth + 1),
        LeanName::Num {
            prefix,
            value_decimal,
        } => {
            validate_name(prefix, depth + 1)?;
            validate_decimal(value_decimal, "Lean Name numeric component")
        }
    }
}

fn is_anonymous_name(name: &LeanName) -> bool {
    matches!(name, LeanName::Anonymous)
}

fn validate_level(level: &Level, depth: usize) -> Result<()> {
    if depth > MAX_EXPR_DEPTH {
        bail!("universe Level exceeds the maximum nesting depth");
    }
    match level {
        Level::Zero => Ok(()),
        Level::Succ { of } => validate_level(of, depth + 1),
        Level::Max { left, right } | Level::IMax { left, right } => {
            validate_level(left, depth + 1)?;
            validate_level(right, depth + 1)
        }
        Level::Param { name } => {
            validate_name(name, depth + 1)?;
            if is_anonymous_name(name) {
                bail!("universe parameter name cannot be anonymous");
            }
            Ok(())
        }
    }
}

fn validate_expr(
    expr: &Expr,
    context_len: u64,
    max_local_index_exclusive: Option<u64>,
    bound_depth: u64,
    depth: usize,
    nodes: &mut usize,
) -> Result<()> {
    *nodes = (*nodes).checked_add(1).context("expression node count overflow")?;
    if *nodes > MAX_EXPR_NODES || depth > MAX_EXPR_DEPTH {
        bail!("expression exceeds the configured complexity limit");
    }
    let child = |expr: &Expr,
                 bound_depth: u64,
                 depth: usize,
                 nodes: &mut usize|
     -> Result<()> {
        validate_expr(
            expr,
            context_len,
            max_local_index_exclusive,
            bound_depth,
            depth,
            nodes,
        )
    };
    match expr {
        Expr::BVar { index } => {
            if *index >= bound_depth {
                bail!("bound variable index {index} escapes its binder scope");
            }
        }
        Expr::FVar { local_index } => {
            if *local_index >= context_len {
                bail!("free variable index {local_index} is outside the captured context");
            }
            if max_local_index_exclusive.is_some_and(|limit| *local_index >= limit) {
                bail!("local declaration references itself or a later telescope declaration");
            }
        }
        Expr::Sort { level } => validate_level(level, depth + 1)?,
        Expr::Const { name, levels } => {
            validate_name(name, depth + 1)?;
            if is_anonymous_name(name) {
                bail!("constant name cannot be anonymous");
            }
            for level in levels {
                validate_level(level, depth + 1)?;
            }
        }
        Expr::App { function, argument } => {
            child(function, bound_depth, depth + 1, nodes)?;
            child(argument, bound_depth, depth + 1, nodes)?;
        }
        Expr::Lam {
            binder_info: _,
            domain,
            body,
        }
        | Expr::ForallE {
            binder_info: _,
            domain,
            body,
        } => {
            child(domain, bound_depth, depth + 1, nodes)?;
            let next_depth = bound_depth.checked_add(1).context("binder depth overflow")?;
            child(body, next_depth, depth + 1, nodes)?;
        }
        Expr::LetE {
            domain,
            value,
            body,
            nondep: _,
        } => {
            child(domain, bound_depth, depth + 1, nodes)?;
            child(value, bound_depth, depth + 1, nodes)?;
            let next_depth = bound_depth.checked_add(1).context("binder depth overflow")?;
            child(body, next_depth, depth + 1, nodes)?;
        }
        Expr::LiteralNat { value_decimal } => {
            validate_decimal(value_decimal, "natural-number literal")?;
        }
        Expr::LiteralString { .. } => {}
        Expr::Proj {
            type_name,
            field_index: _,
            structure,
        } => {
            validate_name(type_name, depth + 1)?;
            if is_anonymous_name(type_name) {
                bail!("projection type name cannot be anonymous");
            }
            child(structure, bound_depth, depth + 1, nodes)?;
        }
    }
    Ok(())
}

fn encode_name(encoder: &mut Encoder, name: &LeanName) {
    match name {
        LeanName::Anonymous => encoder.tag(0),
        LeanName::Str { prefix, value } => {
            encoder.tag(1);
            encode_name(encoder, prefix);
            encoder.string(value);
        }
        LeanName::Num {
            prefix,
            value_decimal,
        } => {
            encoder.tag(2);
            encode_name(encoder, prefix);
            encoder.string(value_decimal);
        }
    }
}

fn encode_level(encoder: &mut Encoder, level: &Level) {
    match level {
        Level::Zero => encoder.tag(0),
        Level::Succ { of } => {
            encoder.tag(1);
            encode_level(encoder, of);
        }
        Level::Max { left, right } => {
            encoder.tag(2);
            encode_level(encoder, left);
            encode_level(encoder, right);
        }
        Level::IMax { left, right } => {
            encoder.tag(3);
            encode_level(encoder, left);
            encode_level(encoder, right);
        }
        Level::Param { name } => {
            encoder.tag(4);
            encode_name(encoder, name);
        }
    }
}

fn encode_binder_info(encoder: &mut Encoder, binder_info: BinderInfo) {
    let tag = match binder_info {
        BinderInfo::Explicit => 0,
        BinderInfo::Implicit => 1,
        BinderInfo::StrictImplicit => 2,
        BinderInfo::InstanceImplicit => 3,
    };
    encoder.tag(tag);
}

fn encode_expr(encoder: &mut Encoder, expr: &Expr) {
    match expr {
        Expr::BVar { index } => {
            encoder.tag(0);
            encoder.u64(*index);
        }
        Expr::FVar { local_index } => {
            encoder.tag(1);
            encoder.u64(*local_index);
        }
        Expr::Sort { level } => {
            encoder.tag(2);
            encode_level(encoder, level);
        }
        Expr::Const { name, levels } => {
            encoder.tag(3);
            encode_name(encoder, name);
            encoder.u64(levels.len() as u64);
            for level in levels {
                encode_level(encoder, level);
            }
        }
        Expr::App { function, argument } => {
            encoder.tag(4);
            encode_expr(encoder, function);
            encode_expr(encoder, argument);
        }
        Expr::Lam {
            binder_info,
            domain,
            body,
        } => {
            encoder.tag(5);
            encode_binder_info(encoder, *binder_info);
            encode_expr(encoder, domain);
            encode_expr(encoder, body);
        }
        Expr::ForallE {
            binder_info,
            domain,
            body,
        } => {
            encoder.tag(6);
            encode_binder_info(encoder, *binder_info);
            encode_expr(encoder, domain);
            encode_expr(encoder, body);
        }
        Expr::LetE {
            domain,
            value,
            body,
            nondep,
        } => {
            encoder.tag(7);
            encoder.boolean(*nondep);
            encode_expr(encoder, domain);
            encode_expr(encoder, value);
            encode_expr(encoder, body);
        }
        Expr::LiteralNat { value_decimal } => {
            encoder.tag(8);
            encoder.string(value_decimal);
        }
        Expr::LiteralString { value } => {
            encoder.tag(9);
            encoder.string(value);
        }
        Expr::Proj {
            type_name,
            field_index,
            structure,
        } => {
            encoder.tag(10);
            encode_name(encoder, type_name);
            encoder.u64(*field_index);
            encode_expr(encoder, structure);
        }
    }
}

fn encode_environment(encoder: &mut Encoder, environment: &Environment) {
    encoder.string(&environment.lean_toolchain);
    encoder.string(&environment.lean_version);
    encoder.string(&environment.lean_binary_sha256);
    encoder.string(&environment.lean_runtime_manifest_sha256);
    encoder.string(&environment.imported_olean_manifest_sha256);
    encoder.string(&environment.task_id);
    encoder.string(&environment.source_repository);
    encoder.string(&environment.source_path);
    encoder.string(&environment.source_file_sha256);
    encoder.string(&environment.source_commit);
    encoder.string(&environment.lean_toolchain_file_sha256);
    encoder.string(&environment.lakefile_sha256);
    encoder.string(&environment.lake_manifest_sha256);
    encoder.string(&environment.root_module_sha256);
    encoder.string(&environment.context_exporter_source_sha256);
    encoder.boolean(environment.clean_checkout);
    encoder.string(&environment.serializer_id);
}

fn encode_context(encoder: &mut Encoder, context: &LocalContext) {
    encoder.u64(context.declarations.len() as u64);
    for declaration in &context.declarations {
        encoder.u64(declaration.local_index);
        // The Lean shim maps generated FVarIds to these ordered indices before
        // serialization. FVarIds themselves are deliberately absent from the
        // stable wire format; tactic-visible user names remain bound.
        encoder.string(&declaration.user_name);
        encode_binder_info(encoder, declaration.binder_info);
        encoder.boolean(declaration.is_aux_declaration);
        encoder.boolean(declaration.is_implementation_detail);
        encoder.boolean(declaration.is_let_declaration);
        encoder.boolean(declaration.is_nondep);
        encode_expr(encoder, &declaration.ty);
        match &declaration.transparent_value {
            Some(value) => {
                encoder.tag(1);
                encode_expr(encoder, value);
            }
            None => encoder.tag(0),
        }
        match &declaration.nondep_value_provenance {
            Some(value) => {
                encoder.tag(1);
                encode_expr(encoder, value);
            }
            None => encoder.tag(0),
        }
    }
    encoder.u64(context.local_instances.len() as u64);
    for instance in &context.local_instances {
        encode_name(encoder, &instance.class_name);
        encoder.u64(instance.local_index);
    }
}

fn sha256_domain(domain: &[u8], append: impl FnOnce(&mut Encoder)) -> String {
    let mut encoder = Encoder::default();
    encoder.raw(b"PITB-LEAN-EXPR-WIRE-V1\0");
    encoder.raw(domain);
    append(&mut encoder);
    hex::encode(Sha256::digest(&encoder.0))
}

fn validate_digest(value: &str, label: &str) -> Result<()> {
    if !valid_hex(value, 64) {
        bail!("{label} must be 64 lowercase hexadecimal characters");
    }
    Ok(())
}

fn compute_digests(evidence: &Evidence) -> DigestsComputed {
    let environment_sha256 = sha256_domain(b"ENVIRONMENT\0", |encoder| {
        encode_environment(encoder, &evidence.environment);
    });
    let environment_bytes = hex::decode(&environment_sha256)
        .expect("SHA-256 output is always valid hexadecimal");
    let target_expr_sha256 = sha256_domain(b"TARGET\0", |encoder| {
        encoder.raw(&environment_bytes);
        encode_expr(encoder, &evidence.target);
    });
    let context_sha256 = sha256_domain(b"CONTEXT\0", |encoder| {
        encoder.raw(&environment_bytes);
        encode_context(encoder, &evidence.local_context);
    });
    let target_bytes = hex::decode(&target_expr_sha256)
        .expect("SHA-256 output is always valid hexadecimal");
    let context_bytes = hex::decode(&context_sha256)
        .expect("SHA-256 output is always valid hexadecimal");
    let proof_state_sha256 = sha256_domain(b"PROOF-STATE\0", |encoder| {
        encoder.raw(&environment_bytes);
        encoder.raw(&target_bytes);
        encoder.raw(&context_bytes);
    });
    DigestsComputed {
        environment_sha256,
        target_expr_sha256,
        context_sha256,
        proof_state_sha256,
    }
}

struct DigestsComputed {
    environment_sha256: String,
    target_expr_sha256: String,
    context_sha256: String,
    proof_state_sha256: String,
}

fn collect_expr_fvars(expr: &Expr, result: &mut BTreeSet<u64>) {
    match expr {
        Expr::FVar { local_index } => {
            result.insert(*local_index);
        }
        Expr::App { function, argument } => {
            collect_expr_fvars(function, result);
            collect_expr_fvars(argument, result);
        }
        Expr::Lam { domain, body, .. } | Expr::ForallE { domain, body, .. } => {
            collect_expr_fvars(domain, result);
            collect_expr_fvars(body, result);
        }
        Expr::LetE { domain, value, body, .. } => {
            collect_expr_fvars(domain, result);
            collect_expr_fvars(value, result);
            collect_expr_fvars(body, result);
        }
        Expr::Proj { structure, .. } => collect_expr_fvars(structure, result),
        Expr::BVar { .. }
        | Expr::Sort { .. }
        | Expr::Const { .. }
        | Expr::LiteralNat { .. }
        | Expr::LiteralString { .. } => {}
    }
}

fn dependency_edge_counts(context: &LocalContext) -> (usize, usize) {
    let mut typing_edges = 0;
    let mut provenance_edges = 0;
    for declaration in &context.declarations {
        let mut typing = BTreeSet::new();
        collect_expr_fvars(&declaration.ty, &mut typing);
        if let Some(value) = &declaration.transparent_value {
            collect_expr_fvars(value, &mut typing);
        }
        typing_edges += typing.len();

        if let Some(value) = &declaration.nondep_value_provenance {
            let mut provenance = BTreeSet::new();
            collect_expr_fvars(value, &mut provenance);
            provenance_edges += provenance.len();
        }
    }
    (typing_edges, provenance_edges)
}

fn validate_relative_source_path(value: &str) -> Result<()> {
    let path = std::path::Path::new(value);
    if value.is_empty()
        || path.is_absolute()
        || value.contains('\\')
        || value.contains('\0')
        || value.contains(':')
        || value.split('/').any(|part| part.is_empty() || part == "." || part == "..")
        || path.components().any(|component| {
            matches!(
                component,
                std::path::Component::ParentDir
                    | std::path::Component::CurDir
                    | std::path::Component::RootDir
                    | std::path::Component::Prefix(_)
            )
        })
    {
        bail!("source_path must be a safe repository-relative path");
    }
    Ok(())
}

fn compute_runtime_manifest_digest(rows: &[RuntimeArtifact]) -> String {
    sha256_domain(b"RUNTIME-MANIFEST\0", |encoder| {
        encoder.u64(rows.len() as u64);
        for row in rows {
            encoder.string(&row.path);
            encoder.string(&row.kind);
            encoder.string(&row.sha256);
        }
    })
}

fn compute_imported_olean_manifest_digest(rows: &[ImportedOleanArtifact]) -> String {
    sha256_domain(b"IMPORTED-OLEAN-MANIFEST\0", |encoder| {
        encoder.u64(rows.len() as u64);
        for row in rows {
            encoder.string(&row.module_name);
            encoder.string(&row.path);
            encoder.string(&row.sha256);
        }
    })
}

fn validate_artifact_manifests(environment: &Environment) -> Result<()> {
    if environment.lean_runtime_artifacts.is_empty()
        || environment.imported_olean_artifacts.is_empty()
    {
        bail!("runtime and imported OLean artifact manifests must be nonempty");
    }
    let mut previous_runtime_path: Option<&str> = None;
    let mut executable_count = 0usize;
    for row in &environment.lean_runtime_artifacts {
        validate_relative_source_path(&row.path)?;
        validate_digest(&row.sha256, "runtime artifact SHA-256")?;
        if previous_runtime_path.is_some_and(|previous| previous >= row.path.as_str()) {
            bail!("runtime artifact paths must be unique and strictly sorted");
        }
        previous_runtime_path = Some(&row.path);
        match row.kind.as_str() {
            "lean_executable" => {
                executable_count += 1;
                if row.sha256 != environment.lean_binary_sha256 {
                    bail!("runtime manifest executable hash differs from lean_binary_sha256");
                }
            }
            "shared_library" => {}
            _ => bail!("unknown runtime artifact kind: {}", row.kind),
        }
    }
    if executable_count != 1 {
        bail!("runtime manifest must contain exactly one Lean executable");
    }
    let runtime_digest = compute_runtime_manifest_digest(&environment.lean_runtime_artifacts);
    if runtime_digest != environment.lean_runtime_manifest_sha256 {
        bail!("Lean runtime manifest digest does not match its artifact rows");
    }

    let mut previous_pair: Option<(&str, &str)> = None;
    let mut modules = BTreeSet::new();
    let mut paths = BTreeSet::new();
    for row in &environment.imported_olean_artifacts {
        if row.module_name.trim().is_empty() {
            bail!("imported OLean module name cannot be empty");
        }
        validate_relative_source_path(&row.path)?;
        if !row.path.ends_with(".olean") {
            bail!("imported artifact path must end in .olean");
        }
        validate_digest(&row.sha256, "imported OLean SHA-256")?;
        let pair = (row.module_name.as_str(), row.path.as_str());
        if previous_pair.is_some_and(|previous| previous >= pair) {
            bail!("imported OLean artifacts must be strictly sorted by module and path");
        }
        previous_pair = Some(pair);
        if !modules.insert(row.module_name.as_str()) || !paths.insert(row.path.as_str()) {
            bail!("imported OLean module names and artifact paths must be unique");
        }
    }
    let imported_digest =
        compute_imported_olean_manifest_digest(&environment.imported_olean_artifacts);
    if imported_digest != environment.imported_olean_manifest_sha256 {
        bail!("imported OLean manifest digest does not match its artifact rows");
    }
    Ok(())
}

fn validate_evidence(evidence: &Evidence) -> Result<DigestsComputed> {
    if evidence.schema_version != SCHEMA_VERSION {
        bail!("unsupported schema version");
    }
    if !matches!(
        evidence.evidence_class.as_str(),
        "synthetic_contract_fixture" | "lean_capture"
    ) {
        bail!("unsupported evidence class");
    }
    let environment = &evidence.environment;
    for (label, value) in [
        ("lean_toolchain", environment.lean_toolchain.as_str()),
        ("lean_version", environment.lean_version.as_str()),
    ] {
        if value.trim().is_empty() {
            bail!("{label} cannot be empty");
        }
    }
    if !valid_hex(&environment.source_commit, 40) {
        bail!("source_commit must be a 40-character lowercase Git SHA-1");
    }
    for (label, value) in [
        ("lean_binary_sha256", environment.lean_binary_sha256.as_str()),
        (
            "lean_runtime_manifest_sha256",
            environment.lean_runtime_manifest_sha256.as_str(),
        ),
        (
            "imported_olean_manifest_sha256",
            environment.imported_olean_manifest_sha256.as_str(),
        ),
        (
            "lean_toolchain_file_sha256",
            environment.lean_toolchain_file_sha256.as_str(),
        ),
        ("lakefile_sha256", environment.lakefile_sha256.as_str()),
        (
            "lake_manifest_sha256",
            environment.lake_manifest_sha256.as_str(),
        ),
        ("root_module_sha256", environment.root_module_sha256.as_str()),
        (
            "context_exporter_source_sha256",
            environment.context_exporter_source_sha256.as_str(),
        ),
    ] {
        validate_digest(value, label)?;
    }
    if !environment.clean_checkout {
        bail!("evidence from a dirty checkout is not eligible for identity qualification");
    }
    if environment.serializer_id != SERIALIZER_ID {
        bail!("unsupported serializer_id");
    }

    if !valid_hex(&environment.task_id, 64)
        || !valid_hex(&environment.source_file_sha256, 64)
    {
        bail!("task_id and source_file_sha256 must be 64-character lowercase SHA-256 digests");
    }
    if environment.source_repository.trim().is_empty() {
        bail!("source_repository cannot be empty");
    }
    validate_relative_source_path(&environment.source_path)?;
    validate_artifact_manifests(environment)?;

    let declarations = &evidence.local_context.declarations;
    if declarations.len() > MAX_CONTEXT_DECLARATIONS {
        bail!("local context exceeds the declaration cap");
    }
    let context_len = declarations.len() as u64;
    let mut total_expr_nodes = 0;
    for (expected_index, declaration) in declarations.iter().enumerate() {
        if declaration.local_index != expected_index as u64 {
            bail!("local declaration indices must be contiguous and ordered from zero");
        }
        if declaration.user_name.trim().is_empty() {
            bail!("local declaration user_name cannot be empty");
        }
        if declaration.transparent_value.is_some() && !declaration.is_let_declaration {
            bail!("transparent_value is only valid for a let declaration");
        }
        if declaration.is_nondep && !declaration.is_let_declaration {
            bail!("nondependent declarations must be represented as let/have declarations");
        }
        if declaration.transparent_value.is_some()
            && declaration.nondep_value_provenance.is_some()
        {
            bail!("typing-visible and provenance-only values must remain separate");
        }
        if declaration.is_nondep != declaration.nondep_value_provenance.is_some() {
            bail!(concat!(
                "nondependent value provenance must be present exactly ",
                "for nondependent declarations"
            ));
        }
        validate_expr(
            &declaration.ty,
            context_len,
            Some(declaration.local_index),
            0,
            0,
            &mut total_expr_nodes,
        )
        .with_context(|| format!("validate type of local declaration {expected_index}"))?;
        if let Some(value) = &declaration.transparent_value {
            validate_expr(
                value,
                context_len,
                Some(declaration.local_index),
                0,
                0,
                &mut total_expr_nodes,
            )
            .with_context(|| {
                format!("validate transparent value of local declaration {expected_index}")
            })?;
        }
        if let Some(value) = &declaration.nondep_value_provenance {
            validate_expr(
                value,
                context_len,
                Some(declaration.local_index),
                0,
                0,
                &mut total_expr_nodes,
            )
            .with_context(|| {
                format!(
                    "validate nondependent value provenance of local declaration {expected_index}"
                )
            })?;
        }
    }

    let mut instance_indices = BTreeSet::new();
    for instance in &evidence.local_context.local_instances {
        validate_name(&instance.class_name, 0)?;
        if is_anonymous_name(&instance.class_name) {
            bail!("local instance class name cannot be anonymous");
        }
        if instance.local_index >= context_len {
            bail!("local instance points outside the captured context");
        }
        if !instance_indices.insert(instance.local_index) {
            bail!("local instance registry repeats a local declaration index");
        }
    }

    validate_expr(
        &evidence.target,
        context_len,
        None,
        0,
        0,
        &mut total_expr_nodes,
    )
    .context("validate target expression")?;

    let actual = compute_digests(evidence);
    for (label, claimed, computed) in [
        (
            "environment_sha256",
            evidence.digests.environment_sha256.as_str(),
            actual.environment_sha256.as_str(),
        ),
        (
            "target_expr_sha256",
            evidence.digests.target_expr_sha256.as_str(),
            actual.target_expr_sha256.as_str(),
        ),
        (
            "context_sha256",
            evidence.digests.context_sha256.as_str(),
            actual.context_sha256.as_str(),
        ),
        (
            "proof_state_sha256",
            evidence.digests.proof_state_sha256.as_str(),
            actual.proof_state_sha256.as_str(),
        ),
    ] {
        if !valid_hex(claimed, 64) || claimed != computed {
            bail!("{label} does not match the independently recomputed transcript digest");
        }
    }
    Ok(actual)
}

fn validate_explicit_nullable_fields(value: &Value) -> Result<()> {
    let declarations = value
        .pointer("/local_context/declarations")
        .and_then(Value::as_array)
        .context("local_context.declarations must be an array")?;
    for (index, declaration) in declarations.iter().enumerate() {
        let object = declaration
            .as_object()
            .with_context(|| format!("local declaration {index} must be an object"))?;
        for field in ["transparent_value", "nondep_value_provenance"] {
            if !object.contains_key(field) {
                bail!("local declaration {index} omits required nullable field {field}");
            }
        }
    }
    Ok(())
}

fn load(path: &PathBuf) -> Result<(Value, Evidence)> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    if bytes.len() > MAX_EVIDENCE_BYTES {
        bail!("evidence exceeds the 64 MiB input cap");
    }
    let value: Value = serde_json::from_slice(&bytes).context("parse evidence JSON")?;
    validate_explicit_nullable_fields(&value)?;
    let evidence: Evidence = serde_json::from_value(value.clone())
        .context("decode typed evidence")?;
    Ok((value, evidence))
}

fn reject_value(label: &str, value: Value) -> Result<()> {
    match serde_json::from_value::<Evidence>(value) {
        Err(_) => {
            println!("PITB_LEAN_EXPR_WIRE_NEGATIVE_PASS: {label}");
            Ok(())
        }
        Ok(evidence) => {
            if validate_evidence(&evidence).is_err() {
                println!("PITB_LEAN_EXPR_WIRE_NEGATIVE_PASS: {label}");
                Ok(())
            } else {
                bail!("negative control unexpectedly validated: {label}")
            }
        }
    }
}

fn self_test(path: &PathBuf) -> Result<()> {
    let (fixture, evidence) = load(path)?;
    if evidence.evidence_class != "synthetic_contract_fixture" {
        bail!("self-test fixture must remain synthetic");
    }
    let digests = validate_evidence(&evidence)?;
    let (typing_edges, provenance_edges) = dependency_edge_counts(&evidence.local_context);
    if typing_edges != 3 || provenance_edges != 1 {
        bail!(concat!(
            "synthetic value-role calibration changed: expected typing=3/provenance=1, ",
            "got typing={typing_edges}/provenance={provenance_edges}"
        ));
    }
    println!(
        concat!(
            "PITB_LEAN_EXPR_WIRE_FIXTURE_PASS: declarations={} local_instances={} ",
            "typing_edges={} provenance_edges={} target_nodes_hash={}"
        ),
        evidence.local_context.declarations.len(),
        evidence.local_context.local_instances.len(),
        typing_edges,
        provenance_edges,
        digests.target_expr_sha256
    );

    let mut bad = fixture.clone();
    bad["target"] = serde_json::json!({"tag":"fvar","local_index":99});
    reject_value("target FVar escapes the captured context", bad)?;

    let mut bad = fixture.clone();
    bad["target"] = serde_json::json!({"tag":"bvar","index":0});
    reject_value("unbound de Bruijn index", bad)?;

    let mut bad = fixture.clone();
    bad["local_context"]["declarations"][1]["type"] =
        serde_json::json!({"tag":"fvar","local_index":1});
    reject_value("self-referential telescope declaration", bad)?;

    let mut bad = fixture.clone();
    bad["local_context"]["local_instances"] = serde_json::json!([]);
    reject_value("local-instance removal changes context identity", bad)?;

    let mut bad = fixture.clone();
    bad["environment"]["lean_version"] = Value::String("different-toolchain".into());
    reject_value("environment drift changes identity", bad)?;

    let mut bad = fixture.clone();
    bad["environment"]["imported_olean_manifest_sha256"] =
        Value::String("c".repeat(64));
    reject_value("imported OLean closure drift changes identity", bad)?;
    let mut bad = fixture.clone();
    bad["environment"]["lean_runtime_artifacts"][1]["sha256"] =
        Value::String("c".repeat(64));
    reject_value("runtime artifact bytes differ from manifest digest", bad)?;

    let mut bad = fixture.clone();
    bad["environment"]["imported_olean_artifacts"][1]["sha256"] =
        Value::String("c".repeat(64));
    reject_value("imported OLean bytes differ from manifest digest", bad)?;

    let mut bad = fixture.clone();
    bad["environment"]["imported_olean_artifacts"][1]["path"] =
        Value::String("../outside.olean".into());
    reject_value("imported OLean path traversal", bad)?;


    let mut bad = fixture.clone();
    bad["environment"]["source_path"] = Value::String("../outside/Target.lean".into());
    reject_value("source path traversal", bad)?;

    let mut bad = fixture.clone();
    bad["local_context"]["local_instances"] = serde_json::json!([
        {
            "class_name": {
                "tag": "str",
                "prefix": {"tag": "anonymous"},
                "value": "PITBLocalInstanceProbe"
            },
            "local_index": 2
        },
        {
            "class_name": {
                "tag": "str",
                "prefix": {"tag": "anonymous"},
                "value": "PITBLocalInstanceProbe"
            },
            "local_index": 2
        }
    ]);
    reject_value("duplicate local instance registry entry", bad)?;

    let mut bad = fixture.clone();
    bad["local_context"]["declarations"][4]["is_nondep"] = Value::Bool(false);
    reject_value("nondependent value cannot become a typing dependency", bad)?;

    let mut bad = fixture.clone();
    bad["target"] = serde_json::json!({"tag":"mvar","id":"?m.1"});
    reject_value("metavariable expression is not kernel evidence", bad)?;

    let mut bad = fixture.clone();
    bad["target"] = serde_json::json!({"tag":"sort","level":{"tag":"mvar","id":"?u.1"}});
    reject_value("unresolved universe metavariable is not kernel evidence", bad)?;

    let mut bad = fixture.clone();
    bad["unreviewed_extra_field"] = Value::Bool(true);
    reject_value("unknown top-level field", bad)?;

    let mut missing_nullable = fixture.clone();
    missing_nullable["local_context"]["declarations"][0]
        .as_object_mut()
        .context("synthetic declaration must be an object")?
        .remove("transparent_value");
    if validate_explicit_nullable_fields(&missing_nullable).is_ok() {
        bail!("missing explicit nullable field was accepted");
    }
    println!("PITB_LEAN_EXPR_WIRE_NEGATIVE_PASS: omitted nullable field");
    Ok(())
}

enum Mode {
    Help,
    Verify(PathBuf),
    SelfTest(PathBuf),
}

fn parse_mode() -> Result<Mode> {
    let mut args = env::args().skip(1);
    let mut verify = None;
    let mut self_test = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--verify" => verify = args.next().map(PathBuf::from),
            "--self-test" => self_test = args.next().map(PathBuf::from),
            "--help" | "-h" => return Ok(Mode::Help),
            _ => bail!("unknown argument: {arg}"),
        }
    }
    match (verify, self_test) {
        (Some(path), None) => Ok(Mode::Verify(path)),
        (None, Some(path)) => Ok(Mode::SelfTest(path)),
        (None, None) => bail!("--verify FILE or --self-test FILE is required"),
        (Some(_), Some(_)) => bail!("--verify and --self-test are mutually exclusive"),
    }
}

fn run() -> Result<()> {
    match parse_mode()? {
        Mode::Help => {
            println!("Usage: verify_lean_expr_wire_v1 --verify FILE | --self-test FILE");
            Ok(())
        }
        Mode::Verify(path) => {
            let (_, evidence) = load(&path)?;
            let digests = validate_evidence(&evidence)?;
            let (typing_edges, provenance_edges) = dependency_edge_counts(&evidence.local_context);
            println!(
                concat!(
                    "PITB_LEAN_EXPR_WIRE_PASS: declarations={} local_instances={} ",
                    "typing_edges={} provenance_edges={} environment={} target={} ",
                    "context={} proof_state={}"
                ),
                evidence.local_context.declarations.len(),
                evidence.local_context.local_instances.len(),
                typing_edges,
                provenance_edges,
                digests.environment_sha256,
                digests.target_expr_sha256,
                digests.context_sha256,
                digests.proof_state_sha256
            );
            Ok(())
        }
        Mode::SelfTest(path) => self_test(&path),
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("PITB_LEAN_EXPR_WIRE_FAIL: {error:#}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decimal_literals_are_exact_and_canonical() {
        assert!(validate_decimal("0", "test").is_ok());
        assert!(validate_decimal("123456789012345678901234567890", "test").is_ok());
        assert!(validate_decimal("00", "test").is_err());
        assert!(validate_decimal("+1", "test").is_err());
    }

    #[test]
    fn binder_info_has_four_stable_codes() {
        let mut enc = Encoder::default();
        for value in [
            BinderInfo::Explicit,
            BinderInfo::Implicit,
            BinderInfo::StrictImplicit,
            BinderInfo::InstanceImplicit,
        ] {
            encode_binder_info(&mut enc, value);
        }
        assert_eq!(enc.0, vec![0, 1, 2, 3]);
    }

    #[test]
    fn expression_domain_is_separated_from_environment_domain() {
        let env = sha256_domain(b"ENVIRONMENT\0", |_| {});
        let target = sha256_domain(b"TARGET\0", |_| {});
        assert_ne!(env, target);
    }
}
