// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root.
//
// Loss-aware interoperability with the public Lanyon specification surface.
//
// The public Lanyon solver repositories expose compact Racket/Lisp
// specifications for coordinates, state, assumptions, parameters, fluxes,
// wave speeds, and diffusive fluxes. This module targets that observable
// subset only; it does not claim compatibility with Lanyon's private compiler.
//
// Symthaea keeps its richer physical ontology in a separate semantic envelope
// so exporting to the compact public surface cannot silently discard meaning.

use serde::{Deserialize, Serialize};
use symthaea_core::hdc::conjecture_engine::{BinOp, Expr, UnaryFn};
use symthaea_types::{PhysicalType, PhysicalTypeError};

const ADAPTER_SCHEMA: &str = "LANYON_PUBLIC_SPEC_ADAPTER.v1";
const ENVELOPE_SCHEMA: &str = "LANYON_SEMANTIC_ENVELOPE.v1";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum LanyonForm {
    Symbol(String),
    Number(f64),
    List(Vec<LanyonForm>),
}

impl LanyonForm {
    pub fn symbol(name: impl Into<String>) -> Result<Self, String> {
        let name = name.into();
        validate_symbol(&name)?;
        Ok(Self::Symbol(name))
    }

    pub fn number(value: f64) -> Result<Self, String> {
        if !value.is_finite() {
            return Err("Lanyon numeric literals must be finite".into());
        }
        Ok(Self::Number(value))
    }

    pub fn list(items: impl IntoIterator<Item = LanyonForm>) -> Self {
        Self::List(items.into_iter().collect())
    }

    fn render(&self) -> String {
        match self {
            Self::Symbol(value) => value.clone(),
            Self::Number(value) => render_number(*value),
            Self::List(values) => format!(
                "({})",
                values.iter().map(Self::render).collect::<Vec<_>>().join(" ")
            ),
        }
    }

    fn validate(&self) -> Result<(), String> {
        match self {
            Self::Symbol(value) => validate_symbol(value),
            Self::Number(value) if value.is_finite() => Ok(()),
            Self::Number(_) => Err("Lanyon numeric literals must be finite".into()),
            Self::List(values) => {
                for value in values {
                    value.validate()?;
                }
                Ok(())
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LanyonSystemSpec {
    pub name: String,
    pub coordinates: Vec<String>,
    pub state: Vec<String>,
    pub state_assumptions: Vec<LanyonForm>,
    pub parameters: Vec<String>,
    pub parameter_assumptions: Vec<LanyonForm>,
    pub fluxes: Vec<Vec<LanyonForm>>,
    pub wavespeeds: Vec<Vec<LanyonForm>>,
    pub diffusive_fluxes: Vec<Vec<LanyonForm>>,
}

impl LanyonSystemSpec {
    pub fn validate(&self) -> Result<(), String> {
        validate_symbol(&self.name)?;
        validate_names("coordinate", &self.coordinates, false)?;
        validate_names("state", &self.state, false)?;
        validate_names("parameter", &self.parameters, true)?;
        reject_duplicates("coordinate", &self.coordinates)?;
        reject_duplicates("state", &self.state)?;
        reject_duplicates("parameter", &self.parameters)?;

        let declared: std::collections::HashSet<&str> = self
            .coordinates
            .iter()
            .chain(self.state.iter())
            .chain(self.parameters.iter())
            .map(String::as_str)
            .collect();

        for form in self.state_assumptions.iter().chain(self.parameter_assumptions.iter()) {
            form.validate()?;
            validate_form_references(form, &declared)?;
        }
        let dimensions = self.coordinates.len();
        validate_matrix("fluxes", &self.fluxes, dimensions, self.state.len(), &declared)?;
        validate_matrix("wavespeeds", &self.wavespeeds, dimensions, self.state.len(), &declared)?;
        validate_matrix(
            "diffusive-fluxes",
            &self.diffusive_fluxes,
            dimensions,
            self.state.len(),
            &declared,
        )?;
        Ok(())
    }

    pub fn render_racket(&self) -> Result<String, String> {
        self.validate()?;
        let mut out = String::from("#lang racket\\n\\n");
        out.push_str(&format!("(define {}\\n  (hash\\n", self.name));
        out.push_str(&format!("   'name {}\\n", racket_string(&self.name)));
        out.push_str(&format!("   'coordinates {}\\n", render_symbols(&self.coordinates)));
        out.push_str(&format!("   'state {}\\n", render_symbols(&self.state)));
        out.push_str(&format!("   'state-assumptions {}\\n", render_forms(&self.state_assumptions)));
        out.push_str(&format!("   'parameters {}\\n", render_symbols(&self.parameters)));
        out.push_str(&format!("   'parameters-assumptions {}\\n", render_forms(&self.parameter_assumptions)));
        out.push_str(&format!("   'fluxes {}\\n", render_matrix(&self.fluxes)));
        out.push_str(&format!("   'wavespeeds {}\\n", render_matrix(&self.wavespeeds)));
        out.push_str(&format!("   'diffusive-fluxes {}\\n", render_matrix(&self.diffusive_fluxes)));
        out.push_str("   ))\\n");
        Ok(out)
    }

    pub fn digest_hex(&self) -> Result<String, String> {
        Ok(blake3::hash(self.render_racket()?.as_bytes()).to_hex().to_string())
    }

    fn contains_name(&self, name: &str) -> bool {
        self.coordinates.iter().chain(self.state.iter()).chain(self.parameters.iter()).any(|entry| entry == name)
    }
}

pub fn expr_to_lanyon_form(expr: &Expr) -> Result<LanyonForm, String> {
    match expr {
        Expr::Var(name) => LanyonForm::symbol(name),
        Expr::Const(value) => LanyonForm::number(*value),
        Expr::BinOp(op, left, right) => {
            match op {
                BinOp::Pow => {
                    let Expr::Const(exponent) = right.as_ref() else {
                        return Err(
                            "variable powers have no stable mapping in the observed public Lanyon subset"
                                .into(),
                        );
                    };
                    if !exponent.is_finite() || (exponent - exponent.round()).abs() >= 1e-9 {
                        return Err(
                            "only finite integer powers have a lossless mapping in the observed public Lanyon subset"
                                .into(),
                        );
                    }
                    let n = exponent.round();
                    if !(-32.0..=32.0).contains(&n) {
                        return Err("integer power outside adapter expansion bound".into());
                    }
                    integer_power_to_form(left, n as i32)
                }
                _ => {
                    let operator = match op {
                        BinOp::Add => "+",
                        BinOp::Sub => "-",
                        BinOp::Mul => "*",
                        BinOp::Div => "/",
                        BinOp::Pow => unreachable!(),
                    };
                    Ok(LanyonForm::list([
                        LanyonForm::symbol(operator)?,
                        expr_to_lanyon_form(left)?,
                        expr_to_lanyon_form(right)?,
                    ]))
                }
            }
        }
        Expr::Func(function, arg) => {
            let name = match function {
                UnaryFn::Sqrt => "sqrt",
                UnaryFn::Log => "log",
                UnaryFn::Exp => "exp",
                UnaryFn::Sin => "sin",
                UnaryFn::Cos => "cos",
                UnaryFn::Abs => "abs",
                UnaryFn::Floor => "floor",
            };
            Ok(LanyonForm::list([
                LanyonForm::symbol(name)?,
                expr_to_lanyon_form(arg)?,
            ]))
        }
        Expr::Sum(_, _) => Err("Expr::Sum has no stable mapping in the observed public Lanyon subset".into()),
    }
}

fn integer_power_to_form(base: &Expr, exponent: i32) -> Result<LanyonForm, String> {
    let base = expr_to_lanyon_form(base)?;
    match exponent {
        0 => LanyonForm::number(1.0),
        1 => Ok(base),
        n if n > 1 => {
            let mut result = base.clone();
            for _ in 1..n {
                result = LanyonForm::list([
                    LanyonForm::symbol("*")?,
                    result,
                    base.clone(),
                ]);
            }
            Ok(result)
        }
        n => {
            let positive = integer_power_to_form_inner(base, -n)?;
            Ok(LanyonForm::list([
                LanyonForm::symbol("/")?,
                LanyonForm::number(1.0)?,
                positive,
            ]))
        }
    }
}

fn integer_power_to_form_inner(base: LanyonForm, exponent: i32) -> Result<LanyonForm, String> {
    if exponent == 0 {
        return LanyonForm::number(1.0);
    }
    let mut result = base.clone();
    for _ in 1..exponent {
        result = LanyonForm::list([
            LanyonForm::symbol("*")?,
            result,
            base.clone(),
        ]);
    }
    Ok(result)
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LanyonPhysicalBinding {
    pub name: String,
    pub physical_type: PhysicalType,
    pub physical_type_digest: String,
}

impl LanyonPhysicalBinding {
    pub fn new(name: impl Into<String>, physical_type: PhysicalType) -> Result<Self, String> {
        let name = name.into();
        validate_symbol(&name)?;
        physical_type.validate().map_err(format_physical_error)?;
        Ok(Self {
            name,
            physical_type_digest: physical_type.digest_hex(),
            physical_type,
        })
    }

    fn validate(&self) -> Result<(), String> {
        validate_symbol(&self.name)?;
        self.physical_type.validate().map_err(format_physical_error)?;
        if self.physical_type_digest != self.physical_type.digest_hex() {
            return Err(format!("physical binding {:?} has a mismatched digest", self.name));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LanyonSemanticEnvelope {
    pub schema_revision: String,
    pub specification_digest: String,
    pub bindings: Vec<LanyonPhysicalBinding>,
}

impl LanyonSemanticEnvelope {
    pub fn new(specification: &LanyonSystemSpec, mut bindings: Vec<LanyonPhysicalBinding>) -> Result<Self, String> {
        specification.validate()?;
        for binding in &bindings {
            binding.validate()?;
            if !specification.contains_name(&binding.name) {
                return Err(format!("physical binding {:?} is not declared by the Lanyon system", binding.name));
            }
        }
        bindings.sort_by(|a, b| a.name.cmp(&b.name));
        let envelope = Self {
            schema_revision: ENVELOPE_SCHEMA.into(),
            specification_digest: specification.digest_hex()?,
            bindings,
        };
        envelope.validate()?;
        Ok(envelope)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema_revision != ENVELOPE_SCHEMA {
            return Err("unsupported Lanyon semantic-envelope schema".into());
        }
        validate_digest(&self.specification_digest)?;
        reject_binding_duplicates(&self.bindings)?;
        for binding in &self.bindings {
            binding.validate()?;
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> Vec<u8> {
        serde_json::to_vec(self).expect("Lanyon semantic envelope serialization is infallible")
    }

    pub fn digest_hex(&self) -> String {
        blake3::hash(&self.canonical_bytes()).to_hex().to_string()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LanyonSpecificationBundle {
    pub schema_revision: String,
    pub specification: LanyonSystemSpec,
    pub semantic_envelope: LanyonSemanticEnvelope,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_candidate_digest: Option<String>,
}

impl LanyonSpecificationBundle {
    pub fn new(specification: LanyonSystemSpec, bindings: Vec<LanyonPhysicalBinding>) -> Result<Self, String> {
        Self::new_with_candidate_digest(specification, bindings, None)
    }

    pub fn new_with_candidate_digest(
        specification: LanyonSystemSpec,
        bindings: Vec<LanyonPhysicalBinding>,
        source_candidate_digest: Option<String>,
    ) -> Result<Self, String> {
        let semantic_envelope = LanyonSemanticEnvelope::new(&specification, bindings)?;
        if let Some(digest) = &source_candidate_digest {
            validate_digest(digest)?;
        }
        let bundle = Self {
            schema_revision: ADAPTER_SCHEMA.into(),
            specification,
            semantic_envelope,
            source_candidate_digest,
        };
        bundle.validate()?;
        Ok(bundle)
    }

    pub fn with_source_candidate_digest(
        specification: LanyonSystemSpec,
        bindings: Vec<LanyonPhysicalBinding>,
        source_candidate_digest: impl Into<String>,
    ) -> Result<Self, String> {
        Self::new_with_candidate_digest(
            specification,
            bindings,
            Some(source_candidate_digest.into()),
        )
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema_revision != ADAPTER_SCHEMA {
            return Err("unsupported Lanyon adapter schema".into());
        }
        self.specification.validate()?;
        self.semantic_envelope.validate()?;
        if let Some(digest) = &self.source_candidate_digest {
            validate_digest(digest)?;
        }
        if self.semantic_envelope.specification_digest != self.specification.digest_hex()? {
            return Err("semantic envelope is bound to a different specification".into());
        }
        Ok(())
    }

    pub fn racket_source(&self) -> Result<String, String> {
        self.validate()?;
        self.specification.render_racket()
    }

    pub fn semantic_envelope_json(&self) -> Result<String, String> {
        self.validate()?;
        serde_json::to_string_pretty(&self.semantic_envelope)
            .map_err(|error| format!("semantic envelope serialization failed: {error}"))
    }

    pub fn digest_hex(&self) -> Result<String, String> {
        self.validate()?;
        let mut bytes = self.racket_source()?.into_bytes();
        bytes.push(b'\\n');
        if let Some(digest) = &self.source_candidate_digest {
            bytes.extend_from_slice(digest.as_bytes());
            bytes.push(b'\\n');
        }
        bytes.extend_from_slice(&self.semantic_envelope.canonical_bytes());
        Ok(blake3::hash(&bytes).to_hex().to_string())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LanyonVerificationStatus {
    NotObserved,
    Passed,
    Failed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LanyonArtifactKind {
    LeanProof,
    CImplementation,
    Other,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LanyonVerificationArtifact {
    pub kind: LanyonArtifactKind,
    pub digest: String,
    pub locator: String,
}

impl LanyonVerificationArtifact {
    pub fn new(
        kind: LanyonArtifactKind,
        digest: impl Into<String>,
        locator: impl Into<String>,
    ) -> Result<Self, String> {
        let artifact = Self {
            kind,
            digest: digest.into(),
            locator: locator.into(),
        };
        artifact.validate()?;
        Ok(artifact)
    }

    fn validate(&self) -> Result<(), String> {
        validate_digest(&self.digest)?;
        if self.locator.trim().is_empty() {
            return Err("verification artifact locator cannot be empty".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LanyonVerificationReceipt {
    pub schema_revision: String,
    pub bundle_digest: String,
    pub status: LanyonVerificationStatus,
    pub verifier: String,
    pub verifier_revision: String,
    pub claims: Vec<String>,
    pub artifacts: Vec<LanyonVerificationArtifact>,
}

impl LanyonVerificationReceipt {
    pub const SCHEMA_REVISION: &str = "LANYON_VERIFICATION_RECEIPT.v1";

    pub fn new(
        bundle: &LanyonSpecificationBundle,
        status: LanyonVerificationStatus,
        verifier: impl Into<String>,
        verifier_revision: impl Into<String>,
        claims: Vec<String>,
        artifacts: Vec<LanyonVerificationArtifact>,
    ) -> Result<Self, String> {
        bundle.validate()?;
        let receipt = Self {
            schema_revision: Self::SCHEMA_REVISION.into(),
            bundle_digest: bundle.digest_hex()?,
            status,
            verifier: verifier.into(),
            verifier_revision: verifier_revision.into(),
            claims,
            artifacts,
        };
        receipt.validate()?;
        Ok(receipt)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema_revision != Self::SCHEMA_REVISION {
            return Err("unsupported Lanyon verification receipt schema".into());
        }
        validate_digest(&self.bundle_digest)?;
        if self.verifier.trim().is_empty() || self.verifier_revision.trim().is_empty() {
            return Err("verification receipt requires verifier identity and revision".into());
        }
        if self.claims.is_empty() || self.claims.iter().any(|claim| claim.trim().is_empty()) {
            return Err("verification receipt requires non-empty claims".into());
        }
        for artifact in &self.artifacts {
            artifact.validate()?;
        }
        if self.status == LanyonVerificationStatus::Passed {
            if !self.artifacts.iter().any(|artifact| artifact.kind == LanyonArtifactKind::LeanProof) {
                return Err("passed verification requires a Lean proof artifact digest".into());
            }
            if !self.artifacts.iter().any(|artifact| artifact.kind == LanyonArtifactKind::CImplementation) {
                return Err("passed verification requires a C implementation artifact digest".into());
            }
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> Vec<u8> {
        serde_json::to_vec(self).expect("Lanyon verification receipt serialization is infallible")
    }

    pub fn digest_hex(&self) -> String {
        blake3::hash(&self.canonical_bytes()).to_hex().to_string()
    }
}
fn validate_names(kind: &str, names: &[String], allow_empty: bool) -> Result<(), String> {
    if names.is_empty() && !allow_empty {
        return Err(format!("Lanyon {kind} list cannot be empty"));
    }
    for name in names {
        validate_identifier(name).map_err(|error| format!("{kind} {error}"))?;
    }
    Ok(())
}

fn validate_form_references(
    form: &LanyonForm,
    declared: &std::collections::HashSet<&str>,
) -> Result<(), String> {
    match form {
        LanyonForm::Symbol(name) if is_allowed_operator(name) => Ok(()),
        LanyonForm::Symbol(name) if declared.contains(name.as_str()) => Ok(()),
        LanyonForm::Symbol(name) => Err(format!(
            "Lanyon expression references undeclared symbol {name:?}"
        )),
        LanyonForm::Number(value) if value.is_finite() => Ok(()),
        LanyonForm::Number(_) => Err("Lanyon expression contains a non-finite number".into()),
        LanyonForm::List(items) => {
            let Some(LanyonForm::Symbol(operator)) = items.first() else {
                return Err("Lanyon expression list must begin with an operator symbol".into());
            };
            if !is_allowed_operator(operator) {
                return Err(format!("unsupported Lanyon expression operator {operator:?}"));
            }
            for item in items.iter().skip(1) {
                validate_form_references(item, declared)?;
            }
            Ok(())
        }
    }
}

fn is_allowed_operator(name: &str) -> bool {
    matches!(
        name,
        "+" | "-" | "*" | "/" | ">" | "<" | ">=" | "<=" | "="
            | "abs" | "max" | "min" | "sqrt" | "log" | "exp" | "sin" | "cos" | "floor"
    )
}

fn validate_identifier(name: &str) -> Result<(), String> {
    if name.is_empty() {
        return Err("identifier cannot be empty".into());
    }
    let mut chars = name.chars();
    let first = chars.next().expect("non-empty");
    if !(first.is_ascii_alphabetic() || first == '_') {
        return Err(format!("identifier {name:?} must start with a letter or underscore"));
    }
    if !chars.all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '?' | '!')) {
        return Err(format!("identifier {name:?} contains unsupported characters"));
    }
    Ok(())
}

fn reject_duplicates(kind: &str, names: &[String]) -> Result<(), String> {
    for (index, name) in names.iter().enumerate() {
        if names[index + 1..].iter().any(|other| other == name) {
            return Err(format!("duplicate {kind} name {name:?}"));
        }
    }
    Ok(())
}

fn reject_binding_duplicates(bindings: &[LanyonPhysicalBinding]) -> Result<(), String> {
    for (index, binding) in bindings.iter().enumerate() {
        if bindings[index + 1..].iter().any(|other| other.name == binding.name) {
            return Err(format!("duplicate physical binding {:?}", binding.name));
        }
    }
    Ok(())
}

fn validate_matrix(
    label: &str,
    matrix: &[Vec<LanyonForm>],
    dimensions: usize,
    state_len: usize,
    declared: &std::collections::HashSet<&str>,
) -> Result<(), String> {
    if matrix.len() != dimensions {
        return Err(format!("{label} must contain one row per coordinate direction"));
    }
    for row in matrix {
        if row.len() != state_len {
            return Err(format!("{label} rows must contain one entry per state field"));
        }
        for form in row {
            form.validate()?;
            validate_form_references(form, declared)?;
        }
    }
    Ok(())
}

fn render_symbols(names: &[String]) -> String {
    if names.is_empty() {
        return "(list)".into();
    }
    let body = names.iter().map(|name| format!("`{name}")).collect::<Vec<_>>().join(" ");
    format!("(list {body})")
}

fn render_forms(forms: &[LanyonForm]) -> String {
    if forms.is_empty() { return "(list)".into(); }
    format!("(list {})", forms.iter().map(|form| format!("`{}", form.render())).collect::<Vec<_>>().join(" "))
}

fn render_matrix(matrix: &[Vec<LanyonForm>]) -> String {
    let rows = matrix.iter().map(|row| {
        let body = row.iter().map(|form| format!("`{}", form.render())).collect::<Vec<_>>().join("\\n                  ");
        format!("(list {body})")
    }).collect::<Vec<_>>().join("\\n            ");
    format!("(list\\n            {rows}\\n           )")
}

fn validate_symbol(name: &str) -> Result<(), String> {
    if name.trim().is_empty() { return Err("Lanyon symbol cannot be empty".into()); }
    if name.chars().any(|c| c.is_whitespace() || matches!(c, '(' | ')' | '"' | ';' | ',' | '[' | ']') || c as u32 == 96) {
        return Err(format!("unsafe Lanyon symbol {name:?}"));
    }
    Ok(())
}

fn validate_digest(value: &str) -> Result<(), String> {
    if value.len() != 64 || !value.as_bytes().iter().all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase()) {
        return Err("digest must be canonical lowercase 64-hex".into());
    }
    Ok(())
}

fn render_number(value: f64) -> String {
    if value == 0.0 { "0.0".into() }
    else if value.fract() == 0.0 && value.abs() < 1e12 { format!("{value:.1}") }
    else { format!("{value:.17e}") }
}

fn racket_string(value: &str) -> String {
    let escaped = value.replace('\\\\', "\\\\\\\\").replace('"', "\\\"");
    format!("\"{escaped}\"")
}

fn format_physical_error(error: PhysicalTypeError) -> String {
    format!("invalid physical type: {}", error.reason)
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_types::{PhysicalDimension, QuantityKind};

    fn fixture() -> LanyonSystemSpec {
        let symbol = |name: &str| LanyonForm::symbol(name).unwrap();
        let zero = LanyonForm::number(0.0).unwrap();
        let mul = |a: LanyonForm, b: LanyonForm| LanyonForm::list([symbol("*"), a, b]);
        LanyonSystemSpec {
            name: "maxwell-1d".into(),
            coordinates: vec!["x".into()],
            state: vec!["Ex".into(), "Ey".into(), "Ez".into()],
            state_assumptions: vec![],
            parameters: vec!["c".into()],
            parameter_assumptions: vec![LanyonForm::list([symbol(">"), symbol("c"), zero.clone()])],
            fluxes: vec![vec![zero.clone(), mul(symbol("c"), symbol("Ey")), mul(symbol("c"), symbol("Ez"))]],
            wavespeeds: vec![vec![symbol("c"), symbol("c"), zero.clone()]],
            diffusive_fluxes: vec![vec![zero.clone(), zero.clone(), zero]],
        }
    }

    #[test]
    fn renders_public_lanyon_shape() {
        let source = fixture().render_racket().unwrap();
        let bt = char::from(96);
        assert!(source.contains("#lang racket"));
        assert!(source.contains(&format!("'coordinates (list {bt}x)")));
        assert!(source.contains(&format!("'state (list {bt}Ex {bt}Ey {bt}Ez)")));
        assert!(source.contains("'fluxes"));
    }

    #[test]
    fn parameter_free_systems_match_public_lanyon_shape() {
        let mut spec = fixture();
        spec.parameters.clear();
        spec.parameter_assumptions.clear();
        assert!(spec.validate().is_ok());
        assert!(spec.render_racket().unwrap().contains("'parameters (list)"));
    }

    #[test]
    fn undeclared_expression_symbol_is_rejected() {
        let mut spec = fixture();
        spec.fluxes[0][0] = LanyonForm::symbol("ghost").unwrap();
        assert!(spec.validate().is_err());
    }

    #[test]
    fn parameter_free_shape_renders_exact_empty_list() {
        let mut spec = fixture();
        spec.parameters.clear();
        spec.parameter_assumptions.clear();
        let source = spec.render_racket().unwrap();
        assert!(source.contains("'parameters (list)"));
    }

    #[test]
    fn expression_export_preserves_structure() {
        let expr = Expr::BinOp(
            BinOp::Mul,
            Box::new(Expr::Const(0.5)),
            Box::new(Expr::BinOp(
                BinOp::Pow,
                Box::new(Expr::Var("v".into())),
                Box::new(Expr::Const(2.0)),
            )),
        );
        assert_eq!(expr_to_lanyon_form(&expr).unwrap().render(), "(* 0.5 (^ v 2.0))");
    }

    #[test]
    fn integer_power_exports_using_observed_core_operators() {
        let expr = Expr::BinOp(
            BinOp::Pow,
            Box::new(Expr::Var("x".into())),
            Box::new(Expr::Const(2.0)),
        );
        assert_eq!(expr_to_lanyon_form(&expr).unwrap().render(), "(* x x)");
    }

    #[test]
    fn fractional_power_fails_closed_in_exporter() {
        let expr = Expr::BinOp(
            BinOp::Pow,
            Box::new(Expr::Var("x".into())),
            Box::new(Expr::Const(0.5)),
        );
        assert!(expr_to_lanyon_form(&expr).is_err());
    }

    #[test]
    fn sum_rejects_unsupported_ast() {
        let expr = Expr::Sum(Box::new(Expr::Var("f".into())), "k".into());
        assert!(expr_to_lanyon_form(&expr).is_err());
    }

    #[test]
    fn source_candidate_digest_is_optional_but_exact_when_present() {
        let bundle = LanyonSpecificationBundle::with_source_candidate_digest(
            fixture(),
            vec![],
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        ).unwrap();
        assert_eq!(
            bundle.source_candidate_digest.as_deref(),
            Some("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"),
        );
        assert!(LanyonSpecificationBundle::with_source_candidate_digest(
            fixture(),
            vec![],
            "not-a-digest",
        ).is_err());
    }

    #[test]
    fn candidate_lineage_changes_bundle_digest() {
        let base = LanyonSpecificationBundle::new(fixture(), vec![]).unwrap();
        let linked = LanyonSpecificationBundle::with_source_candidate_digest(
            fixture(), vec![],
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        ).unwrap();
        assert_ne!(base.digest_hex().unwrap(), linked.digest_hex().unwrap());
    }

    #[test]
    fn verification_receipt_binds_exact_bundle_digest() {
        let bundle = LanyonSpecificationBundle::new(fixture(), vec![LanyonPhysicalBinding::new(
            "c",
            PhysicalType::with_kind(QuantityKind::Velocity, PhysicalDimension::VELOCITY),
        ).unwrap()]).unwrap();

        let receipt = LanyonVerificationReceipt::new(
            &bundle,
            LanyonVerificationStatus::Passed,
            "lanyon",
            "public-verifier-v1",
            vec!["formal verification observed for this exact bundle".into()],
            vec![
                LanyonVerificationArtifact::new(
                    LanyonArtifactKind::LeanProof,
                    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                    "artifact://proof",
                ).unwrap(),
                LanyonVerificationArtifact::new(
                    LanyonArtifactKind::CImplementation,
                    "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                    "artifact://solver",
                ).unwrap(),
            ],
        ).unwrap();

        assert_eq!(receipt.bundle_digest, bundle.digest_hex().unwrap());
        assert!(receipt.validate().is_ok());
    }

    #[test]
    fn passed_verification_requires_both_public_artifact_classes() {
        let bundle = LanyonSpecificationBundle::new(fixture(), Vec::new()).unwrap();
        let result = LanyonVerificationReceipt::new(
            &bundle,
            LanyonVerificationStatus::Passed,
            "lanyon",
            "public-verifier-v1",
            vec!["formal verification observed".into()],
            vec![LanyonVerificationArtifact::new(
                LanyonArtifactKind::LeanProof,
                "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "artifact://proof",
            ).unwrap()],
        );
        assert!(result.is_err());
    }

    #[test]
    fn semantic_binding_is_digest_bound_and_attached() {
        let spec = fixture();
        let binding = LanyonPhysicalBinding::new(
            "c",
            PhysicalType::with_kind(QuantityKind::Velocity, PhysicalDimension::VELOCITY),
        ).unwrap();
        let envelope = LanyonSemanticEnvelope::new(&spec, vec![binding]).unwrap();
        assert!(envelope.validate().is_ok());

        let detached = LanyonPhysicalBinding::new(
            "not_declared",
            PhysicalType::with_kind(QuantityKind::Velocity, PhysicalDimension::VELOCITY),
        ).unwrap();
        assert!(LanyonSemanticEnvelope::new(&spec, vec![detached]).is_err());
    }

    #[test]
    fn bundle_digest_changes_with_semantic_type() {
        let spec = fixture();
        let first = LanyonSpecificationBundle::new(
            spec.clone(),
            vec![LanyonPhysicalBinding::new(
                "c",
                PhysicalType::with_kind(QuantityKind::Velocity, PhysicalDimension::VELOCITY),
            ).unwrap()],
        ).unwrap();
        let second = LanyonSpecificationBundle::new(
            spec,
            vec![LanyonPhysicalBinding::new(
                "c",
                PhysicalType::with_kind(QuantityKind::Velocity, PhysicalDimension::VELOCITY)
                    .with_semantic_id(symthaea_types::SemanticIdentifier::new("example", "different").unwrap()),
            ).unwrap()],
        ).unwrap();
        assert_ne!(first.digest_hex().unwrap(), second.digest_hex().unwrap());
    }
}
