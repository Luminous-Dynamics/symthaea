#![deny(unsafe_code)]
//! Provider-neutral canonical framing for qualification receipt cores.
//!
//! This crate deliberately starts *after* PASS selection / receipt-candidate eligibility.
//! The resulting core is a portable immutable claim object, not trusted qualification authority.
//! It does not authenticate a provider, verify evidence bytes, witness execution, create a
//! detached attestation, decide current admission, or authorize merge/execution.
//!
//! The portable semantic identity excludes occurrence-specific retry/provider evidence.
//! Those remain mandatory evidence in PassSelection / AttemptHistory / evidence bundles.

use std::{error::Error, fmt, str::FromStr};

pub const RECEIPT_CORE_SCHEMA_V1: &str = "symthaea.qualification-receipt-core.v1";
pub const RECEIPT_CORE_DOMAIN_V1: &[u8] = b"symthaea.qualification-receipt-core.v1";
pub const SELECTION_DISPOSITION_V1: &str = "SelectedRequiredRecipesReportedPassedV1";
pub const CROSS_CUTTING_DISPOSITION_V1: &str = "NoGenericCrossCuttingRulesV1";

pub const NON_CLAIM_TAGS_V1: [&str; 7] = [
    "AttemptHistoryExcludedFromSemanticIdentity",
    "NoCurrentAdmission",
    "NoDetachedAttestation",
    "NoMergeOrExecutionAuthority",
    "NoProviderAuthenticity",
    "NoScientificValidity",
    "NoTrustedPassEstablished",
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReceiptCoreError {
    InvalidSha256Id { field: &'static str },
    InvalidRecipeId,
    InvalidReceiptId,
    EmptyRecipes,
    DuplicateRecipe,
}

impl fmt::Display for ReceiptCoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSha256Id { field } => {
                write!(f, "{field}: expected canonical sha256:<64 lowercase hex> identity")
            }
            Self::InvalidRecipeId => write!(
                f,
                "recipe_id: expected sha256:<64 lowercase hex> or git-blob-sha1:<40 lowercase hex>"
            ),
            Self::InvalidReceiptId => {
                write!(f, "qualification_receipt_id: expected blake3:<64 lowercase hex>")
            }
            Self::EmptyRecipes => write!(f, "required recipe set must be non-empty"),
            Self::DuplicateRecipe => write!(f, "recipe identities must be unique"),
        }
    }
}

impl Error for ReceiptCoreError {}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct RecipeTheoremV1 {
    recipe_id: String,
    attempt_subject_id: String,
}

impl RecipeTheoremV1 {
    pub fn try_new(
        recipe_id: impl Into<String>,
        attempt_subject_id: impl Into<String>,
    ) -> Result<Self, ReceiptCoreError> {
        let recipe_id = recipe_id.into();
        let attempt_subject_id = attempt_subject_id.into();
        if !is_recipe_id(&recipe_id) {
            return Err(ReceiptCoreError::InvalidRecipeId);
        }
        require_sha256_id(&attempt_subject_id, "attempt_subject_id")?;
        Ok(Self {
            recipe_id,
            attempt_subject_id,
        })
    }

    pub fn recipe_id(&self) -> &str {
        &self.recipe_id
    }

    pub fn attempt_subject_id(&self) -> &str {
        &self.attempt_subject_id
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QualificationReceiptCoreV1 {
    qualification_subject_id: String,
    qualification_profile_id: String,
    input_closure_id: String,
    qualification_environment_id: String,
    recipes: Vec<RecipeTheoremV1>,
}

impl QualificationReceiptCoreV1 {
    pub fn try_new(
        qualification_subject_id: impl Into<String>,
        qualification_profile_id: impl Into<String>,
        input_closure_id: impl Into<String>,
        qualification_environment_id: impl Into<String>,
        mut recipes: Vec<RecipeTheoremV1>,
    ) -> Result<Self, ReceiptCoreError> {
        let qualification_subject_id = qualification_subject_id.into();
        let qualification_profile_id = qualification_profile_id.into();
        let input_closure_id = input_closure_id.into();
        let qualification_environment_id = qualification_environment_id.into();

        require_sha256_id(&qualification_subject_id, "qualification_subject_id")?;
        require_sha256_id(&qualification_profile_id, "qualification_profile_id")?;
        require_sha256_id(&input_closure_id, "input_closure_id")?;
        require_sha256_id(
            &qualification_environment_id,
            "qualification_environment_id",
        )?;

        if recipes.is_empty() {
            return Err(ReceiptCoreError::EmptyRecipes);
        }
        recipes.sort();
        if recipes
            .windows(2)
            .any(|pair| pair[0].recipe_id == pair[1].recipe_id)
        {
            return Err(ReceiptCoreError::DuplicateRecipe);
        }

        Ok(Self {
            qualification_subject_id,
            qualification_profile_id,
            input_closure_id,
            qualification_environment_id,
            recipes,
        })
    }

    pub fn qualification_subject_id(&self) -> &str {
        &self.qualification_subject_id
    }

    pub fn qualification_profile_id(&self) -> &str {
        &self.qualification_profile_id
    }

    pub fn input_closure_id(&self) -> &str {
        &self.input_closure_id
    }

    pub fn qualification_environment_id(&self) -> &str {
        &self.qualification_environment_id
    }

    pub fn recipes(&self) -> &[RecipeTheoremV1] {
        &self.recipes
    }

    /// This is deliberately a statement about the selected observations, not trusted PASS.
    pub fn selection_disposition(&self) -> &'static str {
        SELECTION_DISPOSITION_V1
    }

    pub fn cross_cutting_disposition(&self) -> &'static str {
        CROSS_CUTTING_DISPOSITION_V1
    }

    pub fn non_claim_tags(&self) -> &'static [&'static str] {
        &NON_CLAIM_TAGS_V1
    }

    /// Canonical semantic framing.
    ///
    /// Strings are exact UTF-8 bytes prefixed by a u64 little-endian byte length.
    /// Collections are prefixed by a u64 little-endian element count. Recipe entries are sorted
    /// by `(recipe_id, attempt_subject_id)` during construction. V1 has no optional fields.
    pub fn canonical_frame(&self) -> Vec<u8> {
        let mut out = Vec::new();
        frame_bytes(&mut out, RECEIPT_CORE_DOMAIN_V1);
        frame_bytes(&mut out, RECEIPT_CORE_SCHEMA_V1.as_bytes());
        frame_bytes(&mut out, self.qualification_subject_id.as_bytes());
        frame_bytes(&mut out, self.qualification_profile_id.as_bytes());
        frame_bytes(&mut out, self.input_closure_id.as_bytes());
        frame_bytes(&mut out, self.qualification_environment_id.as_bytes());

        push_count(&mut out, self.recipes.len());
        for recipe in &self.recipes {
            frame_bytes(&mut out, recipe.recipe_id.as_bytes());
            frame_bytes(&mut out, recipe.attempt_subject_id.as_bytes());
        }

        frame_bytes(&mut out, SELECTION_DISPOSITION_V1.as_bytes());
        frame_bytes(&mut out, CROSS_CUTTING_DISPOSITION_V1.as_bytes());

        push_count(&mut out, NON_CLAIM_TAGS_V1.len());
        for tag in NON_CLAIM_TAGS_V1 {
            frame_bytes(&mut out, tag.as_bytes());
        }
        out
    }

    pub fn receipt_id(&self) -> QualificationReceiptId {
        QualificationReceiptId(*blake3::hash(&self.canonical_frame()).as_bytes())
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct QualificationReceiptId([u8; 32]);

impl QualificationReceiptId {
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl fmt::Debug for QualificationReceiptId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Display for QualificationReceiptId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "blake3:{}", blake3::Hash::from_bytes(self.0).to_hex())
    }
}

impl FromStr for QualificationReceiptId {
    type Err = ReceiptCoreError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        if !is_prefixed_lower_hex(value, "blake3:", 64) {
            return Err(ReceiptCoreError::InvalidReceiptId);
        }
        let hex = value
            .strip_prefix("blake3:")
            .ok_or(ReceiptCoreError::InvalidReceiptId)?;
        let hash = blake3::Hash::from_hex(hex).map_err(|_| ReceiptCoreError::InvalidReceiptId)?;
        Ok(Self(*hash.as_bytes()))
    }
}

fn push_count(out: &mut Vec<u8>, count: usize) {
    let count = u64::try_from(count).expect("usize fits u64 on supported Rust targets");
    out.extend_from_slice(&count.to_le_bytes());
}

fn frame_bytes(out: &mut Vec<u8>, bytes: &[u8]) {
    push_count(out, bytes.len());
    out.extend_from_slice(bytes);
}

fn require_sha256_id(value: &str, field: &'static str) -> Result<(), ReceiptCoreError> {
    if is_prefixed_lower_hex(value, "sha256:", 64) {
        Ok(())
    } else {
        Err(ReceiptCoreError::InvalidSha256Id { field })
    }
}

fn is_recipe_id(value: &str) -> bool {
    is_prefixed_lower_hex(value, "sha256:", 64)
        || is_prefixed_lower_hex(value, "git-blob-sha1:", 40)
}

fn is_prefixed_lower_hex(value: &str, prefix: &str, digits: usize) -> bool {
    let Some(hex) = value.strip_prefix(prefix) else {
        return false;
    };
    hex.len() == digits
        && hex
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[cfg(test)]
mod tests {
    use super::*;

    const GOLDEN_FRAME_HEX: &str = "260000000000000073796d74686165612e7175616c696669636174696f6e2d726563656970742d636f72652e7631260000000000000073796d74686165612e7175616c696669636174696f6e2d726563656970742d636f72652e763147000000000000007368613235363a3131313131313131313131313131313131313131313131313131313131313131313131313131313131313131313131313131313131313131313131313131313147000000000000007368613235363a3232323232323232323232323232323232323232323232323232323232323232323232323232323232323232323232323232323232323232323232323232323247000000000000007368613235363a3333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333347000000000000007368613235363a34343434343434343434343434343434343434343434343434343434343434343434343434343434343434343434343434343434343434343434343434343434020000000000000036000000000000006769742d626c6f622d736861313a3535353535353535353535353535353535353535353535353535353535353535353535353535353547000000000000007368613235363a3636363636363636363636363636363636363636363636363636363636363636363636363636363636363636363636363636363636363636363636363636363647000000000000007368613235363a3737373737373737373737373737373737373737373737373737373737373737373737373737373737373737373737373737373737373737373737373737373747000000000000007368613235363a38383838383838383838383838383838383838383838383838383838383838383838383838383838383838383838383838383838383838383838383838383838270000000000000053656c65637465645265717569726564526563697065735265706f7274656450617373656456311c000000000000004e6f47656e6572696343726f737343757474696e6752756c6573563107000000000000002a00000000000000417474656d7074486973746f72794578636c7564656446726f6d53656d616e7469634964656e7469747912000000000000004e6f43757272656e7441646d697373696f6e15000000000000004e6f44657461636865644174746573746174696f6e1b000000000000004e6f4d657267654f72457865637574696f6e417574686f7269747916000000000000004e6f50726f766964657241757468656e74696369747914000000000000004e6f536369656e746966696356616c696469747918000000000000004e6f547275737465645061737345737461626c6973686564";
    const GOLDEN_ID: &str =
        "blake3:10b18b9ab46ae0c75b4cfb322ff6dfed48f7b45541cf3de2a4128812e28cf69d";

    fn golden_core(reverse_recipes: bool) -> QualificationReceiptCoreV1 {
        let mut recipes = vec![
            RecipeTheoremV1::try_new(
                format!("git-blob-sha1:{}", "5".repeat(40)),
                format!("sha256:{}", "6".repeat(64)),
            )
            .unwrap(),
            RecipeTheoremV1::try_new(
                format!("sha256:{}", "7".repeat(64)),
                format!("sha256:{}", "8".repeat(64)),
            )
            .unwrap(),
        ];
        if reverse_recipes {
            recipes.reverse();
        }
        QualificationReceiptCoreV1::try_new(
            format!("sha256:{}", "1".repeat(64)),
            format!("sha256:{}", "2".repeat(64)),
            format!("sha256:{}", "3".repeat(64)),
            format!("sha256:{}", "4".repeat(64)),
            recipes,
        )
        .unwrap()
    }

    #[test]
    fn golden_frame_and_id_are_stable() {
        let core = golden_core(false);
        assert_eq!(core.canonical_frame().len(), 1036);
        assert_eq!(hex_lower(&core.canonical_frame()), GOLDEN_FRAME_HEX);
        assert_eq!(core.receipt_id().to_string(), GOLDEN_ID);
        assert_eq!(core.selection_disposition(), SELECTION_DISPOSITION_V1);
        assert!(core.non_claim_tags().contains(&"NoTrustedPassEstablished"));
    }

    #[test]
    fn recipe_input_order_does_not_change_semantic_identity() {
        assert_eq!(
            golden_core(false).receipt_id(),
            golden_core(true).receipt_id()
        );
    }

    #[test]
    fn semantic_drift_changes_receipt_identity() {
        let base = golden_core(false);
        let changed = QualificationReceiptCoreV1::try_new(
            format!("sha256:{}", "9".repeat(64)),
            base.qualification_profile_id().to_owned(),
            base.input_closure_id().to_owned(),
            base.qualification_environment_id().to_owned(),
            base.recipes().to_vec(),
        )
        .unwrap();
        assert_ne!(base.receipt_id(), changed.receipt_id());
    }

    #[test]
    fn duplicate_recipe_identity_fails_closed() {
        let recipe = RecipeTheoremV1::try_new(
            format!("sha256:{}", "7".repeat(64)),
            format!("sha256:{}", "8".repeat(64)),
        )
        .unwrap();
        let err = QualificationReceiptCoreV1::try_new(
            format!("sha256:{}", "1".repeat(64)),
            format!("sha256:{}", "2".repeat(64)),
            format!("sha256:{}", "3".repeat(64)),
            format!("sha256:{}", "4".repeat(64)),
            vec![recipe.clone(), recipe],
        )
        .unwrap_err();
        assert_eq!(err, ReceiptCoreError::DuplicateRecipe);
    }

    #[test]
    fn malformed_identity_fails_closed() {
        let err = RecipeTheoremV1::try_new(
            "sha256:not-a-digest",
            format!("sha256:{}", "8".repeat(64)),
        )
        .unwrap_err();
        assert_eq!(err, ReceiptCoreError::InvalidRecipeId);
    }

    #[test]
    fn receipt_id_parser_is_canonical_lowercase_only() {
        let parsed: QualificationReceiptId = GOLDEN_ID.parse().unwrap();
        assert_eq!(parsed.to_string(), GOLDEN_ID);
        assert!(GOLDEN_ID.to_uppercase().parse::<QualificationReceiptId>().is_err());
        assert!((GOLDEN_ID.to_owned() + " ").parse::<QualificationReceiptId>().is_err());
    }

    fn hex_lower(bytes: &[u8]) -> String {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut out = String::with_capacity(bytes.len() * 2);
        for &byte in bytes {
            out.push(HEX[(byte >> 4) as usize] as char);
            out.push(HEX[(byte & 0x0f) as usize] as char);
        }
        out
    }
}
