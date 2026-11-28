#![forbid(unsafe_code)]

//! Typed null/calibration contracts and SHA-256 content-addressed manifests.
//!
//! This crate distinguishes direct conditional paired equality from
//! reference-calibrated bounded-mean monitoring. It canonicalizes the actual
//! CMAQ-054 expert inventory (Fourier banks, targets, IDs and rational priors)
//! and computes digests from those contents rather than trusting digest-shaped
//! caller strings.
//!
//! Digests provide content identity only. They do not authenticate the source,
//! prove independence, or establish stochastic assumptions.

use std::collections::HashSet;

use cmaq_053_compositional_risk_budget::{
    ComposedRiskPlan, CompositionReceipt, FleetBudgetPlan, FleetScope, Rational,
    RepresentationWeight, RiskBudgetError,
};
use cmaq_054_evidence_adaptive_kernel_family::KernelExpertConfig;
use sha2::{Digest, Sha256};

const FORMAT_VERSION: u32 = 1;
const MAX_EXPERTS: usize = 32;
const MAX_ID_BYTES: usize = 256;
const MAX_THEOREM_ID_BYTES: usize = 128;
const DIGEST_HEX_LEN: usize = 64;
const INVENTORY_DOMAIN: &[u8] = b"LUMINOUS-DYNAMICS/CMAQ-056/PORTFOLIO-INVENTORY/V1\0";
const TARGET_DOMAIN: &[u8] = b"LUMINOUS-DYNAMICS/CMAQ-056/TARGET-ENVELOPE/V1\0";
const REFERENCE_DOMAIN: &[u8] = b"LUMINOUS-DYNAMICS/CMAQ-056/REFERENCE-BUNDLE/V1\0";
const CALIBRATION_DOMAIN: &[u8] = b"LUMINOUS-DYNAMICS/CMAQ-056/CALIBRATION-CONTRACT/V1\0";
const FEATURE_MAP_SEMANTICS: &str = "rff-cos-sin-interleaved-unit-norm-v1";
const TARGET_SEMANTICS: &str = "convex-box-support-eprocess-v1";

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ManifestError {
    EmptyExpertInventory,
    TooManyExperts,
    InvalidIdentifier(&'static str),
    DuplicateExpertId,
    InvalidDigest(&'static str),
    InvalidTheoremVersion,
    InvalidPrior,
    PriorWeightsDoNotSumToOne,
    TargetDimensionMismatch,
    TargetExcludesConditionalNull,
    FeatureDimensionOverflow,
    NonFiniteFeature,
    InvalidProbability,
    RiskBudgetExceeded,
    ManifestModeMismatch,
    ManifestDigestMismatch(&'static str),
    LegacyReceiptRequiresReferenceCalibration,
    LegacyReceiptMismatch(&'static str),
    ArithmeticOverflow,
    RiskBudget(RiskBudgetError),
}

impl From<RiskBudgetError> for ManifestError {
    fn from(value: RiskBudgetError) -> Self {
        Self::RiskBudget(value)
    }
}

impl std::fmt::Display for ManifestError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for ManifestError {}

/// Statistical validity mode for one complete monitor family.
///
/// Direct paired equality uses the conditional null itself; it does not
/// allocate a reference-envelope failure probability. Reference-calibrated
/// mode spends a positive delta for the coverage event from its explicitly
/// versioned calibration theorem.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NullCalibration {
    DirectConditionalPairedEquality {
        theorem_version: String,
    },
    ReferenceCalibratedBoundedMean {
        theorem_version: String,
        delta_reference: Rational,
    },
}

impl NullCalibration {
    pub fn direct_v1() -> Self {
        Self::DirectConditionalPairedEquality {
            theorem_version: "conditional-paired-equality-v1".into(),
        }
    }

    pub fn reference_cmaq052_v1(delta_reference: Rational) -> Self {
        Self::ReferenceCalibratedBoundedMean {
            theorem_version: "cmaq-052-common-conditional-mean-azuma-v1".into(),
            delta_reference,
        }
    }

    pub fn reference_failure_budget(&self) -> Option<Rational> {
        match self {
            Self::DirectConditionalPairedEquality { .. } => None,
            Self::ReferenceCalibratedBoundedMean {
                delta_reference, ..
            } => Some(*delta_reference),
        }
    }

    fn validate(&self) -> Result<(), ManifestError> {
        let version = match self {
            Self::DirectConditionalPairedEquality { theorem_version }
            | Self::ReferenceCalibratedBoundedMean {
                theorem_version, ..
            } => theorem_version,
        };
        if version.trim().is_empty() || version.len() > MAX_THEOREM_ID_BYTES {
            return Err(ManifestError::InvalidTheoremVersion);
        }
        if let Self::ReferenceCalibratedBoundedMean {
            delta_reference, ..
        } = self
        {
            validate_probability(*delta_reference)?;
        }
        Ok(())
    }
}

/// Content digests computed from the canonical manifest encoding.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ManifestDigests {
    /// Full ordered expert inventory, including weights, Fourier maps,
    /// targets, reference IDs and null/calibration mode.
    pub portfolio_inventory_sha256: String,
    /// All expert target boxes, with expert order and IDs.
    pub target_envelope_sha256: String,
    /// All per-expert response/covariate reference identifiers.
    pub reference_bundle_sha256: String,
    /// Exact null/calibration mode and theorem version, including delta.
    pub calibration_contract_sha256: String,
}

/// One canonical, immutable inventory generated from actual CMAQ-054 configs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelFamilyManifest {
    format_version: u32,
    calibration: NullCalibration,
    digests: ManifestDigests,
    canonical_bytes: Vec<u8>,
    target_bytes: Vec<u8>,
    reference_bytes: Vec<u8>,
    representation_weights: Vec<RepresentationWeight>,
    expert_ids: Vec<String>,
    total_feature_count: usize,
}

impl KernelFamilyManifest {
    /// Builds a canonical manifest from the actual feature banks and targets.
    ///
    /// Expert order is significant and preserved. Floating point values are
    /// encoded as IEEE-754 f64 bits in big-endian order, after rejecting
    /// non-finite values and normalizing negative zero to positive zero.
    pub fn build(
        configurations: &[KernelExpertConfig],
        calibration: NullCalibration,
    ) -> Result<Self, ManifestError> {
        calibration.validate()?;
        if configurations.is_empty() {
            return Err(ManifestError::EmptyExpertInventory);
        }
        if configurations.len() > MAX_EXPERTS {
            return Err(ManifestError::TooManyExperts);
        }

        let mut ids = HashSet::with_capacity(configurations.len());
        let mut weights = Vec::with_capacity(configurations.len());
        let mut expert_ids = Vec::with_capacity(configurations.len());
        let mut total_feature_count = 0usize;
        let mut prior_sum = None;

        let mut inventory = Encoder::with_domain(INVENTORY_DOMAIN);
        inventory.u32(FORMAT_VERSION);
        inventory.string(FEATURE_MAP_SEMANTICS)?;
        inventory.string(TARGET_SEMANTICS)?;
        encode_calibration(&mut inventory, &calibration)?;

        let mut target_encoder = Encoder::with_domain(TARGET_DOMAIN);
        target_encoder.u32(FORMAT_VERSION);
        target_encoder.string(TARGET_SEMANTICS)?;
        target_encoder.u64(configurations.len() as u64);

        let mut reference_encoder = Encoder::with_domain(REFERENCE_DOMAIN);
        reference_encoder.u32(FORMAT_VERSION);
        reference_encoder.u64(configurations.len() as u64);

        inventory.u64(configurations.len() as u64);

        for config in configurations {
            validate_identifier(&config.expert_id, "expert_id")?;
            if !ids.insert(config.expert_id.clone()) {
                return Err(ManifestError::DuplicateExpertId);
            }
            validate_digest(config.response_map.map_sha256(), "response_map_sha256")?;
            validate_digest(config.covariate_map.map_sha256(), "covariate_map_sha256")?;
            validate_digest(&config.target_sha256, "target_sha256")?;
            validate_digest(
                &config.response_reference_sha256,
                "response_reference_sha256",
            )?;
            validate_digest(
                &config.covariate_reference_sha256,
                "covariate_reference_sha256",
            )?;
            validate_prior(config.prior_weight)?;

            prior_sum = Some(match prior_sum {
                None => config.prior_weight,
                Some(previous) => previous
                    .checked_add(config.prior_weight)
                    .map_err(|_| ManifestError::PriorWeightsDoNotSumToOne)?,
            });

            let response_dimension = config.response_map.feature_dimension();
            let covariate_dimension = config.covariate_map.feature_dimension();
            let product_dimension = response_dimension
                .checked_mul(covariate_dimension)
                .ok_or(ManifestError::FeatureDimensionOverflow)?;
            if config.target.dimension() != product_dimension {
                return Err(ManifestError::TargetDimensionMismatch);
            }

            // Under direct conditional paired equality, the expected product
            // feature difference is zero. A target excluding zero would
            // reject under that null. Reference-calibrated bounded-mean mode
            // may describe a nonzero target and is governed by CMAQ-052.
            if matches!(
                &calibration,
                NullCalibration::DirectConditionalPairedEquality { .. }
            ) && config
                .target
                .lower()
                .iter()
                .zip(config.target.upper())
                .any(|(lower, upper)| *lower > 0.0 || *upper < 0.0)
            {
                return Err(ManifestError::TargetExcludesConditionalNull);
            }

            total_feature_count = total_feature_count
                .checked_add(product_dimension)
                .ok_or(ManifestError::FeatureDimensionOverflow)?;

            encode_expert(&mut inventory, config)?;
            encode_target(&mut target_encoder, config)?;
            encode_reference_ids(&mut reference_encoder, config)?;

            weights.push(RepresentationWeight {
                member_id: config.expert_id.clone(),
                weight: config.prior_weight,
            });
            expert_ids.push(config.expert_id.clone());
        }

        if prior_sum != Some(Rational::one()) {
            return Err(ManifestError::PriorWeightsDoNotSumToOne);
        }

        let canonical_bytes = inventory.finish();
        let target_bytes = target_encoder.finish();
        let reference_bytes = reference_encoder.finish();
        let calibration_bytes = encode_calibration_only(&calibration)?;

        let digests = ManifestDigests {
            portfolio_inventory_sha256: sha256_hex(&canonical_bytes),
            target_envelope_sha256: sha256_hex(&target_bytes),
            reference_bundle_sha256: sha256_hex(&reference_bytes),
            calibration_contract_sha256: sha256_hex(&calibration_bytes),
        };

        Ok(Self {
            format_version: FORMAT_VERSION,
            calibration,
            digests,
            canonical_bytes,
            target_bytes,
            reference_bytes,
            representation_weights: weights,
            expert_ids,
            total_feature_count,
        })
    }

    pub fn format_version(&self) -> u32 {
        self.format_version
    }

    pub fn calibration(&self) -> &NullCalibration {
        &self.calibration
    }

    pub fn digests(&self) -> &ManifestDigests {
        &self.digests
    }

    pub fn canonical_bytes(&self) -> &[u8] {
        &self.canonical_bytes
    }

    pub fn target_bytes(&self) -> &[u8] {
        &self.target_bytes
    }

    pub fn reference_bytes(&self) -> &[u8] {
        &self.reference_bytes
    }

    pub fn representation_weights(&self) -> &[RepresentationWeight] {
        &self.representation_weights
    }

    pub fn expert_ids(&self) -> &[String] {
        &self.expert_ids
    }

    pub fn total_feature_count(&self) -> usize {
        self.total_feature_count
    }

    pub fn expert_count(&self) -> usize {
        self.expert_ids.len()
    }
}

/// A risk plan that charges only the statistical resources used by its mode.
///
/// Direct conditional paired equality pays alpha_test <= alpha_local and has
/// no reference-failure allocation. Reference-calibrated mode pays
/// alpha_test + delta_reference <= alpha_local.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TypedRiskPlan {
    fleet_plan_sha256: String,
    fleet_inventory_sha256: String,
    global_alpha: Rational,
    scope: FleetScope,
    local_alpha: Rational,
    alpha_test: Rational,
    calibration: NullCalibration,
    manifest_digests: ManifestDigests,
    representation_weights: Vec<RepresentationWeight>,
}

impl TypedRiskPlan {
    pub fn new(
        fleet_plan: &FleetBudgetPlan,
        scope: FleetScope,
        alpha_test: Rational,
        manifest: &KernelFamilyManifest,
    ) -> Result<Self, ManifestError> {
        validate_probability(alpha_test)?;
        let local_alpha = fleet_plan.local_alpha_for(&scope)?;

        match manifest.calibration() {
            NullCalibration::DirectConditionalPairedEquality { .. } => {
                if alpha_test.compare(local_alpha).is_gt() {
                    return Err(ManifestError::RiskBudgetExceeded);
                }
            }
            NullCalibration::ReferenceCalibratedBoundedMean {
                delta_reference, ..
            } => {
                if !Rational::checked_sum_le(alpha_test, *delta_reference, local_alpha)? {
                    return Err(ManifestError::RiskBudgetExceeded);
                }
            }
        }

        Ok(Self {
            fleet_plan_sha256: fleet_plan.fleet_plan_sha256().to_string(),
            fleet_inventory_sha256: fleet_plan.fleet_inventory_sha256().to_string(),
            global_alpha: fleet_plan.global_alpha(),
            scope,
            local_alpha,
            alpha_test,
            calibration: manifest.calibration().clone(),
            manifest_digests: manifest.digests().clone(),
            representation_weights: manifest.representation_weights().to_vec(),
        })
    }

    /// Rechecks every manifest digest and the full calibration mode against
    /// the immutable risk plan before a monitor is constructed.
    pub fn verify_manifest(
        &self,
        manifest: &KernelFamilyManifest,
    ) -> Result<(), ManifestError> {
        if self.calibration != *manifest.calibration() {
            return Err(ManifestError::ManifestModeMismatch);
        }
        if self.manifest_digests != *manifest.digests() {
            return Err(ManifestError::ManifestDigestMismatch("manifest"));
        }
        Ok(())
    }

    /// Rebuilds the canonical manifest from the actual monitor configuration
    /// and verifies that every Fourier bank, target, expert ID, prior and
    /// reference binding is identical to the risk plan.
    pub fn verify_expert_inventory(
        &self,
        configurations: &[KernelExpertConfig],
    ) -> Result<(), ManifestError> {
        let rebuilt = KernelFamilyManifest::build(
            configurations,
            self.calibration.clone(),
        )?;
        self.verify_manifest(&rebuilt)
    }

    /// Verifies a legacy CMAQ-053 receipt. The legacy receipt requires a
    /// positive reference-failure budget, so it can only represent Mode B.
    /// Direct Mode A must use TypedRiskPlan until CMAQ-053 gains an explicit
    /// no-reference calibration variant.
    pub fn verify_legacy_cmaq053_receipt(
        &self,
        receipt: &CompositionReceipt,
    ) -> Result<(), ManifestError> {
        let delta_reference = match &self.calibration {
            NullCalibration::DirectConditionalPairedEquality { .. } => {
                return Err(ManifestError::LegacyReceiptRequiresReferenceCalibration);
            }
            NullCalibration::ReferenceCalibratedBoundedMean {
                delta_reference, ..
            } => *delta_reference,
        };

        if receipt.fleet_plan_sha256 != self.fleet_plan_sha256
            || receipt.fleet_inventory_sha256 != self.fleet_inventory_sha256
            || receipt.global_alpha != self.global_alpha
            || receipt.scope != self.scope
            || receipt.local_alpha != self.local_alpha
            || receipt.alpha_test != self.alpha_test
            || receipt.delta_reference_failure != delta_reference
        {
            return Err(ManifestError::LegacyReceiptMismatch("risk-allocation"));
        }

        if receipt.bindings.representation_inventory_sha256
            != self.manifest_digests.portfolio_inventory_sha256
        {
            return Err(ManifestError::LegacyReceiptMismatch("portfolio-inventory"));
        }
        if receipt.bindings.target_envelope_sha256
            != self.manifest_digests.target_envelope_sha256
        {
            return Err(ManifestError::LegacyReceiptMismatch("target-envelope"));
        }
        if receipt.bindings.reference_manifest_sha256
            != self.manifest_digests.reference_bundle_sha256
        {
            return Err(ManifestError::LegacyReceiptMismatch("reference-bundle"));
        }
        if receipt.bindings.calibration_method_sha256
            != self.manifest_digests.calibration_contract_sha256
        {
            return Err(ManifestError::LegacyReceiptMismatch("calibration-contract"));
        }

        if receipt.representation_weights != self.representation_weights {
            return Err(ManifestError::LegacyReceiptMismatch("representation-weights"));
        }
        Ok(())
    }

    pub fn fleet_plan_sha256(&self) -> &str {
        &self.fleet_plan_sha256
    }

    pub fn fleet_inventory_sha256(&self) -> &str {
        &self.fleet_inventory_sha256
    }

    pub fn global_alpha(&self) -> Rational {
        self.global_alpha
    }

    pub fn scope(&self) -> &FleetScope {
        &self.scope
    }

    pub fn local_alpha(&self) -> Rational {
        self.local_alpha
    }

    pub fn alpha_test(&self) -> Rational {
        self.alpha_test
    }

    pub fn reference_failure_budget(&self) -> Option<Rational> {
        self.calibration.reference_failure_budget()
    }

    pub fn manifest_digests(&self) -> &ManifestDigests {
        &self.manifest_digests
    }
}

fn validate_prior(value: Rational) -> Result<(), ManifestError> {
    if value.numerator() == 0 || value.numerator() > value.denominator() {
        Err(ManifestError::InvalidPrior)
    } else {
        Ok(())
    }
}

fn validate_probability(value: Rational) -> Result<(), ManifestError> {
    if value.numerator() == 0 || value.denominator() == 0 || value.numerator() >= value.denominator() {
        Err(ManifestError::InvalidProbability)
    } else {
        Ok(())
    }
}

fn validate_identifier(value: &str, label: &'static str) -> Result<(), ManifestError> {
    if value.trim().is_empty() || value.len() > MAX_ID_BYTES {
        Err(ManifestError::InvalidIdentifier(label))
    } else {
        Ok(())
    }
}

fn validate_digest(value: &str, label: &'static str) -> Result<(), ManifestError> {
    if value.len() == DIGEST_HEX_LEN
        && value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        Ok(())
    } else {
        Err(ManifestError::InvalidDigest(label))
    }
}

struct Encoder {
    bytes: Vec<u8>,
}

impl Encoder {
    fn with_domain(domain: &[u8]) -> Self {
        Self {
            bytes: domain.to_vec(),
        }
    }

    fn u8(&mut self, value: u8) {
        self.bytes.push(value);
    }

    fn u32(&mut self, value: u32) {
        self.bytes.extend_from_slice(&value.to_be_bytes());
    }

    fn u64(&mut self, value: u64) {
        self.bytes.extend_from_slice(&value.to_be_bytes());
    }

    fn u128(&mut self, value: u128) {
        self.bytes.extend_from_slice(&value.to_be_bytes());
    }

    fn string(&mut self, value: &str) -> Result<(), ManifestError> {
        let length = u64::try_from(value.len()).map_err(|_| ManifestError::ArithmeticOverflow)?;
        self.u64(length);
        self.bytes.extend_from_slice(value.as_bytes());
        Ok(())
    }

    fn rational(&mut self, value: Rational) {
        self.u128(value.numerator());
        self.u128(value.denominator());
    }

    fn f64(&mut self, value: f64) -> Result<(), ManifestError> {
        if !value.is_finite() {
            return Err(ManifestError::NonFiniteFeature);
        }
        let canonical = if value == 0.0 { 0.0 } else { value };
        self.u64(canonical.to_bits());
        Ok(())
    }

    fn finish(self) -> Vec<u8> {
        self.bytes
    }
}

fn encode_calibration(
    encoder: &mut Encoder,
    calibration: &NullCalibration,
) -> Result<(), ManifestError> {
    match calibration {
        NullCalibration::DirectConditionalPairedEquality { theorem_version } => {
            encoder.u8(1);
            encoder.string(theorem_version)?;
        }
        NullCalibration::ReferenceCalibratedBoundedMean {
            theorem_version,
            delta_reference,
        } => {
            encoder.u8(2);
            encoder.string(theorem_version)?;
            encoder.rational(*delta_reference);
        }
    }
    Ok(())
}

fn encode_calibration_only(
    calibration: &NullCalibration,
) -> Result<Vec<u8>, ManifestError> {
    let mut encoder = Encoder::with_domain(CALIBRATION_DOMAIN);
    encoder.u32(FORMAT_VERSION);
    encode_calibration(&mut encoder, calibration)?;
    Ok(encoder.finish())
}

fn encode_expert(
    encoder: &mut Encoder,
    config: &KernelExpertConfig,
) -> Result<(), ManifestError> {
    encoder.string(&config.expert_id)?;
    encoder.rational(config.prior_weight);

    // Response map: declared digest is retained as a provenance label, while
    // the actual dimensions and frequency values are also hashed.
    encoder.u64(config.response_map.input_dimension() as u64);
    encoder.u64(config.response_map.frequency_count() as u64);
    encoder.string(config.response_map.map_sha256())?;
    for frequency in config.response_map.frequencies() {
        encoder.u64(frequency.len() as u64);
        for value in frequency {
            encoder.f64(*value)?;
        }
    }

    // Covariate map.
    encoder.u64(config.covariate_map.input_dimension() as u64);
    encoder.u64(config.covariate_map.frequency_count() as u64);
    encoder.string(config.covariate_map.map_sha256())?;
    for frequency in config.covariate_map.frequencies() {
        encoder.u64(frequency.len() as u64);
        for value in frequency {
            encoder.f64(*value)?;
        }
    }

    encode_target_values(encoder, config)?;
    encoder.string(&config.target_sha256)?;
    encoder.string(&config.response_reference_sha256)?;
    encoder.string(&config.covariate_reference_sha256)?;
    Ok(())
}

fn encode_target(
    encoder: &mut Encoder,
    config: &KernelExpertConfig,
) -> Result<(), ManifestError> {
    encoder.string(&config.expert_id)?;
    encode_target_values(encoder, config)?;
    encoder.string(&config.target_sha256)?;
    Ok(())
}

fn encode_target_values(
    encoder: &mut Encoder,
    config: &KernelExpertConfig,
) -> Result<(), ManifestError> {
    encoder.u64(config.target.dimension() as u64);
    for value in config.target.lower() {
        encoder.f64(*value)?;
    }
    for value in config.target.upper() {
        encoder.f64(*value)?;
    }
    Ok(())
}

fn encode_reference_ids(
    encoder: &mut Encoder,
    config: &KernelExpertConfig,
) -> Result<(), ManifestError> {
    encoder.string(&config.expert_id)?;
    encoder.string(&config.response_reference_sha256)?;
    encoder.string(&config.covariate_reference_sha256)?;
    Ok(())
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(64);
    const HEX: &[u8; 16] = b"0123456789abcdef";
    for byte in digest {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

fn gcd(mut left: u128, mut right: u128) -> u128 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

pub fn multiply_probabilities(
    left: Rational,
    right: Rational,
) -> Result<Rational, ManifestError> {
    validate_probability(left)?;
    validate_prior(right)?;
    let first_cancel = gcd(left.numerator(), right.denominator());
    let second_cancel = gcd(right.numerator(), left.denominator());
    let numerator = (left.numerator() / first_cancel)
        .checked_mul(right.numerator() / second_cancel)
        .ok_or(ManifestError::ArithmeticOverflow)?;
    let denominator = (left.denominator() / second_cancel)
        .checked_mul(right.denominator() / first_cancel)
        .ok_or(ManifestError::ArithmeticOverflow)?;
    Rational::probability(numerator, denominator).map_err(ManifestError::RiskBudget)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cmaq_046_convex_target_approachability::BoxTarget;
    use cmaq_048_fourier_kernel_approachability::FourierFeatureMap;
    use cmaq_053_compositional_risk_budget::{
        FleetBudgetPlan, RiskPlanBindings,
    };

    const SHA_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const SHA_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const SHA_C: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const SHA_D: &str = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
    const SHA_E: &str = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";
    const SHA_F: &str = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";
    fn config(
        id: &str,
        response_frequency: f64,
        weight: Rational,
        response_map_id: &str,
        covariate_map_id: &str,
        target_id: &str,
    ) -> KernelExpertConfig {
        KernelExpertConfig {
            expert_id: id.into(),
            response_map: FourierFeatureMap::new(
                1,
                vec![vec![response_frequency]],
                response_map_id.into(),
            )
            .unwrap(),
            covariate_map: FourierFeatureMap::new(
                1,
                vec![vec![0.5]],
                covariate_map_id.into(),
            )
            .unwrap(),
            target: BoxTarget::new(vec![-0.01; 4], vec![0.01; 4]).unwrap(),
            prior_weight: weight,
            target_sha256: target_id.into(),
            response_reference_sha256: SHA_E.into(),
            covariate_reference_sha256: SHA_F.into(),
        }
    }

    fn configs() -> Vec<KernelExpertConfig> {
        vec![
            config("low", 1.0, Rational::new(1, 2).unwrap(), SHA_A, SHA_B, SHA_C),
            config("high", 0.5, Rational::new(1, 2).unwrap(), SHA_D, SHA_E, SHA_F),
        ]
    }

    fn scope() -> FleetScope {
        FleetScope {
            stream_id: "cmaq-056-test".into(),
            asset_id: "asset-1".into(),
            channel_id: "conditional-pair".into(),
            deployment_id: "deployment-1".into(),
            asset_index: 1,
            channel_index: 1,
            deployment_index: 1,
        }
    }

    fn fleet() -> FleetBudgetPlan {
        FleetBudgetPlan::new(
            Rational::probability(1, 20).unwrap(),
            SHA_A.into(),
            SHA_B.into(),
        )
        .unwrap()
    }

    #[test]
    fn sha256_known_answer_vectors_match_standard() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn manifest_is_deterministic_and_sums_exact_priors() {
        let first = KernelFamilyManifest::build(&configs(), NullCalibration::direct_v1()).unwrap();
        let second = KernelFamilyManifest::build(&configs(), NullCalibration::direct_v1()).unwrap();
        assert_eq!(first.canonical_bytes(), second.canonical_bytes());
        assert_eq!(first.digests(), second.digests());
        assert_eq!(first.expert_count(), 2);
        assert_eq!(first.total_feature_count(), 8);
        assert_eq!(first.representation_weights().len(), 2);
    }

    #[test]
    fn frequency_target_reference_weight_and_order_mutations_change_digest() {
        let base = KernelFamilyManifest::build(&configs(), NullCalibration::direct_v1()).unwrap();

        let mut changed_frequency = configs();
        changed_frequency[0].response_map = FourierFeatureMap::new(
            1,
            vec![vec![1.25]],
            SHA_A.into(),
        )
        .unwrap();
        let changed_frequency =
            KernelFamilyManifest::build(&changed_frequency, NullCalibration::direct_v1()).unwrap();
        assert_ne!(
            base.digests().portfolio_inventory_sha256,
            changed_frequency.digests().portfolio_inventory_sha256
        );

        let mut changed_target = configs();
        changed_target[0].target =
            BoxTarget::new(vec![-0.02; 4], vec![0.02; 4]).unwrap();
        let changed_target =
            KernelFamilyManifest::build(&changed_target, NullCalibration::direct_v1()).unwrap();
        assert_ne!(
            base.digests().target_envelope_sha256,
            changed_target.digests().target_envelope_sha256
        );

        let mut changed_reference = configs();
        changed_reference[0].response_reference_sha256 = SHA_C.into();
        let changed_reference =
            KernelFamilyManifest::build(&changed_reference, NullCalibration::direct_v1()).unwrap();
        assert_ne!(
            base.digests().reference_bundle_sha256,
            changed_reference.digests().reference_bundle_sha256
        );

        let mut changed_weight = configs();
        changed_weight[0].prior_weight = Rational::new(1, 3).unwrap();
        changed_weight[1].prior_weight = Rational::new(2, 3).unwrap();
        let changed_weight =
            KernelFamilyManifest::build(&changed_weight, NullCalibration::direct_v1()).unwrap();
        assert_ne!(
            base.digests().portfolio_inventory_sha256,
            changed_weight.digests().portfolio_inventory_sha256
        );

        let mut changed_order = configs();
        changed_order.swap(0, 1);
        let changed_order =
            KernelFamilyManifest::build(&changed_order, NullCalibration::direct_v1()).unwrap();
        assert_ne!(
            base.digests().portfolio_inventory_sha256,
            changed_order.digests().portfolio_inventory_sha256
        );
    }

    #[test]
    fn expert_identity_and_theorem_version_are_content_bound() {
        let baseline = KernelFamilyManifest::build(
            &configs(),
            NullCalibration::direct_v1(),
        )
        .unwrap();

        let mut changed_id = configs();
        changed_id[0].expert_id = "renamed-expert".into();
        let changed_id_manifest = KernelFamilyManifest::build(
            &changed_id,
            NullCalibration::direct_v1(),
        )
        .unwrap();
        assert_ne!(
            baseline.digests().portfolio_inventory_sha256,
            changed_id_manifest.digests().portfolio_inventory_sha256
        );

        let changed_theorem = KernelFamilyManifest::build(
            &configs(),
            NullCalibration::DirectConditionalPairedEquality {
                theorem_version: "conditional-paired-equality-v2".into(),
            },
        )
        .unwrap();
        assert_ne!(
            baseline.digests().calibration_contract_sha256,
            changed_theorem.digests().calibration_contract_sha256
        );
        assert_ne!(
            baseline.digests().portfolio_inventory_sha256,
            changed_theorem.digests().portfolio_inventory_sha256
        );
    }

    #[test]
    fn negative_zero_is_canonicalized() {
        let positive = KernelFamilyManifest::build(&configs(), NullCalibration::direct_v1()).unwrap();
        let mut negative_configs = configs();
        negative_configs[0].response_map = FourierFeatureMap::new(
            1,
            vec![vec![-0.0]],
            SHA_A.into(),
        )
        .unwrap();
        let mut positive_configs = configs();
        positive_configs[0].response_map = FourierFeatureMap::new(
            1,
            vec![vec![0.0]],
            SHA_A.into(),
        )
        .unwrap();
        let negative = KernelFamilyManifest::build(&negative_configs, NullCalibration::direct_v1()).unwrap();
        let positive_zero = KernelFamilyManifest::build(&positive_configs, NullCalibration::direct_v1()).unwrap();
        assert_eq!(
            negative.digests().portfolio_inventory_sha256,
            positive_zero.digests().portfolio_inventory_sha256
        );
        assert_ne!(
            positive.digests().portfolio_inventory_sha256,
            positive_zero.digests().portfolio_inventory_sha256
        );
    }

    #[test]
    fn zero_excluding_target_is_rejected() {
        let mut invalid = configs();
        invalid[0].target =
            BoxTarget::new(vec![0.1; 4], vec![0.2; 4]).unwrap();
        assert_eq!(
            KernelFamilyManifest::build(&invalid, NullCalibration::direct_v1()).err(),
            Some(ManifestError::TargetExcludesConditionalNull)
        );
    }

    #[test]
    fn direct_mode_spends_no_reference_failure_budget() {
        let manifest =
            KernelFamilyManifest::build(&configs(), NullCalibration::direct_v1()).unwrap();
        let local = fleet().local_alpha_for(&scope()).unwrap();
        let alpha_test = local;
        let plan = TypedRiskPlan::new(&fleet(), scope(), alpha_test, &manifest).unwrap();
        assert_eq!(plan.alpha_test(), local);
        assert_eq!(plan.reference_failure_budget(), None);
        plan.verify_manifest(&manifest).unwrap();
    }

    #[test]
    fn typed_plan_recomputes_manifest_from_actual_expert_inventory() {
        let configs = configs();
        let manifest =
            KernelFamilyManifest::build(&configs, NullCalibration::direct_v1()).unwrap();
        let local = fleet().local_alpha_for(&scope()).unwrap();
        let plan = TypedRiskPlan::new(&fleet(), scope(), local, &manifest).unwrap();
        assert_eq!(plan.verify_expert_inventory(&configs), Ok(()));

        let mut mutated = configs;
        mutated[0].response_map = FourierFeatureMap::new(
            1,
            vec![vec![1.75]],
            SHA_A.into(),
        )
        .unwrap();
        assert_eq!(
            plan.verify_expert_inventory(&mutated),
            Err(ManifestError::ManifestDigestMismatch("manifest"))
        );
    }

    #[test]
    fn reference_mode_reserves_exact_delta_and_rejects_over_budget() {
        let delta = Rational::new(1, 320).unwrap();
        let calibration = NullCalibration::reference_cmaq052_v1(delta);
        let manifest = KernelFamilyManifest::build(&configs(), calibration.clone()).unwrap();
        let local = fleet().local_alpha_for(&scope()).unwrap();
        let alpha_test = local.divided_by_integer(2).unwrap();
        let plan = TypedRiskPlan::new(&fleet(), scope(), alpha_test, &manifest).unwrap();
        assert_eq!(plan.reference_failure_budget(), Some(delta));
        assert_eq!(
            TypedRiskPlan::new(
                &fleet(),
                scope(),
                Rational::new(1, 80).unwrap(),
                &manifest,
            )
            .err(),
            Some(ManifestError::RiskBudgetExceeded)
        );
    }

    #[test]
    fn legacy_cmaq053_receipt_only_verifies_when_every_content_digest_matches() {
        let delta = Rational::new(1, 320).unwrap();
        let calibration = NullCalibration::reference_cmaq052_v1(delta);
        let configs = configs();
        let manifest = KernelFamilyManifest::build(&configs, calibration).unwrap();
        let local = fleet().local_alpha_for(&scope()).unwrap();
        let alpha_test = local.divided_by_integer(2).unwrap();
        let bindings = RiskPlanBindings {
            deployment_plan_sha256: SHA_C.into(),
            reference_manifest_sha256: manifest.digests().reference_bundle_sha256.clone(),
            target_envelope_sha256: manifest.digests().target_envelope_sha256.clone(),
            monitor_build_sha256: SHA_D.into(),
            representation_inventory_sha256: manifest.digests().portfolio_inventory_sha256.clone(),
            calibration_method_sha256: manifest.digests().calibration_contract_sha256.clone(),
        };
        let legacy = ComposedRiskPlan::new(
            &fleet(),
            scope(),
            alpha_test,
            delta,
            manifest.representation_weights().to_vec(),
            bindings,
        )
        .unwrap();
        let typed = TypedRiskPlan::new(&fleet(), scope(), alpha_test, &manifest).unwrap();
        typed.verify_legacy_cmaq053_receipt(legacy.receipt()).unwrap();

        let mut bad_receipt = legacy.receipt().clone();
        bad_receipt.bindings.target_envelope_sha256 = SHA_A.into();
        assert_eq!(
            typed.verify_legacy_cmaq053_receipt(&bad_receipt),
            Err(ManifestError::LegacyReceiptMismatch("target-envelope"))
        );
    }

    #[test]
    fn legacy_receipt_representation_weight_order_must_match() {
        let delta = Rational::new(1, 320).unwrap();
        let manifest = KernelFamilyManifest::build(
            &configs(),
            NullCalibration::reference_cmaq052_v1(delta),
        )
        .unwrap();
        let local = fleet().local_alpha_for(&scope()).unwrap();
        let alpha_test = local.divided_by_integer(2).unwrap();
        let bindings = RiskPlanBindings {
            deployment_plan_sha256: SHA_C.into(),
            reference_manifest_sha256: manifest.digests().reference_bundle_sha256.clone(),
            target_envelope_sha256: manifest.digests().target_envelope_sha256.clone(),
            monitor_build_sha256: SHA_D.into(),
            representation_inventory_sha256: manifest.digests().portfolio_inventory_sha256.clone(),
            calibration_method_sha256: manifest.digests().calibration_contract_sha256.clone(),
        };
        let legacy = ComposedRiskPlan::new(
            &fleet(),
            scope(),
            alpha_test,
            delta,
            manifest.representation_weights().to_vec(),
            bindings,
        )
        .unwrap();
        let typed = TypedRiskPlan::new(&fleet(), scope(), alpha_test, &manifest).unwrap();
        let mut reordered = legacy.receipt().clone();
        reordered.representation_weights.swap(0, 1);
        assert_eq!(
            typed.verify_legacy_cmaq053_receipt(&reordered),
            Err(ManifestError::LegacyReceiptMismatch("representation-weights"))
        );
    }

    #[test]
    fn direct_mode_refuses_legacy_receipt_with_fake_reference_budget() {
        let manifest =
            KernelFamilyManifest::build(&configs(), NullCalibration::direct_v1()).unwrap();
        let local = fleet().local_alpha_for(&scope()).unwrap();
        let typed = TypedRiskPlan::new(&fleet(), scope(), local, &manifest).unwrap();
        let delta = Rational::new(1, 320).unwrap();
        let legacy = ComposedRiskPlan::new(
            &fleet(),
            scope(),
            Rational::new(1, 320).unwrap(),
            delta,
            manifest.representation_weights().to_vec(),
            RiskPlanBindings {
                deployment_plan_sha256: SHA_C.into(),
                reference_manifest_sha256: manifest.digests().reference_bundle_sha256.clone(),
                target_envelope_sha256: manifest.digests().target_envelope_sha256.clone(),
                monitor_build_sha256: SHA_D.into(),
                representation_inventory_sha256: manifest.digests().portfolio_inventory_sha256.clone(),
                calibration_method_sha256: manifest.digests().calibration_contract_sha256.clone(),
            },
        )
        .unwrap();
        assert_eq!(
            typed.verify_legacy_cmaq053_receipt(legacy.receipt()),
            Err(ManifestError::LegacyReceiptRequiresReferenceCalibration)
        );
    }
}
