//! Minimum context-mixture shift needed to erase a source-cohort improvement.
//!
//! The robustness radius is the smallest total-variation (TV) movement in group prevalence
//! away from an observed source mixture that makes the weighted context effect non-positive,
//! while respecting declared target-mixture prevalence bounds.
//!
//! Two radii are reported separately: a nominal radius using context point estimates and a
//! conservative radius using each context's lower effect bound. Bounds/effects are inputs,
//! not estimated here. This is sensitivity analysis, not a confidence interval, shift detector,
//! guarantee under unobserved contexts, or biological/clinical safety claim.
//!
//! TV is half the L1 distance between source and target weights. A radius of 0.05 means
//! five percentage points of population mass must be reassigned.
//!
//! Compile without Cargo/network:
//! rustc --edition=2024 --test scripts/crispr_shift_robustness_radius.rs -o /tmp/crispr-shift-radius-tests
//! /tmp/crispr-shift-radius-tests

use std::collections::HashSet;
use std::fmt;

const TOLERANCE: f64 = 1e-12;

#[derive(Clone, Debug, PartialEq)]
pub struct ContextShiftEffect {
    pub context_id: String,
    pub nominal_delta: f64,
    pub lower_delta: f64,
    pub upper_delta: f64,
    pub source_weight: f64,
    pub target_weight_lower: f64,
    pub target_weight_upper: f64,
    /// False blocks analysis; it is a gate, not proof of transportability.
    pub context_supported: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RadiusConclusion {
    AlreadyNonPositive,
    ErasableWithinBounds,
    NotErasableWithinBounds,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RadiusWitness {
    pub target_weights: Vec<(String, f64)>,
    pub mixture_delta: f64,
    pub total_variation_distance: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct OneRadius {
    pub source_mixture_delta: f64,
    pub critical_total_variation_distance: Option<f64>,
    pub conclusion: RadiusConclusion,
    pub witness: Option<RadiusWitness>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ShiftRobustnessReport {
    pub nominal: OneRadius,
    pub conservative_lower_bound: OneRadius,
    pub context_count: usize,
    pub source_weight_sum: f64,
    pub target_lower_weight_sum: f64,
    pub target_upper_weight_sum: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ShiftRadiusError {
    NoContexts,
    EmptyContextId,
    DuplicateContextId(String),
    UnsupportedTargetContext(String),
    InvalidDelta { context_id: String },
    InvalidWeight { context_id: String },
    SourceWeightSumNotOne(f64),
    SourceWeightOutsideTargetBounds { context_id: String, source: f64, lower: f64, upper: f64 },
    LowerWeightSumExceedsOne(f64),
    UpperWeightSumBelowOne(f64),
    NumericalInfeasibility(String),
}

impl fmt::Display for ShiftRadiusError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoContexts => write!(f, "at least one context is required"),
            Self::EmptyContextId => write!(f, "context ID must not be blank"),
            Self::DuplicateContextId(id) => write!(f, "duplicate context ID: {id}"),
            Self::UnsupportedTargetContext(id) => write!(f, "target context {id} lacks supported permitted evidence"),
            Self::InvalidDelta { context_id } => write!(f, "context {context_id} has invalid nominal/lower/upper effect values"),
            Self::InvalidWeight { context_id } => write!(f, "context {context_id} has invalid source or target prevalence bounds"),
            Self::SourceWeightSumNotOne(sum) => write!(f, "source weights sum to {sum}, expected one"),
            Self::SourceWeightOutsideTargetBounds { context_id, source, lower, upper } =>
                write!(f, "source weight {source} for {context_id} falls outside target bounds [{lower}, {upper}]"),
            Self::LowerWeightSumExceedsOne(sum) => write!(f, "target lower-bound mass {sum} exceeds one"),
            Self::UpperWeightSumBelowOne(sum) => write!(f, "target upper-bound mass {sum} is below one"),
            Self::NumericalInfeasibility(message) => write!(f, "shift-radius calculation infeasible: {message}"),
        }
    }
}

impl std::error::Error for ShiftRadiusError {}

fn validate(contexts: &[ContextShiftEffect]) -> Result<(f64, f64, f64), ShiftRadiusError> {
    if contexts.is_empty() {
        return Err(ShiftRadiusError::NoContexts);
    }
    let mut ids = HashSet::with_capacity(contexts.len());
    let mut source_sum = 0.0;
    let mut lower_sum = 0.0;
    let mut upper_sum = 0.0;

    for c in contexts {
        if c.context_id.trim().is_empty() {
            return Err(ShiftRadiusError::EmptyContextId);
        }
        if !ids.insert(c.context_id.as_str()) {
            return Err(ShiftRadiusError::DuplicateContextId(c.context_id.clone()));
        }
        if !c.context_supported {
            return Err(ShiftRadiusError::UnsupportedTargetContext(c.context_id.clone()));
        }
        if !c.nominal_delta.is_finite()
            || !c.lower_delta.is_finite()
            || !c.upper_delta.is_finite()
            || c.lower_delta < -1.0
            || c.upper_delta > 1.0
            || c.lower_delta > c.nominal_delta
            || c.nominal_delta > c.upper_delta
        {
            return Err(ShiftRadiusError::InvalidDelta { context_id: c.context_id.clone() });
        }
        let weights_valid = c.source_weight.is_finite()
            && c.target_weight_lower.is_finite()
            && c.target_weight_upper.is_finite()
            && c.source_weight >= 0.0
            && c.source_weight <= 1.0
            && c.target_weight_lower >= 0.0
            && c.target_weight_upper <= 1.0
            && c.target_weight_lower <= c.target_weight_upper;
        if !weights_valid {
            return Err(ShiftRadiusError::InvalidWeight { context_id: c.context_id.clone() });
        }
        if c.source_weight < c.target_weight_lower - TOLERANCE
            || c.source_weight > c.target_weight_upper + TOLERANCE
        {
            return Err(ShiftRadiusError::SourceWeightOutsideTargetBounds {
                context_id: c.context_id.clone(),
                source: c.source_weight,
                lower: c.target_weight_lower,
                upper: c.target_weight_upper,
            });
        }
        source_sum += c.source_weight;
        lower_sum += c.target_weight_lower;
        upper_sum += c.target_weight_upper;
    }

    if (source_sum - 1.0).abs() > TOLERANCE {
        return Err(ShiftRadiusError::SourceWeightSumNotOne(source_sum));
    }
    if lower_sum > 1.0 + TOLERANCE {
        return Err(ShiftRadiusError::LowerWeightSumExceedsOne(lower_sum));
    }
    if upper_sum < 1.0 - TOLERANCE {
        return Err(ShiftRadiusError::UpperWeightSumBelowOne(upper_sum));
    }
    Ok((source_sum, lower_sum, upper_sum))
}

fn weighted_delta(contexts: &[ContextShiftEffect], weights: &[f64], use_lower: bool) -> f64 {
    contexts.iter().zip(weights).map(|(c, w)| {
        let d = if use_lower { c.lower_delta } else { c.nominal_delta };
        d * w
    }).sum()
}

fn one_radius(contexts: &[ContextShiftEffect], use_lower: bool) -> Result<OneRadius, ShiftRadiusError> {
    let mut weights: Vec<f64> = contexts.iter().map(|c| c.source_weight).collect();
    let source_delta = weighted_delta(contexts, &weights, use_lower);
    if source_delta <= TOLERANCE {
        let mut witness_weights: Vec<(String, f64)> = contexts.iter()
            .map(|c| (c.context_id.clone(), c.source_weight)).collect();
        witness_weights.sort_by(|a,b| a.0.cmp(&b.0));
        return Ok(OneRadius {
            source_mixture_delta: source_delta,
            critical_total_variation_distance: Some(0.0),
            conclusion: RadiusConclusion::AlreadyNonPositive,
            witness: Some(RadiusWitness {
                target_weights: witness_weights,
                mixture_delta: source_delta,
                total_variation_distance: 0.0,
            }),
        });
    }

    let mut donor_remaining: Vec<f64> = contexts.iter()
        .map(|c| (c.source_weight - c.target_weight_lower).max(0.0)).collect();
    let mut receiver_remaining: Vec<f64> = contexts.iter()
        .map(|c| (c.target_weight_upper - c.source_weight).max(0.0)).collect();
    let mut donors: Vec<usize> = (0..contexts.len()).filter(|i| donor_remaining[*i] > TOLERANCE).collect();
    let mut receivers: Vec<usize> = (0..contexts.len()).filter(|i| receiver_remaining[*i] > TOLERANCE).collect();

    donors.sort_by(|i,j| {
        let di=if use_lower {contexts[*i].lower_delta}else{contexts[*i].nominal_delta};
        let dj=if use_lower {contexts[*j].lower_delta}else{contexts[*j].nominal_delta};
        dj.total_cmp(&di).then_with(||contexts[*i].context_id.cmp(&contexts[*j].context_id))
    });
    receivers.sort_by(|i,j| {
        let di=if use_lower {contexts[*i].lower_delta}else{contexts[*i].nominal_delta};
        let dj=if use_lower {contexts[*j].lower_delta}else{contexts[*j].nominal_delta};
        di.total_cmp(&dj).then_with(||contexts[*i].context_id.cmp(&contexts[*j].context_id))
    });

    let (mut donor_pos, mut receiver_pos) = (0, 0);
    let mut current_delta = source_delta;
    let mut total_moved = 0.0;

    while current_delta > TOLERANCE {
        while donor_pos < donors.len() && donor_remaining[donors[donor_pos]] <= TOLERANCE { donor_pos += 1; }
        while receiver_pos < receivers.len() && receiver_remaining[receivers[receiver_pos]] <= TOLERANCE { receiver_pos += 1; }
        if donor_pos >= donors.len() || receiver_pos >= receivers.len() { break; }

        let donor=donors[donor_pos];
        let receiver=receivers[receiver_pos];
        let dd=if use_lower {contexts[donor].lower_delta}else{contexts[donor].nominal_delta};
        let rd=if use_lower {contexts[receiver].lower_delta}else{contexts[receiver].nominal_delta};
        if donor == receiver || dd <= rd + TOLERANCE { break; }

        let gain=dd-rd;
        let transfer=donor_remaining[donor].min(receiver_remaining[receiver]).min((current_delta/gain).max(0.0));
        if transfer <= TOLERANCE { break; }
        weights[donor]-=transfer;
        weights[receiver]+=transfer;
        donor_remaining[donor]-=transfer;
        receiver_remaining[receiver]-=transfer;
        current_delta-=transfer*gain;
        total_moved+=transfer;
    }

    if current_delta > TOLERANCE {
        return Ok(OneRadius {
            source_mixture_delta:source_delta,
            critical_total_variation_distance:None,
            conclusion:RadiusConclusion::NotErasableWithinBounds,
            witness:None
        });
    }
    let tv=weights.iter().zip(contexts).map(|(w,c)|(w-c.source_weight).abs()).sum::<f64>()*0.5;
    if (tv-total_moved).abs()>1e-9 {
        return Err(ShiftRadiusError::NumericalInfeasibility(format!("witness TV {tv} differs from transferred mass {total_moved}")));
    }
    let effect=weighted_delta(contexts,&weights,use_lower);
    let mut witness_weights:Vec<(String,f64)>=contexts.iter().zip(&weights).map(|(c,w)|(c.context_id.clone(),*w)).collect();
    witness_weights.sort_by(|a,b|a.0.cmp(&b.0));
    Ok(OneRadius {
        source_mixture_delta:source_delta,
        critical_total_variation_distance:Some(tv),
        conclusion:RadiusConclusion::ErasableWithinBounds,
        witness:Some(RadiusWitness {target_weights:witness_weights,mixture_delta:effect,total_variation_distance:tv})
    })
}

pub fn shift_robustness_radius(contexts:&[ContextShiftEffect])->Result<ShiftRobustnessReport,ShiftRadiusError>{
    let (source_weight_sum,target_lower_weight_sum,target_upper_weight_sum)=validate(contexts)?;
    Ok(ShiftRobustnessReport {
        nominal:one_radius(contexts,false)?,
        conservative_lower_bound:one_radius(contexts,true)?,
        context_count:contexts.len(),
        source_weight_sum,
        target_lower_weight_sum,
        target_upper_weight_sum
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn c(id:&str, nominal:f64, lower:f64, upper:f64, source:f64, lo:f64, hi:f64)->ContextShiftEffect{
        ContextShiftEffect{context_id:id.to_owned(),nominal_delta:nominal,lower_delta:lower,upper_delta:upper,source_weight:source,target_weight_lower:lo,target_weight_upper:hi,context_supported:true}
    }
    fn f12()->Vec<ContextShiftEffect>{
        vec![c("A",0.10,0.08,0.12,0.35,0.2,0.5),c("B",0.045,0.02,0.07,0.45,0.3,0.6),c("C",-0.20,-0.25,-0.15,0.20,0.1,0.4)]
    }

    #[test]
    fn nominal_effect_can_be_erased_by_small_population_shift(){
        let r=shift_robustness_radius(&f12()).unwrap();
        assert!((r.nominal.source_mixture_delta-0.01525).abs()<1e-12);
        assert!((r.nominal.critical_total_variation_distance.unwrap()-0.0508333333333333).abs()<1e-9);
        assert_eq!(r.nominal.conclusion,RadiusConclusion::ErasableWithinBounds);
        let w=r.nominal.witness.unwrap();
        assert!(w.mixture_delta.abs()<1e-9);
        assert!((w.total_variation_distance-0.0508333333333333).abs()<1e-9);
    }
    #[test]
    fn conservative_effect_interval_has_zero_radius_at_source(){
        let r=shift_robustness_radius(&f12()).unwrap();
        assert!((r.conservative_lower_bound.source_mixture_delta+0.013).abs()<1e-12);
        assert_eq!(r.conservative_lower_bound.critical_total_variation_distance,Some(0.0));
        assert_eq!(r.conservative_lower_bound.conclusion,RadiusConclusion::AlreadyNonPositive);
    }
    #[test]
    fn all_positive_context_effects_cannot_be_erased(){
        let xs=vec![c("A",0.1,0.08,0.12,0.4,0.3,0.5),c("B",0.05,0.04,0.06,0.6,0.5,0.7)];
        let r=shift_robustness_radius(&xs).unwrap();
        assert_eq!(r.nominal.conclusion,RadiusConclusion::NotErasableWithinBounds);
        assert_eq!(r.nominal.critical_total_variation_distance,None);
    }
    #[test]
    fn source_already_nonpositive_has_zero_radius(){
        let xs=vec![c("A",-0.1,-0.2,-0.05,0.5,0.2,0.8),c("B",0.05,0.0,0.08,0.5,0.2,0.8)];
        assert_eq!(shift_robustness_radius(&xs).unwrap().nominal.critical_total_variation_distance,Some(0.0));
    }
    #[test]
    fn witness_weights_are_feasible_and_sum_to_one(){
        let r=shift_robustness_radius(&f12()).unwrap();
        let w=r.nominal.witness.unwrap();
        assert!((w.target_weights.iter().map(|x|x.1).sum::<f64>()-1.0).abs()<1e-12);
        for (id,weight) in w.target_weights {
            let ctx=f12().into_iter().find(|c|c.context_id==id).unwrap();
            assert!(weight>=ctx.target_weight_lower-TOLERANCE&&weight<=ctx.target_weight_upper+TOLERANCE);
        }
    }
    #[test]
    fn unsupported_context_fails_closed(){
        let mut xs=f12(); xs[2].context_supported=false;
        assert_eq!(shift_robustness_radius(&xs),Err(ShiftRadiusError::UnsupportedTargetContext("C".to_owned())));
    }
    #[test]
    fn source_mix_must_be_within_target_bounds(){
        let mut xs=f12(); xs[0].target_weight_lower=0.4;
        assert!(matches!(shift_robustness_radius(&xs),Err(ShiftRadiusError::SourceWeightOutsideTargetBounds{..})));
    }
    #[test]
    fn infeasible_target_lower_mass_fails_closed(){
        let mut xs=f12(); xs[0].target_weight_lower=0.6; xs[1].target_weight_lower=0.6;
        assert!(matches!(shift_robustness_radius(&xs),Err(ShiftRadiusError::LowerWeightSumExceedsOne(_))));
    }
    #[test]
    fn invalid_delta_interval_fails_closed(){
        let mut xs=f12(); xs[0].lower_delta=0.11;
        assert!(matches!(shift_robustness_radius(&xs),Err(ShiftRadiusError::InvalidDelta{..})));
    }
    #[test]
    fn nonfinite_effect_fails_closed(){
        let mut xs=f12(); xs[0].nominal_delta=f64::NAN;
        assert!(matches!(shift_robustness_radius(&xs),Err(ShiftRadiusError::InvalidDelta{..})));
    }
    #[test]
    fn duplicate_context_ids_fail_closed(){
        let mut xs=f12(); xs[1].context_id="A".to_owned();
        assert!(matches!(shift_robustness_radius(&xs),Err(ShiftRadiusError::DuplicateContextId(_))));
    }
    #[test]
    fn blank_context_id_fails_closed(){
        let mut xs=f12(); xs[0].context_id=" ".to_owned();
        assert_eq!(shift_robustness_radius(&xs),Err(ShiftRadiusError::EmptyContextId));
    }
    #[test]
    fn empty_contexts_fail_closed(){ assert_eq!(shift_robustness_radius(&[]),Err(ShiftRadiusError::NoContexts)); }
}
