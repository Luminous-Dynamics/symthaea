//! Exact finite-support policy clipping and risk diagnostics.
//!
//! Evaluator-side research reference. This is not a full CPC implementation,
//! an action-conditional conformal method, or a production safety gate.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

pub const PPM: u128 = 1_000_000;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolicyError(pub String);
impl std::fmt::Display for PolicyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.write_str(&self.0) }
}
impl std::error::Error for PolicyError {}
fn fail(message: impl Into<String>) -> PolicyError { PolicyError(message.into()) }
fn gcd(mut a: u128, mut b: u128) -> u128 {
    while b != 0 { let r = a % b; a = b; b = r; }
    a.max(1)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Fraction { pub numerator: u128, pub denominator: u128 }
impl Fraction {
    pub fn new(numerator: u128, denominator: u128) -> Result<Self, PolicyError> {
        if denominator == 0 { return Err(fail("fraction denominator cannot be zero")); }
        let g = gcd(numerator, denominator);
        Ok(Self { numerator: numerator / g, denominator: denominator / g })
    }
    pub fn zero() -> Self { Self { numerator: 0, denominator: 1 } }
    pub fn one() -> Self { Self { numerator: 1, denominator: 1 } }
    pub fn add(self, rhs: Self) -> Result<Self, PolicyError> {
        let g = gcd(self.denominator, rhs.denominator);
        let ls = rhs.denominator / g; let rs = self.denominator / g;
        let n = self.numerator.checked_mul(ls)
            .and_then(|x| rhs.numerator.checked_mul(rs).and_then(|y| x.checked_add(y)))
            .ok_or_else(|| fail("fraction addition overflow"))?;
        let d = self.denominator.checked_mul(ls).ok_or_else(|| fail("fraction denominator overflow"))?;
        Self::new(n, d)
    }
    pub fn sub(self, rhs: Self) -> Result<Self, PolicyError> {
        if self.cmp_checked(rhs)? == Ordering::Less { return Err(fail("negative fraction result")); }
        let g = gcd(self.denominator, rhs.denominator);
        let ls = rhs.denominator / g; let rs = self.denominator / g;
        let left = self.numerator.checked_mul(ls).ok_or_else(|| fail("fraction subtraction overflow"))?;
        let right = rhs.numerator.checked_mul(rs).ok_or_else(|| fail("fraction subtraction overflow"))?;
        let d = self.denominator.checked_mul(ls).ok_or_else(|| fail("fraction denominator overflow"))?;
        Self::new(left - right, d)
    }
    pub fn mul(self, rhs: Self) -> Result<Self, PolicyError> {
        let g1 = gcd(self.numerator, rhs.denominator);
        let g2 = gcd(rhs.numerator, self.denominator);
        let n = (self.numerator / g1).checked_mul(rhs.numerator / g2)
            .ok_or_else(|| fail("fraction multiplication overflow"))?;
        let d = (self.denominator / g2).checked_mul(rhs.denominator / g1)
            .ok_or_else(|| fail("fraction denominator overflow"))?;
        Self::new(n, d)
    }
    pub fn div(self, rhs: Self) -> Result<Self, PolicyError> {
        if rhs.numerator == 0 { return Err(fail("division by zero fraction")); }
        self.mul(Self::new(rhs.denominator, rhs.numerator)?)
    }
    pub fn cmp_checked(self, rhs: Self) -> Result<Ordering, PolicyError> {
        let g1 = gcd(self.numerator, rhs.numerator);
        let g2 = gcd(self.denominator, rhs.denominator);
        let left = (self.numerator / g1).checked_mul(rhs.denominator / g2)
            .ok_or_else(|| fail("fraction comparison overflow"))?;
        let right = (rhs.numerator / g1).checked_mul(self.denominator / g2)
            .ok_or_else(|| fail("fraction comparison overflow"))?;
        Ok(left.cmp(&right))
    }
    pub fn min(self, rhs: Self) -> Result<Self, PolicyError> {
        Ok(if self.cmp_checked(rhs)? == Ordering::Greater { rhs } else { self })
    }
    pub fn max(self, rhs: Self) -> Result<Self, PolicyError> {
        Ok(if self.cmp_checked(rhs)? == Ordering::Less { rhs } else { self })
    }
    pub fn to_string_pair(self) -> String { format!("{}/{}", self.numerator, self.denominator) }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Beta { pub numerator: u64, pub denominator: u64 }
impl Beta {
    pub fn validate(self) -> Result<(), PolicyError> {
        if self.numerator == 0 || self.denominator == 0 { return Err(fail("beta must be strictly positive")); }
        Ok(())
    }
    pub fn cmp_checked(self, rhs: Self) -> Result<Ordering, PolicyError> {
        let l = u128::from(self.numerator).checked_mul(u128::from(rhs.denominator)).ok_or_else(|| fail("beta comparison overflow"))?;
        let r = u128::from(rhs.numerator).checked_mul(u128::from(self.denominator)).ok_or_else(|| fail("beta comparison overflow"))?;
        Ok(l.cmp(&r))
    }
    pub fn label(self) -> String { format!("{}/{}", self.numerator, self.denominator) }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DiscretePolicy { pub action_ids: Vec<u32>, pub masses: Vec<u64> }
impl DiscretePolicy {
    pub fn validate(&self) -> Result<(), PolicyError> {
        if self.action_ids.is_empty() || self.action_ids.len() != self.masses.len() {
            return Err(fail("policy action and mass vectors must have same non-zero length"));
        }
        let mut seen = BTreeSet::new();
        if self.action_ids.iter().any(|id| !seen.insert(*id)) { return Err(fail("duplicate policy action ID")); }
        if self.masses.iter().all(|m| *m == 0) { return Err(fail("policy mass total must be positive")); }
        Ok(())
    }
    pub fn total_mass(&self) -> Result<u128, PolicyError> {
        self.validate()?;
        self.masses.iter().try_fold(0u128, |a, x| a.checked_add(u128::from(*x)).ok_or_else(|| fail("policy total mass overflow")))
    }
    pub fn mass_map(&self) -> Result<BTreeMap<u32, u128>, PolicyError> {
        self.validate()?;
        Ok(self.action_ids.iter().copied().zip(self.masses.iter().map(|x| u128::from(*x))).collect())
    }
    pub fn probability(&self, id: u32) -> Result<Fraction, PolicyError> {
        let mass = *self.mass_map()?.get(&id).ok_or_else(|| fail(format!("unknown action ID {id}")))?;
        Fraction::new(mass, self.total_mass()?)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ActionProfile { pub action_id: u32, pub name: String, pub reward_milli: i64, pub loss_ppm: u32 }
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CalibrationSample { pub action_id: u32, pub loss_ppm: u32 }
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ExternalBaselineReference {
    pub id: String,
    pub role: String,
    pub paper_url: String,
    pub code_url: String,
    pub code_commit: String,
    pub code_commit_date: String,
    pub license: String,
    pub status: String,
    pub audit_note: String,
}

impl ExternalBaselineReference {
    fn validate(&self) -> Result<(), PolicyError> {
        if self.id.is_empty() || self.role.is_empty() || self.license.is_empty() || self.audit_note.is_empty() {
            return Err(fail("external baseline reference has empty required text"));
        }
        if !self.paper_url.starts_with("https://") || !self.code_url.starts_with("https://") {
            return Err(fail("external baseline paper/code URLs must use HTTPS"));
        }
        if self.code_commit.len() != 40
            || !self.code_commit.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) {
            return Err(fail("external baseline commit must be 40 lowercase hexadecimal characters"));
        }
        if self.code_commit_date.len() != 10
            || !self.code_commit_date.as_bytes().get(4).is_some_and(|b| *b == b'-')
            || !self.code_commit_date.as_bytes().get(7).is_some_and(|b| *b == b'-') {
            return Err(fail("external baseline commit date must use YYYY-MM-DD"));
        }
        if !matches!(self.status.as_str(),
            "reference_implementation_pinned_reproduction_required"
            | "paper_matched_reproduction_verified"
            | "preliminary_reference_only_no_license") {
            return Err(fail("external baseline reference status is unsupported"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PolicyFixture {
    pub schema: String, pub version: String, pub status: String, pub scenario_id: String,
    pub risk_budget_ppm: u32, pub cvar_alpha_ppm: u32, pub worst_case_loss_ppm: u32,
    pub action_support: Vec<ActionProfile>, pub safe_reference_policy: DiscretePolicy,
    pub optimized_candidate_policy: DiscretePolicy, pub calibration_mixture_policy: DiscretePolicy,
    pub beta_grid: Vec<Beta>, pub calibration_samples: Vec<CalibrationSample>, pub notes: Vec<String>,
    pub external_baseline_references: Vec<ExternalBaselineReference>,
}
impl PolicyFixture {
    pub fn validate(&self) -> Result<(), PolicyError> {
        if self.schema != "symthaea.risk_qualified_policy_benchmark.v0.1" || self.version != "0.1.0" {
            return Err(fail("unsupported policy fixture schema/version"));
        }
        if self.risk_budget_ppm > 1_000_000 || self.cvar_alpha_ppm == 0 || self.cvar_alpha_ppm > 1_000_000 || self.worst_case_loss_ppm > 1_000_000 {
            return Err(fail("risk/alpha/loss values outside [0,1] PPM"));
        }
        let support: BTreeSet<u32> = self.action_support.iter().map(|x| x.action_id).collect();
        if support.is_empty() || support.len() != self.action_support.len() { return Err(fail("action support must be non-empty and unique")); }
        if self.action_support.iter().any(|x| x.name.is_empty() || x.loss_ppm > 1_000_000) { return Err(fail("invalid action profile")); }
        for p in [&self.safe_reference_policy, &self.optimized_candidate_policy, &self.calibration_mixture_policy] {
            p.validate()?;
            if p.action_ids.iter().copied().collect::<BTreeSet<_>>() != support { return Err(fail("policy IDs must exactly match support")); }
        }
        if self.beta_grid.is_empty() { return Err(fail("beta grid cannot be empty")); }
        let mut prior: Option<Beta> = None;
        for beta in &self.beta_grid {
            beta.validate()?;
            if let Some(p) = prior { if beta.cmp_checked(p)? != Ordering::Greater { return Err(fail("beta grid must be strictly increasing")); } }
            prior = Some(*beta);
        }
        for sample in &self.calibration_samples {
            if !support.contains(&sample.action_id) || sample.loss_ppm > 1_000_000 { return Err(fail("invalid calibration sample")); }
            if self.calibration_mixture_policy.probability(sample.action_id)?.numerator == 0 { return Err(fail("sampled action has zero mixture probability")); }
        }
        if self.calibration_samples.is_empty() { return Err(fail("calibration sample must be non-empty")); }
        if self.external_baseline_references.len() < 2 {
            return Err(fail("at least two pinned external baseline references are required"));
        }
        let mut reference_ids = BTreeSet::new();
        for reference in &self.external_baseline_references {
            reference.validate()?;
            if !reference_ids.insert(reference.id.as_str()) {
                return Err(fail("duplicate external baseline reference ID"));
            }
        }
        Ok(())
    }
}

pub const BUILTIN_FIXTURE: &str = include_str!("../fixtures/risk_qualified_policy_v0.1.json");
pub fn load_builtin_fixture() -> Result<PolicyFixture, PolicyError> {
    let f: PolicyFixture = serde_json::from_str(BUILTIN_FIXTURE).map_err(|e| fail(format!("fixture parse failed: {e}")))?;
    f.validate()?; Ok(f)
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ExactPolicy { pub action_ids: Vec<u32>, pub masses: Vec<u128> }
impl ExactPolicy {
    pub fn validate(&self) -> Result<(), PolicyError> {
        if self.action_ids.is_empty() || self.action_ids.len() != self.masses.len() || self.masses.iter().all(|x| *x == 0) {
            return Err(fail("exact policy support/mass invalid"));
        }
        let mut seen=BTreeSet::new();
        if self.action_ids.iter().any(|x| !seen.insert(*x)) { return Err(fail("duplicate exact policy action")); }
        Ok(())
    }
    pub fn total_mass(&self) -> Result<u128, PolicyError> {
        self.validate()?;
        self.masses.iter().try_fold(0u128, |a,x| a.checked_add(*x).ok_or_else(|| fail("exact policy total overflow")))
    }
    pub fn probability(&self, id: u32) -> Result<Fraction, PolicyError> {
        let i=self.action_ids.iter().position(|x|*x==id).ok_or_else(||fail(format!("unknown action {id}")))?;
        Fraction::new(self.masses[i],self.total_mass()?)
    }
}

/// Exact discrete counterpart of pi_beta proportional to min(pi_candidate, beta*pi_safe).
/// Every action mass shares a common denominator; normalization is therefore exact.
pub fn clip_policy(safe:&DiscretePolicy,candidate:&DiscretePolicy,beta:Beta)->Result<ExactPolicy,PolicyError>{
    safe.validate()?; candidate.validate()?; beta.validate()?;
    if safe.action_ids.iter().copied().collect::<BTreeSet<_>>() != candidate.action_ids.iter().copied().collect::<BTreeSet<_>>() {
        return Err(fail("safe and candidate policies must share action support"));
    }
    let sm=safe.mass_map()?; let cm=candidate.mass_map()?;
    let st=safe.total_mass()?; let ct=candidate.total_mass()?;
    let mut masses=Vec::with_capacity(safe.action_ids.len());
    for id in &safe.action_ids {
        let s=*sm.get(id).ok_or_else(||fail("safe mass missing"))?;
        let c=*cm.get(id).ok_or_else(||fail("candidate mass missing"))?;
        let left=c.checked_mul(st).and_then(|x|x.checked_mul(u128::from(beta.denominator))).ok_or_else(||fail("candidate clip overflow"))?;
        let right=u128::from(beta.numerator).checked_mul(s).and_then(|x|x.checked_mul(ct)).ok_or_else(||fail("safe clip overflow"))?;
        masses.push(left.min(right));
    }
    let p=ExactPolicy{action_ids:safe.action_ids.clone(),masses}; p.validate()?; Ok(p)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SignedFraction { pub numerator:i128, pub denominator:u128 }
impl SignedFraction {
    fn new(n:i128,d:u128)->Result<Self,PolicyError>{
        if d==0{return Err(fail("signed fraction denominator zero"));}
        let g=gcd(n.unsigned_abs(),d);
        let gi=i128::try_from(g).map_err(|_|fail("signed fraction gcd overflow"))?;
        Ok(Self{numerator:n/gi,denominator:d/g})
    }
    fn add(self,rhs:Self)->Result<Self,PolicyError>{
        let g=gcd(self.denominator,rhs.denominator);
        let ls=i128::try_from(rhs.denominator/g).map_err(|_|fail("signed scale overflow"))?;
        let rs=i128::try_from(self.denominator/g).map_err(|_|fail("signed scale overflow"))?;
        let n=self.numerator.checked_mul(ls).and_then(|x|rhs.numerator.checked_mul(rs).and_then(|y|x.checked_add(y))).ok_or_else(||fail("signed add overflow"))?;
        let d=self.denominator.checked_mul(rhs.denominator/g).ok_or_else(||fail("signed denominator overflow"))?;
        Self::new(n,d)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActionRiskReceipt { pub action_id:u32, pub probability:Fraction, pub reward_milli:i64, pub loss_ppm:u32 }
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PolicyEvaluation {
    pub expected_loss_ppm:Fraction,
    pub expected_reward_milli:SignedFraction,
    pub cvar_loss_ppm:Fraction,
    pub total_variation_from_safe:Fraction,
    pub actions:Vec<ActionRiskReceipt>,
}
fn total_variation(policy:&ExactPolicy,safe:&DiscretePolicy)->Result<Fraction,PolicyError>{
    let mut sum=Fraction::zero();
    for id in &policy.action_ids{
        let p=policy.probability(*id)?;let q=safe.probability(*id)?;
        let d=if p.cmp_checked(q)?==Ordering::Greater{p.sub(q)?}else{q.sub(p)?};
        sum=sum.add(d)?;
    }
    sum.div(Fraction::new(2,1)?)
}
pub fn evaluate_policy(policy:&ExactPolicy,safe:&DiscretePolicy,profiles:&[ActionProfile],alpha_ppm:u32)->Result<PolicyEvaluation,PolicyError>{
    policy.validate()?;
    if alpha_ppm==0||alpha_ppm>1_000_000{return Err(fail("CVaR alpha out of range"));}
    let map:BTreeMap<u32,&ActionProfile>=profiles.iter().map(|x|(x.action_id,x)).collect();
    if map.len()!=profiles.len()||policy.action_ids.iter().any(|id|!map.contains_key(id)){return Err(fail("action profile/support mismatch"));}
    let mut risk=Fraction::zero();let mut reward=SignedFraction::new(0,1)?;let mut actions=Vec::new();
    for id in &policy.action_ids{
        let p=policy.probability(*id)?;let a=map[id];
        risk=risk.add(p.mul(Fraction::new(u128::from(a.loss_ppm),1)?)?)?;
        let pn=i128::try_from(p.numerator).map_err(|_|fail("probability exceeds i128"))?;
        let product=pn.checked_mul(i128::from(a.reward_milli)).ok_or_else(||fail("reward multiplication overflow"))?;
        reward=reward.add(SignedFraction::new(product,p.denominator)?)?;
        actions.push(ActionRiskReceipt{action_id:*id,probability:p,reward_milli:a.reward_milli,loss_ppm:a.loss_ppm});
    }
    let alpha=Fraction::new(u128::from(alpha_ppm),PPM)?;
    let mut ordered=actions.clone();
    ordered.sort_by(|a,b|b.loss_ppm.cmp(&a.loss_ppm).then(a.action_id.cmp(&b.action_id)));
    let mut remaining=alpha;let mut tail=Fraction::zero();
    for a in &ordered{
        if remaining.numerator==0{break;}
        let take=a.probability.min(remaining)?;
        tail=tail.add(take.mul(Fraction::new(u128::from(a.loss_ppm),1)?)?)?;
        remaining=remaining.sub(take)?;
    }
    if remaining.numerator!=0{return Err(fail("CVaR tail mass not filled"));}
    Ok(PolicyEvaluation{
        expected_loss_ppm:risk, expected_reward_milli:reward,
        cvar_loss_ppm:tail.div(alpha)?,
        total_variation_from_safe:total_variation(policy,safe)?,
        actions
    })
}

#[derive(Debug,Clone,Serialize,Deserialize,PartialEq,Eq)]
pub struct WeightedCalibrationDiagnostic{
    pub weighted_empirical_risk_with_worst_case_test:Fraction,
    pub supremum_test_weight:Fraction,
    pub sum_calibration_weights:Fraction,
    pub calibration_sample_count:usize,
    pub status:String
}
/// CPC-inspired weighted empirical diagnostic for one fixed beta. The calibration
/// sample is assumed drawn from calibration_mixture. The test weight is the exact
/// finite-support supremum of pi_beta/q_mix. The unknown test loss is set to its
/// declared upper bound. This does not qualify data-adaptive selection across betas.
pub fn weighted_risk_diagnostic(
    policy:&ExactPolicy, mixture:&DiscretePolicy, samples:&[CalibrationSample], worst_loss_ppm:u32
)->Result<WeightedCalibrationDiagnostic,PolicyError>{
    if samples.is_empty()||worst_loss_ppm>1_000_000{return Err(fail("invalid weighted diagnostic inputs"));}
    let mt=mixture.total_mass()?;let pt=policy.total_mass()?;
    let mm=mixture.mass_map()?;
    let pm:BTreeMap<u32,u128>=policy.action_ids.iter().copied().zip(policy.masses.iter().copied()).collect();
    let mut supremum=Fraction::zero();
    for id in &policy.action_ids{
        let p=*pm.get(id).ok_or_else(||fail("policy mass missing"))?;
        if p==0{continue;}
        let q=*mm.get(id).ok_or_else(||fail("mixture mass missing"))?;
        if q==0{return Err(fail(format!("mixture has zero support for action {id}")));}
        let w=Fraction::new(p.checked_mul(mt).ok_or_else(||fail("test weight overflow"))?,
            pt.checked_mul(q).ok_or_else(||fail("test weight denominator overflow"))?)?;
        supremum=supremum.max(w)?;
    }
    let mut sum=Fraction::zero();let mut weighted=Fraction::zero();
    for s in samples{
        if s.loss_ppm>1_000_000{return Err(fail("sample loss exceeds [0,1]"));}
        let p=*pm.get(&s.action_id).ok_or_else(||fail("sample action not in candidate support"))?;
        let q=*mm.get(&s.action_id).ok_or_else(||fail("sample action not in mixture support"))?;
        if q==0{return Err(fail("sample has zero mixture probability"));}
        let w=Fraction::new(p.checked_mul(mt).ok_or_else(||fail("calibration weight overflow"))?,
            pt.checked_mul(q).ok_or_else(||fail("calibration weight denominator overflow"))?)?;
        sum=sum.add(w)?;
        weighted=weighted.add(w.mul(Fraction::new(u128::from(s.loss_ppm),PPM)?)?)?;
    }
    let test=Fraction::new(u128::from(worst_loss_ppm),PPM)?;
    let bound=weighted.add(supremum.mul(test)?)?.div(sum.add(supremum)?)?;
    Ok(WeightedCalibrationDiagnostic{
        weighted_empirical_risk_with_worst_case_test:bound,
        supremum_test_weight:supremum,
        sum_calibration_weights:sum,
        calibration_sample_count:samples.len(),
        status:"diagnostic_only_no_formal_beta_selection_or_policy_guarantee".into()
    })
}

#[derive(Debug,Clone,Serialize,Deserialize)]
pub struct BetaEvaluation{
    pub beta:Beta,pub clipped_policy:ExactPolicy,pub evaluator:PolicyEvaluation,
    pub weighted_calibration:WeightedCalibrationDiagnostic,
    pub evaluator_risk_within_budget:bool,pub calibration_diagnostic_within_budget:bool
}
pub fn evaluate_beta_grid(f:&PolicyFixture)->Result<Vec<BetaEvaluation>,PolicyError>{
    f.validate()?;let mut results=Vec::new();
    for beta in &f.beta_grid{
        let policy=clip_policy(&f.safe_reference_policy,&f.optimized_candidate_policy,*beta)?;
        let eval=evaluate_policy(&policy,&f.safe_reference_policy,&f.action_support,f.cvar_alpha_ppm)?;
        let diag=weighted_risk_diagnostic(&policy,&f.calibration_mixture_policy,&f.calibration_samples,f.worst_case_loss_ppm)?;
        let risk_fraction=eval.expected_loss_ppm.div(Fraction::new(PPM,1)?)?;
        let budget=Fraction::new(u128::from(f.risk_budget_ppm),PPM)?;
        let calibration_diagnostic_within_budget =
            diag.weighted_empirical_risk_with_worst_case_test.cmp_checked(budget)? != Ordering::Greater;
        results.push(BetaEvaluation{
            beta:*beta,clipped_policy:policy,evaluator:eval,
            weighted_calibration:diag,
            evaluator_risk_within_budget:risk_fraction.cmp_checked(budget)?!=Ordering::Greater,
            calibration_diagnostic_within_budget
        });
    }
    Ok(results)
}
pub fn sha256_hex(bytes:&[u8])->String{
    let d=Sha256::digest(bytes);let mut s=String::with_capacity(64);
    for b in d{use std::fmt::Write;let _=write!(s,"{b:02x}");}s
}

#[cfg(test)]
mod tests{
    use super::*;
    fn p(ids:Vec<u32>,m:Vec<u64>)->DiscretePolicy{DiscretePolicy{action_ids:ids,masses:m}}
    #[test]fn fractions_reduce_add_and_compare(){
        assert_eq!(Fraction::new(2,4).unwrap(),Fraction::new(1,2).unwrap());
        assert_eq!(Fraction::new(1,3).unwrap().add(Fraction::new(1,6).unwrap()).unwrap(),Fraction::new(1,2).unwrap());
        assert_eq!(Fraction::new(1,2).unwrap().cmp_checked(Fraction::new(2,3).unwrap()).unwrap(),Ordering::Less);
    }
    #[test]fn invalid_fraction_and_beta_fail_closed(){
        assert!(Fraction::new(1,0).is_err());assert!(Fraction::new(1,2).unwrap().sub(Fraction::new(2,3).unwrap()).is_err());
        assert!(Beta{numerator:0,denominator:1}.validate().is_err());
    }
    #[test]fn clipping_uses_exact_common_mass_and_normalizes(){
        let safe=p(vec![0,1,2],vec![8,1,1]);let cand=p(vec![0,1,2],vec![1,2,7]);
        let x=clip_policy(&safe,&cand,Beta{numerator:1,denominator:2}).unwrap();
        assert_eq!(x.masses,vec![20,10,10]);
        assert_eq!(x.probability(0).unwrap(),Fraction::new(1,2).unwrap());
        assert_eq!(x.probability(1).unwrap(),Fraction::new(1,4).unwrap());
        assert_eq!(x.probability(2).unwrap(),Fraction::new(1,4).unwrap());
    }
    #[test]fn large_beta_recovers_candidate_when_ratio_cap_is_inactive(){
        let safe=p(vec![0,1,2],vec![8,1,1]);let cand=p(vec![0,1,2],vec![1,2,7]);
        let x=clip_policy(&safe,&cand,Beta{numerator:100,denominator:1}).unwrap();
        for id in 0..3{assert_eq!(x.probability(id).unwrap(),cand.probability(id).unwrap());}
    }
    #[test]fn candidate_mass_outside_safe_support_is_clipped_to_zero(){
        let safe=p(vec![0,1],vec![10,0]);let cand=p(vec![0,1],vec![1,9]);
        let x=clip_policy(&safe,&cand,Beta{numerator:1,denominator:1}).unwrap();
        assert_eq!(x.probability(1).unwrap(),Fraction::zero());
        assert_eq!(x.probability(0).unwrap(),Fraction::one());
    }
    #[test]fn finite_fixture_exposes_reward_risk_tradeoff_and_nonmonotone_action_loss(){
        let f=load_builtin_fixture().unwrap();let xs=evaluate_beta_grid(&f).unwrap();
        assert_eq!(xs.len(),8);
        assert!(xs[0].evaluator_risk_within_budget);
        assert!(xs[1].evaluator_risk_within_budget);
        assert!(!xs[2].evaluator_risk_within_budget);
        assert_eq!(f.action_support[2].loss_ppm,900000);
        assert_eq!(f.action_support[3].loss_ppm,20000);
        assert!(xs[0].evaluator.expected_reward_milli.numerator>0);
        assert!(xs[0].evaluator.expected_loss_ppm.cmp_checked(Fraction::new(0,1).unwrap()).unwrap()==Ordering::Greater);
        assert_ne!(xs[0].evaluator.expected_loss_ppm,xs[4].evaluator.expected_loss_ppm);
        assert!(xs[6].evaluator.expected_loss_ppm.cmp_checked(xs[5].evaluator.expected_loss_ppm).unwrap()==Ordering::Greater);
        assert!(xs[7].evaluator.expected_loss_ppm.cmp_checked(xs[6].evaluator.expected_loss_ppm).unwrap()==Ordering::Less);
    }
    #[test]fn weighted_diagnostic_uses_exact_finite_support_supremum(){
        let f=load_builtin_fixture().unwrap();
        let pol=clip_policy(&f.safe_reference_policy,&f.optimized_candidate_policy,f.beta_grid[2]).unwrap();
        let d=weighted_risk_diagnostic(&pol,&f.calibration_mixture_policy,&f.calibration_samples,f.worst_case_loss_ppm).unwrap();
        assert!(d.supremum_test_weight.numerator>0);
        assert_eq!(d.status,"diagnostic_only_no_formal_beta_selection_or_policy_guarantee");
    }
    #[test]fn missing_mixture_support_for_candidate_fails_closed(){
        let q=p(vec![0,1],vec![1,0]);let candidate=ExactPolicy{action_ids:vec![0,1],masses:vec![1,9]};
        let sample=vec![CalibrationSample{action_id:0,loss_ppm:0}];
        assert!(weighted_risk_diagnostic(&candidate,&q,&sample,1000000).is_err());
    }
    #[test]fn beta_grid_order_is_preregistered(){
        let mut f=load_builtin_fixture().unwrap();f.beta_grid.swap(0,1);assert!(f.validate().is_err());
    }
    #[test]fn external_baseline_provenance_is_pinned_and_retained(){
        let fixture=load_builtin_fixture().unwrap();
        assert_eq!(fixture.external_baseline_references.len(),2);
        assert!(fixture.external_baseline_references.iter().any(|x|x.id=="cpc_reference_cbo"));
        assert!(fixture.external_baseline_references.iter().any(|x|x.id=="ac_rac_action_conditional"));
    }
    #[test]fn malformed_or_duplicate_external_baseline_reference_fails_closed(){
        let mut fixture=load_builtin_fixture().unwrap();
        fixture.external_baseline_references[0].code_commit="not-a-commit".into();
        assert!(fixture.validate().is_err());
        let mut fixture=load_builtin_fixture().unwrap();
        fixture.external_baseline_references[1].id=fixture.external_baseline_references[0].id.clone();
        assert!(fixture.validate().is_err());
    }
    #[test]fn total_variation_is_exact(){
        let safe=p(vec![0,1],vec![1,1]);let candidate=ExactPolicy{action_ids:vec![0,1],masses:vec![3,1]};
        assert_eq!(total_variation(&candidate,&safe).unwrap(),Fraction::new(1,4).unwrap());
    }
    #[test]fn discrete_cvar_uses_fractional_tail_mass(){
        let pol=ExactPolicy{action_ids:vec![0,1],masses:vec![9,1]};
        let safe=p(vec![0,1],vec![9,1]);
        let profiles=vec![
            ActionProfile{action_id:0,name:"low".into(),reward_milli:0,loss_ppm:0},
            ActionProfile{action_id:1,name:"high".into(),reward_milli:1,loss_ppm:1000000}
        ];
        let e=evaluate_policy(&pol,&safe,&profiles,50000).unwrap();
        assert_eq!(e.cvar_loss_ppm,Fraction::new(1000000,1).unwrap());
    }
}
