//! Controlled detector-concealment evaluation under post-training optimization.
//!
//! Tests whether detector TPR declines after an identified optimization stage
//! while an exposure-associated advantage remains on fresh, paired tasks.
//! Pre-concealment frozen thresholds and stage-specific recalibration are kept
//! separate. The module consumes evaluator-produced receipts and scores; it
//! does not train models or authenticate model identities.

use crate::causal_analysis::{MeanIntervalV1, mean_interval};
use crate::hash_domain;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const DETECTOR_CONCEALMENT_STUDY_SCHEMA: &str = "symthaea.detector_concealment_study.v1";
pub const DETECTOR_CONCEALMENT_REPORT_SCHEMA: &str = "symthaea.detector_concealment_report.v1";
pub const MAX_MODEL_PAIRS: usize = 1_000;
pub const MAX_DETECTORS: usize = 64;
pub const MAX_SCORE_ROWS: usize = 10_000_000;

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum ConcealmentStageKindV1 { Grpo, Ppo, Dapo, SftWithCot, Other }

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum ModelArmV1 { ExposureTreatment, MatchedPlacebo }

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum EvaluationCohortV1 { TreatmentAssignedSourceFamily, HeldoutCleanControl }

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum ThresholdPolicyV1 { FrozenPreConcealment, StageSpecificRecalibration }

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DetectorMethodV1 {
    pub detector_id: String,
    pub detector_version: String,
    pub detector_config_sha256: String,
    pub scoring_access_regime: String,
    /// True when larger raw scores indicate more contamination evidence.
    pub higher_is_more_suspicious: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ConcealmentModelPairV1 {
    pub replicate_id: String,
    pub stage_kind: ConcealmentStageKindV1,
    pub base_checkpoint_sha256: String,
    pub treatment_model_id: String,
    pub placebo_model_id: String,
    pub exposure_assignment_sha256: String,
    pub placebo_assignment_sha256: String,
    pub pre_treatment_checkpoint_sha256: String,
    pub pre_placebo_checkpoint_sha256: String,
    pub post_treatment_checkpoint_sha256: String,
    pub post_placebo_checkpoint_sha256: String,
    pub exposure_training_receipt_sha256: String,
    pub placebo_training_receipt_sha256: String,
    pub concealment_training_receipt_sha256: String,
    pub concealment_config_sha256: String,
    pub checkpoint_freeze_receipt_sha256: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationScoreV1 {
    pub replicate_id: String,
    pub model_arm: ModelArmV1,
    pub model_id: String,
    pub detector_id: String,
    pub post_concealment: bool,
    pub item_id: String,
    /// Every calibration record is from a separately adjudicated clean cohort.
    pub score: f64,
    pub checkpoint_sha256: String,
    pub manifest_sha256: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DetectorTestScoreV1 {
    pub replicate_id: String,
    pub model_arm: ModelArmV1,
    pub model_id: String,
    pub detector_id: String,
    pub post_concealment: bool,
    pub item_id: String,
    pub cohort: EvaluationCohortV1,
    pub score: f64,
    pub checkpoint_sha256: String,
    pub manifest_sha256: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FreshBehaviorScoreV1 {
    pub replicate_id: String,
    pub model_arm: ModelArmV1,
    pub model_id: String,
    pub post_concealment: bool,
    pub item_id: String,
    pub task_family: String,
    /// Verifier-backed performance in permille.
    pub score_permille: u16,
    pub checkpoint_sha256: String,
    pub manifest_sha256: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DetectorConcealmentStudyV1 {
    pub schema: String,
    pub study_id: String,
    pub benchmark_id: String,
    pub benchmark_version: String,
    pub base_benchmark_manifest_sha256: String,
    pub calibration_manifest_sha256: String,
    pub detector_test_manifest_sha256: String,
    pub fresh_behavior_manifest_sha256: String,
    pub target_fpr_permille: Vec<u16>,
    pub minimum_replicates: u16,
    pub minimum_calibration_negatives: u32,
    pub minimum_test_negatives: u32,
    pub minimum_behavior_advantage_permille: u16,
    pub detector_methods: Vec<DetectorMethodV1>,
    pub model_pairs: Vec<ConcealmentModelPairV1>,
    pub calibration_scores: Vec<CalibrationScoreV1>,
    pub detector_test_scores: Vec<DetectorTestScoreV1>,
    pub fresh_behavior_scores: Vec<FreshBehaviorScoreV1>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ConfusionMetricsV1 {
    pub positive_count: u64,
    pub true_positive_count: u64,
    pub negative_count: u64,
    pub false_positive_count: u64,
    pub tpr_permille: Option<f64>,
    pub fpr_permille: Option<f64>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DetectorPairResultV1 {
    pub replicate_id: String,
    pub detector_id: String,
    pub stage_kind: ConcealmentStageKindV1,
    pub threshold_policy: ThresholdPolicyV1,
    pub target_fpr_permille: u16,
    pub pre_treatment_threshold: f64,
    pub post_treatment_threshold: f64,
    pub pre_placebo_threshold: f64,
    pub post_placebo_threshold: f64,
    pub pre_calibration_count_per_arm: u64,
    pub post_calibration_count_per_arm: u64,
    pub pre_treatment_members: ConfusionMetricsV1,
    pub post_treatment_members: ConfusionMetricsV1,
    pub pre_treatment_controls: ConfusionMetricsV1,
    pub post_treatment_controls: ConfusionMetricsV1,
    pub pre_placebo_pseudo_members: ConfusionMetricsV1,
    pub post_placebo_pseudo_members: ConfusionMetricsV1,
    pub pre_placebo_controls: ConfusionMetricsV1,
    pub post_placebo_controls: ConfusionMetricsV1,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DetectorConcealmentAggregateV1 {
    pub detector_id: String,
    pub stage_kind: ConcealmentStageKindV1,
    pub threshold_policy: ThresholdPolicyV1,
    pub target_fpr_permille: u16,
    pub independent_model_pair_count: u64,
    pub pre_treatment_member_tpr: Option<MeanIntervalV1>,
    pub post_treatment_member_tpr: Option<MeanIntervalV1>,
    pub pre_to_post_tpr_change: Option<MeanIntervalV1>,
    pub pre_treatment_clean_fpr: Option<MeanIntervalV1>,
    pub post_treatment_clean_fpr: Option<MeanIntervalV1>,
    pub pre_placebo_pseudo_member_fpr: Option<MeanIntervalV1>,
    pub post_placebo_pseudo_member_fpr: Option<MeanIntervalV1>,
    pub pre_placebo_control_fpr: Option<MeanIntervalV1>,
    pub post_placebo_control_fpr: Option<MeanIntervalV1>,
    pub concealment_signal: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BehavioralRetentionV1 {
    pub stage_kind: ConcealmentStageKindV1,
    pub post_concealment: bool,
    pub independent_model_pair_count: u64,
    pub treatment_minus_placebo_advantage_permille: MeanIntervalV1,
    pub task_family_count: u64,
    pub minimum_fresh_items_per_pair: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ConcealmentClaimV1 {
    pub detector_id: String,
    pub stage_kind: ConcealmentStageKindV1,
    pub threshold_policy: ThresholdPolicyV1,
    pub target_fpr_permille: u16,
    pub mean_tpr_change_permille: f64,
    pub tpr_change_upper_95_permille: f64,
    pub retained_behavioral_advantage_permille: f64,
    pub behavioral_advantage_lower_95_permille: f64,
    pub concealment_event_supported: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DetectorConcealmentReportV1 {
    pub schema: String,
    pub study_id: String,
    pub benchmark_id: String,
    pub benchmark_version: String,
    pub study_sha256: String,
    pub independent_model_pair_count: u64,
    pub detector_count: u64,
    pub stage_count: u64,
    pub target_fpr_permille: Vec<u16>,
    pub pair_results: Vec<DetectorPairResultV1>,
    pub detector_aggregates: Vec<DetectorConcealmentAggregateV1>,
    pub behavioral_retention: Vec<BehavioralRetentionV1>,
    pub supported_concealment_claims: Vec<ConcealmentClaimV1>,
    pub independently_authenticated: bool,
    pub confirmatory_eligible: bool,
    pub inference_status: String,
    pub issues: Vec<String>,
    pub claim_boundary: String,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct ScoreKey {
    replicate_id: String,
    model_arm: ModelArmV1,
    model_id: String,
    detector_id: String,
    post: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct BehaviorKey {
    replicate_id: String,
    model_arm: ModelArmV1,
    model_id: String,
    post: bool,
}

fn valid_id(value: &str, max: usize) -> bool {
    !value.trim().is_empty() && value.len() <= max
        && value.bytes().all(|b| b.is_ascii_alphanumeric() || b"._:-".contains(&b))
}

fn valid_digest(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn digest(value: &str, label: &str) -> Result<(), String> {
    if valid_digest(value) { Ok(()) } else { Err(format!("{label} must be lowercase SHA-256")) }
}

fn model_id(pair: &ConcealmentModelPairV1, arm: ModelArmV1) -> &str {
    match arm { ModelArmV1::ExposureTreatment => &pair.treatment_model_id, ModelArmV1::MatchedPlacebo => &pair.placebo_model_id }
}

fn checkpoint(pair: &ConcealmentModelPairV1, arm: ModelArmV1, post: bool) -> &str {
    match (arm,post) {
        (ModelArmV1::ExposureTreatment,false)=>&pair.pre_treatment_checkpoint_sha256,
        (ModelArmV1::ExposureTreatment,true)=>&pair.post_treatment_checkpoint_sha256,
        (ModelArmV1::MatchedPlacebo,false)=>&pair.pre_placebo_checkpoint_sha256,
        (ModelArmV1::MatchedPlacebo,true)=>&pair.post_placebo_checkpoint_sha256,
    }
}

/// Returns a raw-score threshold after orienting scores so larger means more suspicious.
pub fn calibrated_threshold(scores: &[f64], method: &DetectorMethodV1, target_fpr_permille: u16) -> Result<f64,String> {
    if scores.is_empty() || target_fpr_permille==0 || target_fpr_permille>1000 || scores.iter().any(|x|!x.is_finite()) {
        return Err("calibration scores/target FPR are invalid".to_owned());
    }
    let mut oriented:Vec<f64>=scores.iter().map(|x|if method.higher_is_more_suspicious{*x}else{-*x}).collect();
    oriented.sort_by(f64::total_cmp);
    let allowed=(target_fpr_permille as usize*oriented.len())/1000;
    let threshold=oriented[oriented.len()-allowed-1];
    Ok(if method.higher_is_more_suspicious{threshold}else{-threshold})
}

fn classify(score:f64,threshold:f64,method:&DetectorMethodV1)->bool {
    if method.higher_is_more_suspicious {score>threshold} else {score<threshold}
}

fn confusion(rows:&[&DetectorTestScoreV1],method:&DetectorMethodV1,threshold:f64,positive:bool)->ConfusionMetricsV1{
    let predicted=rows.iter().filter(|row|classify(row.score,threshold,method)).count() as u64;
    let n=rows.len() as u64;
    ConfusionMetricsV1{
        positive_count:if positive{n}else{0},
        true_positive_count:if positive{predicted}else{0},
        negative_count:if positive{0}else{n},
        false_positive_count:if positive{0}else{predicted},
        tpr_permille:if positive&&n>0{Some(predicted as f64*1000.0/n as f64)}else{None},
        fpr_permille:if !positive&&n>0{Some(predicted as f64*1000.0/n as f64)}else{None},
    }
}

fn pair_behavior_advantage(
    treatment:&[&FreshBehaviorScoreV1],
    placebo:&[&FreshBehaviorScoreV1],
)->Result<(f64,BTreeSet<String>,u64),String>{
    let mut t=BTreeMap::<String,(u16,String)>::new();
    let mut p=BTreeMap::<String,(u16,String)>::new();
    for row in treatment {
        if t.insert(row.item_id.clone(),(row.score_permille,row.task_family.clone())).is_some(){return Err("duplicate fresh treatment item".to_owned());}
    }
    for row in placebo {
        if p.insert(row.item_id.clone(),(row.score_permille,row.task_family.clone())).is_some(){return Err("duplicate fresh placebo item".to_owned());}
    }
    if t.is_empty()||t.keys().collect::<BTreeSet<_>>()!=p.keys().collect::<BTreeSet<_>>(){return Err("fresh treatment/placebo item sets must match and be nonempty".to_owned());}
    let mut diff=Vec::new();let mut families=BTreeSet::new();
    for (id,(score,family)) in &t {
        let (control,control_family)=p.get(id).ok_or_else(||"fresh placebo item missing".to_owned())?;
        if family!=control_family{return Err("fresh task-family labels differ across paired models".to_owned());}
        families.insert(family.clone());
        diff.push(*score as f64-*control as f64);
    }
    Ok((diff.iter().sum::<f64>()/diff.len() as f64,families,diff.len() as u64))
}

fn validate_pair(pair:&ConcealmentModelPairV1)->Result<(),String>{
    if !valid_id(&pair.replicate_id,128)||!valid_id(&pair.treatment_model_id,128)||!valid_id(&pair.placebo_model_id,128)
        ||pair.treatment_model_id==pair.placebo_model_id{return Err("model-pair IDs are invalid".to_owned());}
    for (v,l) in [
        (&pair.base_checkpoint_sha256,"base checkpoint"),
        (&pair.exposure_assignment_sha256,"exposure assignment"),
        (&pair.placebo_assignment_sha256,"placebo assignment"),
        (&pair.pre_treatment_checkpoint_sha256,"pre treatment checkpoint"),
        (&pair.pre_placebo_checkpoint_sha256,"pre placebo checkpoint"),
        (&pair.post_treatment_checkpoint_sha256,"post treatment checkpoint"),
        (&pair.post_placebo_checkpoint_sha256,"post placebo checkpoint"),
        (&pair.exposure_training_receipt_sha256,"exposure receipt"),
        (&pair.placebo_training_receipt_sha256,"placebo receipt"),
        (&pair.concealment_training_receipt_sha256,"concealment receipt"),
        (&pair.concealment_config_sha256,"concealment config"),
        (&pair.checkpoint_freeze_receipt_sha256,"checkpoint freeze receipt"),
    ]{digest(v,l)?;}
    Ok(())
}

pub fn analyze_detector_concealment(study:&DetectorConcealmentStudyV1)->Result<DetectorConcealmentReportV1,String>{
    if study.schema!=DETECTOR_CONCEALMENT_STUDY_SCHEMA||!valid_id(&study.study_id,128)||!valid_id(&study.benchmark_id,128)
        ||!valid_id(&study.benchmark_version,64)||study.minimum_replicates<2||study.minimum_calibration_negatives==0
        ||study.minimum_test_negatives==0{return Err("study schema/identity/minimum sample counts are invalid".to_owned());}
    for (v,l) in [
        (&study.base_benchmark_manifest_sha256,"base manifest"),
        (&study.calibration_manifest_sha256,"calibration manifest"),
        (&study.detector_test_manifest_sha256,"test manifest"),
        (&study.fresh_behavior_manifest_sha256,"behavior manifest"),
    ]{digest(v,l)?;}
    let targets:BTreeSet<u16>=study.target_fpr_permille.iter().copied().collect();
    if targets.is_empty()||targets.len()!=study.target_fpr_permille.len()||targets.iter().any(|x|*x==0||*x>1000){
        return Err("target FPR values must be unique and in 1..=1000 permille".to_owned());
    }
    if study.model_pairs.len()<study.minimum_replicates as usize||study.model_pairs.len()>MAX_MODEL_PAIRS
        ||study.detector_methods.is_empty()||study.detector_methods.len()>MAX_DETECTORS
        ||study.calibration_scores.is_empty()||study.calibration_scores.len()>MAX_SCORE_ROWS
        ||study.detector_test_scores.is_empty()||study.detector_test_scores.len()>MAX_SCORE_ROWS
        ||study.fresh_behavior_scores.is_empty()||study.fresh_behavior_scores.len()>MAX_SCORE_ROWS{
        return Err("study collections are empty, underpowered or exceed limits".to_owned());
    }
    let mut methods=BTreeMap::<String,&DetectorMethodV1>::new();
    for method in &study.detector_methods{
        if !valid_id(&method.detector_id,128)||!valid_id(&method.detector_version,64)||!valid_id(&method.scoring_access_regime,64)
            ||methods.insert(method.detector_id.clone(),method).is_some(){return Err("detector identity must be unique and valid".to_owned());}
        digest(&method.detector_config_sha256,"detector config")?;
    }
    let mut pairs=BTreeMap::<String,&ConcealmentModelPairV1>::new();
    let mut all_model_ids=BTreeSet::new();let mut stages=BTreeSet::new();
    for pair in &study.model_pairs{
        validate_pair(pair)?;
        if pairs.insert(pair.replicate_id.clone(),pair).is_some()||!all_model_ids.insert(pair.treatment_model_id.clone())
            ||!all_model_ids.insert(pair.placebo_model_id.clone()){return Err("model pair/replicate IDs must be globally unique".to_owned());}
        stages.insert(pair.stage_kind);
    }

    let mut calibration=BTreeMap::<ScoreKey,Vec<f64>>::new();
    let mut calibration_ids=BTreeMap::<(String,String),BTreeSet<String>>::new();
    for row in &study.calibration_scores{
        let pair=pairs.get(&row.replicate_id).ok_or_else(||"unknown calibration replicate".to_owned())?;
        if !methods.contains_key(&row.detector_id)||row.model_id!=model_id(pair,row.model_arm)
            ||row.checkpoint_sha256!=checkpoint(pair,row.model_arm,row.post_concealment)||!row.score.is_finite()
            ||!valid_id(&row.item_id,128)||row.manifest_sha256!=study.calibration_manifest_sha256{
            return Err("calibration identity/checkpoint/score/manifest mismatch".to_owned());}
        calibration.entry(ScoreKey{replicate_id:row.replicate_id.clone(),model_arm:row.model_arm,model_id:row.model_id.clone(),detector_id:row.detector_id.clone(),post:row.post_concealment})
            .or_default().push(row.score);
        calibration_ids.entry((row.replicate_id.clone(),row.detector_id.clone())).or_default().insert(row.item_id.clone());
    }
    let mut thresholds=BTreeMap::<(String,ModelArmV1,String,String,bool,u16),(f64,u64)>::new();
    for (key,scores) in &calibration{
        if scores.len()<study.minimum_calibration_negatives as usize{return Err("clean calibration cohort below configured minimum".to_owned());}
        let method=methods.get(&key.detector_id).unwrap();
        for target in &study.target_fpr_permille{
            thresholds.insert((key.replicate_id.clone(),key.model_arm,key.model_id.clone(),key.detector_id.clone(),key.post,*target),
                (calibrated_threshold(scores,method,*target)?,scores.len() as u64));
        }
    }

    let mut tests=BTreeMap::<ScoreKey,BTreeMap<EvaluationCohortV1,Vec<&DetectorTestScoreV1>>>::new();
    let mut test_item_sets=BTreeMap::<(String,String,bool,ModelArmV1),BTreeMap<EvaluationCohortV1,BTreeSet<String>>>::new();
    for row in &study.detector_test_scores{
        let pair=pairs.get(&row.replicate_id).ok_or_else(||"unknown detector test replicate".to_owned())?;
        if !methods.contains_key(&row.detector_id)||row.model_id!=model_id(pair,row.model_arm)
            ||row.checkpoint_sha256!=checkpoint(pair,row.model_arm,row.post_concealment)||!row.score.is_finite()
            ||!valid_id(&row.item_id,128)||row.manifest_sha256!=study.detector_test_manifest_sha256{
            return Err("detector test identity/checkpoint/score/manifest mismatch".to_owned());}
        if calibration_ids.get(&(row.replicate_id.clone(),row.detector_id.clone())).is_some_and(|ids|ids.contains(&row.item_id)){
            return Err("calibration and test item IDs overlap".to_owned());}
        tests.entry(ScoreKey{replicate_id:row.replicate_id.clone(),model_arm:row.model_arm,model_id:row.model_id.clone(),detector_id:row.detector_id.clone(),post:row.post_concealment})
            .or_default().entry(row.cohort).or_default().push(row);
        test_item_sets.entry((row.replicate_id.clone(),row.detector_id.clone(),row.post_concealment,row.model_arm))
            .or_default().entry(row.cohort).or_default().insert(row.item_id.clone());
    }
    for pair in &study.model_pairs{for detector in methods.keys(){
        let reference=test_item_sets.get(&(pair.replicate_id.clone(),detector.clone(),false,ModelArmV1::ExposureTreatment)).ok_or_else(||"missing pre treatment test items".to_owned())?;
        for post in [false,true]{for arm in [ModelArmV1::ExposureTreatment,ModelArmV1::MatchedPlacebo]{
            let current=test_item_sets.get(&(pair.replicate_id.clone(),detector.clone(),post,arm)).ok_or_else(||"missing test cohort matrix cell".to_owned())?;
            if current!=reference{return Err("test item/cohort manifests differ across pre/post or treatment/placebo arms".to_owned());}
            if current.get(&EvaluationCohortV1::TreatmentAssignedSourceFamily).map_or(0,BTreeSet::len)<study.minimum_test_negatives as usize
                ||current.get(&EvaluationCohortV1::HeldoutCleanControl).map_or(0,BTreeSet::len)<study.minimum_test_negatives as usize{
                return Err("test positive or clean-control cohort below configured minimum".to_owned());}
        }}
    }}

    let mut behavior=BTreeMap::<BehaviorKey,Vec<&FreshBehaviorScoreV1>>::new();
    let mut behavior_ids=BTreeMap::<String,BTreeSet<String>>::new();
    for row in &study.fresh_behavior_scores{
        let pair=pairs.get(&row.replicate_id).ok_or_else(||"unknown fresh behavior replicate".to_owned())?;
        if row.model_id!=model_id(pair,row.model_arm)||row.checkpoint_sha256!=checkpoint(pair,row.model_arm,row.post_concealment)
            ||!valid_id(&row.item_id,128)||!valid_id(&row.task_family,64)||row.score_permille>1000
            ||row.manifest_sha256!=study.fresh_behavior_manifest_sha256{return Err("fresh behavior identity/checkpoint/score/manifest mismatch".to_owned());}
        behavior.entry(BehaviorKey{replicate_id:row.replicate_id.clone(),model_arm:row.model_arm,model_id:row.model_id.clone(),post:row.post_concealment})
            .or_default().push(row);
        behavior_ids.entry(row.replicate_id.clone()).or_default().insert(row.item_id.clone());
    }
    for pair in &study.model_pairs{
        for post in [false,true]{
            let treatment=behavior.get(&BehaviorKey{replicate_id:pair.replicate_id.clone(),model_arm:ModelArmV1::ExposureTreatment,model_id:pair.treatment_model_id.clone(),post}).ok_or_else(||"fresh treatment behavior missing".to_owned())?;
            let placebo=behavior.get(&BehaviorKey{replicate_id:pair.replicate_id.clone(),model_arm:ModelArmV1::MatchedPlacebo,model_id:pair.placebo_model_id.clone(),post}).ok_or_else(||"fresh placebo behavior missing".to_owned())?;
            let _=pair_behavior_advantage(treatment,placebo)?;
            let tset:BTreeSet<String>=treatment.iter().map(|r|r.item_id.clone()).collect();
            let pset:BTreeSet<String>=placebo.iter().map(|r|r.item_id.clone()).collect();
            if tset!=pset{return Err("fresh behavior item IDs differ across model arms".to_owned());}
            let testset=test_item_sets.iter().filter(|((rep,_,_,_),_)|rep==&pair.replicate_id)
                .flat_map(|(_,cohorts)|cohorts.values().flatten().cloned()).collect::<BTreeSet<_>>();
            if tset.iter().any(|id|testset.contains(id)||calibration_ids.iter().any(|((rep,_),ids)|rep==&pair.replicate_id&&ids.contains(id))){
                return Err("fresh behavior items overlap detector test/calibration".to_owned());}
        }
    }

    let mut pair_results=Vec::new();
    let mut grouped=BTreeMap::<(String,ConcealmentStageKindV1,ThresholdPolicyV1,u16),Vec<[f64;8]>>::new();
    for pair in &study.model_pairs{for method in &study.detector_methods{for target in &study.target_fpr_permille{for policy in [ThresholdPolicyV1::FrozenPreConcealment,ThresholdPolicyV1::StageSpecificRecalibration]{
        let threshold_for=|arm:ModelArmV1,post:bool|->Result<(f64,u64),String>{
            let model=model_id(pair,arm);
            let calibration_post=post&&policy==ThresholdPolicyV1::StageSpecificRecalibration;
            thresholds.get(&(pair.replicate_id.clone(),arm,model.to_owned(),method.detector_id.clone(),calibration_post,*target))
                .copied().ok_or_else(||"missing calibrated threshold".to_owned())
        };
        let (tpre,ntpre)=threshold_for(ModelArmV1::ExposureTreatment,false)?;
        let (tpost,ntpost)=threshold_for(ModelArmV1::ExposureTreatment,true)?;
        let (ppre,nppre)=threshold_for(ModelArmV1::MatchedPlacebo,false)?;
        let (ppost,nppost)=threshold_for(ModelArmV1::MatchedPlacebo,true)?;
        let rows=|arm:ModelArmV1,post:bool,cohort:EvaluationCohortV1|->Result<Vec<&DetectorTestScoreV1>,String>{
            tests.get(&ScoreKey{replicate_id:pair.replicate_id.clone(),model_arm:arm,model_id:model_id(pair,arm).to_owned(),detector_id:method.detector_id.clone(),post})
                .and_then(|map|map.get(&cohort)).cloned().ok_or_else(||"missing detector test cohort".to_owned())
        };
        let tm_pre=confusion(&rows(ModelArmV1::ExposureTreatment,false,EvaluationCohortV1::TreatmentAssignedSourceFamily)?,method,tpre,true);
        let tm_post=confusion(&rows(ModelArmV1::ExposureTreatment,true,EvaluationCohortV1::TreatmentAssignedSourceFamily)?,method,tpost,true);
        let tc_pre=confusion(&rows(ModelArmV1::ExposureTreatment,false,EvaluationCohortV1::HeldoutCleanControl)?,method,tpre,false);
        let tc_post=confusion(&rows(ModelArmV1::ExposureTreatment,true,EvaluationCohortV1::HeldoutCleanControl)?,method,tpost,false);
        let pm_pre=confusion(&rows(ModelArmV1::MatchedPlacebo,false,EvaluationCohortV1::TreatmentAssignedSourceFamily)?,method,ppre,false);
        let pm_post=confusion(&rows(ModelArmV1::MatchedPlacebo,true,EvaluationCohortV1::TreatmentAssignedSourceFamily)?,method,ppost,false);
        let pc_pre=confusion(&rows(ModelArmV1::MatchedPlacebo,false,EvaluationCohortV1::HeldoutCleanControl)?,method,ppre,false);
        let pc_post=confusion(&rows(ModelArmV1::MatchedPlacebo,true,EvaluationCohortV1::HeldoutCleanControl)?,method,ppost,false);
        let tpre_v=tm_pre.tpr_permille.ok_or_else(||"treatment pre TPR non-estimable".to_owned())?;
        let tpost_v=tm_post.tpr_permille.ok_or_else(||"treatment post TPR non-estimable".to_owned())?;
        let metrics=[tpre_v,tpost_v,tc_pre.fpr_permille.unwrap(),tc_post.fpr_permille.unwrap(),pm_pre.fpr_permille.unwrap(),pm_post.fpr_permille.unwrap(),pc_pre.fpr_permille.unwrap(),pc_post.fpr_permille.unwrap()];
        grouped.entry((method.detector_id.clone(),pair.stage_kind,policy,*target)).or_default().push(metrics);
        pair_results.push(DetectorPairResultV1{
            replicate_id:pair.replicate_id.clone(),detector_id:method.detector_id.clone(),stage_kind:pair.stage_kind,threshold_policy:policy,target_fpr_permille:*target,
            pre_treatment_threshold:tpre,post_treatment_threshold:tpost,pre_placebo_threshold:ppre,post_placebo_threshold:ppost,
            pre_calibration_count_per_arm:ntpre+nppre,post_calibration_count_per_arm:ntpost+nppost,
            pre_treatment_members:tm_pre,post_treatment_members:tm_post,pre_treatment_controls:tc_pre,post_treatment_controls:tc_post,
            pre_placebo_pseudo_members:pm_pre,post_placebo_pseudo_members:pm_post,pre_placebo_controls:pc_pre,post_placebo_controls:pc_post,
        });
    }}}}

    let mut behavior_effects=BTreeMap::<(ConcealmentStageKindV1,bool),Vec<f64>>::new();
    let mut family_sets=BTreeMap::<(ConcealmentStageKindV1,bool),BTreeSet<String>>::new();
    let mut min_fresh_items=BTreeMap::<(ConcealmentStageKindV1,bool),u64>::new();
    for pair in &study.model_pairs{for post in [false,true]{
        let t=behavior.get(&BehaviorKey{replicate_id:pair.replicate_id.clone(),model_arm:ModelArmV1::ExposureTreatment,model_id:pair.treatment_model_id.clone(),post}).unwrap();
        let p=behavior.get(&BehaviorKey{replicate_id:pair.replicate_id.clone(),model_arm:ModelArmV1::MatchedPlacebo,model_id:pair.placebo_model_id.clone(),post}).unwrap();
        let (effect,families,n)=pair_behavior_advantage(t,p)?;
        let key=(pair.stage_kind,post);
        behavior_effects.entry(key).or_default().push(effect);
        family_sets.entry(key).or_default().extend(families);
        min_fresh_items.entry(key).and_modify(|v|*v=(*v).min(n)).or_insert(n);
    }}
    let mut behavior_reports=Vec::new();
    let mut behavior_intervals=BTreeMap::<ConcealmentStageKindV1,MeanIntervalV1>::new();
    for stage in &stages{for post in [false,true]{
        let key=(*stage,post);let interval=mean_interval(behavior_effects.get(&key).unwrap())?;
        if post{behavior_intervals.insert(*stage,interval.clone());}
        behavior_reports.push(BehavioralRetentionV1{
            stage_kind:*stage,post_concealment:post,independent_model_pair_count:behavior_effects.get(&key).unwrap().len() as u64,
            treatment_minus_placebo_advantage_permille:interval,task_family_count:family_sets.get(&key).map_or(0,BTreeSet::len) as u64,
            minimum_fresh_items_per_pair:*min_fresh_items.get(&key).unwrap_or(&0),
        });
    }}

    let mut aggregates=Vec::new();let mut claims=Vec::new();
    for ((detector,stage,policy,target),values) in grouped{
        let get=|i:usize|values.iter().map(|row|row[i]).collect::<Vec<_>>();
        let pre=get(0);let post=get(1);let delta=values.iter().map(|row|row[1]-row[0]).collect::<Vec<_>>();
        let tpre=get(2);let tpost=get(3);let ppre=get(4);let ppost=get(5);let pcpre=get(6);let pcpost=get(7);
        let delta_ci=mean_interval(&delta)?;
        let behavior=behavior_intervals.get(&stage).unwrap();
        let decline=delta_ci.upper_95<0.0;
        let retained=behavior.lower_95>=study.minimum_behavior_advantage_permille as f64;
        if decline{
            claims.push(ConcealmentClaimV1{
                detector_id:detector.clone(),stage_kind:stage,threshold_policy:policy,target_fpr_permille:target,
                mean_tpr_change_permille:delta_ci.mean,tpr_change_upper_95_permille:delta_ci.upper_95,
                retained_behavioral_advantage_permille:behavior.mean,behavioral_advantage_lower_95_permille:behavior.lower_95,
                concealment_event_supported:retained,
            });
        }
        aggregates.push(DetectorConcealmentAggregateV1{
            detector_id:detector,stage_kind:stage,threshold_policy:policy,target_fpr_permille:target,
            independent_model_pair_count:values.len() as u64,
            pre_treatment_member_tpr:Some(mean_interval(&pre)?),post_treatment_member_tpr:Some(mean_interval(&post)?),
            pre_to_post_tpr_change:Some(delta_ci),pre_treatment_clean_fpr:Some(mean_interval(&tpre)?),
            post_treatment_clean_fpr:Some(mean_interval(&tpost)?),pre_placebo_pseudo_member_fpr:Some(mean_interval(&ppre)?),
            post_placebo_pseudo_member_fpr:Some(mean_interval(&ppost)?),pre_placebo_control_fpr:Some(mean_interval(&pcpre)?),
            post_placebo_control_fpr:Some(mean_interval(&pcpost)?),
            concealment_signal:if decline{"tpr_decline_ci_below_zero"}else{"no_confirmed_tpr_decline"}.to_owned(),
        });
    }

    let bytes=serde_json::to_vec(study).map_err(|e|format!("cannot serialize study: {e}"))?;
    Ok(DetectorConcealmentReportV1{
        schema:DETECTOR_CONCEALMENT_REPORT_SCHEMA.to_owned(),study_id:study.study_id.clone(),
        benchmark_id:study.benchmark_id.clone(),benchmark_version:study.benchmark_version.clone(),
        study_sha256:hash_domain(b"detector-concealment-study.v1",&bytes),
        independent_model_pair_count:study.model_pairs.len() as u64,detector_count:study.detector_methods.len() as u64,
        stage_count:stages.len() as u64,target_fpr_permille:study.target_fpr_permille.clone(),
        pair_results,detector_aggregates:aggregates,behavioral_retention:behavior_reports,
        supported_concealment_claims:claims,independently_authenticated:false,confirmatory_eligible:false,
        inference_status:"exploratory_until_external_receipt_authentication_and_preregistered_power".to_owned(),
        issues:vec![
            "Threshold selection is performed only on disjoint clean calibration cohorts; threshold uncertainty is not included in replicate-level intervals.".to_owned(),
            "Independent model-pair replicates, not item rows, define training-level uncertainty.".to_owned(),
            "A supported concealment event requires a negative TPR-change 95% interval and a fresh-task treatment advantage whose lower interval exceeds the preregistered margin.".to_owned(),
            "Placebo pseudo-member false-positive rates and treatment heldout-control false-positive rates are reported separately.".to_owned(),
            "Receipt digests provide content consistency, not authenticated model identity or training-data custody.".to_owned(),
        ],
        claim_boundary:"This report measures detector sensitivity under the declared post-training stage and frozen/recalibrated threshold policies, alongside fresh-task treatment-minus-placebo performance. It does not prove item-level membership beyond the declared randomized training artifacts, prove task-family equivalence, or establish generalization outside the registered distribution. Model-pair replicates are the independent units; calibration uncertainty is not included in the replicate intervals. Authentication and confirmatory flags remain false.".to_owned(),
    })
}

#[cfg(test)]
mod tests{
    use super::*;

    fn method(higher:bool)->DetectorMethodV1{
        DetectorMethodV1{detector_id:"detector".into(),detector_version:"v1".into(),detector_config_sha256:"a".repeat(64),scoring_access_regime:"black_box".into(),higher_is_more_suspicious:higher}
    }

    #[test]
    fn threshold_obeys_empirical_false_positive_budget(){
        let threshold=calibrated_threshold(&[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],&method(true),100).unwrap();
        assert_eq!(threshold,0.9);
    }

    #[test]
    fn low_is_more_suspicious_uses_low_raw_threshold(){
        let threshold=calibrated_threshold(&[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],&method(false),100).unwrap();
        assert_eq!(threshold,0.2);
        assert!(classify(0.1,threshold,&method(false)));
        assert!(!classify(0.9,threshold,&method(false)));
    }

    #[test]
    fn calibration_ties_do_not_exceed_declared_empirical_budget(){
        let threshold=calibrated_threshold(&[0.1,0.1,0.1,0.2,0.2,0.2,0.3,0.3,0.3,0.3],&method(true),100).unwrap();
        let false_positives=[0.1,0.1,0.1,0.2,0.2,0.2,0.3,0.3,0.3,0.3].iter().filter(|x|classify(**x,threshold,&method(true))).count();
        assert!(false_positives<=1);
    }

    #[test]
    fn confusion_separates_positive_and_negative_denominators(){
        let row=DetectorTestScoreV1{replicate_id:"r1".into(),model_arm:ModelArmV1::ExposureTreatment,model_id:"m1".into(),
            detector_id:"detector".into(),post_concealment:false,item_id:"i1".into(),cohort:EvaluationCohortV1::TreatmentAssignedSourceFamily,
            score:0.9,checkpoint_sha256:"b".repeat(64),manifest_sha256:"c".repeat(64)};
        let positive=confusion(&[&row],&method(true),0.5,true);
        let negative=confusion(&[&row],&method(true),0.5,false);
        assert_eq!(positive.tpr_permille,Some(1000.0));
        assert_eq!(negative.fpr_permille,Some(1000.0));
    }

    #[test]
    fn behavior_advantage_requires_paired_fresh_item_ids(){
        let t=FreshBehaviorScoreV1{replicate_id:"r1".into(),model_arm:ModelArmV1::ExposureTreatment,model_id:"t".into(),post_concealment:true,item_id:"fresh-1".into(),task_family:"math".into(),score_permille:800,checkpoint_sha256:"a".repeat(64),manifest_sha256:"b".repeat(64)};
        let p=FreshBehaviorScoreV1{model_arm:ModelArmV1::MatchedPlacebo,model_id:"p".into(),score_permille:500,..t.clone()};
        let (adv,domains,n)=pair_behavior_advantage(&[&t],&[&p]).unwrap();
        assert_eq!(adv,300.0);assert_eq!(domains.len(),1);assert_eq!(n,1);
    }
}
