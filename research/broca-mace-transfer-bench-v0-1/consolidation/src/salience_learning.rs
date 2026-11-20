// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Deterministic learned future-use forecasting with source/task-held-out splits.
//! Future-reuse labels are evaluator-owned; inference receives features only.

use std::collections::BTreeSet;
use broca_mace_policy::SourceRootDigest;
use sha2::{Digest, Sha256};

pub const MIN_TRAIN_CLUSTERS: usize = 32;
pub const MIN_CALIBRATION_CLUSTERS: usize = 16;
pub const MIN_HELDOUT_CLUSTERS: usize = 16;
pub const MIN_TASK_FAMILIES: usize = 4;
const FEATURE_COUNT: usize = 7;
const ECE_BINS: usize = 10;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FutureUseFeaturesV1 {
    pub recency_bps: u16,
    pub frequency_bps: u16,
    pub task_similarity_bps: u16,
    pub exact_scope_match: bool,
    pub source_support_bps: u16,
    pub measurement_quality_bps: u16,
    pub failure_memory: bool,
}
impl FutureUseFeaturesV1 {
    fn valid(self) -> bool {
        self.recency_bps <= 10_000
            && self.frequency_bps <= 10_000
            && self.task_similarity_bps <= 10_000
            && self.source_support_bps <= 10_000
            && self.measurement_quality_bps <= 10_000
    }
    fn vector(self) -> [f64; FEATURE_COUNT + 1] {
        [
            1.0,
            f64::from(self.recency_bps) / 10_000.0,
            f64::from(self.frequency_bps) / 10_000.0,
            f64::from(self.task_similarity_bps) / 10_000.0,
            f64::from(self.exact_scope_match),
            f64::from(self.source_support_bps) / 10_000.0,
            f64::from(self.measurement_quality_bps) / 10_000.0,
            f64::from(self.failure_memory),
        ]
    }
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FutureUseExampleV1 {
    pub event_id: u32,
    pub cluster_id: u64,
    pub task_family_id: u32,
    pub source_root: SourceRootDigest,
    pub features: FutureUseFeaturesV1,
    pub future_reused: bool,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ModelConfigV1 {
    pub epochs: u32,
    pub learning_rate_micros: u32,
    pub l2_micros: u32,
}
impl Default for ModelConfigV1 {
    fn default() -> Self {
        Self { epochs: 2_000, learning_rate_micros: 30_000, l2_micros: 1_000 }
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ForecastReject {
    Empty, BadId, UnknownRoot, BadFeature, DuplicateEvent,
    TooFewClusters, TooFewFamilies, TooFewRoots, MissingClass,
    ClusterOverlap, FamilyOverlap, RootOverlap, EventOverlap,
    BadConfig, ModelMismatch,
}
#[derive(Clone, Debug, Eq, PartialEq)]
struct Identity {
    clusters: BTreeSet<u64>,
    families: BTreeSet<u32>,
    roots: BTreeSet<SourceRootDigest>,
    events: BTreeSet<u32>,
}
#[derive(Clone, Debug, Eq, PartialEq)]
struct Split {
    identity: Identity,
    digest: String,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SplitManifestV1 {
    pub training_sha256: String,
    pub calibration_sha256: String,
    pub heldout_sha256: String,
    pub manifest_sha256: String,
    pub training_clusters: usize,
    pub calibration_clusters: usize,
    pub heldout_clusters: usize,
}
#[derive(Clone, Debug, PartialEq)]
pub struct LearnedFutureUseModelV1 {
    weights: [f64; FEATURE_COUNT + 1],
    training: Identity,
    pub model_sha256: String,
}
#[derive(Clone, Debug, PartialEq)]
pub struct FutureUseCalibratorV1 {
    slope: f64,
    intercept: f64,
    model_sha256: String,
    calibration: Identity,
    pub calibrator_sha256: String,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ForecastMetricsV1 {
    pub count: u32,
    pub clusters: u32,
    pub task_families: u32,
    pub source_roots: u32,
    pub brier_micros: u32,
    pub base_rate_brier_micros: u32,
    pub ece_micros: u32,
    pub worst_family_ece_micros: u32,
}

fn sha(bytes: &[u8]) -> String {
    Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect()
}
fn hash_field(h: &mut Sha256, value: &[u8]) {
    h.update((value.len() as u64).to_be_bytes());
    h.update(value);
}
fn sigmoid(x: f64) -> f64 { 1.0 / (1.0 + (-x.clamp(-30.0, 30.0)).exp()) }
fn dot(w: &[f64; 8], x: &[f64; 8]) -> f64 {
    w.iter().zip(x).map(|(a,b)| a*b).sum()
}
fn identity(rows: &[FutureUseExampleV1]) -> Identity {
    Identity {
        clusters: rows.iter().map(|r| r.cluster_id).collect(),
        families: rows.iter().map(|r| r.task_family_id).collect(),
        roots: rows.iter().map(|r| r.source_root).collect(),
        events: rows.iter().map(|r| r.event_id).collect(),
    }
}
fn split_digest(rows: &[FutureUseExampleV1]) -> String {
    let mut ordered: Vec<_> = rows.iter().collect();
    ordered.sort_by_key(|r| r.event_id);
    let mut h=Sha256::new();
    h.update(b"luminous:mace:future-use-split:v1\0");
    for r in ordered {
        h.update(r.event_id.to_be_bytes());
        h.update(r.cluster_id.to_be_bytes());
        h.update(r.task_family_id.to_be_bytes());
        h.update(r.source_root.as_bytes());
        h.update(r.features.recency_bps.to_be_bytes());
        h.update(r.features.frequency_bps.to_be_bytes());
        h.update(r.features.task_similarity_bps.to_be_bytes());
        h.update([u8::from(r.features.exact_scope_match)]);
        h.update(r.features.source_support_bps.to_be_bytes());
        h.update(r.features.measurement_quality_bps.to_be_bytes());
        h.update([u8::from(r.features.failure_memory)]);
        h.update([u8::from(r.future_reused)]);
    }
    sha(&h.finalize())
}
fn validate(rows: &[FutureUseExampleV1], min_clusters: usize) -> Result<Split, ForecastReject> {
    if rows.is_empty() { return Err(ForecastReject::Empty); }
    let mut seen=BTreeSet::new();
    let mut positive=false;
    let mut negative=false;
    for r in rows {
        if r.event_id==0 || r.cluster_id==0 || r.task_family_id==0 {
            return Err(ForecastReject::BadId);
        }
        if !r.source_root.is_known() { return Err(ForecastReject::UnknownRoot); }
        if !r.features.valid() { return Err(ForecastReject::BadFeature); }
        if !seen.insert(r.event_id) { return Err(ForecastReject::DuplicateEvent); }
        positive |= r.future_reused;
        negative |= !r.future_reused;
    }
    if !positive || !negative { return Err(ForecastReject::MissingClass); }
    let id=identity(rows);
    if id.clusters.len()<min_clusters { return Err(ForecastReject::TooFewClusters); }
    if id.families.len()<MIN_TASK_FAMILIES { return Err(ForecastReject::TooFewFamilies); }
    if id.roots.len()<min_clusters/2 { return Err(ForecastReject::TooFewRoots); }
    Ok(Split { identity:id, digest:split_digest(rows) })
}
fn disjoint(a: &Identity,b: &Identity) -> Result<(),ForecastReject> {
    if !a.clusters.is_disjoint(&b.clusters) { return Err(ForecastReject::ClusterOverlap); }
    if !a.families.is_disjoint(&b.families) { return Err(ForecastReject::FamilyOverlap); }
    if !a.roots.is_disjoint(&b.roots) { return Err(ForecastReject::RootOverlap); }
    if !a.events.is_disjoint(&b.events) { return Err(ForecastReject::EventOverlap); }
    Ok(())
}
pub fn qualify_splits(
    train: &[FutureUseExampleV1],
    calibration: &[FutureUseExampleV1],
    heldout: &[FutureUseExampleV1],
) -> Result<SplitManifestV1,ForecastReject> {
    let a=validate(train,MIN_TRAIN_CLUSTERS)?;
    let b=validate(calibration,MIN_CALIBRATION_CLUSTERS)?;
    let c=validate(heldout,MIN_HELDOUT_CLUSTERS)?;
    disjoint(&a.identity,&b.identity)?;
    disjoint(&a.identity,&c.identity)?;
    disjoint(&b.identity,&c.identity)?;
    let mut h=Sha256::new();
    h.update(b"luminous:mace:future-use-manifest:v1\0");
    hash_field(&mut h,a.digest.as_bytes());
    hash_field(&mut h,b.digest.as_bytes());
    hash_field(&mut h,c.digest.as_bytes());
    for id in [&a.identity,&b.identity,&c.identity] {
        for f in &id.families { h.update(f.to_be_bytes()); }
        for r in &id.roots { h.update(r.as_bytes()); }
    }
    Ok(SplitManifestV1 {
        training_sha256:a.digest,
        calibration_sha256:b.digest,
        heldout_sha256:c.digest,
        manifest_sha256:sha(&h.finalize()),
        training_clusters:a.identity.clusters.len(),
        calibration_clusters:b.identity.clusters.len(),
        heldout_clusters:c.identity.clusters.len(),
    })
}
pub fn train(
    rows: &[FutureUseExampleV1],
    config: ModelConfigV1,
) -> Result<LearnedFutureUseModelV1,ForecastReject> {
    if config.epochs==0 || config.epochs>50_000
        || !(1_000..=200_000).contains(&config.learning_rate_micros)
        || config.l2_micros>100_000 {
        return Err(ForecastReject::BadConfig);
    }
    let split=validate(rows,MIN_TRAIN_CLUSTERS)?;
    let mut ordered:Vec<_>=rows.iter().collect();
    ordered.sort_by_key(|r| r.event_id);
    let n=ordered.len() as f64;
    let rate=f64::from(config.learning_rate_micros)/1_000_000.0;
    let l2=f64::from(config.l2_micros)/1_000_000.0;
    let mut w=[0.0_f64;8];
    for _ in 0..config.epochs {
        let mut grad=[0.0_f64;8];
        for row in &ordered {
            let x=row.features.vector();
            let y=if row.future_reused {1.0} else {0.0};
            let e=sigmoid(dot(&w,&x))-y;
            for i in 0..8 { grad[i]+=e*x[i]; }
        }
        for i in 0..8 {
            let penalty=if i==0 {0.0} else {l2*w[i]};
            w[i]=(w[i]-rate*(grad[i]/n+penalty)).clamp(-12.0,12.0);
        }
    }
    let mut h=Sha256::new();
    h.update(b"luminous:mace:future-use-model:v1\0");
    hash_field(&mut h,split.digest.as_bytes());
    h.update(config.epochs.to_be_bytes());
    h.update(config.learning_rate_micros.to_be_bytes());
    h.update(config.l2_micros.to_be_bytes());
    for v in w { h.update(v.to_bits().to_be_bytes()); }
    Ok(LearnedFutureUseModelV1 {
        weights:w,
        training:split.identity,
        model_sha256:sha(&h.finalize()),
    })
}
impl LearnedFutureUseModelV1 {
    fn logit(&self,x:FutureUseFeaturesV1)->Result<f64,ForecastReject> {
        if !x.valid() { return Err(ForecastReject::BadFeature); }
        Ok(dot(&self.weights,&x.vector()))
    }
    pub fn model_sha256(&self)->&str { &self.model_sha256 }
}
pub fn calibrate(
    model:&LearnedFutureUseModelV1,
    rows:&[FutureUseExampleV1],
)->Result<FutureUseCalibratorV1,ForecastReject> {
    let split=validate(rows,MIN_CALIBRATION_CLUSTERS)?;
    disjoint(&model.training,&split.identity)?;
    let mut ordered:Vec<_>=rows.iter().collect();
    ordered.sort_by_key(|r|r.event_id);
    let n=ordered.len() as f64;
    let mut slope=1.0_f64;
    let mut intercept=0.0_f64;
    for _ in 0..2_000 {
        let mut gs=0.0;
        let mut gi=0.0;
        for row in &ordered {
            let z=model.logit(row.features)?;
            let y=if row.future_reused {1.0} else {0.0};
            let e=sigmoid(slope*z+intercept)-y;
            gs+=e*z;
            gi+=e;
        }
        slope=(slope-0.02*gs/n).clamp(0.05,20.0);
        intercept=(intercept-0.02*gi/n).clamp(-10.0,10.0);
    }
    let mut h=Sha256::new();
    h.update(b"luminous:mace:future-use-calibrator:v1\0");
    hash_field(&mut h,model.model_sha256.as_bytes());
    hash_field(&mut h,split.digest.as_bytes());
    h.update(slope.to_bits().to_be_bytes());
    h.update(intercept.to_bits().to_be_bytes());
    Ok(FutureUseCalibratorV1 {
        slope,intercept,
        model_sha256:model.model_sha256.clone(),
        calibration:split.identity,
        calibrator_sha256:sha(&h.finalize()),
    })
}
impl FutureUseCalibratorV1 {
    pub fn calibrator_sha256(&self)->&str { &self.calibrator_sha256 }
    pub fn predict_bps(
        &self,model:&LearnedFutureUseModelV1,features:FutureUseFeaturesV1
    )->Result<u16,ForecastReject> {
        if self.model_sha256!=model.model_sha256 {
            return Err(ForecastReject::ModelMismatch);
        }
        Ok((sigmoid(self.slope*model.logit(features)?+self.intercept)
            *10_000.0).round().clamp(0.0,10_000.0) as u16)
    }
}
#[derive(Clone,Copy,Debug,Eq,PartialEq)]
pub struct ForecastMetricsV1 {
    pub count:u32,
    pub clusters:u32,
    pub task_families:u32,
    pub source_roots:u32,
    pub brier_micros:u32,
    pub base_rate_brier_micros:u32,
    pub ece_micros:u32,
    pub worst_family_ece_micros:u32,
}
fn ece(rows:&[FutureUseExampleV1],p:&[f64])->f64 {
    let mut counts=[0_u32;ECE_BINS];
    let mut ps=[0.0;ECE_BINS];
    let mut ys=[0.0;ECE_BINS];
    for (row,prob) in rows.iter().zip(p) {
        let bin=((*prob*ECE_BINS as f64).floor() as usize).min(ECE_BINS-1);
        counts[bin]+=1;
        ps[bin]+=prob;
        ys[bin]+=if row.future_reused {1.0} else {0.0};
    }
    let n=rows.len() as f64;
    let mut result=0.0;
    for i in 0..ECE_BINS {
        if counts[i]>0 {
            let count=f64::from(counts[i]);
            result+=count/n*(ps[i]/count-ys[i]/count).abs();
        }
    }
    result
}
pub fn evaluate(
    model:&LearnedFutureUseModelV1,
    calibrator:&FutureUseCalibratorV1,
    rows:&[FutureUseExampleV1],
)->Result<ForecastMetricsV1,ForecastReject> {
    if calibrator.model_sha256!=model.model_sha256 {
        return Err(ForecastReject::ModelMismatch);
    }
    let split=validate(rows,MIN_HELDOUT_CLUSTERS)?;
    disjoint(&model.training,&split.identity)?;
    disjoint(&calibrator.calibration,&split.identity)?;
    let probs:Vec<f64>=rows.iter().map(|r|
        calibrator.predict_bps(model,r.features).map(|v|f64::from(v)/10_000.0)
    ).collect::<Result<_,_>>()?;
    let rate=rows.iter().filter(|r|r.future_reused).count() as f64/rows.len() as f64;
    let mut brier=0.0;
    let mut base=0.0;
    for (r,p) in rows.iter().zip(&probs) {
        let y=if r.future_reused {1.0} else {0.0};
        brier+=(p-y).powi(2);
        base+=(rate-y).powi(2);
    }
    brier/=rows.len() as f64;
    base/=rows.len() as f64;
    let overall_ece=ece(rows,&probs);
    let mut groups:std::collections::BTreeMap<u32,Vec<usize>>=Default::default();
    for (i,r) in rows.iter().enumerate() {
        groups.entry(r.task_family_id).or_default().push(i);
    }
    let mut worst=0.0_f64;
    for indices in groups.values() {
        if indices.len()<4 { continue; }
        let group_rows:Vec<_>=indices.iter().map(|i|rows[*i].clone()).collect();
        let group_probs:Vec<_>=indices.iter().map(|i|probs[*i]).collect();
        worst=worst.max(ece(&group_rows,&group_probs));
    }
    let micros=|v:f64|(v*1_000_000.0).round().clamp(0.0,1_000_000.0) as u32;
    Ok(ForecastMetricsV1 {
        count:rows.len() as u32,
        clusters:split.identity.clusters.len() as u32,
        task_families:split.identity.families.len() as u32,
        source_roots:split.identity.roots.len() as u32,
        brier_micros:micros(brier),
        base_rate_brier_micros:micros(base),
        ece_micros:micros(overall_ece),
        worst_family_ece_micros:micros(worst),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn rows(event:u32,cluster:u64,family:u32)->Vec<FutureUseExampleV1> {
        let mut out=Vec::new();
        for c in 0..32_u32 {
            for j in 0..8_u32 {
                let id=event+c*8+j+1;
                let mut root=[0_u8;32];
                root[..4].copy_from_slice(&(id+50_000).to_be_bytes());
                out.push(FutureUseExampleV1 {
                    event_id:id,cluster_id:cluster+u64::from(c)+1,
                    task_family_id:family+c%8,
                    source_root:SourceRootDigest::from_bytes(root),
                    features:FutureUseFeaturesV1 {
                        recency_bps:7_000,frequency_bps:6_000,
                        task_similarity_bps:8_000,exact_scope_match:true,
                        source_support_bps:7_000,measurement_quality_bps:9_000,
                        failure_memory:false,
                    },
                    future_reused:(c+j)%3!=0,
                });
            }
        }
        out
    }
    #[test]
    fn independent_train_calibration_and_heldout_splits_are_required() {
        let train=rows(1,1,10);
        let calibration=rows(10_000,100,30);
        let heldout=rows(20_000,200,50);
        let manifest=qualify_splits(&train,&calibration,&heldout).unwrap();
        assert_eq!(manifest.training_clusters,32);
        assert_eq!(manifest.calibration_clusters,32);
        assert_eq!(manifest.heldout_clusters,32);
    }
    #[test]
    fn source_root_overlap_is_rejected_even_when_other_ids_are_disjoint() {
        let train=rows(1,1,10);
        let calibration=rows(10_000,100,30);
        let mut heldout=rows(20_000,200,50);
        heldout[0].source_root=train[0].source_root;
        assert_eq!(
            qualify_splits(&train,&calibration,&heldout),
            Err(ForecastReject::RootOverlap)
        );
    }
    #[test]
    fn learned_model_is_calibrated_and_evaluated_without_heldout_updates() {
        let train=rows(1,1,10);
        let calibration=rows(10_000,100,30);
        let heldout=rows(20_000,200,50);
        let model=train(&train,ModelConfigV1::default()).unwrap();
        let calibrated=calibrate(&model,&calibration).unwrap();
        let metrics=evaluate(&model,&calibrated,&heldout).unwrap();
        assert_eq!(metrics.count,heldout.len() as u32);
        assert_eq!(metrics.clusters,32);
        assert!(metrics.brier_micros<=1_000_000);
        assert!(metrics.ece_micros<=1_000_000);
    }
    #[test]
    fn calibrator_is_bound_to_exact_model_digest() {
        let train_rows=rows(1,1,10);
        let calibration=rows(10_000,100,30);
        let model=train(&train_rows,ModelConfigV1::default()).unwrap();
        let calibrated=calibrate(&model,&calibration).unwrap();
        let other=train(&train_rows,ModelConfigV1 {epochs:100,..ModelConfigV1::default()}).unwrap();
        assert_eq!(
            calibrated.predict_bps(&other,train_rows[0].features),
            Err(ForecastReject::ModelMismatch)
        );
    }
}
