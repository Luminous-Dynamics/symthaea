// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Effective static-potential analysis from correlated Wilson-loop trajectories.
//!
//! This module deliberately does not choose a plateau window. Callers supply an
//! exact Euclidean-time window, and the analysis reports the correlated GLS fit
//! for that window. Neighboring windows may be evaluated explicitly by callers,
//! but no automatic picker has scientific authority here.

use std::collections::BTreeSet;
use crate::lattice_covariance::{CorrelatedConstantFit, CovarianceError, correlated_constant_fit};

pub const EFFECTIVE_POTENTIAL_ANALYSIS_ID: &str = "wilson_effective_potential_declared_plateau_gls_v1";

#[derive(Debug, Clone, PartialEq)]
pub struct WilsonLoopTrajectory { pub chain_id: String, pub chain_position: usize, pub values: Vec<f64> }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlateauWindow { pub start_t: usize, pub end_t: usize }

#[derive(Debug, Clone, PartialEq)]
pub struct EffectivePotentialSeries {
    pub analysis_id: &'static str,
    pub chain_count: usize,
    pub configuration_count: usize,
    pub block_size: usize,
    pub block_count: usize,
    pub values: Vec<f64>,
    pub covariance: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DeclaredPlateauFit { pub analysis_id: &'static str, pub window: PlateauWindow, pub fit: CorrelatedConstantFit }

#[derive(Debug, Clone, PartialEq)]
pub enum StaticPotentialError {
    EmptyTrajectories,
    EmptyChainId { index: usize },
    DuplicateOrReappearingChain(String),
    InvalidChainPosition { chain_id: String, expected: usize, actual: usize },
    EmptyTrajectory { chain_id: String, chain_position: usize },
    TrajectoryWidthMismatch { expected: usize, actual: usize },
    NonFiniteWilsonLoop { chain_id: String, chain_position: usize, time_index: usize },
    NonPositiveMeanWilsonLoop { time_index: usize, value: f64 },
    InvalidBlockSize(usize),
    IncompleteChainBlock { chain_id: String, chain_length: usize, block_size: usize },
    TooFewBlocks(usize),
    InvalidWindow(PlateauWindow),
    Covariance(CovarianceError),
}
impl From<CovarianceError> for StaticPotentialError { fn from(value: CovarianceError) -> Self { Self::Covariance(value) } }

fn validate_and_chain_ranges(trajectories: &[WilsonLoopTrajectory]) -> Result<(usize, Vec<(String, usize, usize)>), StaticPotentialError> {
    if trajectories.is_empty() { return Err(StaticPotentialError::EmptyTrajectories); }
    let width = trajectories[0].values.len();
    if width < 2 { return Err(StaticPotentialError::EmptyTrajectory { chain_id: trajectories[0].chain_id.clone(), chain_position: trajectories[0].chain_position }); }
    let mut seen_closed = BTreeSet::new(); let mut ranges = Vec::new(); let mut start = 0usize;
    while start < trajectories.len() {
        let chain_id = trajectories[start].chain_id.clone();
        if chain_id.trim().is_empty() { return Err(StaticPotentialError::EmptyChainId { index: start }); }
        if seen_closed.contains(&chain_id) { return Err(StaticPotentialError::DuplicateOrReappearingChain(chain_id)); }
        let mut end = start;
        while end < trajectories.len() && trajectories[end].chain_id == chain_id {
            let row = &trajectories[end]; let expected_position = end - start;
            if row.chain_position != expected_position { return Err(StaticPotentialError::InvalidChainPosition { chain_id: chain_id.clone(), expected: expected_position, actual: row.chain_position }); }
            if row.values.is_empty() { return Err(StaticPotentialError::EmptyTrajectory { chain_id: chain_id.clone(), chain_position: row.chain_position }); }
            if row.values.len() != width { return Err(StaticPotentialError::TrajectoryWidthMismatch { expected: width, actual: row.values.len() }); }
            for (time_index, value) in row.values.iter().copied().enumerate() {
                if !value.is_finite() { return Err(StaticPotentialError::NonFiniteWilsonLoop { chain_id: chain_id.clone(), chain_position: row.chain_position, time_index }); }
            }
            end += 1;
        }
        seen_closed.insert(chain_id.clone()); ranges.push((chain_id, start, end)); start = end;
    }
    Ok((width, ranges))
}

fn effective_potential_for_indices(trajectories: &[WilsonLoopTrajectory], included: impl Iterator<Item=usize>, width: usize) -> Result<Vec<f64>, StaticPotentialError> {
    let mut sums = vec![0.0; width]; let mut count = 0usize;
    for index in included { count += 1; for (sum, value) in sums.iter_mut().zip(&trajectories[index].values) { *sum += *value; } }
    if count == 0 { return Err(StaticPotentialError::EmptyTrajectories); }
    let means = sums.into_iter().map(|sum| sum / count as f64).collect::<Vec<_>>();
    for (time_index, value) in means.iter().copied().enumerate() { if !value.is_finite() || value <= 0.0 { return Err(StaticPotentialError::NonPositiveMeanWilsonLoop { time_index, value }); } }
    Ok(means.windows(2).map(|pair| (pair[0] / pair[1]).ln()).collect())
}

pub fn blocked_effective_potential(trajectories: &[WilsonLoopTrajectory], block_size: usize) -> Result<EffectivePotentialSeries, StaticPotentialError> {
    let (width, ranges) = validate_and_chain_ranges(trajectories)?;
    if block_size == 0 { return Err(StaticPotentialError::InvalidBlockSize(block_size)); }
    let mut blocks = Vec::<(usize,usize)>::new();
    for (chain_id,start,end) in &ranges {
        let chain_length = end - start;
        if chain_length % block_size != 0 { return Err(StaticPotentialError::IncompleteChainBlock { chain_id: chain_id.clone(), chain_length, block_size }); }
        for block_start in (*start..*end).step_by(block_size) { blocks.push((block_start, block_start + block_size)); }
    }
    if blocks.len() < 2 { return Err(StaticPotentialError::TooFewBlocks(blocks.len())); }
    let full = effective_potential_for_indices(trajectories, 0..trajectories.len(), width)?;
    let mut replicates = Vec::with_capacity(blocks.len());
    for (remove_start,remove_end) in &blocks {
        let included = (0..trajectories.len()).filter(|index| *index < *remove_start || *index >= *remove_end);
        replicates.push(effective_potential_for_indices(trajectories, included, width)?);
    }
    let block_count = replicates.len(); let mut replicate_mean = vec![0.0; full.len()];
    for replicate in &replicates { for (mean,value) in replicate_mean.iter_mut().zip(replicate) { *mean += *value; } }
    for mean in &mut replicate_mean { *mean /= block_count as f64; }
    let factor = (block_count - 1) as f64 / block_count as f64; let mut covariance = vec![vec![0.0; full.len()]; full.len()];
    for replicate in &replicates { for i in 0..full.len() { let di = replicate[i]-replicate_mean[i]; for j in 0..full.len() { covariance[i][j] += factor * di * (replicate[j]-replicate_mean[j]); } } }
    Ok(EffectivePotentialSeries { analysis_id:EFFECTIVE_POTENTIAL_ANALYSIS_ID, chain_count:ranges.len(), configuration_count:trajectories.len(), block_size, block_count, values:full, covariance })
}

pub fn fit_declared_plateau(series: &EffectivePotentialSeries, window: PlateauWindow) -> Result<DeclaredPlateauFit, StaticPotentialError> {
    if window.start_t == 0 || window.end_t < window.start_t || window.end_t > series.values.len() || window.end_t-window.start_t+1 < 2 { return Err(StaticPotentialError::InvalidWindow(window)); }
    if series.covariance.len()!=series.values.len() || series.covariance.iter().any(|row| row.len()!=series.values.len()) { return Err(StaticPotentialError::Covariance(CovarianceError::DimensionMismatch)); }
    let start=window.start_t-1; let end=window.end_t; let values=series.values[start..end].to_vec();
    let covariance=(start..end).map(|i| (start..end).map(|j| series.covariance[i][j]).collect::<Vec<_>>()).collect::<Vec<_>>();
    let fit=correlated_constant_fit(&values,&covariance)?;
    Ok(DeclaredPlateauFit { analysis_id:EFFECTIVE_POTENTIAL_ANALYSIS_ID, window, fit })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn oracle_fixture() -> Vec<WilsonLoopTrajectory> {
        let mut out=Vec::new();
        for i in 0..96usize {
            let k=(i+1) as f64; let coeff=[0.06*(2.0*std::f64::consts::PI*k/13.0).sin(),0.04*(2.0*std::f64::consts::PI*k/11.0).cos(),0.03*(2.0*std::f64::consts::PI*k/7.0).sin(),0.025*(2.0*std::f64::consts::PI*k/5.0).cos(),0.02*(2.0*std::f64::consts::PI*k/17.0).sin(),0.018*(2.0*std::f64::consts::PI*k/19.0).cos(),0.015*(2.0*std::f64::consts::PI*k/23.0).sin()];
            let mut values=Vec::new();
            for t in 1..=8usize {
                let tf=t as f64; let x=(tf-4.5)/4.0; let features=[1.0,x,x*x-0.3,(0.6*tf).sin(),(0.8*tf).cos(),0.4*if t%2==0{1.0}else{-1.0},0.5*(1.2*tf).sin()];
                let perturb=coeff.iter().zip(features).map(|(a,b)| a*b).sum::<f64>(); let ground=0.82*(-0.43*tf).exp(); let excited=0.22*(-1.43*tf).exp(); values.push((ground+excited)*perturb.exp());
            }
            out.push(WilsonLoopTrajectory { chain_id:"chain-0".into(), chain_position:i, values });
        }
        out
    }
    #[test] fn reproduces_independent_lqcd_020e_effective_potential_and_plateau() {
        let series=blocked_effective_potential(&oracle_fixture(),8).unwrap(); let expected=[0.48933012423645994,0.45370913450403266,0.4389356119288337,0.4333102520823892,0.43047873133636405,0.43011889162947936,0.4298681776318562];
        for (actual,target) in series.values.iter().zip(expected) { assert!((actual-target).abs()<2.0e-13); }
        assert_eq!(series.block_count,12); let fit=fit_declared_plateau(&series,PlateauWindow{start_t:5,end_t:7}).unwrap();
        assert!((fit.fit.value-0.43001421487448777).abs()<2.0e-12); assert!((fit.fit.sigma-0.0014347332153113823).abs()<2.0e-12); assert!((fit.fit.chi_square_per_dof-0.020678206772295253).abs()<1.0e-10);
    }
    #[test] fn contaminated_early_window_is_not_silently_selected() { let series=blocked_effective_potential(&oracle_fixture(),8).unwrap(); let fit=fit_declared_plateau(&series,PlateauWindow{start_t:1,end_t:3}).unwrap(); assert!(fit.fit.chi_square_per_dof>600.0); }
    #[test] fn block_geometry_never_crosses_chain_boundaries() {
        let mut rows=oracle_fixture(); rows.truncate(12); for (index,row) in rows.iter_mut().enumerate() { row.chain_id=if index<6{"a".into()}else{"b".into()}; row.chain_position=if index<6{index}else{index-6}; }
        assert!(matches!(blocked_effective_potential(&rows,4),Err(StaticPotentialError::IncompleteChainBlock{..}))); assert!(blocked_effective_potential(&rows,3).is_ok());
    }
    #[test] fn reappearing_chain_fails_closed() {
        let mut rows=oracle_fixture(); rows.truncate(6); let ids=["a","a","b","b","a","a"]; for (index,row) in rows.iter_mut().enumerate() { row.chain_id=ids[index].into(); row.chain_position=if index%2==0{0}else{1}; }
        assert!(matches!(blocked_effective_potential(&rows,1),Err(StaticPotentialError::DuplicateOrReappearingChain(_))));
    }
    #[test] fn nonpositive_ensemble_mean_fails_before_log() {
        let rows=vec![WilsonLoopTrajectory{chain_id:"a".into(),chain_position:0,values:vec![1.0,-1.0]},WilsonLoopTrajectory{chain_id:"a".into(),chain_position:1,values:vec![1.0,-1.0]}];
        assert!(matches!(blocked_effective_potential(&rows,1),Err(StaticPotentialError::NonPositiveMeanWilsonLoop{..})));
    }
}
