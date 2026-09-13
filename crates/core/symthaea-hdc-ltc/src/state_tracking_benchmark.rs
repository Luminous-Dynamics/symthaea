// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Deterministic irregular-time compositional state-tracking benchmark.
//!
//! This module is intentionally model-agnostic. It generates an exact symbolic
//! world, event stream, query stream, and oracle so HLS, legacy HDC-LTC, CfC,
//! recurrent, SSM, and attention baselines can be evaluated on the same task.
//!
//! World structure:
//!
//! ```text
//! object --owned_by--> entity --located_at--> location
//! ```
//!
//! Every generated episode begins with an **observable initialization prefix**:
//! one `MoveEntity` fact for every entity and one `TransferObject` fact for every
//! object, deterministically shuffled. Only after that complete world snapshot
//! has been presented do ordinary mutations and scored queries begin. This keeps
//! train/test initial worlds seed-randomized without making any scored fact
//! unknowable to the model.
//!
//! Ownership and location then mutate independently. `ObjectLocation` requires
//! composition across two mutable relations. Historical queries ask about an
//! earlier fully-observed world state after later events have been seen.

use serde::{Deserialize, Serialize};
use std::fmt;

pub type EntityId = u16;
pub type ObjectId = u16;
pub type LocationId = u16;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrackingEventKind { MoveEntity { entity: EntityId, to: LocationId }, TransferObject { object: ObjectId, to: EntityId } }
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrackingEvent { pub time: f64, pub kind: TrackingEventKind }
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrackingQueryKind { EntityLocation { entity: EntityId }, ObjectOwner { object: ObjectId }, ObjectLocation { object: ObjectId } }
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrackingAnswer { Entity(EntityId), Location(LocationId) }
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrackingQuery { pub asked_after_event: usize, pub as_of_event: usize, pub as_of_time: f64, pub kind: TrackingQueryKind, pub expected: TrackingAnswer }
impl TrackingQuery { pub fn is_historical(&self) -> bool { self.as_of_event < self.asked_after_event } }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateTrackingBenchmarkConfig { pub entities: usize, pub objects: usize, pub locations: usize, pub events: usize, pub query_every: usize, pub min_dt: f64, pub max_dt: f64, pub historical_query_rate: f64, pub seed: u64 }
impl Default for StateTrackingBenchmarkConfig { fn default() -> Self { Self { entities:32, objects:64, locations:16, events:10_000, query_every:10, min_dt:1e-3, max_dt:1e2, historical_query_rate:0.5, seed:42 } } }

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StateTrackingBenchmarkError { EmptyDomain(&'static str), TooManyIds(&'static str), InvalidQueryCadence, InvalidTimeRange, InvalidHistoricalRate, EventIndexOutOfRange { index: usize, len: usize }, PredictionCountMismatch { expected: usize, actual: usize } }
impl fmt::Display for StateTrackingBenchmarkError { fn fmt(&self,f:&mut fmt::Formatter<'_>)->fmt::Result{match self{Self::EmptyDomain(n)=>write!(f,"benchmark domain {n} must be non-empty"),Self::TooManyIds(n)=>write!(f,"benchmark domain {n} exceeds u16 id capacity"),Self::InvalidQueryCadence=>write!(f,"query_every must be non-zero"),Self::InvalidTimeRange=>write!(f,"dt range must satisfy 0 < min_dt <= max_dt"),Self::InvalidHistoricalRate=>write!(f,"historical_query_rate must be finite and in [0, 1]"),Self::EventIndexOutOfRange{index,len}=>write!(f,"event index {index} out of range for event length {len}"),Self::PredictionCountMismatch{expected,actual}=>write!(f,"prediction count mismatch: expected {expected}, got {actual}")}} }
impl std::error::Error for StateTrackingBenchmarkError {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrackingScore { pub correct:usize,pub total:usize,pub current_correct:usize,pub current_total:usize,pub historical_correct:usize,pub historical_total:usize,pub entity_location_correct:usize,pub entity_location_total:usize,pub object_owner_correct:usize,pub object_owner_total:usize,pub object_location_correct:usize,pub object_location_total:usize }
impl TrackingScore { pub fn accuracy(&self)->f64{ratio(self.correct,self.total)} pub fn current_accuracy(&self)->f64{ratio(self.current_correct,self.current_total)} pub fn historical_accuracy(&self)->f64{ratio(self.historical_correct,self.historical_total)} pub fn compositional_accuracy(&self)->f64{ratio(self.object_location_correct,self.object_location_total)} }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateTrackingBenchmark { pub config:StateTrackingBenchmarkConfig,pub events:Vec<TrackingEvent>,pub queries:Vec<TrackingQuery>,#[serde(default)] pub initialization_events:usize,initial_entity_locations:Vec<LocationId>,initial_object_owners:Vec<EntityId> }
impl StateTrackingBenchmark {
    pub fn generate(config:StateTrackingBenchmarkConfig)->Result<Self,StateTrackingBenchmarkError>{validate_config(&config)?;let mut rng=XorShift64::new(config.seed);let initial_entity_locations=(0..config.entities).map(|_|rng.index(config.locations)as LocationId).collect::<Vec<_>>();let initial_object_owners=(0..config.objects).map(|_|rng.index(config.entities)as EntityId).collect::<Vec<_>>();let mut bootstrap=Vec::with_capacity(config.entities+config.objects);for(entity,&to)in initial_entity_locations.iter().enumerate(){bootstrap.push(TrackingEventKind::MoveEntity{entity:entity as EntityId,to})}for(object,&to)in initial_object_owners.iter().enumerate(){bootstrap.push(TrackingEventKind::TransferObject{object:object as ObjectId,to})}rng.shuffle(&mut bootstrap);let initialization_events=bootstrap.len();let mut events=Vec::with_capacity(initialization_events+config.events);let mut time=0.0;for kind in bootstrap{time+=sample_log_uniform(&mut rng,config.min_dt,config.max_dt);events.push(TrackingEvent{time,kind})}for _ in 0..config.events{time+=sample_log_uniform(&mut rng,config.min_dt,config.max_dt);let kind=if rng.next_bool(){TrackingEventKind::MoveEntity{entity:rng.index(config.entities)as EntityId,to:rng.index(config.locations)as LocationId}}else{TrackingEventKind::TransferObject{object:rng.index(config.objects)as ObjectId,to:rng.index(config.entities)as EntityId}};events.push(TrackingEvent{time,kind})}let mut b=Self{config,events,queries:Vec::new(),initialization_events,initial_entity_locations,initial_object_owners};b.queries=b.generate_queries(&mut rng)?;Ok(b)}
    pub fn first_fully_observed_event(&self)->Option<usize>{self.initialization_events.checked_sub(1)}
    pub fn oracle_answer(&self,as_of:usize,kind:&TrackingQueryKind)->Result<TrackingAnswer,StateTrackingBenchmarkError>{if as_of>=self.events.len(){return Err(StateTrackingBenchmarkError::EventIndexOutOfRange{index:as_of,len:self.events.len()})}Ok(answer(&self.replay_through(as_of),kind))}
    pub fn score(&self,predictions:&[TrackingAnswer])->Result<TrackingScore,StateTrackingBenchmarkError>{if predictions.len()!=self.queries.len(){return Err(StateTrackingBenchmarkError::PredictionCountMismatch{expected:self.queries.len(),actual:predictions.len()})}let mut s=TrackingScore{correct:0,total:self.queries.len(),current_correct:0,current_total:0,historical_correct:0,historical_total:0,entity_location_correct:0,entity_location_total:0,object_owner_correct:0,object_owner_total:0,object_location_correct:0,object_location_total:0};for(p,q)in predictions.iter().zip(&self.queries){let h=(*p==q.expected)as usize;s.correct+=h;if q.is_historical(){s.historical_total+=1;s.historical_correct+=h}else{s.current_total+=1;s.current_correct+=h}match q.kind{TrackingQueryKind::EntityLocation{..}=>{s.entity_location_total+=1;s.entity_location_correct+=h},TrackingQueryKind::ObjectOwner{..}=>{s.object_owner_total+=1;s.object_owner_correct+=h},TrackingQueryKind::ObjectLocation{..}=>{s.object_location_total+=1;s.object_location_correct+=h}}}Ok(s)}
    fn generate_queries(&self,rng:&mut XorShift64)->Result<Vec<TrackingQuery>,StateTrackingBenchmarkError>{let mut q=Vec::new();let Some(first)=self.first_fully_observed_event()else{return Ok(q)};let start=self.initialization_events+self.config.query_every-1;if start>=self.events.len(){return Ok(q)}for asked in(start..self.events.len()).step_by(self.config.query_every){let e=rng.index(self.config.entities)as EntityId;let o=rng.index(self.config.objects)as ObjectId;self.push_query(&mut q,asked,asked,TrackingQueryKind::EntityLocation{entity:e})?;self.push_query(&mut q,asked,asked,TrackingQueryKind::ObjectOwner{object:o})?;self.push_query(&mut q,asked,asked,TrackingQueryKind::ObjectLocation{object:o})?;if asked>first&&rng.next_f64()<self.config.historical_query_rate{let as_of=first+rng.index(asked-first);let o=rng.index(self.config.objects)as ObjectId;self.push_query(&mut q,asked,as_of,TrackingQueryKind::ObjectLocation{object:o})?}}Ok(q)}
    fn push_query(&self,q:&mut Vec<TrackingQuery>,asked:usize,as_of:usize,kind:TrackingQueryKind)->Result<(),StateTrackingBenchmarkError>{let expected=self.oracle_answer(as_of,&kind)?;q.push(TrackingQuery{asked_after_event:asked,as_of_event:as_of,as_of_time:self.events[as_of].time,kind,expected});Ok(())}
    fn replay_through(&self,as_of:usize)->WorldState{let mut s=WorldState{entity_locations:self.initial_entity_locations.clone(),object_owners:self.initial_object_owners.clone()};for e in self.events.iter().take(as_of+1){apply_event(&mut s,&e.kind)}s}
}
#[derive(Debug,Clone)]struct WorldState{entity_locations:Vec<LocationId>,object_owners:Vec<EntityId>}
fn apply_event(s:&mut WorldState,e:&TrackingEventKind){match *e{TrackingEventKind::MoveEntity{entity,to}=>s.entity_locations[entity as usize]=to,TrackingEventKind::TransferObject{object,to}=>s.object_owners[object as usize]=to}}
fn answer(s:&WorldState,k:&TrackingQueryKind)->TrackingAnswer{match *k{TrackingQueryKind::EntityLocation{entity}=>TrackingAnswer::Location(s.entity_locations[entity as usize]),TrackingQueryKind::ObjectOwner{object}=>TrackingAnswer::Entity(s.object_owners[object as usize]),TrackingQueryKind::ObjectLocation{object}=>{let owner=s.object_owners[object as usize];TrackingAnswer::Location(s.entity_locations[owner as usize])}}}
fn validate_config(c:&StateTrackingBenchmarkConfig)->Result<(),StateTrackingBenchmarkError>{for(n,v)in[("entities",c.entities),("objects",c.objects),("locations",c.locations),("events",c.events)]{if v==0{return Err(StateTrackingBenchmarkError::EmptyDomain(n))}}for(n,v)in[("entities",c.entities),("objects",c.objects),("locations",c.locations)]{if v>u16::MAX as usize+1{return Err(StateTrackingBenchmarkError::TooManyIds(n))}}if c.query_every==0{return Err(StateTrackingBenchmarkError::InvalidQueryCadence)}if !c.min_dt.is_finite()||!c.max_dt.is_finite()||c.min_dt<=0.0||c.max_dt<c.min_dt{return Err(StateTrackingBenchmarkError::InvalidTimeRange)}if !c.historical_query_rate.is_finite()||!(0.0..=1.0).contains(&c.historical_query_rate){return Err(StateTrackingBenchmarkError::InvalidHistoricalRate)}Ok(())}
fn sample_log_uniform(r:&mut XorShift64,min:f64,max:f64)->f64{if min==max{return min}(min.ln()+r.next_f64()*(max.ln()-min.ln())).exp()}
fn ratio(c:usize,t:usize)->f64{if t==0{0.0}else{c as f64/t as f64}}
#[derive(Debug,Clone)]struct XorShift64{state:u64}
impl XorShift64{fn new(seed:u64)->Self{let m=seed^0x9E3779B97F4A7C15;Self{state:if m==0{0xD1B54A32D192ED03}else{m}}}fn next_u64(&mut self)->u64{self.state^=self.state<<13;self.state^=self.state>>7;self.state^=self.state<<17;self.state}fn next_bool(&mut self)->bool{self.next_u64()&1==1}fn next_f64(&mut self)->f64{let b=self.next_u64()>>11;b as f64*(1.0/((1_u64<<53)as f64))}fn index(&mut self,len:usize)->usize{debug_assert!(len>0);(self.next_u64()as usize)%len}fn shuffle<T>(&mut self,v:&mut[T]){for i in(1..v.len()).rev(){let j=self.index(i+1);v.swap(i,j)}}}

#[cfg(test)]mod tests{use super::*;#[test]fn bootstrap_precedes_queries(){let b=StateTrackingBenchmark::generate(StateTrackingBenchmarkConfig{entities:8,objects:12,events:100,query_every:1,historical_query_rate:1.0,..Default::default()}).unwrap();let f=b.first_fully_observed_event().unwrap();assert_eq!(b.initialization_events,20);assert!(b.queries.iter().all(|q|q.asked_after_event>=b.initialization_events&&q.as_of_event>=f))}#[test]fn perfect_oracle_scores_one(){let b=StateTrackingBenchmark::generate(StateTrackingBenchmarkConfig{events:100,historical_query_rate:1.0,..Default::default()}).unwrap();let p=b.queries.iter().map(|q|q.expected).collect::<Vec<_>>();let s=b.score(&p).unwrap();assert_eq!(s.accuracy(),1.0);assert_eq!(s.historical_accuracy(),1.0);assert_eq!(s.compositional_accuracy(),1.0)}}
