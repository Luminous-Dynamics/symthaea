// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! MATH-RET-RUNTIME-001E guarded F-arm fusion fixture emitter.

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use symthaea_math_retrieval_membership_guard::{FrozenCandidateUniverse, MembershipGuardBackend};
use symthaea_math_retrieval_runtime_seam::{
    BackendExecution, BackendRetrieval, CanonicalSourceMaterializer, ChannelExecution, ChannelTrace,
    ControlBinding, ControlTransform, FusionExecution, FusionMethod, GraphIdentity,
    InMemoryEvidenceSink, PayloadAuditBatch, PayloadRole, QualifiedRetrievalBackend,
    QualifiedRetrievalRequest, RetrievalBudget, RetrievalChannel, RetrievalError, RetrievalExecutor,
    RetrievalTraceReceipt, Sha256Digest, StopReason, TraceRetrieval,
};

fn required<'a>(m: &'a BTreeMap<String, String>, key: &str) -> &'a str {
    m.get(key).unwrap_or_else(|| panic!("missing config key: {key}"))
}
fn u32v(m: &BTreeMap<String, String>, k: &str) -> u32 { required(m,k).parse().unwrap() }
fn u64v(m: &BTreeMap<String, String>, k: &str) -> u64 { required(m,k).parse().unwrap() }
fn usizev(m: &BTreeMap<String, String>, k: &str) -> usize { required(m,k).parse().unwrap() }
fn dig(m: &BTreeMap<String, String>, k: &str) -> Sha256Digest { Sha256Digest::parse(required(m,k)).unwrap() }
fn digests(m: &BTreeMap<String,String>, k: &str) -> Vec<Sha256Digest> {
    required(m,k).split(',').map(|x| Sha256Digest::parse(x).unwrap()).collect()
}
fn fixed(n:u8)->Sha256Digest { Sha256Digest::parse(format!("sha256:{n:064x}")).unwrap() }

fn load(path:&Path)->BTreeMap<String,String>{
    fs::read_to_string(path).unwrap().lines().filter(|x|!x.trim().is_empty()&&!x.trim().starts_with('#')).map(|line|{
        let (k,v)=line.split_once('=').expect("key=value");(k.trim().into(),v.trim().into())
    }).collect()
}
fn j(s:&str)->String{
    let mut o=String::from("\"");
    for c in s.chars(){match c{'"'=>o.push_str("\\\""),'\\'=>o.push_str("\\\\"),'\n'=>o.push_str("\\n"),'\r'=>o.push_str("\\r"),'\t'=>o.push_str("\\t"),c if c<' '=>o.push_str(&format!("\\u{:04x}",c as u32)),c=>o.push(c)}}
    o.push('"');o
}
fn dj(d:&Sha256Digest)->String{j(d.as_str())}
fn dl(xs:&[Sha256Digest])->String{format!("[{}]",xs.iter().map(dj).collect::<Vec<_>>().join(","))}
fn dec(m:u64)->String{let w=m/1_000_000;let f=m%1_000_000;if f==0{return w.to_string()}let mut x=format!("{f:06}");while x.ends_with('0'){x.pop();}format!("{w}.{x}")}
fn channel_json(x:&ChannelTrace)->String{
    format!("{{\"channel\":{},\"index_manifest_sha256\":{},\"index_artifact_sha256\":{},\"requested_k\":{},\"input_bytes_used\":{},\"ranked_source_object_digests\":{}}}",
        j(match x.channel{RetrievalChannel::Syntax=>"Syntax",RetrievalChannel::ExactNormalForm=>"ExactNormalForm"}),dj(&x.index_manifest_sha256),dj(&x.index_artifact_sha256),x.requested_k,x.input_bytes_used,dl(&x.ranked_source_object_digests))
}
fn serialize_trace(t:&RetrievalTraceReceipt)->String{
    let retrieval=match &t.retrieval{
        TraceRetrieval::Fusion{fusion_policy_sha256,channels,fused_ranked_source_object_digests}=>format!("{{\"mode\":\"Fusion\",\"fusion\":{{\"fusion_policy_sha256\":{},\"channels\":[{},{}],\"fused_ranked_source_object_digests\":{}}}}}",dj(fusion_policy_sha256),channel_json(&channels[0]),channel_json(&channels[1]),dl(fused_ranked_source_object_digests)),
        TraceRetrieval::SingleIndex{..}=>panic!("001E requires Fusion"),
    };
    let output=t.packing.output.iter().map(|x|format!("{{\"rank\":{},\"source_object_sha256\":{},\"canonical_payload_bytes\":{}}}",x.rank,dj(&x.source_object_sha256),x.canonical_payload_bytes)).collect::<Vec<_>>().join(",");
    let mut packing=format!("{{\"context_packer_sha256\":{},\"input_ranked_source_object_digests\":{},\"output\":[{}],\"output_items_used\":{},\"output_bytes_used\":{},\"stop_reason\":{}",dj(&t.packing.context_packer_sha256),dl(&t.packing.input_ranked_source_object_digests),output,t.packing.output_items_used,t.packing.output_bytes_used,j(match t.packing.stop_reason{StopReason::Empty=>"Empty",StopReason::RankedCandidatesExhausted=>"RankedCandidatesExhausted",StopReason::ItemLimitReached=>"ItemLimitReached",StopReason::FirstNonFittingItem=>"FirstNonFittingItem"}));
    if let Some(x)=&t.packing.first_nonfitting{packing.push_str(&format!(",\"first_nonfitting\":{{\"rank\":{},\"source_object_sha256\":{},\"canonical_payload_bytes\":{}}}",x.rank,dj(&x.source_object_sha256),x.canonical_payload_bytes));}packing.push('}');
    let controls=t.control_bindings.iter().map(|x|{let tr=match x.control_transform{ControlTransform::None=>"None",ControlTransform::RandomRetrieval=>"RandomRetrieval",ControlTransform::ShuffledHdcVectors=>"ShuffledHdcVectors",ControlTransform::PermutedChallengeAssociations=>"PermutedChallengeAssociations"};format!("{{\"index_manifest_sha256\":{},\"control_transform\":{}}}",dj(&x.index_manifest_sha256),j(tr))}).collect::<Vec<_>>().join(",");
    format!("{{\"version\":{},\"trace_id\":{},\"authority\":{},\"graph\":{{\"bundle_sha256\":{},\"graph_report_sha256\":{},\"experiment_sha256\":{},\"arm_id\":{},\"retrieval_binding_sha256\":{},\"candidate_set_sha256\":{},\"candidate_count\":{},\"context_packer_sha256\":{}}},\"experiment_seed\":{},\"query\":{{\"query_id\":{},\"query_source_object_sha256\":{}}},\"retrieval\":{},\"packing\":{},\"resources\":{{\"retrieval_queries_used\":{},\"wall_time_ms_used\":{},\"normalized_compute_units_used_decimal\":{}}},\"control_bindings\":[{}]}}\n",
        j(t.version),j(&t.trace_id),j(t.authority),dj(&t.graph.bundle_sha256),dj(&t.graph.graph_report_sha256),dj(&t.graph.experiment_sha256),j(&t.arm_id),dj(&t.graph.retrieval_binding_sha256),dj(&t.graph.candidate_set_sha256),t.graph.candidate_count,dj(&t.graph.context_packer_sha256),t.experiment_seed,j(&t.query_id),dj(&t.query_source_object_sha256),retrieval,packing,t.resources.retrieval_queries_used,t.resources.wall_time_ms_used,j(&dec(t.resources.normalized_compute_microunits_used)),controls)
}
fn write_payloads(out:&Path,a:&PayloadAuditBatch){
    let p=out.join("payloads");fs::create_dir_all(&p).unwrap();
    let mut rows=String::from("role\trank\tsource_object_sha256\tpayload_path\tcanonical_payload_bytes\n");
    for e in &a.entries{let role=match e.role{PayloadRole::Delivered=>"Delivered",PayloadRole::FirstNonFitting=>"FirstNonFitting"};let rel=format!("payloads/rank-{:02}.txt",e.rank);fs::write(out.join(&rel),&e.canonical_payload_utf8).unwrap();rows.push_str(&format!("{role}\t{}\t{}\t{rel}\t{}\n",e.rank,e.source_object_sha256,e.canonical_payload_utf8.len()));}fs::write(out.join("payload-plan.tsv"),rows).unwrap();
}

#[derive(Clone)]struct Backend{syntax_index:Sha256Digest,syntax_artifact:Sha256Digest,normal_index:Sha256Digest,normal_artifact:Sha256Digest,fusion_policy:Sha256Digest,syntax:Vec<Sha256Digest>,normal:Vec<Sha256Digest>,rrf_k:u32}
impl QualifiedRetrievalBackend for Backend{
    fn execute(&mut self,_:&QualifiedRetrievalRequest)->Result<BackendExecution,RetrievalError>{
        let ctl=|ix:Sha256Digest|ControlBinding{index_manifest_sha256:ix,control_transform:ControlTransform::None,control_seed:None,control_artifact_sha256:None};
        Ok(BackendExecution{retrieval:BackendRetrieval::Fusion(FusionExecution{fusion_policy_sha256:self.fusion_policy.clone(),method:FusionMethod::ReciprocalRankFusion{rrf_k:self.rrf_k},syntax:ChannelExecution{channel:RetrievalChannel::Syntax,index_manifest_sha256:self.syntax_index.clone(),index_artifact_sha256:self.syntax_artifact.clone(),requested_k:self.syntax.len() as u32,input_bytes_used:64,ranked_source_object_digests:self.syntax.clone(),control:ctl(self.syntax_index.clone())},normal_form:ChannelExecution{channel:RetrievalChannel::ExactNormalForm,index_manifest_sha256:self.normal_index.clone(),index_artifact_sha256:self.normal_artifact.clone(),requested_k:self.normal.len() as u32,input_bytes_used:64,ranked_source_object_digests:self.normal.clone(),control:ctl(self.normal_index.clone())}}),retrieval_queries_used:2,wall_time_ms_used:1,normalized_compute_microunits_used:1_000_000})
    }
}
struct Materializer{payloads:BTreeMap<Sha256Digest,Vec<u8>>,calls:usize}
impl CanonicalSourceMaterializer for Materializer{fn materialize(&mut self,s:&Sha256Digest)->Result<Vec<u8>,RetrievalError>{self.calls+=1;self.payloads.get(s).cloned().ok_or_else(||RetrievalError::Materialization(format!("missing payload {s}")))}}
fn request(c:&BTreeMap<String,String>)->QualifiedRetrievalRequest{QualifiedRetrievalRequest{trace_id:"math-ret-runtime-001e-fixture-trace".into(),audit_id:"math-ret-runtime-001e-fixture-audit".into(),experiment_id:required(c,"experiment_id").into(),arm_id:"F".into(),experiment_seed:u64v(c,"experiment_seed"),query_id:"fixture-query-001".into(),query_source_object_sha256:fixed(0xf0),graph:GraphIdentity{bundle_sha256:dig(c,"bundle_sha256"),graph_report_sha256:dig(c,"graph_report_sha256"),experiment_sha256:dig(c,"experiment_sha256"),retrieval_binding_sha256:dig(c,"retrieval_binding_sha256"),candidate_set_sha256:dig(c,"candidate_set_sha256"),candidate_count:usizev(c,"candidate_count"),context_packer_sha256:dig(c,"context_packer_sha256"),source_object_contract_sha256:dig(c,"source_object_contract_sha256"),source_fetch_policy_sha256:dig(c,"source_fetch_policy_sha256"),payload_serialization_sha256:dig(c,"payload_serialization_sha256")},budget:RetrievalBudget{max_output_items:usizev(c,"max_output_items"),max_output_bytes:usizev(c,"max_output_bytes"),max_output_item_bytes:usizev(c,"max_output_item_bytes"),max_retrieval_queries:u32v(c,"max_retrieval_queries"),max_normalized_compute_microunits:u64v(c,"max_normalized_compute_microunits"),max_wall_time_ms:u64v(c,"max_wall_time_ms")}}}

fn main(){
    let mut a=env::args_os();let _=a.next();let cfg=PathBuf::from(a.next().expect("config"));let out=PathBuf::from(a.next().expect("out-dir"));let bad=matches!(a.next().as_deref(),Some(x) if x==std::ffi::OsStr::new("--inject-illegal-normal"));assert!(a.next().is_none());
    let c=load(&cfg);let u=FrozenCandidateUniverse::new(dig(&c,"candidate_set_sha256"),digests(&c,"candidate_source_sha256s")).unwrap();assert_eq!(u.candidate_count(),usizev(&c,"candidate_count"));
    let syntax=vec![fixed(0xa1),fixed(0xa2)];let mut normal=vec![fixed(0xa2),fixed(0xa3)];if bad{normal[1]=fixed(0xee)}
    let inner=Backend{syntax_index:dig(&c,"syntax_index_manifest_sha256"),syntax_artifact:dig(&c,"syntax_index_artifact_sha256"),normal_index:dig(&c,"normal_index_manifest_sha256"),normal_artifact:dig(&c,"normal_index_artifact_sha256"),fusion_policy:dig(&c,"fusion_policy_sha256"),syntax,normal,rrf_k:u32v(&c,"rrf_k")};
    let mut backend=MembershipGuardBackend::new(inner,u);let mut mat=Materializer{payloads:BTreeMap::from([(fixed(0xa1),b"alpha theorem context\n".to_vec()),(fixed(0xa2),b"beta lemma context\n".to_vec()),(fixed(0xa3),b"gamma definition context\n".to_vec())]),calls:0};let mut sink=InMemoryEvidenceSink::default();let req=request(&c);
    if bad{match RetrievalExecutor::execute(&mut backend,&mut mat,&mut sink,req){Err(RetrievalError::Candidate(_))=>{},Err(e)=>panic!("wrong negative failure: {e}"),Ok(_)=>panic!("illegal normal-form candidate qualified")};assert_eq!(mat.calls,0);assert!(sink.committed().is_empty());assert!(!sink.has_staged_state());assert!(!out.exists());println!("guarded F illegal-normal candidate canary: PASS");return}
    let o=RetrievalExecutor::execute(&mut backend,&mut mat,&mut sink,req).expect("guarded F execution");let expected=vec![fixed(0xa2),fixed(0xa1),fixed(0xa3)];match &o.trace.retrieval{TraceRetrieval::Fusion{fused_ranked_source_object_digests,..}=>assert_eq!(fused_ranked_source_object_digests,&expected),_=>panic!("expected fusion")};assert_eq!(o.trace.packing.input_ranked_source_object_digests,expected);assert_eq!(mat.calls,3);assert_eq!(sink.committed().len(),1);fs::create_dir_all(&out).unwrap();fs::write(out.join("trace.json"),serialize_trace(&o.trace)).unwrap();write_payloads(&out,&o.payload_audit);
}
