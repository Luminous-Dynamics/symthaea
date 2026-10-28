//! Complete, deterministic PITB family-to-component mapping for conservative resampling.
use std::{collections::{BTreeMap,BTreeSet,VecDeque},env,fs};
use anyhow::{Context,Result,bail};
use serde::{Deserialize,Serialize};
use serde_json::Value;
use sha2::{Digest,Sha256};
const UPSTREAM:&str="1581d4cbdd69a0faf34b2089614fe2a814316a6a";
const SEED:&str="pitb-v0-openai-math-2026-10-19";
const ALGORITHM:&str="sha256(seed || NUL || family_id), sort by digest then family ID, floor 70% train and floor 85% train+validation";
const PARTITION:&str="e7ac417556cbd47cee1ddd51b98f9684fadfc8cd097587ae3beacc4dfb59f970";
#[derive(Debug,Deserialize)]
#[serde(deny_unknown_fields)]
struct Inventory{schema_version:String,source_partition_digest_sha256:String,coverage_status:String,coverage_receipt_sha256:Value,edges:Vec<Edge>}
#[derive(Debug,Deserialize)]
#[serde(deny_unknown_fields)]
struct Edge{edge_id:String,source_family_id:String,target_family_id:String,relation_kind:String,assessment_status:String,evidence_sha256:String,reviewer_ids:Vec<String>}
#[derive(Debug,Serialize)]
struct Counts{train:usize,validation:usize,test:usize}
#[derive(Debug,Serialize)]
struct EdgeCounts{total_edges:usize,hard_edges_included:usize,confirmed_hard_edges:usize,unresolved_hard_edges:usize,dismissed_hard_edges:usize,soft_edges:usize,cross_split_hard_edges:usize,unresolved_cross_split_hard_edges:usize}
#[derive(Debug,Serialize)]
struct Component{component_id:String,split_names:Vec<String>,family_ids:Vec<String>,hard_edge_ids:Vec<String>,confirmed_edge_ids:Vec<String>,unresolved_edge_ids:Vec<String>}
#[derive(Debug,Serialize)]
struct FamilyRow{family_id:String,split:String,component_id:String}
#[derive(Debug,Serialize)]
struct Manifest{schema_version:&'static str,upstream_commit:String,source_partition_digest_sha256:String,relationship_inventory_sha256:String,relationship_inventory_status:String,coverage_receipt_sha256:Option<String>,total_family_count:usize,total_component_count:usize,independent_component_count_by_split:Counts,cross_split_component_count:usize,edge_counts:EdgeCounts,families:Vec<FamilyRow>,components:Vec<Component>,qualification_status:String,coverage_receipt_verification_status:&'static str,qualified_for_confirmatory_resampling:bool,qualification_limitations:Vec<String>}
fn sha(b:&[u8])->String{format!("{:x}",Sha256::digest(b))}
fn digest_ok(s:&str)->bool{s.len()==64&&s.bytes().all(|b|b.is_ascii_hexdigit())}
fn hard(s:&str)->bool{matches!(s,"exact_duplicate"|"shared_source_artifact"|"formal_dependency"|"reused_lemma"|"equivalent_statement")}
fn deterministic(ids:&[String])->Result<BTreeMap<String,String>>{
 if ids.len()<3{bail!("at least three family IDs required")}let mut unique=BTreeSet::new();for id in ids{if id.trim().is_empty()||!unique.insert(id.clone()){bail!("family IDs must be unique and nonblank")}}
 let mut ranked:Vec<(String,String)>=ids.iter().map(|id|(sha(format!("{SEED}\0{id}").as_bytes()),id.clone())).collect();ranked.sort_unstable();let n=ranked.len();let tr=n*70/100;let va=n*85/100;
 Ok(ranked.into_iter().enumerate().map(|(i,(_,id))|(id,(if i<tr{"train"}else if i<va{"validation"}else{"test"}).to_owned())).collect())
}
fn partition_digest(m:&BTreeMap<String,String>)->String{let mut s=String::new();for(id,split)in m{s.push_str(id);s.push('\t');s.push_str(split);s.push('\n')}sha(s.as_bytes())}
fn validate_splits(doc:&Value)->Result<(String,BTreeMap<String,String>)>{
 if doc["schema_version"].as_str()!=Some("symthaea.proof-insight-transfer.splits.v0")||doc["upstream"]["commit"].as_str()!=Some(UPSTREAM){bail!("wrong pinned split manifest")}
 let p=&doc["split_plan"];if p["seed"].as_str()!=Some(SEED)||p["algorithm"].as_str()!=Some(ALGORITHM)||p["train_percent"].as_u64()!=Some(70)||p["validation_percent"].as_u64()!=Some(15)||p["test_percent"].as_u64()!=Some(15){bail!("unregistered split plan")}
 let rows=doc["families"].as_array().context("families must be an array")?;let mut m=BTreeMap::new();for r in rows{let id=r["family_id"].as_str().context("missing family_id")?;let split=r["split"].as_str().context("missing split")?;if id.trim().is_empty()||!matches!(split,"train"|"validation"|"test")||m.insert(id.to_owned(),split.to_owned()).is_some(){bail!("invalid/duplicate family row")}}
 let ids:Vec<String>=m.keys().cloned().collect();if m!=deterministic(&ids)?{bail!("split does not reproduce deterministic assignment")}let d=partition_digest(&m);
 if d!=PARTITION||p["partition_digest_sha256"].as_str()!=Some(d.as_str())||m.len()!=372{bail!("pinned partition digest/count mismatch")}
 let counts=(m.values().filter(|x|*x=="train").count(),m.values().filter(|x|*x=="validation").count(),m.values().filter(|x|*x=="test").count());if counts!=(260,56,56)||p["counts"]["train"].as_u64()!=Some(260)||p["counts"]["validation"].as_u64()!=Some(56)||p["counts"]["test"].as_u64()!=Some(56){bail!("split counts do not reconcile")}Ok((d,m))
}
fn validate_edges(inv:&Inventory,d:&str,splits:&BTreeMap<String,String>)->Result<Option<String>>{
 if inv.schema_version!="pitb.family-leakage-edges.v0"||!digest_ok(&inv.source_partition_digest_sha256)||inv.source_partition_digest_sha256!=d{bail!("edge inventory schema/partition mismatch")}
 let receipt=if inv.coverage_receipt_sha256.is_null(){None}else{let x=inv.coverage_receipt_sha256.as_str().context("receipt must be a digest or null")?;if !digest_ok(x){bail!("bad coverage receipt digest")}Some(x.to_owned())};match inv.coverage_status.as_str(){"incomplete"if receipt.is_none()=>{},"incomplete"=>bail!("incomplete inventory cannot have receipt"),"independently_reviewed_complete"if receipt.is_some()=>{},"independently_reviewed_complete"=>bail!("complete inventory requires receipt"),x=>bail!("unknown coverage status {x}")}
 let mut edge_ids=BTreeSet::new();let mut pairs=BTreeSet::new();for e in &inv.edges{if e.edge_id.trim().is_empty()||!edge_ids.insert(e.edge_id.clone()){bail!("duplicate/blank edge ID")}if e.source_family_id==e.target_family_id||!splits.contains_key(&e.source_family_id)||!splits.contains_key(&e.target_family_id){bail!("self-edge or unknown family")}if !digest_ok(&e.evidence_sha256){bail!("bad evidence digest")}
 if !matches!(e.relation_kind.as_str(),"exact_duplicate"|"shared_source_artifact"|"formal_dependency"|"reused_lemma"|"equivalent_statement"|"conceptual_similarity"|"citation_only")||!matches!(e.assessment_status.as_str(),"confirmed"|"dismissed"|"unresolved"){bail!("unregistered edge relation/status")}
 let rv:BTreeSet<&str>=e.reviewer_ids.iter().map(String::as_str).collect();if e.reviewer_ids.iter().any(|x|x.trim().is_empty())||rv.len()!=e.reviewer_ids.len(){bail!("invalid reviewer IDs")}if e.assessment_status!="unresolved"&&rv.len()<2{bail!("resolved edge needs two reviewers")}
 let(a,b)=if e.source_family_id<e.target_family_id{(&e.source_family_id,&e.target_family_id)}else{(&e.target_family_id,&e.source_family_id)};if !pairs.insert(format!("{a}\0{b}\0{}",e.relation_kind)){bail!("duplicate unordered pair/relation edge")}}
 Ok(receipt)
}
fn component_id(ids:&[String])->String{let mut s=ids.join("\n");s.push('\n');format!("component-{}",sha(s.as_bytes()))}
fn components(splits:&BTreeMap<String,String>,edges:&[Edge])->Vec<Component>{
 let mut adj:BTreeMap<String,BTreeSet<String>>=splits.keys().map(|id|(id.clone(),BTreeSet::new())).collect();for e in edges{if hard(&e.relation_kind)&&e.assessment_status!="dismissed"{adj.get_mut(&e.source_family_id).unwrap().insert(e.target_family_id.clone());adj.get_mut(&e.target_family_id).unwrap().insert(e.source_family_id.clone());}}
 let mut seen=BTreeSet::new();let mut out=Vec::new();for start in splits.keys(){if !seen.insert(start.clone()){continue}let mut q=VecDeque::from([start.clone()]);let mut members=BTreeSet::from([start.clone()]);while let Some(id)=q.pop_front(){for next in &adj[&id]{if seen.insert(next.clone()){members.insert(next.clone());q.push_back(next.clone());}}}
 let family_ids:Vec<String>=members.into_iter().collect();let split_names:Vec<String>=family_ids.iter().map(|id|splits[id].clone()).collect::<BTreeSet<_>>().into_iter().collect();let set:BTreeSet<&str>=family_ids.iter().map(String::as_str).collect();let rel:Vec<&Edge>=edges.iter().filter(|e|hard(&e.relation_kind)&&e.assessment_status!="dismissed"&&set.contains(e.source_family_id.as_str())&&set.contains(e.target_family_id.as_str())).collect();
 let sort=|mut v:Vec<String>|{v.sort();v};let all=sort(rel.iter().map(|e|e.edge_id.clone()).collect());let conf=sort(rel.iter().filter(|e|e.assessment_status=="confirmed").map(|e|e.edge_id.clone()).collect());let unresolved=sort(rel.iter().filter(|e|e.assessment_status=="unresolved").map(|e|e.edge_id.clone()).collect());out.push(Component{component_id:component_id(&family_ids),split_names,family_ids,hard_edge_ids:all,confirmed_edge_ids:conf,unresolved_edge_ids:unresolved})}
 out.sort_by(|a,b|a.family_ids.cmp(&b.family_ids));out
}
fn build(doc:&Value,inv:&Inventory,raw:&[u8])->Result<Manifest>{
 let(d,splits)=validate_splits(doc)?;let receipt=validate_edges(inv,&d,&splits)?;let comps=components(&splits,&inv.edges);let mut ec=EdgeCounts{total_edges:inv.edges.len(),hard_edges_included:0,confirmed_hard_edges:0,unresolved_hard_edges:0,dismissed_hard_edges:0,soft_edges:0,cross_split_hard_edges:0,unresolved_cross_split_hard_edges:0};
 for e in &inv.edges{if !hard(&e.relation_kind){ec.soft_edges+=1;continue}match e.assessment_status.as_str(){"confirmed"=>ec.confirmed_hard_edges+=1,"unresolved"=>ec.unresolved_hard_edges+=1,"dismissed"=>ec.dismissed_hard_edges+=1,_=>unreachable!()}if e.assessment_status=="dismissed"{continue}ec.hard_edges_included+=1;if splits[&e.source_family_id]!=splits[&e.target_family_id]{ec.cross_split_hard_edges+=1;if e.assessment_status=="unresolved"{ec.unresolved_cross_split_hard_edges+=1}}}
 let cross=comps.iter().filter(|c|c.split_names.len()>1).count();let ok=false;let status=if inv.coverage_status!="independently_reviewed_complete"{"not_qualified_incomplete_inventory"}else if cross>0{"not_qualified_cross_split_hard_relationships"}else{"structurally_eligible_receipt_unverified"}.to_owned();
 let mut counts=Counts{train:0,validation:0,test:0};let mut rows=Vec::new();for c in &comps{if c.split_names.len()==1{match c.split_names[0].as_str(){"train"=>counts.train+=1,"validation"=>counts.validation+=1,"test"=>counts.test+=1,_=>unreachable!()}}for id in &c.family_ids{rows.push(FamilyRow{family_id:id.clone(),split:splits[id].clone(),component_id:c.component_id.clone()})}}rows.sort_by(|a,b|a.family_id.cmp(&b.family_id));
 Ok(Manifest{schema_version:"pitb.independent-component-manifest.v0",upstream_commit:UPSTREAM.to_owned(),source_partition_digest_sha256:d,relationship_inventory_sha256:sha(raw),relationship_inventory_status:inv.coverage_status.clone(),coverage_receipt_sha256:receipt,total_family_count:splits.len(),total_component_count:comps.len(),independent_component_count_by_split:counts,cross_split_component_count:cross,edge_counts:ec,families:rows,components:comps,qualification_status:status,coverage_receipt_verification_status:"not_verified_receipt_bytes_not_supplied",qualified_for_confirmatory_resampling:ok,qualification_limitations:vec!["Only the digest was supplied; the underlying coverage receipt bytes and reviewer attestations were not verified, so this builder never authorizes confirmatory resampling.".into(),"Unresolved non-dismissed hard links are joined; any cross-split component blocks qualification.".into(),"Primary scorer, power planner, and test bootstrap must bind and consume this exact manifest digest before confirmatory qualification.".into()]})
}
fn main()->Result<()>{let a:Vec<String>=env::args().skip(1).collect();if a.len()!=3{bail!("usage: build_family_component_manifest <splits.json> <edges.json> <manifest.json>")}let sb=fs::read(&a[0]).with_context(||format!("cannot read {}",a[0]))?;let eb=fs::read(&a[1]).with_context(||format!("cannot read {}",a[1]))?;let doc:Value=serde_json::from_slice(&sb).context("invalid split JSON")?;let inv:Inventory=serde_json::from_slice(&eb).context("invalid relationship JSON")?;let m=build(&doc,&inv,&eb)?;fs::write(&a[2],serde_json::to_vec_pretty(&m)?).with_context(||format!("cannot write {}",a[2]))?;println!("PITB_COMPONENT_MANIFEST: qualification={} families={} components={} cross_split_components={} coverage={}",m.qualification_status,m.total_family_count,m.total_component_count,m.cross_split_component_count,m.relationship_inventory_status);Ok(())}
#[cfg(test)]mod tests{use super::*;fn e(id:&str,a:&str,b:&str,k:&str,s:&str)->Edge{Edge{edge_id:id.into(),source_family_id:a.into(),target_family_id:b.into(),relation_kind:k.into(),assessment_status:s.into(),evidence_sha256:"a".repeat(64),reviewer_ids:if s=="unresolved"{vec![]}else{vec!["r1".into(),"r2".into()]}}}fn sm(p:&[(&str,&str)])->BTreeMap<String,String>{p.iter().map(|(a,b)|((*a).into(),(*b).into())).collect()}
#[test]fn includes_singletons(){let s=sm(&[("001","train"),("002","train"),("003","validation"),("004","test")]);let c=components(&s,&[e("e1","001","002","formal_dependency","confirmed")]);assert_eq!(c.len(),3);assert_eq!(c.iter().map(|x|x.family_ids.len()).sum::<usize>(),4);assert!(c.iter().any(|x|x.family_ids==vec!["004".to_owned()]));}
#[test]fn unresolved_hard_joins_soft_and_dismissed_do_not(){let s=sm(&[("001","train"),("002","train"),("003","train"),("004","train")]);let es=vec![e("e1","001","002","reused_lemma","unresolved"),e("e2","002","003","conceptual_similarity","confirmed"),e("e3","003","004","formal_dependency","dismissed")];let c=components(&s,&es);assert_eq!(c.len(),3);assert_eq!(c[0].family_ids,vec!["001".to_owned(),"002".to_owned()]);assert_eq!(c[0].unresolved_edge_ids,vec!["e1".to_owned()]);}
#[test]fn cross_split_link_is_one_component(){let s=sm(&[("001","train"),("002","test"),("003","train")]);let c=components(&s,&[e("e1","001","002","formal_dependency","confirmed")]);assert!(c.iter().any(|x|x.family_ids==vec!["001".to_owned(),"002".to_owned()]&&x.split_names.len()==2));}
#[test]fn ids_are_content_addressed(){let x=vec!["001".to_owned(),"002".to_owned()];assert_eq!(component_id(&x),component_id(&x));assert_ne!(component_id(&x),component_id(&vec!["001".to_owned(),"003".to_owned()]));}}
