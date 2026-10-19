// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Synthetic quantity genealogy across partial consumption, splitting and merging.
//! Source allocation is explicit; this validator never infers mixing proportions.
use std::collections::{BTreeMap, BTreeSet, HashSet};
use serde_json::Value;

pub const EMBEDDED_QUANTITY_GENEALOGY: &str =
    include_str!("../../../../docs/research/fixtures/habitat-quantity-genealogy-v0.1.json");

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuantityGenealogyIssue { pub code: &'static str, pub path: String, pub message: String }
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuantityGenealogyReport {
    pub issues: Vec<QuantityGenealogyIssue>,
    pub operation_count: usize,
    pub initial_mass_mg: u64,
    pub ending_mass_mg: u64,
    pub terminal_count: usize,
    pub affected_terminal_ids: Vec<String>,
    pub remaining_balance_mg: BTreeMap<String, u64>,
    pub source_lineage_total_mg: BTreeMap<String, u64>,
    pub lineage_by_material: BTreeMap<String, BTreeMap<String, u64>>,
}
impl QuantityGenealogyReport { pub fn is_valid(&self) -> bool { self.issues.is_empty() } }
pub fn validate_quantity_genealogy_str(input: &str) -> Result<QuantityGenealogyReport, serde_json::Error> {
    let root: Value = serde_json::from_str(input)?;
    Ok(validate_quantity_genealogy(&root))
}

pub fn validate_quantity_genealogy(root: &Value) -> QuantityGenealogyReport {
    let mut issues=Vec::new();
    expect_string(root,"schema_version","0.1.0",&mut issues);
    expect_string(root,"campaign_id","HABITAT-QUANTITY-GENEALOGY-001",&mut issues);
    expect_string(root,"program_id","HABITAT-VERTICAL-001",&mut issues);
    expect_string(root,"evidence_status","synthetic_quantity_genealogy_challenge_only_not_executed",&mut issues);
    expect_string(root,"mass_unit","mg",&mut issues);
    if !root.get("claim_ceiling").and_then(Value::as_str).is_some_and(|s|s.contains("never infers mixing proportions")) {
        push(&mut issues,"CLAIM_CEILING_MISSING","claim_ceiling","explicit source allocations and the physical-evidence ceiling are required");
    }
    let suspect=root.get("source_suspect_lot_id").and_then(Value::as_str).unwrap_or_default().to_owned();
    let mut available=BTreeMap::<String,u64>::new();
    let mut kinds=BTreeMap::<String,String>::new();
    let mut lineages=BTreeMap::<String,BTreeMap<String,u64>>::new();
    let mut opening_by_source=BTreeMap::<String,u64>::new();
    let mut opening_total=0u64;
    let Some(initial)=root.get("initial_inventory").and_then(Value::as_array) else {
        push(&mut issues,"INITIAL_INVENTORY_MISSING","initial_inventory","initial inventory must be an array");
        return finish(issues,0,0,0,0,Vec::new(),BTreeMap::new(),BTreeMap::new(),BTreeMap::new());
    };
    for (i,row) in initial.iter().enumerate() {
        let path=format!("initial_inventory[{i}]");
        let Some(id)=row.get("material_id").and_then(Value::as_str).filter(|s|!s.trim().is_empty()) else {
            push(&mut issues,"MATERIAL_ID_MISSING",format!("{path}.material_id"),"material ID must be non-empty"); continue;
        };
        if available.contains_key(id) {
            push(&mut issues,"DUPLICATE_MATERIAL_ID",format!("{path}.material_id"),format!("duplicate initial material {id:?}")); continue;
        }
        if row.get("material_kind").and_then(Value::as_str)!=Some("source_lot") {
            push(&mut issues,"INITIAL_KIND_UNSUPPORTED",format!("{path}.material_kind"),"initial entries must be source_lot records");
        }
        if row.get("unit").and_then(Value::as_str)!=Some("mg") { push(&mut issues,"UNIT_MISMATCH",format!("{path}.unit"),"all quantities must use mg"); }
        let Some(qty)=row.get("opening_mg").and_then(Value::as_u64).filter(|q|*q>0) else {
            push(&mut issues,"QUANTITY_INVALID",format!("{path}.opening_mg"),"opening quantity must be positive"); continue;
        };
        available.insert(id.to_owned(),qty); kinds.insert(id.to_owned(),"source_lot".to_owned());
        lineages.insert(id.to_owned(),BTreeMap::from([(id.to_owned(),qty)]));
        opening_by_source.insert(id.to_owned(),qty);
        add_checked(&mut opening_total,qty,&mut issues,format!("{path}.opening_mg"));
    }
    if !opening_by_source.contains_key(&suspect) { push(&mut issues,"SUSPECT_SOURCE_UNKNOWN","source_suspect_lot_id",format!("unknown suspect source {suspect:?}")); }
    let Some(operations)=root.get("operations").and_then(Value::as_array) else {
        push(&mut issues,"OPERATIONS_MISSING","operations","operations must be an array");
        return finish(issues,0,opening_total,0,0,Vec::new(),BTreeMap::new(),BTreeMap::new(),lineages);
    };
    let mut op_ids=HashSet::new();
    for (op_i,op) in operations.iter().enumerate() {
        let prefix=format!("operations[{op_i}]"); let op_issue_start=issues.len();
        let Some(op_id)=op.get("operation_id").and_then(Value::as_str).filter(|s|!s.trim().is_empty()) else {
            push(&mut issues,"OPERATION_ID_MISSING",format!("{prefix}.operation_id"),"operation ID must be non-empty"); continue;
        };
        if !op_ids.insert(op_id.to_owned()) { push(&mut issues,"DUPLICATE_OPERATION_ID",format!("{prefix}.operation_id"),format!("duplicate operation {op_id:?}")); }
        if op.get("operation_kind").and_then(Value::as_str).is_none_or(|s|s.trim().is_empty()) { push(&mut issues,"OPERATION_KIND_MISSING",format!("{prefix}.operation_kind"),"operation kind must be non-empty"); }
        let Some(ins)=op.get("inputs").and_then(Value::as_array) else { push(&mut issues,"INPUTS_MISSING",format!("{prefix}.inputs"),"inputs must be an array"); continue; };
        let Some(outs)=op.get("outputs").and_then(Value::as_array) else { push(&mut issues,"OUTPUTS_MISSING",format!("{prefix}.outputs"),"outputs must be an array"); continue; };
        let Some(allocs)=op.get("allocations").and_then(Value::as_array) else { push(&mut issues,"ALLOCATIONS_MISSING",format!("{prefix}.allocations"),"allocations must be an array"); continue; };
        if ins.is_empty()||outs.is_empty() { push(&mut issues,"OPERATION_ENDPOINTS_EMPTY",&prefix,"each operation needs inputs and outputs"); }
        let mut input_qty=BTreeMap::<String,u64>::new(); let mut output_qty=BTreeMap::<String,u64>::new(); let mut output_kind=BTreeMap::<String,String>::new();
        for (i,row) in ins.iter().enumerate() {
            let p=format!("{prefix}.inputs[{i}]");
            let Some(id)=row.get("material_id").and_then(Value::as_str).filter(|s|!s.trim().is_empty()) else { push(&mut issues,"MATERIAL_ID_MISSING",format!("{p}.material_id"),"input ID must be non-empty"); continue; };
            if input_qty.contains_key(id) { push(&mut issues,"DUPLICATE_OPERATION_INPUT",format!("{p}.material_id"),"input may occur only once per operation"); continue; }
            let Some(qty)=row.get("quantity_mg").and_then(Value::as_u64).filter(|q|*q>0) else { push(&mut issues,"QUANTITY_INVALID",format!("{p}.quantity_mg"),"input quantity must be positive"); continue; };
            if row.get("unit").and_then(Value::as_str)!=Some("mg") { push(&mut issues,"UNIT_MISMATCH",format!("{p}.unit"),"input unit must be mg"); }
            match available.get(id) {
                Some(on_hand) if *on_hand>=qty => {},
                Some(on_hand) => push(&mut issues,"INPUT_OVERDRAW",format!("{p}.quantity_mg"),format!("requested {qty} mg from {id:?}, only {on_hand} mg available")),
                None => push(&mut issues,"INPUT_MATERIAL_UNKNOWN_OR_FUTURE",format!("{p}.material_id"),format!("material {id:?} is undeclared or not yet produced")),
            }
            if kinds.get(id).is_some_and(|k|k=="packaged_unit") { push(&mut issues,"TERMINAL_RECONSUMED",format!("{p}.material_id"),"terminal packaged product cannot be reconsumed in this fixture"); }
            input_qty.insert(id.to_owned(),qty);
        }
        for (i,row) in outs.iter().enumerate() {
            let p=format!("{prefix}.outputs[{i}]");
            let Some(id)=row.get("material_id").and_then(Value::as_str).filter(|s|!s.trim().is_empty()) else { push(&mut issues,"MATERIAL_ID_MISSING",format!("{p}.material_id"),"output ID must be non-empty"); continue; };
            if available.contains_key(id)||output_qty.contains_key(id)||kinds.contains_key(id) { push(&mut issues,"DUPLICATE_MATERIAL_ID",format!("{p}.material_id"),format!("output ID {id:?} is not fresh")); continue; }
            let Some(qty)=row.get("quantity_mg").and_then(Value::as_u64).filter(|q|*q>0) else { push(&mut issues,"QUANTITY_INVALID",format!("{p}.quantity_mg"),"output quantity must be positive"); continue; };
            if row.get("unit").and_then(Value::as_str)!=Some("mg") { push(&mut issues,"UNIT_MISMATCH",format!("{p}.unit"),"output unit must be mg"); }
            let Some(kind)=row.get("material_kind").and_then(Value::as_str).filter(|s|!s.trim().is_empty()) else { push(&mut issues,"MATERIAL_KIND_MISSING",format!("{p}.material_kind"),"output kind must be non-empty"); continue; };
            output_qty.insert(id.to_owned(),qty); output_kind.insert(id.to_owned(),kind.to_owned());
        }
        let mut input_alloc=BTreeMap::<String,u64>::new(); let mut output_alloc=BTreeMap::<String,u64>::new();
        let mut source_used=BTreeMap::<String,BTreeMap<String,u64>>::new(); let mut source_out=BTreeMap::<String,BTreeMap<String,u64>>::new();
        let mut pairs=HashSet::<(String,String)>::new();
        for (i,a) in allocs.iter().enumerate() {
            let p=format!("{prefix}.allocations[{i}]");
            let Some(input_id)=a.get("input_material_id").and_then(Value::as_str) else { push(&mut issues,"ALLOCATION_INPUT_MISSING",format!("{p}.input_material_id"),"input ID must be string"); continue; };
            let Some(output_id)=a.get("output_material_id").and_then(Value::as_str) else { push(&mut issues,"ALLOCATION_OUTPUT_MISSING",format!("{p}.output_material_id"),"output ID must be string"); continue; };
            if !input_qty.contains_key(input_id) { push(&mut issues,"ALLOCATION_INPUT_UNKNOWN",format!("{p}.input_material_id"),"allocation input is not an operation input"); }
            if !output_qty.contains_key(output_id) { push(&mut issues,"ALLOCATION_OUTPUT_UNKNOWN",format!("{p}.output_material_id"),"allocation output is not an operation output"); }
            if !pairs.insert((input_id.to_owned(),output_id.to_owned())) { push(&mut issues,"DUPLICATE_INPUT_OUTPUT_ALLOCATION",&p,"each input/output pair must occur once"); }
            let Some(qty)=a.get("quantity_mg").and_then(Value::as_u64).filter(|q|*q>0) else { push(&mut issues,"QUANTITY_INVALID",format!("{p}.quantity_mg"),"allocation quantity must be positive"); continue; };
            add_map(&mut input_alloc,input_id.to_owned(),qty,&mut issues,format!("{p}.quantity_mg"));
            add_map(&mut output_alloc,output_id.to_owned(),qty,&mut issues,format!("{p}.quantity_mg"));
            let Some(ls)=a.get("lineage").and_then(Value::as_array) else { push(&mut issues,"ALLOCATION_LINEAGE_MISSING",format!("{p}.lineage"),"explicit source quantities required; never infer a mixing ratio"); continue; };
            let mut seen=HashSet::new(); let mut ltotal=0u64;
            for (j,l) in ls.iter().enumerate() {
                let lp=format!("{p}.lineage[{j}]");
                let Some(src)=l.get("source_lot_id").and_then(Value::as_str).filter(|s|!s.trim().is_empty()) else { push(&mut issues,"SOURCE_ID_MISSING",format!("{lp}.source_lot_id"),"source ID must be non-empty"); continue; };
                if !opening_by_source.contains_key(src) { push(&mut issues,"SOURCE_LOT_UNKNOWN",format!("{lp}.source_lot_id"),format!("unknown source lot {src:?}")); }
                if !seen.insert(src.to_owned()) { push(&mut issues,"DUPLICATE_SOURCE_LINEAGE",format!("{lp}.source_lot_id"),"source may occur only once per allocation"); }
                let Some(sqty)=l.get("quantity_mg").and_then(Value::as_u64).filter(|q|*q>0) else { push(&mut issues,"QUANTITY_INVALID",format!("{lp}.quantity_mg"),"source quantity must be positive"); continue; };
                add_checked(&mut ltotal,sqty,&mut issues,format!("{lp}.quantity_mg"));
                add_nested(&mut source_used,input_id.to_owned(),src.to_owned(),sqty,&mut issues,format!("{lp}.quantity_mg"));
                add_nested(&mut source_out,output_id.to_owned(),src.to_owned(),sqty,&mut issues,format!("{lp}.quantity_mg"));
            }
            if ltotal!=qty { push(&mut issues,"ALLOCATION_LINEAGE_MISMATCH",format!("{p}.lineage"),format!("source lineage totals {ltotal} mg, allocation transfers {qty} mg")); }
        }
        let itotal=sum(input_qty.values().copied(),&mut issues,format!("{prefix}.inputs"));
        let ototal=sum(output_qty.values().copied(),&mut issues,format!("{prefix}.outputs"));
        if itotal!=ototal { push(&mut issues,"OPERATION_MASS_IMBALANCE",&prefix,format!("input {itotal} mg differs from output {ototal} mg; scrap/residue must be explicit outputs")); }
        for (id,qty) in &input_qty {
            if input_alloc.get(id).copied().unwrap_or(0)!=*qty { push(&mut issues,"INPUT_ALLOCATION_MISMATCH",format!("{prefix}.inputs.{id}"),"allocations must account for the full input quantity"); }
            let usage=source_used.get(id).cloned().unwrap_or_default(); let input_lineage=lineages.get(id).cloned().unwrap_or_default();
            for (src,used) in usage {
                if !input_lineage.contains_key(&src) { push(&mut issues,"SOURCE_LINEAGE_UNAVAILABLE",format!("{prefix}.inputs.{id}"),format!("input does not carry source lineage {src:?}")); }
                else if input_lineage.get(&src).copied().unwrap_or(0)<used { push(&mut issues,"SOURCE_LINEAGE_OVERDRAW",format!("{prefix}.inputs.{id}"),format!("source {src:?} is overdrawn")); }
            }
        }
        for (id,qty) in &output_qty {
            if output_alloc.get(id).copied().unwrap_or(0)!=*qty { push(&mut issues,"OUTPUT_ALLOCATION_MISMATCH",format!("{prefix}.outputs.{id}"),"allocations must account for full output quantity"); }
            let lt=sum(source_out.get(id).into_iter().flat_map(|m|m.values().copied()),&mut issues,format!("{prefix}.outputs.{id}.lineage"));
            if lt!=*qty { push(&mut issues,"OUTPUT_LINEAGE_MISMATCH",format!("{prefix}.outputs.{id}"),"explicit source quantities must equal output quantity"); }
        }
        if issues.len()==op_issue_start {
            for (id,qty) in &input_qty {
                if let Some(onhand)=available.get_mut(id) { *onhand-=*qty; }
                if let Some(current)=lineages.get_mut(id) {
                    if let Some(used)=source_used.get(id) {
                        for (src,qty) in used { if let Some(q)=current.get_mut(src) {*q-=*qty;} }
                    }
                    current.retain(|_,q|*q>0);
                }
            }
            for (id,qty) in &output_qty {
                available.insert(id.clone(),*qty); kinds.insert(id.clone(),output_kind[id].clone());
                lineages.insert(id.clone(),source_out.get(id).cloned().unwrap_or_default());
            }
        }
    }

    let mut remaining=BTreeMap::new(); let mut ending=0u64; let mut source_totals=BTreeMap::new();
    for (id,qty) in &available {
        if *qty==0 {continue;}
        remaining.insert(id.clone(),*qty); add_checked(&mut ending,*qty,&mut issues,format!("remaining_balance_mg.{id}"));
        if let Some(ls)=lineages.get(id) {
            for (src,q) in ls { add_map(&mut source_totals,src.clone(),*q,&mut issues,format!("lineage_by_material.{id}.{src}")); }
        } else { push(&mut issues,"MATERIAL_LINEAGE_MISSING",format!("lineage_by_material.{id}"),"remaining material has no lineage"); }
    }
    if opening_total!=ending { push(&mut issues,"GLOBAL_MASS_IMBALANCE","remaining_balance_mg",format!("opening {opening_total} mg differs from ending {ending} mg")); }
    if source_totals!=opening_by_source { push(&mut issues,"GLOBAL_SOURCE_LINEAGE_IMBALANCE","source_lineage_total_mg",format!("ending source totals {source_totals:?} differ from source openings {opening_by_source:?}")); }
    let mut terminals=Vec::new(); let mut affected=Vec::new();
    for (id,kind) in &kinds { if kind=="packaged_unit"&&available.get(id).copied().unwrap_or(0)>0 {
        terminals.push(id.clone());
        if lineages.get(id).and_then(|m|m.get(&suspect)).copied().unwrap_or(0)>0 {affected.push(id.clone());}
    }}
    terminals.sort(); affected.sort();
    compare_u64(root,"/expected_results/operation_count",operations.len() as u64,&mut issues);
    compare_u64(root,"/expected_results/initial_mass_mg",opening_total,&mut issues);
    compare_u64(root,"/expected_results/ending_mass_mg",ending,&mut issues);
    compare_u64(root,"/expected_results/terminal_count",terminals.len() as u64,&mut issues);
    let expected:BTreeSet<String>=string_array(root.pointer("/expected_results/affected_terminal_ids"),"expected_results.affected_terminal_ids",&mut issues).into_iter().collect();
    let actual:BTreeSet<String>=affected.iter().cloned().collect();
    if expected!=actual { push(&mut issues,"AFFECTED_TERMINAL_KNOWN_ANSWER_MISMATCH","expected_results.affected_terminal_ids",format!("expected {expected:?}, derived {actual:?}")); }
    let expected_balances=read_quantity_map(root.pointer("/expected_results/remaining_balance_mg"),"expected_results.remaining_balance_mg",&mut issues);
    if expected_balances!=remaining {push(&mut issues,"REMAINING_BALANCE_KNOWN_ANSWER_MISMATCH","expected_results.remaining_balance_mg",format!("expected {expected_balances:?}, derived {remaining:?}"));}
    let expected_sources=read_quantity_map(root.pointer("/expected_results/source_lineage_total_mg"),"expected_results.source_lineage_total_mg",&mut issues);
    if expected_sources!=source_totals {push(&mut issues,"SOURCE_LINEAGE_KNOWN_ANSWER_MISMATCH","expected_results.source_lineage_total_mg",format!("expected {expected_sources:?}, derived {source_totals:?}"));}
    finish(issues,operations.len(),opening_total,ending,terminals.len(),affected,remaining,source_totals,lineages)
}

fn finish(mut issues:Vec<QuantityGenealogyIssue>,operation_count:usize,initial_mass_mg:u64,ending_mass_mg:u64,terminal_count:usize,mut affected_terminal_ids:Vec<String>,remaining_balance_mg:BTreeMap<String,u64>,source_lineage_total_mg:BTreeMap<String,u64>,lineage_by_material:BTreeMap<String,BTreeMap<String,u64>>)->QuantityGenealogyReport{
    issues.sort_by(|a,b|a.code.cmp(b.code).then_with(||a.path.cmp(&b.path)).then_with(||a.message.cmp(&b.message)));
    affected_terminal_ids.sort();
    QuantityGenealogyReport{issues,operation_count,initial_mass_mg,ending_mass_mg,terminal_count,affected_terminal_ids,remaining_balance_mg,source_lineage_total_mg,lineage_by_material}
}
fn push(issues:&mut Vec<QuantityGenealogyIssue>,code:&'static str,path:impl Into<String>,message:impl Into<String>){issues.push(QuantityGenealogyIssue{code,path:path.into(),message:message.into()});}
fn expect_string(root:&Value,field:&str,expected:&str,issues:&mut Vec<QuantityGenealogyIssue>){if root.get(field).and_then(Value::as_str)!=Some(expected){push(issues,"EXPECTED_STRING_MISMATCH",field,format!("required value is {expected:?}"));}}
fn add_checked(total:&mut u64,v:u64,issues:&mut Vec<QuantityGenealogyIssue>,path:String){match total.checked_add(v){Some(n)=>*total=n,None=>push(issues,"QUANTITY_OVERFLOW",path,"checked quantity sum overflowed")}}
fn add_map(map:&mut BTreeMap<String,u64>,id:String,v:u64,issues:&mut Vec<QuantityGenealogyIssue>,path:String){let total=map.entry(id).or_insert(0);add_checked(total,v,issues,path);}
fn add_nested(map:&mut BTreeMap<String,BTreeMap<String,u64>>,parent:String,id:String,v:u64,issues:&mut Vec<QuantityGenealogyIssue>,path:String){add_map(map.entry(parent).or_default(),id,v,issues,path);}
fn sum(values:impl IntoIterator<Item=u64>,issues:&mut Vec<QuantityGenealogyIssue>,path:String)->u64{let mut n=0;for v in values{add_checked(&mut n,v,issues,path.clone());}n}
fn string_array(value:Option<&Value>,path:&str,issues:&mut Vec<QuantityGenealogyIssue>)->Vec<String>{let Some(rows)=value.and_then(Value::as_array)else{push(issues,"STRING_ARRAY_MISSING",path,"expected array of strings");return Vec::new()};let mut out=Vec::new();for(i,row)in rows.iter().enumerate(){match row.as_str().filter(|s|!s.trim().is_empty()){Some(id)=>out.push(id.to_owned()),None=>push(issues,"STRING_ARRAY_VALUE_INVALID",format!("{path}[{i}]"),"IDs must be non-empty strings")}}out}
fn read_quantity_map(value:Option<&Value>,path:&str,issues:&mut Vec<QuantityGenealogyIssue>)->BTreeMap<String,u64>{let Some(obj)=value.and_then(Value::as_object)else{push(issues,"QUANTITY_MAP_MISSING",path,"expected an object of quantities");return BTreeMap::new()};let mut out=BTreeMap::new();for(id,v)in obj{match v.as_u64(){Some(n)if n>0=>{out.insert(id.clone(),n);},_=>push(issues,"QUANTITY_MAP_VALUE_INVALID",format!("{path}.{id}"),"expected positive integer quantity")}}out}
fn compare_u64(root:&Value,pointer:&str,actual:u64,issues:&mut Vec<QuantityGenealogyIssue>){if root.pointer(pointer).and_then(Value::as_u64)!=Some(actual){push(issues,"KNOWN_ANSWER_MISMATCH",pointer.trim_start_matches('/').replace('/',"."),format!("expected {:?}, derived {actual}",root.pointer(pointer).and_then(Value::as_u64)));}}

#[cfg(test)]
mod tests{
 use super::*; use serde_json::json;
 fn fixture()->Value{serde_json::from_str(EMBEDDED_QUANTITY_GENEALOGY).expect("fixture JSON")}
 fn run(v:&Value)->QuantityGenealogyReport{validate_quantity_genealogy(v)}
 fn has(r:&QuantityGenealogyReport,c:&str)->bool{r.issues.iter().any(|x|x.code==c)}
 #[test] fn known_answer_conserves_mass_and_derives_only_affected_products(){let r=run(&fixture());assert!(r.is_valid(),"{:?}",r.issues);assert_eq!(r.operation_count,8);assert_eq!(r.initial_mass_mg,2700);assert_eq!(r.ending_mass_mg,2700);assert_eq!(r.terminal_count,3);assert_eq!(r.affected_terminal_ids,vec!["PACKED-ARTICLE-A","PACKED-ARTICLE-B"]);}
 #[test] fn partial_consumption_preserves_remaining_inventory(){let r=run(&fixture());assert_eq!(r.remaining_balance_mg.get("CUT-SEGMENT-A"),Some(&100));assert_eq!(r.remaining_balance_mg.get("LOT-SUSPECT-FIBER"),Some(&300));}
 #[test] fn merged_output_preserves_each_source_attribution(){let r=run(&fixture());let l=r.lineage_by_material.get("PACKED-ARTICLE-A").unwrap();assert_eq!(l.get("LOT-SUSPECT-FIBER"),Some(&300));assert_eq!(l.get("LOT-CLEAN-FIBER"),Some(&200));assert_eq!(l.get("LOT-CLEAN-LAYER"),Some(&500));}
 #[test] fn split_fragments_retain_distinct_source_totals(){let r=run(&fixture());assert_eq!(r.lineage_by_material.get("CUT-SEGMENT-A").unwrap().get("LOT-SUSPECT-FIBER"),Some(&100));assert_eq!(r.lineage_by_material.get("CUT-SEGMENT-B").unwrap().get("LOT-SUSPECT-FIBER"),None);}
 #[test] fn clean_control_is_not_affected(){assert!(!run(&fixture()).affected_terminal_ids.contains(&"PACKED-UNRELATED-CONTROL".to_owned()));}
 #[test] fn all_opening_source_lineage_is_reconciled(){let r=run(&fixture());assert_eq!(r.source_lineage_total_mg.get("LOT-SUSPECT-FIBER"),Some(&1000));assert_eq!(r.source_lineage_total_mg.get("LOT-CLEAN-LAYER"),Some(&1000));}
 #[test] fn overdrawn_input_is_rejected(){let mut v=fixture();v["operations"][1]["inputs"][0]["quantity_mg"]=json!(401);assert!(has(&run(&v),"INPUT_OVERDRAW"));}
 #[test] fn missing_allocation_is_rejected(){let mut v=fixture();v["operations"][0]["allocations"].as_array_mut().unwrap().pop();let r=run(&v);assert!(has(&r,"INPUT_ALLOCATION_MISMATCH"));assert!(has(&r,"OUTPUT_ALLOCATION_MISMATCH"));}
 #[test] fn duplicate_operation_identity_is_rejected(){let mut v=fixture();v["operations"][1]["operation_id"]=v["operations"][0]["operation_id"].clone();assert!(has(&run(&v),"DUPLICATE_OPERATION_ID"));}
 #[test] fn duplicate_output_identity_is_rejected(){let mut v=fixture();v["operations"][1]["outputs"][0]["material_id"]=json!("ARTICLE-A");assert!(has(&run(&v),"DUPLICATE_MATERIAL_ID"));}
 #[test] fn unknown_source_lineage_is_rejected(){let mut v=fixture();v["operations"][1]["allocations"][0]["lineage"][0]["source_lot_id"]=json!("LOT-UNKNOWN");assert!(has(&run(&v),"SOURCE_LOT_UNKNOWN"));}
 #[test] fn source_lineage_must_equal_allocation_quantity(){let mut v=fixture();v["operations"][0]["allocations"][0]["lineage"][0]["quantity_mg"]=json!(399);assert!(has(&run(&v),"ALLOCATION_LINEAGE_MISMATCH"));}
 #[test] fn input_cannot_claim_lineage_not_present_on_that_material(){let mut v=fixture();v["operations"][1]["allocations"][0]["lineage"]=json!([{"source_lot_id":"LOT-SUSPECT-FIBER","quantity_mg":299},{"source_lot_id":"LOT-CLEAN-LAYER","quantity_mg":1}]);assert!(has(&run(&v),"SOURCE_LINEAGE_UNAVAILABLE"));}
 #[test] fn operation_mass_imbalance_fails_closed(){let mut v=fixture();v["operations"][0]["outputs"][0]["quantity_mg"]=json!(401);assert!(has(&run(&v),"OPERATION_MASS_IMBALANCE"));}
 #[test] fn cross_unit_input_fails_closed(){let mut v=fixture();v["operations"][0]["inputs"][0]["unit"]=json!("kg");assert!(has(&run(&v),"UNIT_MISMATCH"));}
 #[test] fn affected_known_answer_tampering_fails_closed(){let mut v=fixture();v["expected_results"]["affected_terminal_ids"]=json!(["PACKED-ARTICLE-A"]);assert!(has(&run(&v),"AFFECTED_TERMINAL_KNOWN_ANSWER_MISMATCH"));}
 #[test] fn reverse_operation_order_rejects_future_input_use(){let mut v=fixture();v["operations"].as_array_mut().unwrap().reverse();assert!(has(&run(&v),"INPUT_MATERIAL_UNKNOWN_OR_FUTURE"));}
}