#!/usr/bin/env python3
"""Verifier for a nested context-indexed projected-BCE branch trace.

Finite exhaustive reference checker; not a D4 trace replay or formal proof checker.
"""
from __future__ import annotations
import copy, hashlib, itertools, json, pathlib, sys
from fractions import Fraction

HERE=pathlib.Path(__file__).resolve().parent
CERT=HERE/"nested_dynamic_bce_trace_certificate.v2.json"
OUT=HERE/"nested_dynamic_bce_trace_results.v2.json"

def parts(lit):
    return (lit[1:],False) if lit.startswith("!") else (lit,True)

def lit_value(lit,a):
    var,pos=parts(lit)
    return bool(a[var]) if pos else not bool(a[var])

def formula_digest(variables,clauses):
    value={"variables":sorted(variables),"clauses":[
      {"id":c["id"],"literals":sorted(set(c["literals"]),key=lambda z:(parts(z)[0],parts(z)[1]))}
      for c in sorted(clauses,key=lambda c:c["id"])]}
    raw=json.dumps(value,sort_keys=True,separators=(",",":")).encode()
    return "sha256:"+hashlib.sha256(raw).hexdigest()

def context_digest(parent_ctx,parent_formula,node_id,delta):
    value={"parent_context_digest":parent_ctx,"parent_formula_digest":parent_formula,
           "context_id":node_id,"assignment_delta":[[k,delta[k]] for k in sorted(delta)]}
    raw=json.dumps(value,sort_keys=True,separators=(",",":")).encode()
    return "sha256:"+hashlib.sha256(raw).hexdigest()

def root_context_digest(root_formula):
    raw=json.dumps({"context_id":"root","formula_digest":root_formula},sort_keys=True,separators=(",",":")).encode()
    return "sha256:"+hashlib.sha256(raw).hexdigest()

def condition(variables,clauses,assignment):
    residual=[v for v in variables if v not in assignment]
    result=[]
    for c in clauses:
        if any(parts(l)[0] in assignment and lit_value(l,assignment) for l in c["literals"]):
            continue
        result.append({"id":c["id"],"literals":[l for l in c["literals"] if parts(l)[0] not in assignment]})
    return residual,result

def tautology_var(literals):
    values=set(literals)
    for lit in sorted(values,key=lambda z:(parts(z)[0],parts(z)[1])):
        var,pos=parts(lit)
        if ("!"+var if pos else var) in values:
            return var
    return None

def check_bce(variables,hidden,clauses,op):
    errors=[]
    by_id={c["id"]:c for c in clauses}
    if len(by_id)!=len(clauses):
        return ["duplicate clause IDs"]
    rid=op["removed_clause_id"]
    if rid not in by_id:
        return ["removed clause missing"]
    removed=by_id[rid]
    blocker=op["blocking_literal"]
    var,_=parts(blocker)
    if blocker not in removed["literals"]:
        errors.append("blocker absent from removed clause")
    if var not in hidden:
        errors.append("blocker variable not hidden")
    if tautology_var(removed["literals"]) is not None:
        errors.append("removed clause tautological")
    complement=("!"+var) if not blocker.startswith("!") else var
    opposing={c["id"]:c for c in clauses if c["id"]!=rid and complement in c["literals"]}
    listed=op["opposing_clause_checks"]
    supplied={x["clause_id"]:x for x in listed}
    if len(supplied)!=len(listed) or set(supplied)!=set(opposing):
        errors.append("opposing-clause coverage incomplete/duplicated")
    for cid,other in opposing.items():
        item=supplied.get(cid)
        if item is None: continue
        resolvent=[x for x in removed["literals"] if x!=blocker]+[x for x in other["literals"] if x!=complement]
        witness=tautology_var(resolvent)
        canonical=sorted(set(resolvent),key=lambda z:(parts(z)[0],parts(z)[1]))
        if witness is None: errors.append("non-tautological resolvent "+cid)
        if item.get("resolvent")!=canonical: errors.append("resolvent mismatch "+cid)
        if item.get("tautology_variable")!=witness: errors.append("tautology witness mismatch "+cid)
    target=[c for c in clauses if c["id"]!=rid]
    if op.get("target_clauses")!=target: errors.append("target is not exact single-clause deletion")
    if op.get("target_formula_digest")!=formula_digest(variables,target): errors.append("target digest mismatch")
    return errors

def sat(clauses,a):
    return all(any(lit_value(l,a) for l in c["literals"]) for c in clauses)

def oracle(variables,visible,clauses):
    models=0;support=set()
    for bits in itertools.product((False,True),repeat=len(variables)):
        a=dict(zip(variables,bits))
        if sat(clauses,a):
            models+=1;support.add(tuple(a[v] for v in visible))
    return models,support

def weighted(support,visible,weights):
    total=Fraction(0,1)
    for row in support:
        term=Fraction(1,1)
        for v,b in zip(visible,row):
            term*=Fraction(weights[v]["true"] if b else weights[v]["false"])
        total+=term
    return f"{total.numerator}/{total.denominator}"

def verify(cert):
    errors=[]
    if cert.get("format")!="symthaea.nested-dynamic-projected-bce-trace" or cert.get("version")!=1:
        return ["unsupported format/version"],{}
    root=cert["root"];rv=root["variables"];rvis=root["visible"];rhid=root["hidden"];rc=root["clauses"]
    if len(rv)!=len(set(rv)) or len(rvis)!=len(set(rvis)) or len(rhid)!=len(set(rhid)) or set(rvis)&set(rhid) or set(rvis)|set(rhid)!=set(rv):
        errors.append("invalid root variable partition")
    rd=formula_digest(rv,rc)
    if rd!=root["formula_digest"]: errors.append("root formula digest mismatch")
    rctx=root_context_digest(rd)
    if rctx!=root["context_digest"]: errors.append("root context digest mismatch")
    state={"root":{"context_digest":rctx,"final_formula_digest":rd,"variables":rv,"visible":rvis,"hidden":rhid,
                   "clauses":rc,"cumulative_assignment":{},"final_clauses":rc}}
    nodes=cert["nodes"];ids=[n["node_id"] for n in nodes]
    if len(ids)!=len(set(ids)): errors.append("duplicate node IDs")
    children={}
    for node in nodes:
        nid=node["node_id"];parent=node["parent_context_id"]
        if parent not in state:
            errors.append(nid+": missing/non-topological parent");continue
        p=state[parent];delta=node["assignment_delta"]
        if not delta or any(k not in p["variables"] for k in delta) or set(delta)&set(p["cumulative_assignment"]):
            errors.append(nid+": invalid/repeated assignment delta")
        if any(not isinstance(v,bool) for v in delta.values()): errors.append(nid+": non-Boolean branch value")
        cumulative=dict(p["cumulative_assignment"]);cumulative.update(delta)
        if cumulative!=node["cumulative_assignment"]: errors.append(nid+": cumulative assignment mismatch")
        if node["parent_formula_digest"]!=p["final_formula_digest"]: errors.append(nid+": parent formula digest mismatch or sibling leakage")
        ctx=context_digest(p["context_digest"],p["final_formula_digest"],nid,delta)
        if node["context_digest"]!=ctx: errors.append(nid+": context digest mismatch")
        residual,conditioned=condition(p["variables"],p["clauses"],delta)
        declared=node["conditioned"]
        vis=[v for v in p["visible"] if v not in delta];hid=[v for v in p["hidden"] if v not in delta]
        if residual!=declared["variables"] or conditioned!=declared["clauses"]: errors.append(nid+": conditioned formula mismatch")
        if vis!=declared["visible"] or hid!=declared["hidden"]: errors.append(nid+": conditioned quantifier partition mismatch")
        cd=formula_digest(residual,conditioned)
        if cd!=declared["formula_digest"]: errors.append(nid+": conditioned digest mismatch")
        current=conditioned;current_digest=cd
        for op in node["operations"]:
            if op.get("type")!="remove_blocked_clause": errors.append(nid+": unsupported operation");continue
            if op.get("context_digest")!=ctx or op.get("source_formula_digest")!=current_digest:
                errors.append(nid+": operation bound to wrong context/source")
            errors.extend(nid+": "+e for e in check_bce(residual,hid,current,op))
            current=op["target_clauses"];current_digest=op["target_formula_digest"]
        back=node["backtrack"]
        if back.get("restored_context_id")!=parent or back.get("restored_context_digest")!=p["context_digest"]:
            errors.append(nid+": wrong backtrack parent context")
        if back.get("restored_formula_digest")!=p["final_formula_digest"]: errors.append(nid+": wrong backtrack parent formula")
        if back.get("child_final_formula_digest")!=current_digest: errors.append(nid+": child final digest mismatch")
        state[nid]={"context_digest":ctx,"final_formula_digest":current_digest,"variables":residual,"visible":vis,"hidden":hid,
                    "clauses":conditioned,"cumulative_assignment":cumulative,"final_clauses":current}
        children.setdefault(parent,[]).append(node)
    for parent,group in children.items():
        if len(group)!=2: errors.append(parent+": branch must have exactly two children");continue
        keys=[set(n["assignment_delta"]) for n in group]
        if keys[0]!=keys[1] or len(keys[0])!=1: errors.append(parent+": sibling deltas must split same variable");continue
        var=next(iter(keys[0]))
        if sorted(n["assignment_delta"][var] for n in group)!=[False,True]:
            errors.append(parent+": children do not cover false/true exactly")
    leaves=[n for n in nodes if n["node_id"] not in children]
    root_models,root_support=oracle(rv,rvis,rc)
    source_total=target_total=0;source_union=set();target_union=set();leaf_rows=[]
    for leaf in leaves:
        s=state.get(leaf["node_id"])
        if s is None: continue
        bm,bs=oracle(leaf["conditioned"]["variables"],leaf["conditioned"]["visible"],leaf["conditioned"]["clauses"])
        am,ass=oracle(s["variables"],s["visible"],s["final_clauses"])
        if bs!=ass: errors.append(leaf["node_id"]+": projected support changed in leaf")
        source_total+=bm;target_total+=am
        for row in bs:
            full=dict(leaf["cumulative_assignment"]);full.update(dict(zip(leaf["conditioned"]["visible"],row)))
            source_union.add(tuple(full[v] for v in rvis))
        for row in ass:
            full=dict(leaf["cumulative_assignment"]);full.update(dict(zip(s["visible"],row)))
            target_union.add(tuple(full[v] for v in rvis))
        leaf_rows.append({"leaf":leaf["node_id"],"source_models":bm,"processed_models":am,
                          "source_support_size":len(bs),"processed_support_size":len(ass)})
    if source_total!=root_models: errors.append("leaf model total differs from root")
    if source_union!=root_support or target_union!=root_support: errors.append("leaf support union differs from root")
    summary={"root_full_models":root_models,"leaf_source_full_models":source_total,"leaf_processed_full_models":target_total,
             "root_pmc":len(root_support),"processed_pmc":len(target_union),
             "root_pwmc":weighted(root_support,rvis,root["weights"]),"processed_pwmc":weighted(target_union,rvis,root["weights"]),
             "root_support":["".join("1" if b else "0" for b in row) for row in sorted(root_support)],
             "processed_support":["".join("1" if b else "0" for b in row) for row in sorted(target_union)],
             "leaves":leaf_rows}
    return errors,summary

def main():
    cert=json.loads(CERT.read_text(encoding="utf-8"));errors,summary=verify(cert)
    if errors: raise SystemExit("FAIL valid nested trace: "+"; ".join(errors))
    assert summary["root_pwmc"]==summary["processed_pwmc"]
    print("PASS nested branch trace: x split, nested y split, branch-local hidden BCE, sibling isolation and backtrack")
    print(f"PASS support/PMC/PWMC preserved; full models {summary['root_full_models']} -> {summary['leaf_processed_full_models']}")
    mutants=[]
    def mutate(label,fn):
        x=copy.deepcopy(cert);fn(x);mutants.append((label,x))
    mutate("missing nested y=1 sibling",lambda x:x["nodes"].pop(2))
    mutate("duplicate nested assignment",lambda x:x["nodes"][2].update(assignment_delta={"y":False}))
    mutate("wrong nested parent",lambda x:x["nodes"][1].update(parent_context_id="x1"))
    mutate("cross-branch context reuse",lambda x:x["nodes"][1].update(context_digest=x["nodes"][3]["context_digest"]))
    mutate("conditioned formula tampering",lambda x:x["nodes"][1]["conditioned"]["clauses"].pop())
    mutate("blocker made visible",lambda x:x["nodes"][1]["conditioned"].update(hidden=[]))
    mutate("missing opposing clause check",lambda x:x["nodes"][1]["operations"][0].update(opposing_clause_checks=[]))
    mutate("backtrack restores child",lambda x:x["nodes"][1]["backtrack"].update(restored_formula_digest=x["nodes"][1]["operations"][0]["target_formula_digest"]))
    mutate("sibling inherits BCE target",lambda x:x["nodes"][2]["conditioned"].update(clauses=x["nodes"][1]["operations"][0]["target_clauses"]))
    mutate("stale root digest",lambda x:x["root"].update(formula_digest="sha256:"+"0"*64))
    mutate("stale nested context",lambda x:x["nodes"][1].update(context_digest="sha256:"+"f"*64))
    for label,x in mutants:
        e,_=verify(x)
        if not e: raise AssertionError("false accept: "+label)
        print("REJECT",label,"-",e[0])
    raw=pathlib.Path(__file__).read_bytes()
    result={"format":"symthaea.nested-dynamic-projected-bce-trace-results","version":2,
      "claim_ceiling":"nested_branch_reference_checker_plus_exhaustive_oracle_not_formal_proof_or_D4_trace_replay",
      "certificate_sha256":"sha256:"+hashlib.sha256(CERT.read_bytes()).hexdigest(),
      "checker_sha256":"sha256:"+hashlib.sha256(raw).hexdigest(),
      "checker_git_blob_sha1":hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest(),
      "certificate_check":"passed","summary":summary,"negative_controls":len(mutants),"negative_controls_rejected":len(mutants),
      "negative_control_labels":[x[0] for x in mutants],"runtime":{"python":sys.version.split()[0]}}
    OUT.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(f"PASS {len(mutants)}/{len(mutants)} nested branch/context negative controls")

if __name__=="__main__":main()
