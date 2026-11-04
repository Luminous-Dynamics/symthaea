#!/usr/bin/env python3
"""Independent reference model of D4v2 dynamic projected-BCE state transitions.

This models the protected-triple/watch/queue/assignment/active-clause subset.
It deliberately omits D4's packed occurrence arrays, parser and full solver.
"""
from __future__ import annotations
import copy, hashlib, json, pathlib, sys
from fractions import Fraction
from itertools import product

HERE=pathlib.Path(__file__).resolve().parent
CERT=HERE/"d4v2_dynamic_bce_trace_certificate.v1.json"
OUT=HERE/"d4v2_dynamic_bce_trace_results.v1.json"

SOURCE_PINS={
 "src/formulaManager/cnf/CnfManagerDynBlockedCl.cpp":"303817529debd95735a32ca9200791c3a854d992",
 "src/formulaManager/cnf/CnfManagerDynBlockedCl.hpp":"d92fe5e5d25b26f981ff657230ff0a9cb8ad04bf",
 "src/formulaManager/cnf/CnfManagerDyn.cpp":"3f19bc0664ff90d167ee00b1b6321a0015fe1ff8",
 "src/formulaManager/cnf/CnfManagerDyn.hpp":"8c014829c7ccae545b1afb98c10d21b425e1b4b7",
 "src/formulaManager/cnf/ClauseInfo.hpp":"c8d0718fdad78aba0c87701e2e19b5fd2acd7f75",
 "src/formulaManager/cnf/DataOccurrence.hpp":"0a74b2c9db6c8eb0863d4d43d1b13f9c0df5bc29",
 "src/problem/ProblemManager.hpp":"f5c2bbfd3a0951685a7055d1a9b4153395563bc1",
}

def tautological(literals):
    values=set(literals)
    return any(-lit in values for lit in values)

def resolvent(source, opposing, blocker):
    return [lit for lit in source if lit!=blocker]+[lit for lit in opposing if lit!=-blocker]

class D4BCEStateModel:
    """Mirrors constructor, inprocessing, preUpdate and postUpdate control flow."""
    def __init__(self, problem):
        self.nvars=problem["nvars"]
        self.selected=set(problem["selected_vars"])
        self.clauses={c["id"]:list(c["lits"]) for c in problem["clauses"]}
        self.order=[c["id"] for c in problem["clauses"]]
        self.assignment={}
        self.active={cid:True for cid in self.order}
        self.protected=[]
        self.watchlists={cid:[] for cid in self.order}
        self.pending=[]
        self.frames=[]
        self.events=[]
        self._initialize()

    def _initialize(self):
        # The upstream constructor collects initially blocked/satisfied clauses
        # during the scan, but marks them inactive only after the complete scan.
        for cid in self.order:
            clause=self.clauses[cid]
            start=len(self.protected)
            blocked=False
            for blocker in clause:
                variable=abs(blocker)
                if variable in self.assignment or variable in self.selected:
                    continue
                non_taut=[]
                for other_id in self.order:
                    if other_id==cid or not self.active[other_id]:
                        continue
                    opposing=self.clauses[other_id]
                    if -blocker not in opposing:
                        continue
                    if not tautological(resolvent(clause,opposing,blocker)):
                        non_taut.append(other_id)
                if non_taut:
                    self.protected.append({
                        "id":len(self.protected),"blocker":blocker,
                        "target":cid,"candidate_clauses":non_taut.copy()
                    })
                else:
                    blocked=True
                    break
            if blocked:
                del self.protected[start:]
                self.pending.append(cid)
                self.events.append(["initial_blocked",cid])
            else:
                for info in self.protected[start:]:
                    self.watchlists[info["candidate_clauses"][0]].append(info["id"])
        for cid in self.pending:
            self.active[cid]=False
        self._inprocessing()
        # Match constructor cleanup of watches whose target is initially inactive.
        for sentinel in self.order:
            self.watchlists[sentinel]=[
                iid for iid in self.watchlists[sentinel]
                if self.active[self.protected[iid]["target"]]
            ]
        self._assert_watch_invariant()

    def _save_clause(self,cid):
        if not self.frames:
            return
        saved=self.frames[-1]["saved_clause_states"]
        if cid not in saved:
            saved[cid]=self.active[cid]

    def _inprocessing(self):
        # m_indexSatClauses is a LIFO vector in D4's implementation.
        while self.pending:
            inactive_clause=self.pending.pop()
            retained=[]
            for info_id in self.watchlists[inactive_clause]:
                info=self.protected[info_id]
                target=info["target"]
                blocker_var=abs(info["blocker"])
                if not self.active[target] or blocker_var in self.assignment:
                    retained.append(info_id)
                    continue
                replacement=None
                for index in range(1,len(info["candidate_clauses"])):
                    candidate=info["candidate_clauses"][index]
                    if self.active[candidate]:
                        replacement=index
                        break
                if replacement is not None:
                    candidates=info["candidate_clauses"]
                    candidates[0],candidates[replacement]=candidates[replacement],candidates[0]
                    new_sentinel=candidates[0]
                    self.watchlists[new_sentinel].append(info_id)
                    self.events.append(["watch_relocated",info_id,inactive_clause,new_sentinel])
                else:
                    self._save_clause(target)
                    self.active[target]=False
                    self.pending.append(target)
                    retained.append(info_id)
                    self.events.append(["blocked_by_exhausted_opposing_set",target,info_id,inactive_clause])
            self.watchlists[inactive_clause]=retained

    def pre_update(self,true_literals,label):
        frame={"label":label,"saved_clause_states":{},"newly_assigned":[]}
        self.frames.append(frame)
        for literal in true_literals:
            variable=abs(literal)
            value=literal>0
            if variable in self.assignment and self.assignment[variable]!=value:
                raise ValueError("conflicting assignment")
            if variable not in self.assignment:
                self.assignment[variable]=value
                frame["newly_assigned"].append(variable)
        self.events=[]
        # D4 propagateTrue: all clauses containing a newly true literal become inactive.
        for literal in true_literals:
            for cid in self.order:
                if self.active[cid] and literal in self.clauses[cid]:
                    self._save_clause(cid)
                    self.active[cid]=False
                    self.pending.append(cid)
                    self.events.append(["clause_satisfied",cid,literal])
        self._inprocessing()
        self._assert_watch_invariant()
        return self.snapshot()

    def post_update(self):
        if not self.frames:
            raise RuntimeError("postUpdate without preUpdate")
        frame=self.frames.pop()
        # Clause and assignment state are restored, but the watch lists are not.
        for cid,old_value in frame["saved_clause_states"].items():
            self.active[cid]=old_value
        for variable in reversed(frame["newly_assigned"]):
            del self.assignment[variable]
        self.events=[["backtrack_restored",frame["label"]]]
        self._assert_watch_invariant()
        return self.snapshot()

    def _assert_watch_invariant(self):
        seen={}
        for sentinel,info_ids in self.watchlists.items():
            for info_id in info_ids:
                if info_id in seen:
                    raise AssertionError("duplicate watcher for protected record "+str(info_id))
                seen[info_id]=sentinel
        for info in self.protected:
            iid=info["id"]
            if iid not in seen:
                raise AssertionError("protected record has no watch "+str(iid))
            if seen[iid]!=info["candidate_clauses"][0]:
                raise AssertionError("watch sentinel/candidate order mismatch "+str(iid))
            if self.active[info["target"]] and abs(info["blocker"]) not in self.assignment:
                if not self.active[seen[iid]]:
                    raise AssertionError("active target has inactive sentinel "+str(iid))

    def snapshot(self):
        return {
            "assignment":dict(self.assignment),
            "active":dict(self.active),
            "protected_records":copy.deepcopy(self.protected),
            "watchlists":{cid:list(self.watchlists[cid]) for cid in self.order},
            "events":copy.deepcopy(self.events)
        }

def evaluate(clauses,nvars,selected,weights,assignment=None,active_ids=None):
    assignment=dict(assignment or {})
    selected=list(selected)
    all_ids={c["id"] for c in clauses}
    active_ids=set(all_ids if active_ids is None else active_ids)
    hidden=[v for v in range(1,nvars+1) if v not in assignment]
    models=0
    support=set()
    for bits in product((False,True),repeat=len(hidden)):
        complete=dict(assignment)
        complete.update(dict(zip(hidden,bits)))
        if all(
            any(complete[abs(lit)]==(lit>0) for lit in clause["lits"])
            for clause in clauses if clause["id"] in active_ids
        ):
            models+=1
            support.add(tuple(complete[v] for v in selected))
    weighted=Fraction(0,1)
    for row in support:
        term=Fraction(1,1)
        for variable,value in zip(selected,row):
            key="true" if value else "false"
            term*=Fraction(weights[str(variable)][key])
        weighted+=term
    return {
        "full_models":models,
        "projected_support":[list(row) for row in sorted(support)],
        "projected_support_size":len(support),
        "pwmc":f"{weighted.numerator}/{weighted.denominator}"
    }

def verify_certificate(cert):
    errors=[]
    if cert.get("format")!="symthaea.d4v2.dynamic-projected-bce-trace" or cert.get("version")!=1:
        return ["unsupported format/version"],{}
    if cert.get("upstream",{}).get("repository")!="crillab/d4v2":
        errors.append("upstream repository pin mismatch")
    if cert.get("upstream",{}).get("branch")!="main":
        errors.append("upstream branch pin mismatch")
    if cert.get("upstream",{}).get("source_blob_sha1")!=SOURCE_PINS:
        errors.append("upstream source blob pin mismatch")
    problem=cert["problem"]
    nvars=problem["nvars"]
    selected=problem["selected_vars"]
    hidden=problem["hidden_vars"]
    if set(selected)&set(hidden) or set(selected)|set(hidden)!=set(range(1,nvars+1)):
        errors.append("selected/hidden partition is not complete and disjoint")
    clauses=problem["clauses"]
    original_ids=[c["id"] for c in clauses]
    if len(original_ids)!=len(set(original_ids)):
        errors.append("duplicate clause IDs")
    model=D4BCEStateModel(problem)
    initial=model.snapshot()
    initial_expected=cert["expected_initial"]
    for key in ("active","protected_records","watchlists","events"):
        if initial[key]!=initial_expected.get(key):
            errors.append("initial "+key+" mismatch")
    root_source=evaluate(clauses,nvars,selected,problem["visible_weights"])
    root_processed=evaluate(
        clauses,nvars,selected,problem["visible_weights"],
        active_ids=[cid for cid in original_ids if model.active[cid]]
    )
    root_expected=cert["expected_root"]
    for field,actual in [
        ("source_full_models",root_source["full_models"]),
        ("processed_full_models",root_processed["full_models"]),
        ("projected_support_size",root_source["projected_support_size"]),
        ("pwmc",root_source["pwmc"])
    ]:
        if root_expected.get(field)!=actual:
            errors.append("root "+field+" mismatch")
    if root_source["projected_support"]!=root_processed["projected_support"]:
        errors.append("root preprocessing changed projected support")
    root_support={tuple(row) for row in root_source["projected_support"]}
    branch_support_source=set()
    branch_support_processed=set()
    branch_rows=[]
    for branch in cert["branches"]:
        branch_id=branch["id"]
        before_snapshot=model.snapshot()
        after=model.pre_update(branch["true_literals"],branch["label"])
        expected=branch["expected"]
        for key in ("active","watchlists","events"):
            if after[key]!=expected.get(key):
                errors.append(branch_id+" "+key+" mismatch")
        assignment={str(k):v for k,v in model.assignment.items()}
        source=evaluate(clauses,nvars,selected,problem["visible_weights"],assignment=assignment)
        processed=evaluate(
            clauses,nvars,selected,problem["visible_weights"],
            assignment=assignment,
            active_ids=[cid for cid in original_ids if model.active[cid]]
        )
        if source["projected_support"]!=processed["projected_support"]:
            errors.append(branch_id+" changed projected support")
        for field,actual in [
            ("source_full_models",source["full_models"]),
            ("processed_full_models",processed["full_models"]),
            ("projected_support_size",source["projected_support_size"]),
            ("pwmc",source["pwmc"])
        ]:
            if expected.get(field)!=actual:
                errors.append(branch_id+" "+field+" mismatch")
        branch_support_source.update(tuple(row) for row in source["projected_support"])
        branch_support_processed.update(tuple(row) for row in processed["projected_support"])
        branch_rows.append({
            "branch":branch_id,
            "assignment":assignment,
            "source":source,
            "processed":processed,
            "state":after
        })
        restored=model.post_update()
        if restored["active"]!=before_snapshot["active"]:
            errors.append(branch_id+" backtrack failed to restore active-clause state")
        if restored["watchlists"]!=before_snapshot["watchlists"]:
            errors.append(branch_id+" backtrack unexpectedly changed watch placement")
        if restored["assignment"]!=before_snapshot["assignment"]:
            errors.append(branch_id+" backtrack failed to restore assignments")
    if branch_support_source!=root_support:
        errors.append("source sibling support union differs from root")
    if branch_support_processed!=root_support:
        errors.append("processed sibling support union differs from root")
    union_expected=cert["expected_branch_union"]
    if union_expected.get("projected_support_size")!=len(branch_support_processed):
        errors.append("sibling union support size mismatch")
    if union_expected.get("pwmc")!=root_source["pwmc"]:
        errors.append("sibling union weighted value mismatch")
    summary={
        "root_source":root_source,
        "root_processed":root_processed,
        "initial_state":initial,
        "branch_results":branch_rows,
        "sibling_source_support_size":len(branch_support_source),
        "sibling_processed_support_size":len(branch_support_processed),
        "sibling_union_matches_root":branch_support_processed==root_support
    }
    return errors,summary

def main():
    cert=json.loads(CERT.read_text(encoding="utf-8"))
    errors,summary=verify_certificate(cert)
    if errors:
        raise SystemExit("FAIL valid source-pinned trace: "+"; ".join(errors))
    print("PASS source-pinned D4v2 dynamic-BCE manager trace")
    print("PASS initial protected triples, watch sentinels and static hidden BCE")
    print("PASS b=true watcher trigger, dynamic C1 deletion and parked sentinel")
    print("PASS postUpdate restores C1/C2 and leaves watch placement untouched")
    print("PASS b=false sibling remains isolated and does not inherit C1 deletion")
    print("PASS projected support and exact signed-rational PWMC at root and leaves")

    mutations=[]
    def mutate(label,fn):
        candidate=copy.deepcopy(cert)
        fn(candidate)
        mutations.append((label,candidate))
    mutate("selected-hidden partition corruption",lambda c:c["problem"].update(selected_vars=[1,2,3,4],hidden_vars=[]))
    mutate("missing protected record",lambda c:c["expected_initial"]["protected_records"].pop())
    mutate("wrong sentinel",lambda c:c["expected_initial"]["watchlists"].update(C2=[1]))
    mutate("missing dynamic BCE event",lambda c:c["branches"][0]["expected"].update(events=[["clause_satisfied","C2",3]]))
    mutate("wrong branch active state",lambda c:c["branches"][0]["expected"]["active"].update(C1=True))
    mutate("sibling inherits deleted target",lambda c:c["branches"][1]["expected"]["active"].update(C1=False))
    mutate("backtrack support altered",lambda c:c["branches"][1]["expected"].update(projected_support_size=4))
    mutate("upstream source pin drift",lambda c:c["upstream"]["source_blob_sha1"].update({"src/formulaManager/cnf/CnfManagerDynBlockedCl.cpp":"0"*40}))
    mutate("branch assignment duplicated",lambda c:c["branches"][1].update(true_literals=[3]))
    mutate("stale root count",lambda c:c["expected_root"].update(source_full_models=99))
    for label,candidate in mutations:
        mutant_errors,_=verify_certificate(candidate)
        if not mutant_errors:
            raise AssertionError("false accept: "+label)
        print("REJECT",label,"-",mutant_errors[0])

    raw=pathlib.Path(__file__).read_bytes()
    cert_raw=CERT.read_bytes()
    result={
        "format":"symthaea.d4v2.dynamic-projected-bce-trace-results",
        "version":1,
        "claim_ceiling":"independent_reference_model_of_pinned_D4_manager_state_subset_plus_exhaustive_projected_oracle_not_D4_binary_replay_or_formal_proof",
        "certificate_check":"passed",
        "certificate_sha256":"sha256:"+hashlib.sha256(cert_raw).hexdigest(),
        "checker_sha256":"sha256:"+hashlib.sha256(raw).hexdigest(),
        "checker_git_blob_sha1":hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest(),
        "source_blob_pins":SOURCE_PINS,
        "summary":summary,
        "negative_controls":len(mutations),
        "negative_controls_rejected":len(mutations),
        "negative_control_labels":[label for label,_ in mutations],
        "runtime":{"python":sys.version.split()[0],"platform":sys.platform}
    }
    OUT.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(f"PASS {len(mutations)}/{len(mutations)} negative controls rejected")

if __name__=="__main__":
    main()
