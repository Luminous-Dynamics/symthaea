// Package minimize reduces deterministic fault sets while preserving the exact
// observed failure signature. It does not infer causation beyond tested interventions.
package minimize

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"time"
)

const (
	Profile = "mon039-counterfactual-fault-reduction-input-v1"
	Version = "1.0.0"
	MaxFaults = 256
	MaxObservations = 50000
	MaxEvaluations = 50000
	ExhaustiveFaultLimit = 12
)

type Verdict string
const (
	Qualified Verdict = "qualified"
	Blocked Verdict = "blocked"
	Indeterminate Verdict = "indeterminate"
)
type Mode string
const (
	OfflineFixture Mode = "offline_fixture"
	RuntimeCandidate Mode = "runtime_candidate"
)
type Observation struct {
	FaultIDs []string `json:"fault_ids"`
	FaultSetSHA256 string `json:"fault_set_sha256"`
	ScenarioSHA256 string `json:"scenario_sha256"`
	RecoveryBoundarySHA256 string `json:"recovery_boundary_sha256"`
	CodeRevision string `json:"code_revision"`
	ObservedAt time.Time `json:"observed_at"`
	Verdict Verdict `json:"verdict"`
	FailureSignatureSHA256 string `json:"failure_signature_sha256"`
	OutcomeReceiptSHA256 string `json:"outcome_receipt_sha256"`
	Complete bool `json:"complete"`
}
type Input struct {
	Profile string `json:"profile"`
	Version string `json:"version"`
	TrialID string `json:"trial_id"`
	Mode Mode `json:"mode"`
	EvaluatedAt time.Time `json:"evaluated_at"`
	ScenarioSHA256 string `json:"scenario_sha256"`
	RecoveryBoundarySHA256 string `json:"recovery_boundary_sha256"`
	CodeRevision string `json:"code_revision"`
	InitialFaultIDs []string `json:"initial_fault_ids"`
	ExpectedFailureSignatureSHA256 string `json:"expected_failure_signature_sha256,omitempty"`
	Observations []Observation `json:"observations"`
}
type EvaluationStep struct {
	Sequence int `json:"sequence"`
	FaultIDs []string `json:"fault_ids"`
	FaultSetSHA256 string `json:"fault_set_sha256"`
	Verdict Verdict `json:"verdict"`
	FailureSignatureSHA256 string `json:"failure_signature_sha256"`
	OutcomeReceiptSHA256 string `json:"outcome_receipt_sha256"`
	TargetFailurePreserved bool `json:"target_failure_preserved"`
}
type Result struct {
	Profile string `json:"profile"`
	Version string `json:"version"`
	TrialID string `json:"trial_id"`
	Mode Mode `json:"mode"`
	FullFaultSetVerdict Verdict `json:"full_fault_set_verdict"`
	TargetFailureSignatureSHA256 string `json:"target_failure_signature_sha256"`
	MinimalFaultIDs []string `json:"minimal_fault_ids"`
	MinimalFaultSetSHA256 string `json:"minimal_fault_set_sha256"`
	MinimalityKind string `json:"minimality_kind"`
	SubsetEvaluations int `json:"subset_evaluations"`
	EvidenceComplete bool `json:"evidence_complete"`
	CampaignVerdict Verdict `json:"campaign_verdict"`
	ProductionActionPermitted bool `json:"production_action_permitted"`
	Steps []EvaluationStep `json:"steps"`
	ReasonCodes []string `json:"reason_codes"`
}

type flags struct {
	blocked bool
	unknown bool
	reasons map[string]struct{}
}
func newFlags()*flags{return &flags{reasons:map[string]struct{}{}}}
func(f *flags)block(s string){f.blocked=true;f.reasons[s]=struct{}{}}
func(f *flags)hold(s string){f.unknown=true;f.reasons[s]=struct{}{}}
func(f *flags)codes()[]string{out:=make([]string,0,len(f.reasons));for s:=range f.reasons{out=append(out,s)};sort.Strings(out);return out}
func(f *flags)verdict()Verdict{if f.blocked{return Blocked};if f.unknown{return Indeterminate};return Qualified}

func validID(s string)bool{return s!=""&&len(s)<=256&&!strings.ContainsAny(s," 	
")}
func validDigest(s string)bool{if len(s)!=64{return false};_,e:=hex.DecodeString(s);return e==nil&&s==strings.ToLower(s)}
func validVerdict(v Verdict)bool{return v==Qualified||v==Blocked||v==Indeterminate}
func validRevision(s string)bool{if len(s)!=40&&len(s)!=64{return false};_,err:=hex.DecodeString(s);return err==nil}
func digestJSON(v any)string{b,e:=json.Marshal(v);if e!=nil{return ""};d:=sha256.Sum256(b);return hex.EncodeToString(d[:])}
func canonicalFaults(ids []string)([]string,bool){
	out:=make([]string,len(ids));copy(out,ids)
	for _,id:=range out{if !validID(id){return nil,false}}
	sort.Strings(out)
	for i:=1;i<len(out);i++{if out[i]==out[i-1]{return nil,false}}
	return out,true
}
func FaultSetDigest(ids []string)string{
	canonical,ok:=canonicalFaults(ids);if !ok{return ""}
	return digestJSON(canonical)
}
func sameIDs(a,b []string)bool{if len(a)!=len(b){return false};for i:=range a{if a[i]!=b[i]{return false}};return true}
func sameFailure(o Observation,target Verdict,signature string)bool{return o.Verdict==target&&o.FailureSignatureSHA256==signature}

func validateInput(in Input,f *flags)map[string]Observation{
	if in.Profile!=Profile||in.Version!=Version{f.block("profile_version_invalid")}
	if !validID(in.TrialID){f.block("trial_id_invalid")}
	if in.Mode!=OfflineFixture&&in.Mode!=RuntimeCandidate{f.block("mode_invalid")}
	if in.EvaluatedAt.IsZero(){f.hold("evaluation_time_missing")}
	if !validDigest(in.ScenarioSHA256){f.block("scenario_digest_invalid")}
	if !validDigest(in.RecoveryBoundarySHA256){f.block("recovery_boundary_digest_invalid")}
	if !validRevision(in.CodeRevision){f.block("code_revision_invalid")}
	if in.InitialFaultIDs==nil{f.block("initial_fault_ids_missing")}
	initial,ok:=canonicalFaults(in.InitialFaultIDs)
	if !ok{f.block("initial_fault_ids_invalid_or_duplicated")}
	if len(initial)>MaxFaults{f.block("fault_limit_exceeded")}
	if len(in.Observations)>MaxObservations{f.block("observation_limit_exceeded")}
	if in.ExpectedFailureSignatureSHA256!=""&&!validDigest(in.ExpectedFailureSignatureSHA256){f.block("expected_failure_signature_invalid")}
	index:=map[string]Observation{}
	for _,o:=range in.Observations{
		if o.FaultIDs==nil{f.block("observation_fault_ids_missing");continue}
		ids,valid:=canonicalFaults(o.FaultIDs)
		if !valid{f.block("observation_fault_ids_invalid");continue}
		digest:=digestJSON(ids)
		if !validDigest(o.FaultSetSHA256)||o.FaultSetSHA256!=digest{f.block("observation_fault_set_digest_mismatch");continue}
		if o.ScenarioSHA256!=in.ScenarioSHA256||o.RecoveryBoundarySHA256!=in.RecoveryBoundarySHA256||o.CodeRevision!=in.CodeRevision{
			f.block("observation_scope_mismatch");continue
		}
		if o.ObservedAt.IsZero()||(!in.EvaluatedAt.IsZero()&&o.ObservedAt.After(in.EvaluatedAt)){f.block("observation_time_invalid");continue}
		if !validVerdict(o.Verdict)||!validDigest(o.FailureSignatureSHA256)||!validDigest(o.OutcomeReceiptSHA256){f.block("observation_result_invalid");continue}
		if _,exists:=index[digest];exists{f.block("duplicate_fault_set_observation");continue}
		index[digest]=o
	}
	return index
}

func Minimize(in Input)Result{
	out:=Result{
		Profile:"mon039-counterfactual-fault-reduction-result-v1",Version:Version,
		TrialID:in.TrialID,Mode:in.Mode,MinimalFaultIDs:[]string{},
		MinimalityKind:"not_established",EvidenceComplete:false,
		CampaignVerdict:Indeterminate,ProductionActionPermitted:false,
		Steps:[]EvaluationStep{},ReasonCodes:[]string{},
	}
	f:=newFlags()
	index:=validateInput(in,f)
	if f.blocked{
		out.CampaignVerdict=Blocked
		out.ReasonCodes=f.codes()
		return out
	}
	initial,ok:=canonicalFaults(in.InitialFaultIDs)
	if !ok||len(initial)>MaxFaults{
		f.block("initial_fault_set_invalid")
		out.CampaignVerdict=Blocked
		out.ReasonCodes=f.codes()
		return out
	}
	evaluations:=0
	incomplete:=false
	lookup:=func(ids []string)(Observation,bool){
		if evaluations>=MaxEvaluations{incomplete=true;f.hold("evaluation_limit_reached");return Observation{},false}
		canonical,valid:=canonicalFaults(ids)
		if !valid{f.block("candidate_fault_set_invalid");return Observation{},false}
		digest:=digestJSON(canonical)
		o,exists:=index[digest]
		if !exists{incomplete=true;f.hold("subset_observation_missing:"+digest);return Observation{},false}
		if !o.Complete{incomplete=true;f.hold("subset_observation_incomplete:"+digest);return Observation{},false}
		if !sameIDs(o.FaultIDs,canonical){f.block("fault_set_canonicalization_mismatch");return Observation{},false}
		evaluations++
		target:=sameFailure(o,out.FullFaultSetVerdict,out.TargetFailureSignatureSHA256)
		out.Steps=append(out.Steps,EvaluationStep{
			Sequence:evaluations,FaultIDs:canonical,FaultSetSHA256:digest,
			Verdict:o.Verdict,FailureSignatureSHA256:o.FailureSignatureSHA256,
			OutcomeReceiptSHA256:o.OutcomeReceiptSHA256,TargetFailurePreserved:target,
		})
		return o,true
	}
	full,exists:=lookup(initial)
	if !exists{
		out.SubsetEvaluations=evaluations
		out.EvidenceComplete=false
		out.CampaignVerdict=f.verdict()
		out.ReasonCodes=f.codes()
		return out
	}
	out.FullFaultSetVerdict=full.Verdict
	out.TargetFailureSignatureSHA256=full.FailureSignatureSHA256
	if len(out.Steps)>0{out.Steps[0].TargetFailurePreserved=true}
	if full.Verdict==Qualified{
		f.hold("initial_fault_set_did_not_reproduce_failure")
		out.MinimizationKind="not_applicable"
		out.SubsetEvaluations=evaluations
		out.EvidenceComplete=true
		out.CampaignVerdict=Indeterminate
		out.ReasonCodes=f.codes()
		return out
	}
	if in.ExpectedFailureSignatureSHA256!=""&&in.ExpectedFailureSignatureSHA256!=full.FailureSignatureSHA256{
		f.block("initial_failure_signature_mismatch")
		out.SubsetEvaluations=evaluations
		out.CampaignVerdict=Blocked
		out.ReasonCodes=f.codes()
		return out
	}

	current:=append([]string(nil),initial...)
	if len(initial)<=ExhaustiveFaultLimit{
		found:=false
		var candidate []string
		for size:=0;size<len(initial);size++{
			combinations(initial,size,func(ids []string)bool{
				o,ok:=lookup(ids)
				if !ok{return false}
				if sameFailure(o,out.FullFaultSetVerdict,out.TargetFailureSignatureSHA256){
					found=true
					candidate=append([]string(nil),ids...)
					return true
				}
				return false
			})
			if found{break}
			if f.blocked||evaluations>=MaxEvaluations{break}
		}
		if found{
			current=candidate
			if !incomplete&&!f.blocked{out.MinimalityKind="global_candidate_minimum"}else{out.MinimalityKind="not_established"}
		}else if !incomplete&&!f.blocked{
			current=initial
			out.MinimalityKind="global_candidate_minimum"
		}
	}else{
		changed:=true
		for changed&&!f.blocked&&!incomplete{
			changed=false
			for i:=len(current)-1;i>=0;i--{
				candidate:=make([]string,0,len(current)-1)
				candidate=append(candidate,current[:i]...)
				candidate=append(candidate,current[i+1:]...)
				o,ok:=lookup(candidate)
				if !ok{incomplete=true;break}
				if sameFailure(o,out.FullFaultSetVerdict,out.TargetFailureSignatureSHA256){
					current=candidate
					changed=true
					break
				}
			}
		}
		if !incomplete&&!f.blocked&&!changed{out.MinimalityKind="one_minimal"}
	}
	out.MinimalFaultIDs=append([]string(nil),current...)
	out.MinimalFaultSetSHA256=digestJSON(current)
	out.SubsetEvaluations=evaluations
	out.EvidenceComplete=!incomplete&&!f.blocked
	if incomplete||f.blocked{out.MinimalityKind="not_established";f.hold("minimality_not_established")}
	if in.Mode==OfflineFixture{f.hold("offline_fixture_is_not_causal_proof")}
	if in.Mode==RuntimeCandidate{f.hold("runtime_replay_authentication_not_implemented")}
	out.CampaignVerdict=f.verdict()
	out.ReasonCodes=f.codes()
	return out
}

func combinations(ids []string,size int,visit func([]string)bool){
	chosen:=make([]string,0,size)
	var walk func(start int)bool
	walk=func(start int)bool{
		if len(chosen)==size{return visit(append([]string(nil),chosen...))}
		remaining:=size-len(chosen)
		for i:=start;i<=len(ids)-remaining;i++{
			chosen=append(chosen,ids[i])
			if walk(i+1){return true}
			chosen=chosen[:len(chosen)-1]
		}
		return false
	}
	_ = walk(0)
}

func FormatEvaluationError(err error)string{return fmt.Sprint(err)}
