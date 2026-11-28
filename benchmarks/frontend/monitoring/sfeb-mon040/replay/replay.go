// Package replay qualifies the reproducibility of one exact recovery failure
// signature over a preregistered set of replay observations. It is not a fault injector.
package replay

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"strings"
	"time"
)

const (
	Profile = "mon040-replay-reproducibility-input-v1"
	Version = "1.0.0"
	MaxRuns = 200
	MaxFaults = 256
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
	ReplayID string `json:"replay_id"`
	TrialID string `json:"trial_id"`
	Seed int64 `json:"seed"`
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
	FaultIDs []string `json:"fault_ids"`
	FaultSetSHA256 string `json:"fault_set_sha256"`
	TargetFailureSignatureSHA256 string `json:"target_failure_signature_sha256"`
	RequiredRuns int `json:"required_runs"`
	MinimumReliability float64 `json:"minimum_reliability"`
	ConfidenceLevel float64 `json:"confidence_level"`
	Observations []Observation `json:"observations"`
}
type ReplaySummary struct {
	ReplayID string `json:"replay_id"`
	Seed int64 `json:"seed"`
	Verdict Verdict `json:"verdict"`
	FailureSignatureSHA256 string `json:"failure_signature_sha256"`
	TargetSignatureMatched bool `json:"target_signature_matched"`
	OutcomeReceiptSHA256 string `json:"outcome_receipt_sha256"`
}
type Result struct {
	Profile string `json:"profile"`
	Version string `json:"version"`
	TrialID string `json:"trial_id"`
	ReproducibilityVerdict Verdict `json:"reproducibility_verdict"`
	CampaignVerdict Verdict `json:"campaign_verdict"`
	ProductionUsePermitted bool `json:"production_use_permitted"`
	RequiredRuns int `json:"required_runs"`
	ObservedRuns int `json:"observed_runs"`
	TargetSignatureMatches int `json:"target_signature_matches"`
	OtherBlockedOutcomes int `json:"other_blocked_outcomes"`
	QualifiedOutcomes int `json:"qualified_outcomes"`
	IndeterminateOutcomes int `json:"indeterminate_outcomes"`
	ObservedRate float64 `json:"observed_rate"`
	OneSidedConfidenceLowerBound float64 `json:"one_sided_confidence_lower_bound"`
	ConfidenceLevel float64 `json:"confidence_level"`
	MinimumReliability float64 `json:"minimum_reliability"`
	EvidenceComplete bool `json:"evidence_complete"`
	Replays []ReplaySummary `json:"replays"`
	ReasonCodes []string `json:"reason_codes"`
}

type flags struct { blocked bool; unknown bool; reasons map[string]struct{} }
func newFlags()*flags{return &flags{reasons:map[string]struct{}{}}}
func(f *flags)block(s string){f.blocked=true;f.reasons[s]=struct{}{}}
func(f *flags)hold(s string){f.unknown=true;f.reasons[s]=struct{}{}}
func(f *flags)verdict()Verdict{if f.blocked{return Blocked};if f.unknown{return Indeterminate};return Qualified}
func(f *flags)codes()[]string{out:=make([]string,0,len(f.reasons));for s:=range f.reasons{out=append(out,s)};sort.Strings(out);return out}
func validID(s string)bool{return s!=""&&len(s)<=256&&!strings.ContainsAny(s," 	
")}
func validDigest(s string)bool{if len(s)!=64{return false};_,e:=hex.DecodeString(s);return e==nil&&s==strings.ToLower(s)}
func validRevision(s string)bool{if len(s)!=40&&len(s)!=64{return false};if s!=strings.ToLower(s){return false};_,e:=hex.DecodeString(s);return e==nil}
func validVerdict(v Verdict)bool{return v==Qualified||v==Blocked||v==Indeterminate}
func digestJSON(v any)string{b,e:=json.Marshal(v);if e!=nil{return ""};h:=sha256.Sum256(b);return hex.EncodeToString(h[:])}
func ObservationReceiptDigest(o Observation)string{o.OutcomeReceiptSHA256="";return digestJSON(o)}
func binomialTail(n,k int,p float64)float64{
	if k<=0{return 1}
	if k>n{return 0}
	if p<=0{return 0}
	if p>=1{return 1}
	terms:=make([]float64,0,n-k+1)
	maxLog:=math.Inf(-1)
	for x:=k;x<=n;x++{
		ln1,_:=math.Lgamma(float64(n+1))
		ln2,_:=math.Lgamma(float64(x+1))
		ln3,_:=math.Lgamma(float64(n-x+1))
		v:=ln1-ln2-ln3+float64(x)*math.Log(p)+float64(n-x)*math.Log1p(-p)
		terms=append(terms,v)
		if v>maxLog{maxLog=v}
	}
	sum:=0.0
	for _,v:=range terms{sum+=math.Exp(v-maxLog)}
	v:=math.Exp(maxLog)*sum
	if v>1{return 1};if v<0{return 0};return v
}

// OneSidedClopperPearsonLower returns the exact one-sided lower confidence
// bound for a Bernoulli success probability under independent, identically
// distributed trials. Independence is an experiment-design assumption.
func OneSidedClopperPearsonLower(successes,trials int,confidence float64)float64{
	if trials<=0||successes<0||successes>trials||confidence<=0||confidence>=1{return 0}
	if successes==0{return 0}
	alpha:=1-confidence
	if successes==trials{return math.Pow(alpha,1/float64(trials))}
	lo,hi:=0.0,float64(successes)/float64(trials)
	for i:=0;i<100;i++{
		mid:=(lo+hi)/2
		if binomialTail(trials,successes,mid)<alpha{lo=mid}else{hi=mid}
	}
	return (lo+hi)/2
}

func Evaluate(in Input)Result{
	out:=Result{
		Profile:"mon040-replay-reproducibility-result-v1",Version:Version,TrialID:in.TrialID,
		CampaignVerdict:Indeterminate,ProductionUsePermitted:false,
		RequiredRuns:in.RequiredRuns,ConfidenceLevel:in.ConfidenceLevel,MinimumReliability:in.MinimumReliability,
		Replays:[]ReplaySummary{},ReasonCodes:[]string{},
	}
	f:=newFlags()
	if in.Profile!=Profile||in.Version!=Version{f.block("profile_version_invalid")}
	if !validID(in.TrialID){f.block("trial_id_invalid")}
	if in.Mode!=OfflineFixture&&in.Mode!=RuntimeCandidate{f.block("mode_invalid")}
	if in.EvaluatedAt.IsZero(){f.hold("evaluation_time_missing")}
	if !validDigest(in.ScenarioSHA256)||!validDigest(in.RecoveryBoundarySHA256){f.block("scenario_or_boundary_digest_invalid")}
	if !validRevision(in.CodeRevision){f.block("code_revision_invalid")}
	if in.FaultIDs==nil||len(in.FaultIDs)==0{f.block("fault_ids_missing")}else if len(in.FaultIDs)>MaxFaults{f.block("fault_id_limit_exceeded")}else{ids:=append([]string{},in.FaultIDs...);for _,id:=range ids{if !validID(id){f.block("fault_id_invalid")}};sort.Strings(ids);for i:=1;i<len(ids);i++{if ids[i]==ids[i-1]{f.block("fault_ids_duplicated")}};if digestJSON(ids)!=in.FaultSetSHA256{f.block("fault_set_digest_mismatch")}}
	if !validDigest(in.FaultSetSHA256){f.block("fault_set_digest_invalid")}
	if !validDigest(in.TargetFailureSignatureSHA256){f.block("target_failure_signature_invalid")}
	if in.RequiredRuns<1||in.RequiredRuns>MaxRuns{f.block("required_run_count_out_of_range")}
	if in.MinimumReliability<=0||in.MinimumReliability>1||math.IsNaN(in.MinimumReliability){f.block("minimum_reliability_invalid")}
	if in.ConfidenceLevel<=0||in.ConfidenceLevel>=1||math.IsNaN(in.ConfidenceLevel){f.block("confidence_level_invalid")}
	if len(in.Observations)>MaxRuns{f.block("observation_limit_exceeded")}
	if f.blocked{out.ReproducibilityVerdict=Blocked;out.CampaignVerdict=Blocked;out.ReasonCodes=f.codes();return out}
	if len(in.Observations)<in.RequiredRuns{f.hold("preregistered_replay_count_incomplete")}
	if len(in.Observations)>in.RequiredRuns{f.block("replay_count_exceeds_preregistered_run_count")}

	seenIDs:=map[string]struct{}{}
	seenSeeds:=map[int64]struct{}{}
	allObservationsComplete:=true
	for _,o:=range in.Observations{
		if !validID(o.ReplayID)||!validID(o.TrialID){f.block("replay_identity_invalid");continue}
		if _,ok:=seenIDs[o.ReplayID];ok{f.block("duplicate_replay_id");continue};seenIDs[o.ReplayID]=struct{}{}
		if _,ok:=seenSeeds[o.Seed];ok{f.block("duplicate_replay_seed");continue};seenSeeds[o.Seed]=struct{}{}
		if o.Seed<0{f.block("replay_seed_invalid")}
		if o.TrialID!=in.TrialID||o.FaultSetSHA256!=in.FaultSetSHA256||o.ScenarioSHA256!=in.ScenarioSHA256||o.RecoveryBoundarySHA256!=in.RecoveryBoundarySHA256||o.CodeRevision!=in.CodeRevision{f.block("replay_scope_mismatch")}
		if o.ObservedAt.IsZero()||(!in.EvaluatedAt.IsZero()&&o.ObservedAt.After(in.EvaluatedAt)){f.block("replay_time_invalid")}
		if !validVerdict(o.Verdict)||!validDigest(o.FailureSignatureSHA256)||!validDigest(o.OutcomeReceiptSHA256){f.block("replay_outcome_invalid")}
		if o.OutcomeReceiptSHA256!=ObservationReceiptDigest(o){f.block("replay_receipt_digest_mismatch")}
		if !o.Complete{allObservationsComplete=false;f.hold("replay_observation_incomplete")}
		if o.Verdict==Indeterminate{f.hold("replay_outcome_indeterminate")}
		target:=o.Verdict==Blocked&&o.FailureSignatureSHA256==in.TargetFailureSignatureSHA256
		out.Replays=append(out.Replays,ReplaySummary{ReplayID:o.ReplayID,Seed:o.Seed,Verdict:o.Verdict,FailureSignatureSHA256:o.FailureSignatureSHA256,TargetSignatureMatched:target,OutcomeReceiptSHA256:o.OutcomeReceiptSHA256})
		out.ObservedRuns++
		if target{out.TargetSignatureMatches++}else{
			switch o.Verdict{case Blocked:out.OtherBlockedOutcomes++;case Qualified:out.QualifiedOutcomes++;case Indeterminate:out.IndeterminateOutcomes++}
		}
	}
	if f.blocked{out.ReproducibilityVerdict=Blocked;out.CampaignVerdict=Blocked;out.ReasonCodes=f.codes();return out}
	if out.ObservedRuns>0{
		out.ObservedRate=float64(out.TargetSignatureMatches)/float64(out.ObservedRuns)
		out.OneSidedConfidenceLowerBound=OneSidedClopperPearsonLower(out.TargetSignatureMatches,out.ObservedRuns,in.ConfidenceLevel)
	}
	out.EvidenceComplete=len(in.Observations)==in.RequiredRuns&&out.ObservedRuns==in.RequiredRuns&&allObservationsComplete
	if !out.EvidenceComplete{f.hold("replay_evidence_incomplete")}
	if out.EvidenceComplete&&!f.unknown{
		if out.OneSidedConfidenceLowerBound>=in.MinimumReliability{
			out.ReproducibilityVerdict=Qualified
		}else{
			out.ReproducibilityVerdict=Blocked
			f.block("failure_signature_reproducibility_below_threshold")
		}
	}else{
		out.ReproducibilityVerdict=Indeterminate
	}
	if in.Mode==OfflineFixture{f.hold("offline_fixture_is_not_live_reproducibility_evidence")}
	if in.Mode==RuntimeCandidate{f.hold("runtime_replay_custody_not_implemented")}
	out.CampaignVerdict=f.verdict()
	out.ReasonCodes=f.codes()
	return out
}

func FormatEvaluationError(err error)string{return fmt.Sprint(err)}
