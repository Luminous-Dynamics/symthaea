// Package continuity evaluates whether qualified recovery evidence remains
// usable after its supporting dependency history advances.
package continuity

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"sort"
	"strings"
	"time"
	"unicode"
)

const (
	Profile = "mon036-recovery-evidence-continuity-input-v1"
	Version = "1.0.0"
	GradeRuntimeIndependent = "runtime_independently_verified"
	MaxEvents = 100000
	MaxClaims = 10000
	MaxStates = 250000
	MaxSupportPerClaim = 2000
	MaxIdentifierBytes = 256
)

type Disposition string
const (
	Retain Disposition = "retain"
	Revalidate Disposition = "revalidate"
	Invalidate Disposition = "invalidate"
	Unresolved Disposition = "unresolved"
)

type EventKind string
const (
	DependencyChanged EventKind = "dependency_changed"
	VerificationChanged EventKind = "verification_changed"
	AuthorityRevoked EventKind = "authority_revoked"
	ScopeChanged EventKind = "scope_changed"
	Contradiction EventKind = "contradiction"
	UnknownChange EventKind = "unknown_change"
)

type Event struct {
	Sequence uint64 `json:"sequence"`
	EventID string `json:"event_id"`
	DependencyID string `json:"dependency_id,omitempty"`
	Kind EventKind `json:"kind"`
	BeforeRevision uint64 `json:"before_revision,omitempty"`
	AfterRevision uint64 `json:"after_revision,omitempty"`
	BeforeDigest string `json:"before_digest,omitempty"`
	AfterDigest string `json:"after_digest,omitempty"`
	OccurredAt time.Time `json:"occurred_at"`
	PreviousDigest string `json:"previous_digest,omitempty"`
	Digest string `json:"digest"`
}
type Ledger struct {
	Complete bool `json:"complete"`
	ThroughSequence uint64 `json:"through_sequence"`
	HeadDigest string `json:"head_digest,omitempty"`
	CapturedAt time.Time `json:"captured_at"`
	ExpiresAt time.Time `json:"expires_at"`
	Events []Event `json:"events"`
}
type DependencyBinding struct {
	DependencyID string `json:"dependency_id"`
	Revision uint64 `json:"revision"`
	Digest string `json:"digest"`
}
type Claim struct {
	ClaimID string `json:"claim_id"`
	EvidenceID string `json:"evidence_id"`
	EvidenceGrade string `json:"evidence_grade"`
	ClosureComplete bool `json:"closure_complete"`
	ObservedThroughSequence uint64 `json:"observed_through_sequence"`
	ObservedThroughDigest string `json:"observed_through_digest,omitempty"`
	ObservedAt time.Time `json:"observed_at"`
	ExpiresAt time.Time `json:"expires_at"`
	Support []DependencyBinding `json:"support"`
}
type DependencyState struct {
	DependencyID string `json:"dependency_id"`
	Revision uint64 `json:"revision"`
	Digest string `json:"digest"`
}
type Input struct {
	Profile string `json:"profile"`
	Version string `json:"version"`
	Now time.Time `json:"now"`
	Ledger Ledger `json:"ledger"`
	CurrentStates []DependencyState `json:"current_states"`
	Claims []Claim `json:"claims"`
}
type ClaimResult struct {
	ClaimID string `json:"claim_id"`
	Disposition Disposition `json:"disposition"`
	AffectedSequences []uint64 `json:"affected_sequences"`
	ReasonCodes []string `json:"reason_codes"`
}
type Result struct {
	Profile string `json:"profile"`
	Version string `json:"version"`
	LedgerIntegrityValid bool `json:"ledger_integrity_valid"`
	LedgerComplete bool `json:"ledger_complete"`
	Claims []ClaimResult `json:"claims"`
	ReasonCodes []string `json:"reason_codes"`
}
type flags struct { invalid, unknown bool; reasons map[string]struct{} }
func newFlags() *flags { return &flags{reasons:map[string]struct{}{}} }
func (f *flags) invalidate(s string) { f.invalid=true; f.reasons[s]=struct{}{} }
func (f *flags) unresolved(s string) { f.unknown=true; f.reasons[s]=struct{}{} }
func (f *flags) codes() []string {
	out:=make([]string,0,len(f.reasons))
	for s:=range f.reasons { out=append(out,s) }
	sort.Strings(out)
	return out
}
func validID(s string) bool {
	return s!="" && len(s)<=MaxIdentifierBytes &&
		strings.IndexFunc(s,func(r rune)bool{return unicode.IsSpace(r)||unicode.IsControl(r)})<0
}
func validDigest(s string) bool {
	if len(s)!=64 { return false }
	_,err:=hex.DecodeString(s)
	return err==nil && s==strings.ToLower(s)
}
type eventDigestPayload struct {
	Sequence uint64 `json:"sequence"`
	EventID string `json:"event_id"`
	DependencyID string `json:"dependency_id,omitempty"`
	Kind EventKind `json:"kind"`
	BeforeRevision uint64 `json:"before_revision,omitempty"`
	AfterRevision uint64 `json:"after_revision,omitempty"`
	BeforeDigest string `json:"before_digest,omitempty"`
	AfterDigest string `json:"after_digest,omitempty"`
	OccurredAtUTC string `json:"occurred_at_utc"`
	PreviousDigest string `json:"previous_digest,omitempty"`
}
// SealEvent computes the v1 event digest using UTC RFC3339Nano time.
func SealEvent(e Event) Event {
	p:=eventDigestPayload{e.Sequence,e.EventID,e.DependencyID,e.Kind,e.BeforeRevision,e.AfterRevision,e.BeforeDigest,e.AfterDigest,e.OccurredAt.UTC().Format(time.RFC3339Nano),e.PreviousDigest}
	raw,_:=json.Marshal(p)
	sum:=sha256.Sum256(raw)
	e.Digest=hex.EncodeToString(sum[:])
	return e
}
func eventError(e Event) string {
	if !validID(e.EventID) || e.OccurredAt.IsZero() { return "event_identity_or_timestamp_invalid" }
	switch e.Kind {
	case DependencyChanged,VerificationChanged:
		if !validID(e.DependencyID)||e.BeforeRevision==0||e.AfterRevision<=e.BeforeRevision||!validDigest(e.BeforeDigest)||!validDigest(e.AfterDigest) { return "dependency_change_event_malformed" }
	case AuthorityRevoked,ScopeChanged,Contradiction:
		if !validID(e.DependencyID) { return "targeted_event_dependency_missing" }
	case UnknownChange:
		if e.DependencyID!=""&&!validID(e.DependencyID) { return "unknown_event_dependency_invalid" }
	default:
		return "unsupported_event_kind"
	}
	return ""
}
type ledgerView struct {
	integrity,complete bool
	bySequence map[uint64]Event
	byDependency map[string][]Event
	globalUnknown []Event
	flags *flags
}
func validateLedger(l Ledger,now time.Time) ledgerView {
	v:=ledgerView{integrity:true,complete:l.Complete,bySequence:map[uint64]Event{},byDependency:map[string][]Event{},globalUnknown:[]Event{},flags:newFlags()}
	if len(l.Events)>MaxEvents { v.integrity=false; v.flags.invalidate("event_count_limit_exceeded") }
	if now.IsZero()||l.CapturedAt.IsZero()||l.ExpiresAt.IsZero() {
		v.complete=false; v.flags.unresolved("ledger_freshness_missing")
	} else {
		if l.CapturedAt.After(now) { v.integrity=false; v.flags.invalidate("ledger_capture_from_future") }
		if !l.ExpiresAt.After(now) { v.complete=false; v.flags.unresolved("ledger_snapshot_expired") }
		if l.ExpiresAt.Before(l.CapturedAt) { v.integrity=false; v.flags.invalidate("ledger_validity_interval_invalid") }
	}
	if !l.Complete { v.complete=false; v.flags.unresolved("event_ledger_not_complete") }
	prevDigest:=""
	prevSeq:=uint64(0)
	seenIDs:=map[string]struct{}{}
	for _,e:=range l.Events {
		if e.Sequence<=prevSeq { v.integrity=false; v.flags.invalidate("event_sequence_not_monotonic") } else if e.Sequence!=prevSeq+1 { v.complete=false; v.flags.unresolved("event_sequence_gap") }
		if e.PreviousDigest!=prevDigest {
			if e.Sequence==prevSeq+1 { v.integrity=false; v.flags.invalidate("event_previous_digest_mismatch") } else { v.complete=false; v.flags.unresolved("event_chain_gap_unverifiable") }
		}
		if _,ok:=seenIDs[e.EventID];ok { v.integrity=false; v.flags.invalidate("duplicate_event_id") }
		seenIDs[e.EventID]=struct{}{}
		if reason:=eventError(e);reason!="" { v.integrity=false; v.flags.invalidate(reason+":"+e.EventID) }
		if !now.IsZero()&&e.OccurredAt.After(now) { v.integrity=false; v.flags.invalidate("event_timestamp_from_future:"+e.EventID) }
		if !l.CapturedAt.IsZero()&&e.OccurredAt.After(l.CapturedAt) { v.integrity=false; v.flags.invalidate("event_after_ledger_capture:"+e.EventID) }
		if !validDigest(e.Digest)||SealEvent(e).Digest!=e.Digest { v.integrity=false; v.flags.invalidate("event_digest_mismatch:"+e.EventID) }
		v.bySequence[e.Sequence]=e
		if e.Kind==UnknownChange&&e.DependencyID=="" { v.globalUnknown=append(v.globalUnknown,e) } else if e.DependencyID!="" { v.byDependency[e.DependencyID]=append(v.byDependency[e.DependencyID],e) }
		prevDigest=e.Digest
		prevSeq=e.Sequence
	}
	if l.ThroughSequence<prevSeq { v.integrity=false; v.flags.invalidate("ledger_watermark_before_last_event") } else if l.ThroughSequence>prevSeq { v.complete=false; v.flags.unresolved("ledger_watermark_not_fully_observed") }
	if l.HeadDigest!=prevDigest { v.integrity=false; v.flags.invalidate("ledger_head_digest_mismatch") }
	return v
}
func Evaluate(in Input) Result {
	out:=Result{Profile:"mon036-recovery-evidence-continuity-result-v1",Version:Version,LedgerIntegrityValid:true,LedgerComplete:true,Claims:[]ClaimResult{},ReasonCodes:[]string{}}
	global:=newFlags()
	if in.Profile!=Profile||in.Version!=Version { global.invalidate("unsupported_profile_or_version") }
	if in.Now.IsZero() { global.unresolved("evaluation_time_missing") }
	if len(in.Claims)>MaxClaims { global.invalidate("claim_count_limit_exceeded") }
	if len(in.CurrentStates)>MaxStates { global.invalidate("dependency_state_count_limit_exceeded") }
	ledger:=validateLedger(in.Ledger,in.Now)
	for r:=range ledger.flags.reasons { global.reasons[r]=struct{}{} }
	global.invalid=global.invalid||ledger.flags.invalid
	global.unknown=global.unknown||ledger.flags.unknown
	out.LedgerIntegrityValid=ledger.integrity
	out.LedgerComplete=ledger.complete
	states:=map[string]DependencyState{}
	badStates:=map[string]string{}
	for _,s:=range in.CurrentStates {
		if !validID(s.DependencyID) { global.unresolved("dependency_state_identifier_unavailable"); continue }
		if s.Revision==0||!validDigest(s.Digest) { badStates[s.DependencyID]="dependency_state_malformed"; delete(states,s.DependencyID); continue }
		if _,ok:=states[s.DependencyID];ok { badStates[s.DependencyID]="duplicate_dependency_state"; delete(states,s.DependencyID); continue }
		if _,bad:=badStates[s.DependencyID];!bad { states[s.DependencyID]=s }
	}
	idCounts:=map[string]int{}
	for _,c:=range in.Claims { idCounts[c.ClaimID]++ }
	for _,c:=range in.Claims {
		r:=evaluateClaim(c,in,ledger,states,badStates,global)
		if idCounts[c.ClaimID]>1 { r.Disposition=Invalidate; r.ReasonCodes=append(r.ReasonCodes,"duplicate_claim_id"); sort.Strings(r.ReasonCodes) }
		out.Claims=append(out.Claims,r)
	}
	sort.Slice(out.Claims,func(i,j int)bool{return out.Claims[i].ClaimID<out.Claims[j].ClaimID})
	out.ReasonCodes=global.codes()
	return out
}
func evaluateClaim(c Claim,in Input,l ledgerView,states map[string]DependencyState,bad map[string]string,global *flags) ClaimResult {
	f:=newFlags()
	r:=ClaimResult{ClaimID:c.ClaimID,Disposition:Retain,AffectedSequences:[]uint64{},ReasonCodes:[]string{}}
	needsRevalidate:=false
	if !validID(c.ClaimID)||!validID(c.EvidenceID) { f.invalidate("claim_identity_invalid") }
	if c.EvidenceGrade!=GradeRuntimeIndependent { f.unresolved("claim_not_runtime_independently_verified") }
	if !c.ClosureComplete { f.unresolved("claim_support_closure_incomplete") }
	if len(c.Support)==0||len(c.Support)>MaxSupportPerClaim { f.unresolved("claim_support_set_missing_or_over_limit") }
	if c.ObservedAt.IsZero()||c.ExpiresAt.IsZero()||in.Now.IsZero() { f.unresolved("claim_freshness_missing") } else {
		if c.ObservedAt.After(in.Now) { f.invalidate("claim_observed_from_future") }
		if !c.ExpiresAt.After(c.ObservedAt) { f.invalidate("claim_validity_interval_invalid") }
		if !c.ExpiresAt.After(in.Now) { needsRevalidate=true; f.reasons["claim_evidence_expired"]=struct{}{} }
	}
	if !in.Ledger.CapturedAt.IsZero()&&!c.ObservedAt.IsZero()&&in.Ledger.CapturedAt.Before(c.ObservedAt) { f.unresolved("ledger_capture_precedes_claim") }
	if c.ObservedThroughSequence>in.Ledger.ThroughSequence { f.unresolved("claim_watermark_ahead_of_ledger")
	} else if c.ObservedThroughSequence==0 {
		if c.ObservedThroughDigest!="" { f.invalidate("claim_empty_watermark_has_digest") }
	} else if e,ok:=l.bySequence[c.ObservedThroughSequence];!ok { f.unresolved("claim_watermark_event_unavailable")
	} else if e.Digest!=c.ObservedThroughDigest { f.invalidate("claim_watermark_digest_mismatch")
	} else if !e.OccurredAt.Before(c.ObservedAt) { f.unresolved("claim_watermark_event_not_preceding_observation") }

	bindings:=map[string]DependencyBinding{}
	for _,b:=range c.Support {
		if !validID(b.DependencyID)||b.Revision==0||!validDigest(b.Digest) { f.invalidate("claim_support_binding_malformed"); continue }
		if _,ok:=bindings[b.DependencyID];ok { f.invalidate("duplicate_claim_support_dependency:"+b.DependencyID); continue }
		bindings[b.DependencyID]=b
	}
	for id,binding:=range bindings {
		if reason,ok:=bad[id];ok { f.unresolved(reason+":"+id); continue }
		state,ok:=states[id]
		if !ok { f.unresolved("current_dependency_state_missing:"+id); continue }
		wantRev,wantDigest:=binding.Revision,binding.Digest
		changed:=false
		for _,e:=range l.byDependency[id] {
			if e.Sequence<=c.ObservedThroughSequence { continue }
			r.AffectedSequences=append(r.AffectedSequences,e.Sequence)
			switch e.Kind {
			case DependencyChanged,VerificationChanged:
				if e.BeforeRevision!=wantRev||e.BeforeDigest!=wantDigest { f.invalidate("dependency_event_predecessor_mismatch:"+e.EventID) } else { wantRev,wantDigest=e.AfterRevision,e.AfterDigest; changed=true }
			case AuthorityRevoked,ScopeChanged,Contradiction:
				f.invalidate(string(e.Kind)+":"+e.EventID)
			case UnknownChange:
				f.unresolved("unknown_dependency_change:"+e.EventID)
			}
		}
		for _,e:=range l.globalUnknown {
			if e.Sequence>c.ObservedThroughSequence { r.AffectedSequences=append(r.AffectedSequences,e.Sequence); f.unresolved("global_unknown_change:"+e.EventID) }
		}
		if changed {
			if state.Revision!=wantRev||state.Digest!=wantDigest { f.unresolved("current_state_not_matched_to_event_tail:"+id) } else { f.reasons["dependency_changed_since_claim:"+id]=struct{}{} }
		} else if state.Revision!=binding.Revision||state.Digest!=binding.Digest { f.unresolved("untracked_dependency_drift:"+id) }
	}
	if !l.integrity { f.invalidate("event_ledger_integrity_failed") }
	if !l.complete { f.unresolved("event_ledger_completeness_unproven") }
	if global.invalid { f.invalidate("global_input_integrity_failed") }
	if global.unknown { f.unresolved("global_input_incomplete") }
	if len(r.AffectedSequences)>0 {
		sort.Slice(r.AffectedSequences,func(i,j int)bool{return r.AffectedSequences[i]<r.AffectedSequences[j]})
		unique:=r.AffectedSequences[:0]
		for _,s:=range r.AffectedSequences { if len(unique)==0||unique[len(unique)-1]!=s { unique=append(unique,s) } }
		r.AffectedSequences=unique
	}
	switch {
	case f.invalid: r.Disposition=Invalidate
	case f.unknown: r.Disposition=Unresolved
	case needsRevalidate||hasReasonPrefix(f.reasons,"dependency_changed_since_claim:"): r.Disposition=Revalidate
	default: r.Disposition=Retain
	}
	r.ReasonCodes=f.codes()
	return r
}
func hasReasonPrefix(m map[string]struct{},prefix string)bool {
	for reason:=range m { if strings.HasPrefix(reason,prefix) { return true } }
	return false
}
