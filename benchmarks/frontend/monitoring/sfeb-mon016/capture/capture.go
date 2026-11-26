// Package capture evaluates bounded source-capture claims over normalized
// evidence. Cryptographic verification remains the responsibility of maintained
// source, manifest, and witness verifiers.
package capture

import (
	"sort"
)

type Verification string
const (
	Verified Verification = "verified"
	Missing Verification = "missing"
	Invalid Verification = "invalid"
)
type DigestContext struct { Profile, Version, PayloadType, Canonicalization, Encoding, Purpose, Subject string }
type DigestRelation string
const (
	DigestEqual DigestRelation = "equal"
	DigestDifferent DigestRelation = "different"
	DigestIncomparable DigestRelation = "incomparable"
	DigestIndeterminate DigestRelation = "indeterminate"
	DigestInvalid DigestRelation = "invalid"
)
type DigestEvidence struct {
	Present bool
	Valid bool
	Algorithm string
	Value string
	Context DigestContext
	Verification Verification
}
type Boundary struct {
	BoundaryID, SourceID, StreamID, PartitionID, SnapshotToken string
	QueryDigest DigestEvidence
	StartExclusive, EndInclusive uint64
}
type SourceEvidence struct {
	Verification, QueryBindingVerification, WatermarkVerification Verification
	BoundaryID, SourceID, StreamID, PartitionID, SnapshotToken string
	BoundQueryDigest DigestEvidence
	Terminal bool
	Watermark, AttestedWatermark uint64
	ControlRoot, AttestedControlRoot string
}
type CapturedItem struct { Position uint64; EventID string; Digest DigestEvidence }
type Manifest struct { Verification Verification; Boundary Boundary; Source SourceEvidence; Items []CapturedItem; Digest DigestEvidence }
type Witness struct { Verification Verification; BoundaryID string; ManifestDigest DigestEvidence; ControlRoot string }
type CoverageStatus string
const (
	CoverageComplete CoverageStatus = "source_attested_complete_within_boundary"
	CoverageIncomplete CoverageStatus = "incomplete"
	CoverageIndeterminate CoverageStatus = "indeterminate"
	CoverageContradicted CoverageStatus = "contradicted"
)
type WitnessStatus string
const (
	WitnessNotProvided WitnessStatus = "not_provided"
	WitnessIndependent WitnessStatus = "independent_witness_verified"
	WitnessSharedRoot WitnessStatus = "shared_control_root"
	WitnessInvalid WitnessStatus = "invalid_witness"
	WitnessMismatch WitnessStatus = "manifest_or_boundary_mismatch"
	WitnessUnknown WitnessStatus = "witness_uncheckable"
)
type NegativeEvidenceStatus string
const (
	NoEventsNotClaimed NegativeEvidenceStatus = "not_an_empty_capture"
	NoEventsObserved NegativeEvidenceStatus = "zero_events_observed_within_boundary"
	NoEventsWitnessed NegativeEvidenceStatus = "zero_events_observed_in_independently_anchored_manifest"
	NoEventsUnestablished NegativeEvidenceStatus = "absence_not_established"
)
type Report struct {
	Coverage CoverageStatus
	Witness WitnessStatus
	NegativeEvidence NegativeEvidenceStatus
	UniqueItems, DuplicateDelivery, MissingPositions uint64
	FirstMissing, LastMissing uint64
	Findings []string
}

// VerifyBoundedCapture qualifies a source enumeration only inside its exact
// declared boundary. It does not prove that a dishonest source did not omit
// events and never asserts global absence.
func VerifyBoundedCapture(m Manifest, w *Witness) Report {
	r:=Report{Coverage:CoverageIndeterminate,Witness:WitnessNotProvided,NegativeEvidence:NoEventsUnestablished}
	if m.Verification==Invalid || m.Source.Verification==Invalid || m.Source.QueryBindingVerification==Invalid || m.Source.WatermarkVerification==Invalid {
		return fail(r,CoverageContradicted,"source_or_manifest_verification_invalid")
	}
	if m.Verification!=Verified { return fail(r,CoverageIndeterminate,"manifest_verification_missing") }
	if m.Source.Verification!=Verified { return fail(r,CoverageIndeterminate,"source_attestation_not_verified") }
	if m.Source.QueryBindingVerification!=Verified { return fail(r,CoverageIndeterminate,"query_scope_binding_not_verified") }
	b,s:=m.Boundary,m.Source
	if s.BoundaryID!=b.BoundaryID || s.SourceID!=b.SourceID || s.StreamID!=b.StreamID ||
		s.PartitionID!=b.PartitionID || s.SnapshotToken!=b.SnapshotToken ||
		CompareDigest(s.BoundQueryDigest,b.QueryDigest)!=DigestEqual {
		return fail(r,CoverageContradicted,"source_query_attestation_does_not_bind_exact_boundary")
	}
	if s.WatermarkVerification!=Verified { return fail(r,CoverageIndeterminate,"source_watermark_not_verified") }
	if s.AttestedWatermark!=s.Watermark || s.AttestedControlRoot!=s.ControlRoot {
		return fail(r,CoverageContradicted,"source_watermark_or_control_root_differs_from_verified_attestation")
	}
	if !validBoundary(b) || s.ControlRoot=="" { return fail(r,CoverageIndeterminate,"boundary_or_source_control_root_incomplete") }
	if b.EndInclusive<b.StartExclusive { return fail(r,CoverageContradicted,"invalid_position_range") }
	if !s.Terminal { return fail(r,CoverageIncomplete,"capture_did_not_reach_terminal_cursor") }
	if s.Watermark<b.EndInclusive { return fail(r,CoverageIncomplete,"source_watermark_does_not_cover_declared_end") }
	if !verifiedDigest(m.Digest) { return fail(r,CoverageIndeterminate,"manifest_digest_not_verified") }

	items:=append([]CapturedItem(nil),m.Items...)
	sort.Slice(items,func(i,j int)bool{return items[i].Position<items[j].Position})
	seen:=make(map[string]string,len(items))
	unique:=make([]CapturedItem,0,len(items))
	previous:=b.StartExclusive
	for _,it:=range items {
		if it.EventID=="" || it.Position<=b.StartExclusive || it.Position>b.EndInclusive {
			return fail(r,CoverageContradicted,"capture_item_outside_boundary_or_missing_identity")
		}
		if !verifiedDigest(it.Digest) { return fail(r,CoverageContradicted,"captured_event_digest_invalid_or_unverified") }
		if it.Position==previous {
			prior:=unique[len(unique)-1]
			if prior.EventID!=it.EventID || CompareDigest(prior.Digest,it.Digest)!=DigestEqual {
				return fail(r,CoverageContradicted,"same_source_position_has_conflicting_event_binding")
			}
			r.DuplicateDelivery++
			continue
		}
		if it.Position>previous+1 { addGap(&r,previous+1,it.Position-1) }
		if prior,ok:=seen[it.EventID];ok {
			if prior!=it.Digest.Value { return fail(r,CoverageContradicted,"event_id_reused_with_different_digest") }
			return fail(r,CoverageContradicted,"event_id_reused_at_multiple_source_positions")
		}
		seen[it.EventID]=it.Digest.Value
		unique=append(unique,it)
		previous=it.Position
	}
	if b.EndInclusive>previous { addGap(&r,previous+1,b.EndInclusive) }
	r.UniqueItems=uint64(len(unique))
	if r.MissingPositions>0 {
		r.Coverage=CoverageContradicted
		r.Findings=append(r.Findings,"verified_terminal_capture_contains_sequence_gap")
		return finishNegative(r,r.UniqueItems,false)
	}
	r.Coverage=CoverageComplete
	r=assessWitness(r,m,w)
	return finishNegative(r,r.UniqueItems,true)
}
func addGap(r *Report,first,last uint64) {
	if last<first{return}
	if r.MissingPositions==0 { r.FirstMissing=first }
	r.MissingPositions+=last-first+1
	r.LastMissing=last
}
func assessWitness(r Report,m Manifest,w *Witness) Report {
	if w==nil { r.Witness=WitnessNotProvided; return r }
	if w.Verification==Invalid {
		r.Witness=WitnessInvalid; r.Findings=append(r.Findings,"independent_witness_verification_failed"); return r
	}
	if w.Verification!=Verified || w.BoundaryID=="" || w.ControlRoot=="" {
		r.Witness=WitnessUnknown; r.Findings=append(r.Findings,"independent_witness_uncheckable"); return r
	}
	if w.BoundaryID!=m.Boundary.BoundaryID {
		r.Witness=WitnessMismatch; r.Findings=append(r.Findings,"witness_boundary_mismatch"); return r
	}
	if !verifiedDigest(w.ManifestDigest) || CompareDigest(m.Digest,w.ManifestDigest)!=DigestEqual {
		r.Witness=WitnessMismatch; r.Findings=append(r.Findings,"witness_manifest_digest_mismatch_or_incomparable"); return r
	}
	if w.ControlRoot==m.Source.ControlRoot {
		r.Witness=WitnessSharedRoot; r.Findings=append(r.Findings,"witness_shares_source_control_root"); return r
	}
	r.Witness=WitnessIndependent
	return r
}
func finishNegative(r Report,count uint64,complete bool) Report {
	if count!=0 { r.NegativeEvidence=NoEventsNotClaimed; return r }
	if !complete { r.NegativeEvidence=NoEventsUnestablished; return r }
	r.NegativeEvidence=NoEventsObserved
	if r.Witness==WitnessIndependent { r.NegativeEvidence=NoEventsWitnessed }
	return r
}
func validBoundary(b Boundary) bool {
	return b.BoundaryID!="" && b.SourceID!="" && b.StreamID!="" && b.PartitionID!="" &&
		b.SnapshotToken!="" && verifiedDigest(b.QueryDigest)
}
func verifiedDigest(d DigestEvidence) bool { return d.Present && d.Valid && d.Verification==Verified && validDigest(d) }
func validDigest(d DigestEvidence) bool {
	c:=d.Context
	return d.Algorithm=="sha-256" && validHex(d.Value) &&
		c.Profile!="" && c.Version!="" && c.PayloadType!="" && c.Canonicalization!="" &&
		c.Encoding!="" && c.Purpose!="" && c.Subject!=""
}
func validHex(s string) bool {
	if len(s)!=64 { return false }
	for i:=0;i<len(s);i++ { c:=s[i]; if !((c>='0'&&c<='9')||(c>='a'&&c<='f')) { return false } }
	return true
}
func CompareDigest(a,b DigestEvidence) DigestRelation {
	if a.Verification==Invalid || b.Verification==Invalid || (a.Present&&!a.Valid) || (b.Present&&!b.Valid) { return DigestInvalid }
	if a.Verification!=Verified || b.Verification!=Verified || !a.Present || !b.Present { return DigestIndeterminate }
	if !validDigest(a) || !validDigest(b) { return DigestInvalid }
	if a.Algorithm!=b.Algorithm || a.Context!=b.Context { return DigestIncomparable }
	if a.Value==b.Value { return DigestEqual }
	return DigestDifferent
}
func fail(r Report,status CoverageStatus,reason string) Report {
	r.Coverage=status; r.Findings=append(r.Findings,reason); r.NegativeEvidence=NoEventsUnestablished; return r
}
