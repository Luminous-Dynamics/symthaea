package surfacecoverage

import (
	"fmt"
	"strings"
)

type Decision string
const (
	Qualified Decision = "qualified"
	Denied Decision = "denied"
	Indeterminate Decision = "indeterminate"
	Divergent Decision = "divergent"
)

type ClaimScope string
const (
	RowStream ClaimScope = "row_stream_continuity"
	FullDatabaseState ClaimScope = "full_recoverable_database_state"
)

type Mode string
const (
	NativeReplication Mode = "native_replication"
	IndependentComparison Mode = "independent_comparison"
	RecomputedComparison Mode = "recomputed_comparison"
	ExcludedByContract Mode = "excluded_by_contract"
	NotApplicable Mode = "not_applicable"
	Unknown Mode = "unknown"
)

type SurfaceSpec struct {
	ID string
	Class string
	ScopeDigest string
	RequiredForRowStream bool
	RequiredForFullState bool
	OutsideRowStream bool
}

type Attestation struct {
	ID string
	SurfaceID string
	Mode Mode
	SourceSystem string
	TargetSystem string
	SourceControlRoot string
	TargetControlRoot string
	ObserverID string
	ObserverControlRoot string
	ObserverIdentityDigest string
	ObserverControlRootDigest string
	ObserverVerified bool
	InventoryDigest string
	ScopeDigest string
	SourceStateDigest string
	TargetStateDigest string
	StateDigestProfile string
	MethodProfileDigest string
	EvidenceDigest string
	EvidenceSubjectID string
	EvidenceProfile string
	ExclusionReason string
	ApplicabilityProofDigest string
	Verified bool
	Complete bool
	ObservedAt int64
	FreshUntil int64
}

type Request struct {
	Scope ClaimScope
	SourceSystem string
	TargetSystem string
	SourceControlRoot string
	TargetControlRoot string
	InventoryDigest string
	EvaluatedAt int64
	Surfaces []SurfaceSpec
	Attestations []Attestation
}

type Result struct {
	Decision Decision
	Reason string
	Required int
	Covered int
	Excluded int
	Uncovered []string
	ExcludedSurfaces []string
	CoveredSurfaces []string
	Checked []string
}

func nonempty(xs ...string) bool {
	for _,x:=range xs { if strings.TrimSpace(x)=="" { return false } }
	return true
}
func finish(d Decision,reason string,r Result) Result {
	r.Decision=d
	r.Reason=reason
	return r
}

func Evaluate(req Request) Result {
	r:=Result{Checked:[]string{"claim-scope","surface-inventory","source-target-scope","typed-attestations"}}
	if req.Scope!=RowStream&&req.Scope!=FullDatabaseState {
		return finish(Indeterminate,"unknown claim scope",r)
	}
	if !nonempty(req.SourceSystem,req.TargetSystem,req.SourceControlRoot,req.TargetControlRoot,req.InventoryDigest)||req.SourceSystem==req.TargetSystem||req.EvaluatedAt<=0 {
		return finish(Indeterminate,"source/target identity, inventory or evaluation boundary incomplete",r)
	}
	if len(req.Surfaces)==0 { return finish(Indeterminate,"surface inventory is empty",r) }
	specs:=map[string]SurfaceSpec{}
	for _,s:=range req.Surfaces {
		if !nonempty(s.ID,s.Class,s.ScopeDigest) { return finish(Indeterminate,"surface ID/class/scope digest incomplete",r) }
		if _,ok:=specs[s.ID];ok { return finish(Divergent,"duplicate surface ID in declared inventory: "+s.ID,r) }
		specs[s.ID]=s
	}
	att:=map[string][]Attestation{}
	ids:=map[string]bool{}
	for _,a:=range req.Attestations {
		if !nonempty(a.ID,a.SurfaceID) { return finish(Indeterminate,"attestation identity missing",r) }
		if ids[a.ID] { return finish(Divergent,"duplicate attestation ID: "+a.ID,r) }
		ids[a.ID]=true
		if _,ok:=specs[a.SurfaceID];!ok { return finish(Divergent,"attestation references undeclared surface: "+a.SurfaceID,r) }
		att[a.SurfaceID]=append(att[a.SurfaceID],a)
	}
	for _,spec:=range req.Surfaces {
		required:=spec.RequiredForRowStream
		if req.Scope==FullDatabaseState { required=spec.RequiredForFullState }
		if !required {
			if req.Scope==FullDatabaseState {
				return finish(Denied,"surface is absent from full-state required inventory: "+spec.ID,r)
			}
			if !spec.OutsideRowStream {
				return finish(Indeterminate,"non-required row-stream surface is not explicitly marked outside the claim: "+spec.ID,r)
			}
			evidence,ok:=att[spec.ID]
			if !ok { r.Uncovered=append(r.Uncovered,spec.ID); return finish(Indeterminate,"out-of-scope surface lacks explicit exclusion evidence: "+spec.ID,r) }
			if len(evidence)!=1 { return finish(Divergent,"surface must have exactly one unambiguous coverage attestation: "+spec.ID,r) }
			a:=evidence[0]
			d,why:=validateCommon(req,spec,a)
			if d!=Qualified { return finish(d,why,r) }
			if a.Mode!=ExcludedByContract||!nonempty(a.ExclusionReason) {
				return finish(Denied,"row-stream surface outside claim must be explicitly excluded with a reason: "+spec.ID,r)
			}
			r.Excluded++
			r.ExcludedSurfaces=append(r.ExcludedSurfaces,spec.ID)
			continue
		}
		r.Required++
		evidence,ok:=att[spec.ID]
		if !ok { r.Uncovered=append(r.Uncovered,spec.ID);continue }
		if len(evidence)!=1 { return finish(Divergent,"surface must have exactly one unambiguous coverage attestation: "+spec.ID,r) }
		a:=evidence[0]
		d,why:=validateCommon(req,spec,a)
		if d!=Qualified { return finish(d,"surface "+spec.ID+": "+why,r) }
		switch a.Mode {
		case NativeReplication,IndependentComparison,RecomputedComparison:
			if !nonempty(a.SourceStateDigest,a.TargetStateDigest,a.StateDigestProfile,a.MethodProfileDigest) {
				return finish(Indeterminate,"state/method digest profile missing for covered surface: "+spec.ID,r)
			}
			if a.SourceStateDigest!=a.TargetStateDigest {
				return finish(Divergent,"source/target canonical state digest mismatch: "+spec.ID,r)
			}
			if a.Mode==IndependentComparison||a.Mode==RecomputedComparison {
				if !nonempty(a.ObserverID,a.ObserverControlRoot,a.ObserverIdentityDigest,a.ObserverControlRootDigest) ||
					a.ObserverID==req.SourceSystem || a.ObserverID==req.TargetSystem ||
					a.ObserverControlRoot==req.SourceControlRoot || a.ObserverControlRoot==req.TargetControlRoot ||
					!a.ObserverVerified {
					return finish(Indeterminate,"independent observer identity and distinct control-root membership are not separately verified: "+spec.ID,r)
				}
			}
			r.Covered++
			r.CoveredSurfaces=append(r.CoveredSurfaces,spec.ID)
		case NotApplicable:
			if !nonempty(a.ApplicabilityProofDigest) {
				return finish(Indeterminate,"not-applicable classification lacks independent applicability proof: "+spec.ID,r)
			}
			r.Covered++
			r.CoveredSurfaces=append(r.CoveredSurfaces,spec.ID)
		case ExcludedByContract:
			if req.Scope==FullDatabaseState {
				return finish(Denied,"excluded surface cannot satisfy full database state: "+spec.ID,r)
			}
			return finish(Denied,"required row-stream surface was excluded: "+spec.ID,r)
		case Unknown:
			r.Uncovered=append(r.Uncovered,spec.ID)
		default:
			return finish(Indeterminate,"unsupported coverage mode: "+spec.ID,r)
		}
	}
	if len(r.Uncovered)>0 {
		return finish(Indeterminate,"one or more required surfaces lack qualified coverage evidence",r)
	}
	r.Checked=append(r.Checked,"scope-limited-completeness","explicit-exclusion-accounting","independence-control-root-check")
	return finish(Qualified,"all required surfaces qualify within the declared claim scope; excluded surfaces are not represented as covered",r)
}

func validateCommon(req Request,s SurfaceSpec,a Attestation)(Decision,string) {
	if a.InventoryDigest!=req.InventoryDigest { return Indeterminate,"attestation inventory digest does not match declared inventory" }
	if a.ScopeDigest!=s.ScopeDigest { return Indeterminate,"surface scope digest mismatch" }
	if a.EvidenceSubjectID!=a.ID||a.EvidenceProfile!="mon022-surface-coverage-attestation-v1" { return Indeterminate,"attestation digest subject/profile binding is missing or unsupported" }
	if a.SourceSystem!=req.SourceSystem||a.TargetSystem!=req.TargetSystem { return Denied,"source/target system identity mismatch" }
	if a.SourceControlRoot!=req.SourceControlRoot||a.TargetControlRoot!=req.TargetControlRoot { return Denied,"source/target control-root mismatch" }
	if !a.Verified { return Indeterminate,"attestation is not verifier-qualified" }
	if !a.Complete { return Indeterminate,"surface evidence is incomplete" }
	if !nonempty(a.EvidenceDigest,a.MethodProfileDigest) { return Indeterminate,"evidence or method-profile digest missing" }
	if a.ObservedAt<=0||a.ObservedAt>req.EvaluatedAt||a.FreshUntil<=req.EvaluatedAt {
		return Indeterminate,"surface attestation stale or outside evaluation boundary"
	}
	return Qualified,""
}

func ExplainScope(s ClaimScope) string {
	switch s {
	case RowStream:
		return "Only surfaces marked RequiredForRowStream are claimed; all other inventoried surfaces must be explicitly excluded."
	case FullDatabaseState:
		return "Every inventoried surface must be covered or proven not applicable; exclusions are forbidden."
	default:
		return fmt.Sprintf("Unknown scope %q",s)
	}
}
