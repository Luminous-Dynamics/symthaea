package surfacecoverage

import "testing"

func base() Request {
	return Request{
		Scope:FullDatabaseState,
		SourceSystem:"pg-source",
		TargetSystem:"pg-target",
		SourceControlRoot:"source-admin",
		TargetControlRoot:"target-admin",
		InventoryDigest:"inventory-v1",
		EvaluatedAt:100,
		Surfaces:[]SurfaceSpec{
			{ID:"rows",Class:"table_rows",ScopeDigest:"rows-scope",RequiredForRowStream:true,RequiredForFullState:true},
			{ID:"schema",Class:"schema_ddl",ScopeDigest:"schema-scope",RequiredForFullState:true,OutsideRowStream:true},
			{ID:"sequence",Class:"sequence_state",ScopeDigest:"sequence-scope",RequiredForFullState:true,OutsideRowStream:true},
		},
		Attestations:[]Attestation{},
	}
}
func ev(id,surface string,mode Mode,scope string) Attestation {
	return Attestation{ID:id,SurfaceID:surface,Mode:mode,SourceSystem:"pg-source",TargetSystem:"pg-target",SourceControlRoot:"source-admin",TargetControlRoot:"target-admin",ScopeDigest:scope,SourceStateDigest:"state-v1",TargetStateDigest:"state-v1",MethodProfileDigest:"method-v1",EvidenceDigest:"evidence-"+id,Verified:true,Complete:true,ObservedAt:50,FreshUntil:200}
}
func independent(id,surface,scope string) Attestation {
	a:=ev(id,surface,IndependentComparison,scope)
	a.ObserverID="observer"
	a.ObserverControlRoot="third-admin"
	a.ObserverIdentityDigest="observer-identity-digest"
	a.ObserverControlRootDigest="observer-root-membership-digest"
	a.ObserverVerified=true
	return a
}
func addFullStateEvidence(r *Request) {
	r.Attestations=append(r.Attestations,
		ev("a-rows","rows",NativeReplication,"rows-scope"),
		independent("a-schema","schema","schema-scope"),
		independent("a-seq","sequence","sequence-scope"))
}
func check(t *testing.T,r Result,want Decision) {
	t.Helper()
	if r.Decision!=want { t.Fatalf("got %s (%s), want %s",r.Decision,r.Reason,want) }
}
func TestFullStateAllSurfacesQualified(t *testing.T) {
	r:=base();addFullStateEvidence(&r);out:=Evaluate(r);check(t,out,Qualified)
	if out.Required!=3||out.Covered!=3 { t.Fatalf("bad counts: %#v",out) }
}
func TestFullStateMissingSurfaceIndeterminate(t *testing.T) {
	r:=base();r.Attestations=append(r.Attestations,ev("rows","rows",NativeReplication,"rows-scope"));check(t,Evaluate(r),Indeterminate)
}
func TestFullStateExcludedSurfaceDenied(t *testing.T) {
	r:=base();addFullStateEvidence(&r)
	for i:=range r.Attestations { if r.Attestations[i].SurfaceID=="sequence" { r.Attestations[i].Mode=ExcludedByContract;r.Attestations[i].ExclusionReason="not replicated" } }
	check(t,Evaluate(r),Denied)
}
func TestRowStreamWithExplicitExclusionsQualifies(t *testing.T) {
	r:=base();r.Scope=RowStream
	r.Attestations=append(r.Attestations,ev("rows","rows",NativeReplication,"rows-scope"))
	s:=ev("schema-excluded","schema",ExcludedByContract,"schema-scope");s.ExclusionReason="DDL outside row CDC"
	q:=ev("sequence-excluded","sequence",ExcludedByContract,"sequence-scope");q.ExclusionReason="sequence profile handled separately"
	r.Attestations=append(r.Attestations,s,q)
	out:=Evaluate(r);check(t,out,Qualified)
	if out.Excluded!=2||len(out.ExcludedSurfaces)!=2 { t.Fatalf("exclusions not surfaced: %#v",out) }
}
func TestRowStreamMissingExclusionEvidenceIndeterminate(t *testing.T) {
	r:=base();r.Scope=RowStream;r.Attestations=append(r.Attestations,ev("rows","rows",NativeReplication,"rows-scope"));check(t,Evaluate(r),Indeterminate)
}
func TestFullStateUnknownModeIndeterminate(t *testing.T) {
	r:=base();addFullStateEvidence(&r);r.Attestations[2].Mode=Unknown;check(t,Evaluate(r),Indeterminate)
}
func TestSourceTargetDigestMismatchDivergent(t *testing.T) {
	r:=base();addFullStateEvidence(&r);r.Attestations[0].TargetStateDigest="other";check(t,Evaluate(r),Divergent)
}
func TestIndependentObserverSharesSourceRootIndeterminate(t *testing.T) {
	r:=base();addFullStateEvidence(&r);r.Attestations[1].ObserverControlRoot=r.SourceControlRoot;check(t,Evaluate(r),Indeterminate)
}
func TestIndependentObserverSharesTargetRootIndeterminate(t *testing.T) {
	r:=base();addFullStateEvidence(&r);r.Attestations[1].ObserverControlRoot=r.TargetControlRoot;check(t,Evaluate(r),Indeterminate)
}
func TestIndependentObserverMissingIndeterminate(t *testing.T) {
	r:=base();addFullStateEvidence(&r);r.Attestations[1].ObserverID="";check(t,Evaluate(r),Indeterminate)
}
func TestStaleSurfaceEvidenceIndeterminate(t *testing.T) {
	r:=base();addFullStateEvidence(&r);r.Attestations[0].FreshUntil=100;check(t,Evaluate(r),Indeterminate)
}
func TestUnverifiedSurfaceEvidenceIndeterminate(t *testing.T) {
	r:=base();addFullStateEvidence(&r);r.Attestations[0].Verified=false;check(t,Evaluate(r),Indeterminate)
}
func TestIncompleteSurfaceEvidenceIndeterminate(t *testing.T) {
	r:=base();addFullStateEvidence(&r);r.Attestations[0].Complete=false;check(t,Evaluate(r),Indeterminate)
}
func TestWrongScopeDigestIndeterminate(t *testing.T) {
	r:=base();addFullStateEvidence(&r);r.Attestations[0].ScopeDigest="other";check(t,Evaluate(r),Indeterminate)
}
func TestWrongSourceSystemDenied(t *testing.T) {
	r:=base();addFullStateEvidence(&r);r.Attestations[0].SourceSystem="other";check(t,Evaluate(r),Denied)
}
func TestDuplicateInventorySurfaceDivergent(t *testing.T) {
	r:=base();r.Surfaces=append(r.Surfaces,r.Surfaces[0]);addFullStateEvidence(&r);check(t,Evaluate(r),Divergent)
}
func TestDuplicateEvidenceForSurfaceDivergent(t *testing.T) {
	r:=base();addFullStateEvidence(&r);r.Attestations=append(r.Attestations,r.Attestations[0]);check(t,Evaluate(r),Divergent)
}
func TestDuplicateAttestationIDDivergent(t *testing.T) {
	r:=base();addFullStateEvidence(&r);r.Attestations[1].ID=r.Attestations[0].ID;check(t,Evaluate(r),Divergent)
}
func TestNotApplicableRequiresProof(t *testing.T) {
	r:=base();addFullStateEvidence(&r);r.Attestations[2].Mode=NotApplicable;check(t,Evaluate(r),Indeterminate)
}
func TestNotApplicableWithProofQualifies(t *testing.T) {
	r:=base();addFullStateEvidence(&r);r.Attestations[2].Mode=NotApplicable;r.Attestations[2].ApplicabilityProofDigest="n-a-proof";check(t,Evaluate(r),Qualified)
}
func TestExcludedReasonRequired(t *testing.T) {
	r:=base();r.Scope=RowStream;r.Attestations=append(r.Attestations,ev("rows","rows",NativeReplication,"rows-scope"));s:=ev("x","schema",ExcludedByContract,"schema-scope");r.Attestations=append(r.Attestations,s,ev("seq","sequence",ExcludedByContract,"sequence-scope"));check(t,Evaluate(r),Denied)
}
func TestFullStateCannotOmitSurfaceFromRequirement(t *testing.T) {
	r:=base();addFullStateEvidence(&r);r.Surfaces[2].RequiredForFullState=false;check(t,Evaluate(r),Denied)
}
func TestEmptyInventoryIndeterminate(t *testing.T) {
	r:=base();r.Surfaces=nil;check(t,Evaluate(r),Indeterminate)
}
func TestUnknownClaimScopeIndeterminate(t *testing.T) {
	r:=base();r.Scope="whatever";check(t,Evaluate(r),Indeterminate)
}
func TestUnrecordedObserverControlDomainNotAccepted(t *testing.T) {
	r:=base();addFullStateEvidence(&r);r.Attestations[1].ObserverControlRoot="";check(t,Evaluate(r),Indeterminate)
}
func TestObserverIdentityMustNotAliasSourceOrTarget(t *testing.T) {
	r:=base();addFullStateEvidence(&r);r.Attestations[1].ObserverID=r.SourceSystem;check(t,Evaluate(r),Indeterminate)
}
func TestObserverControlRootMembershipMustBeVerified(t *testing.T) {
	r:=base();addFullStateEvidence(&r);r.Attestations[1].ObserverVerified=false;check(t,Evaluate(r),Indeterminate)
}
func TestObserverIdentityEvidenceRequired(t *testing.T) {
	r:=base();addFullStateEvidence(&r);r.Attestations[1].ObserverIdentityDigest="";check(t,Evaluate(r),Indeterminate)
}
func TestObserverControlRootEvidenceRequired(t *testing.T) {
	r:=base();addFullStateEvidence(&r);r.Attestations[1].ObserverControlRootDigest="";check(t,Evaluate(r),Indeterminate)
}
func FuzzSurfaceScopeIdentity(f *testing.F) {
	f.Add("rows-scope")
	f.Add("different-scope")
	f.Fuzz(func(t *testing.T,actual string) {
		r:=base();addFullStateEvidence(&r);r.Attestations[0].ScopeDigest=actual
		out:=Evaluate(r)
		if actual!="rows-scope" && out.Decision==Qualified { t.Fatal("mismatched surface scope qualified") }
	})
}
