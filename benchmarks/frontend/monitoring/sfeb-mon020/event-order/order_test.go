package eventorder

import "testing"

func dig(subject,value,purpose string)DigestRef{return DigestRef{"sha-256","event-order-test","1","application/json","jcs",purpose,subject,value}}
func base() (Input,map[string]VerifierPin) {
	sources:=[]Source{
		{ID:"db",ControlRoot:"db-root",IdentityDigest:dig("db","db-key","source-identity")},
		{ID:"agent",ControlRoot:"agent-root",IdentityDigest:dig("agent","agent-key","source-identity")},
		{ID:"observer",ControlRoot:"observer-root",IdentityDigest:dig("observer","observer-key","source-identity")},
	}
	events:=[]Event{
		{ID:"start-a",Kind:ActivityStarted,SubjectID:"a",ActivityID:"a",SourceID:"agent",Sequence:1,RecordDigest:dig("start-a","s1","event")},
		{ID:"gen-x",Kind:EntityGenerated,SubjectID:"x",ActivityID:"a",SourceID:"agent",Sequence:2,RecordDigest:dig("gen-x","g1","event")},
		{ID:"end-a",Kind:ActivityEnded,SubjectID:"a",ActivityID:"a",SourceID:"agent",Sequence:3,RecordDigest:dig("end-a","e1","event")},
		{ID:"start-b",Kind:ActivityStarted,SubjectID:"b",ActivityID:"b",SourceID:"agent",Sequence:4,RecordDigest:dig("start-b","s2","event")},
		{ID:"use-x",Kind:EntityUsed,SubjectID:"x",ActivityID:"b",SourceID:"agent",Sequence:6,RecordDigest:dig("use-x","u1","event")},
		{ID:"end-b",Kind:ActivityEnded,SubjectID:"b",ActivityID:"b",SourceID:"agent",Sequence:7,RecordDigest:dig("end-b","e2","event")},
	}
	reports:=[]VerifierReport{}
	mk:=func(id string,d DigestRef) VerifierReport{return VerifierReport{SubjectID:id,InputDigest:d,Status:"verified",VerifierID:"v1",Version:"1",BuildDigest:"b1",TrustStoreDigest:"roots1",VerifiedAt:10,FreshUntil:1000}}
	for _,s:=range sources { reports=append(reports,mk("source:"+s.ID,s.IdentityDigest)) }
	for _,e:=range events { reports=append(reports,mk("event:"+e.ID,e.RecordDigest)) }
	return Input{Sources:sources,Events:events,Reports:reports,EvaluationAt:100},map[string]VerifierPin{"v1":{"v1","1","b1","roots1"}}
}
func want(t *testing.T,in Input,p map[string]VerifierPin,d Decision){t.Helper();r:=Evaluate(in,p);if r.Decision!=d{t.Fatalf("got %s (%s), want %s",r.Decision,r.Reason,d)}}
func TestSameSourceGenerationPrecedesUse(t *testing.T){in,p:=base();want(t,in,p,Qualified)}
func TestSameSourceUseBeforeGenerationDenied(t *testing.T){
	in,p:=base();for i:=range in.Events{if in.Events[i].ID=="gen-x"{in.Events[i].Sequence=8};if in.Events[i].ID=="use-x"{in.Events[i].Sequence=5}}
	for i:=range in.Reports{for _,e:=range in.Events{if in.Reports[i].SubjectID=="event:"+e.ID{in.Reports[i].InputDigest=e.RecordDigest}}}
	want(t,in,p,Denied)
}
func TestMissingEventReportIndeterminate(t *testing.T){in,p:=base();in.Reports=in.Reports[:len(in.Reports)-1];want(t,in,p,Indeterminate)}
func TestInvalidEventReportDenied(t *testing.T){in,p:=base();for i:=range in.Reports{if in.Reports[i].SubjectID=="event:start-a"{in.Reports[i].Status="invalid"}};want(t,in,p,Denied)}
func TestUnverifiedSourceIdentityIndeterminate(t *testing.T){in,p:=base();in.Reports=in.Reports[0:0];want(t,in,p,Indeterminate)}
func TestDuplicateSourceSequenceDivergent(t *testing.T){
	in,p:=base();e:=in.Events[1];e.ID="other-event";e.RecordDigest=dig("other-event","z","event");in.Events=append(in.Events,e);in.Reports=append(in.Reports,VerifierReport{SubjectID:"event:other-event",InputDigest:e.RecordDigest,Status:"verified",VerifierID:"v1",Version:"1",BuildDigest:"b1",TrustStoreDigest:"roots1",VerifiedAt:10,FreshUntil:1000});want(t,in,p,Divergent)
}
func TestCrossSourceWithoutWitnessIndeterminate(t *testing.T){
	in,p:=base();for i:=range in.Events{if in.Events[i].ID=="use-x"{in.Events[i].SourceID="db";in.Events[i].Sequence=1};if in.Events[i].ID=="start-b"{in.Events[i].SourceID="db";in.Events[i].Sequence=1};if in.Events[i].ID=="end-b"{in.Events[i].SourceID="db";in.Events[i].Sequence=2}}
	// Rebuild valid unique sequence for db and report bindings.
	in.Events[3].Sequence=1;in.Events[4].Sequence=2;in.Events[5].Sequence=3
	want(t,in,p,Indeterminate)
}
func TestCrossSourceIndependentWitnessQualifies(t *testing.T){
	in,p:=base()
	// move activity b and its events to a second source; witness source remains a distinct control root.
	for i:=range in.Events{if in.Events[i].ActivityID=="b"{in.Events[i].SourceID="db";if in.Events[i].ID=="start-b"{in.Events[i].Sequence=1};if in.Events[i].ID=="use-x"{in.Events[i].Sequence=2};if in.Events[i].ID=="end-b"{in.Events[i].Sequence=3}}}
	d:=dig("handoff","g-before-u","order-witness")
	in.Witnesses=[]OrderWitness{{ID:"handoff-1",BeforeEventID:"gen-x",AfterEventID:"use-x",WitnessSourceID:"observer",EvidenceDigest:d}}
	in.Reports=append(in.Reports,VerifierReport{SubjectID:"witness:handoff-1",InputDigest:d,Status:"verified",VerifierID:"v1",Version:"1",BuildDigest:"b1",TrustStoreDigest:"roots1",VerifiedAt:10,FreshUntil:1000})
	want(t,in,p,Qualified)
}
func TestWitnessSharesEndpointRootIndeterminate(t *testing.T){
	in,p:=base()
	for i:=range in.Events{if in.Events[i].ActivityID=="b"{in.Events[i].SourceID="db";if in.Events[i].ID=="start-b"{in.Events[i].Sequence=1};if in.Events[i].ID=="use-x"{in.Events[i].Sequence=2};if in.Events[i].ID=="end-b"{in.Events[i].Sequence=3}}}
	d:=dig("handoff","h","order-witness")
	in.Witnesses=[]OrderWitness{{ID:"w",BeforeEventID:"gen-x",AfterEventID:"use-x",WitnessSourceID:"db",EvidenceDigest:d}}
	in.Reports=append(in.Reports,VerifierReport{SubjectID:"witness:w",InputDigest:d,Status:"verified",VerifierID:"v1",Version:"1",BuildDigest:"b1",TrustStoreDigest:"roots1",VerifiedAt:10,FreshUntil:1000})
	want(t,in,p,Indeterminate)
}
func TestTimestampAloneCannotOrderCrossSource(t *testing.T){in,p:=base();for i:=range in.Events{if in.Events[i].ID=="use-x"{in.Events[i].SourceID="db";in.Events[i].Sequence=2}};want(t,in,p,Indeterminate)}
func TestActivityBoundaryMissing(t *testing.T){in,p:=base();for i,e:=range in.Events{if e.ID=="start-b"{in.Events=append(in.Events[:i],in.Events[i+1:]...);break}};want(t,in,p,Indeterminate)}
func TestActivityEndBeforeUseDenied(t *testing.T){
	in,p:=base();for i:=range in.Events{if in.Events[i].ID=="end-b"{in.Events[i].Sequence=5};if in.Events[i].ID=="use-x"{in.Events[i].Sequence=6}}
	want(t,in,p,Denied)
}
func TestInvalidationAfterUseQualifies(t *testing.T){
	in,p:=base();e:=Event{ID:"invalidate-x",Kind:EntityInvalidated,SubjectID:"x",SourceID:"agent",Sequence:8,RecordDigest:dig("invalidate-x","i","event")}
	in.Events=append(in.Events,e);in.Reports=append(in.Reports,VerifierReport{SubjectID:"event:invalidate-x",InputDigest:e.RecordDigest,Status:"verified",VerifierID:"v1",Version:"1",BuildDigest:"b1",TrustStoreDigest:"roots1",VerifiedAt:10,FreshUntil:1000})
	want(t,in,p,Qualified)
}
func TestUseAfterInvalidationDenied(t *testing.T){
	in,p:=base();e:=Event{ID:"invalidate-x",Kind:EntityInvalidated,SubjectID:"x",SourceID:"agent",Sequence:5,RecordDigest:dig("invalidate-x","i","event")}
	in.Events=append(in.Events,e);in.Reports=append(in.Reports,VerifierReport{SubjectID:"event:invalidate-x",InputDigest:e.RecordDigest,Status:"verified",VerifierID:"v1",Version:"1",BuildDigest:"b1",TrustStoreDigest:"roots1",VerifiedAt:10,FreshUntil:1000})
	want(t,in,p,Denied)
}
func TestOrderWitnessCycleDivergent(t *testing.T){
	in,p:=base()
	moveActivityBToDB(&in)
	d1:=dig("handoff-1","forward","order-witness")
	d2:=dig("handoff-2","reverse","order-witness")
	in.Witnesses=[]OrderWitness{
		{ID:"forward",BeforeEventID:"gen-x",AfterEventID:"use-x",WitnessSourceID:"observer",EvidenceDigest:d1},
		{ID:"reverse",BeforeEventID:"use-x",AfterEventID:"gen-x",WitnessSourceID:"observer",EvidenceDigest:d2},
	}
	in.Reports=append(in.Reports,
		VerifierReport{SubjectID:"witness:forward",InputDigest:d1,Status:"verified",VerifierID:"v1",Version:"1",BuildDigest:"b1",TrustStoreDigest:"roots1",VerifiedAt:10,FreshUntil:1000},
		VerifierReport{SubjectID:"witness:reverse",InputDigest:d2,Status:"verified",VerifierID:"v1",Version:"1",BuildDigest:"b1",TrustStoreDigest:"roots1",VerifiedAt:10,FreshUntil:1000},
	)
	want(t,in,p,Divergent)
}
func TestSameSourceReverseWitnessDivergence(t *testing.T){
	in,p:=base()
	d:=dig("reverse","contradiction","order-witness")
	in.Witnesses=[]OrderWitness{{ID:"reverse",BeforeEventID:"use-x",AfterEventID:"gen-x",WitnessSourceID:"observer",EvidenceDigest:d}}
	in.Reports=append(in.Reports,VerifierReport{SubjectID:"witness:reverse",InputDigest:d,Status:"verified",VerifierID:"v1",Version:"1",BuildDigest:"b1",TrustStoreDigest:"roots1",VerifiedAt:10,FreshUntil:1000})
	want(t,in,p,Divergent)
}
func FuzzSequenceOrder(f *testing.F){f.Add(uint64(2),uint64(3));f.Fuzz(func(t *testing.T,a,b uint64){if a==0||b==0{return};if a==b{return};if (a<b)!=(b>a){t.Fatal("sequence ordering is not antisymmetric")}})}

func TestConflictingVerifierReportsDivergent(t *testing.T) {
	in, p := base()
	r := in.Reports[len(in.Reports)-1]
	r.InputDigest.Value = "changed"
	in.Reports = append(in.Reports, r)
	want(t, in, p, Divergent)
}
func TestStaleVerifierReportIndeterminate(t *testing.T) {
	in, p := base()
	in.Reports[0].FreshUntil = 100
	want(t, in, p, Indeterminate)
}
func TestOrderWitnessEvidenceMissingIndeterminate(t *testing.T) {
	in, p := base()
	moveActivityBToDB(&in)
	d := dig("handoff", "g-before-u", "order-witness")
	in.Witnesses = []OrderWitness{{ID: "handoff-1", BeforeEventID: "gen-x", AfterEventID: "use-x", WitnessSourceID: "observer", EvidenceDigest: d}}
	want(t, in, p, Indeterminate)
}
