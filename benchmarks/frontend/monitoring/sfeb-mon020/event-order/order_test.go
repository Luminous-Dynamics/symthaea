package eventorder

import ("testing";"strings")

func dig(subject,value,purpose string)DigestRef{return DigestRef{"sha-256","event-order-test","1","application/json","jcs",purpose,subject,value}}
func normalizedReport(in *Input,subject string,d DigestRef) VerifierReport {
 r:=VerifierReport{SubjectID:subject,InputDigest:d,Status:"verified",VerifierID:"v1",Version:"1",BuildDigest:"b1",TrustStoreDigest:"roots1",ClaimsProfile:"mon020-normalized-evidence-v1",VerifiedAt:10,FreshUntil:1000}
 prefix,id,_:=strings.Cut(subject,":")
 switch prefix {
 case "source": r.ClaimSourceID=id
 case "control-root": for _,s:=range in.Sources{if s.ID==id{r.ClaimSourceID=id;r.ClaimControlRoot=s.ControlRoot}}
 case "event": for _,e:=range in.Events{if e.ID==id{r.ClaimEventID=e.ID;r.ClaimEventKind=string(e.Kind);r.ClaimEventSubjectID=e.SubjectID;r.ClaimEventActivityID=e.ActivityID;r.ClaimEventSourceID=e.SourceID;r.ClaimEventSequence=e.Sequence}}
 case "witness": for _,w:=range in.Witnesses{if w.ID==id{r.ClaimWitnessID=w.ID;r.ClaimWitnessBeforeEventID=w.BeforeEventID;r.ClaimWitnessAfterEventID=w.AfterEventID;r.ClaimWitnessSourceID=w.WitnessSourceID}}
 }
 return r
}
func addEventReport(in *Input,e Event){in.Reports=append(in.Reports,normalizedReport(in,"event:"+e.ID,e.RecordDigest))}
func refreshEventReport(in *Input,id string){
 for i,r:=range in.Reports{if r.SubjectID=="event:"+id{in.Reports=append(in.Reports[:i],in.Reports[i+1:]...);break}}
 for _,e:=range in.Events{if e.ID==id{addEventReport(in,e);return}}
}
func setEventSequence(in *Input,id string,seq uint64){for i:=range in.Events{if in.Events[i].ID==id{in.Events[i].Sequence=seq;refreshEventReport(in,id);return}}}
func addWitnessReport(in *Input,w OrderWitness){in.Witnesses=append(in.Witnesses,w);in.Reports=append(in.Reports,normalizedReport(in,"witness:"+w.ID,w.EvidenceDigest))}
func base() (Input,map[string]VerifierPin) {
	sources:=[]Source{
		{ID:"db",ControlRoot:"db-root",IdentityDigest:dig("db","db-key","source-identity"),ControlRootDigest:dig("db","db-root","control-root-membership")},
		{ID:"agent",ControlRoot:"agent-root",IdentityDigest:dig("agent","agent-key","source-identity"),ControlRootDigest:dig("agent","agent-root","control-root-membership")},
		{ID:"observer",ControlRoot:"observer-root",IdentityDigest:dig("observer","observer-key","source-identity"),ControlRootDigest:dig("observer","observer-root","control-root-membership")},
	}
	events:=[]Event{
		{ID:"start-a",Kind:ActivityStarted,SubjectID:"a",ActivityID:"a",SourceID:"agent",Sequence:1,RecordDigest:dig("start-a","s1","event")},
		{ID:"gen-x",Kind:EntityGenerated,SubjectID:"x",ActivityID:"a",SourceID:"agent",Sequence:2,RecordDigest:dig("gen-x","g1","event")},
		{ID:"end-a",Kind:ActivityEnded,SubjectID:"a",ActivityID:"a",SourceID:"agent",Sequence:3,RecordDigest:dig("end-a","e1","event")},
		{ID:"start-b",Kind:ActivityStarted,SubjectID:"b",ActivityID:"b",SourceID:"agent",Sequence:4,RecordDigest:dig("start-b","s2","event")},
		{ID:"use-x",Kind:EntityUsed,SubjectID:"x",ActivityID:"b",SourceID:"agent",Sequence:6,RecordDigest:dig("use-x","u1","event")},
		{ID:"end-b",Kind:ActivityEnded,SubjectID:"b",ActivityID:"b",SourceID:"agent",Sequence:7,RecordDigest:dig("end-b","e2","event")},
	}
	in:=Input{Sources:sources,Events:events,EvaluationAt:100}
	for _,s:=range sources { in.Reports=append(in.Reports,normalizedReport(&in,"source:"+s.ID,s.IdentityDigest),normalizedReport(&in,"control-root:"+s.ID,s.ControlRootDigest)) }
	for _,e:=range events { addEventReport(&in,e) }
	return in,map[string]VerifierPin{"v1":{"v1","1","b1","roots1"}}
}
func want(t *testing.T,in Input,p map[string]VerifierPin,d Decision){t.Helper();r:=Evaluate(in,p);if r.Decision!=d{t.Fatalf("got %s (%s), want %s",r.Decision,r.Reason,d)}}
func moveActivityBToDB(in *Input){for i:=range in.Events{if in.Events[i].ActivityID=="b"{in.Events[i].SourceID="db";switch in.Events[i].ID{case "start-b":in.Events[i].Sequence=1;case "use-x":in.Events[i].Sequence=2;case "end-b":in.Events[i].Sequence=3};refreshEventReport(in,in.Events[i].ID)}}}
func TestSameSourceGenerationPrecedesUse(t *testing.T){in,p:=base();want(t,in,p,Qualified)}
func TestSameSourceUseBeforeGenerationDenied(t *testing.T){
	in,p:=base();setEventSequence(&in,"gen-x",8);setEventSequence(&in,"end-a",9);want(t,in,p,Denied)
}
func TestMissingEventReportIndeterminate(t *testing.T){in,p:=base();in.Reports=in.Reports[:len(in.Reports)-1];want(t,in,p,Indeterminate)}
func TestInvalidEventReportDenied(t *testing.T){in,p:=base();for i:=range in.Reports{if in.Reports[i].SubjectID=="event:start-a"{in.Reports[i].Status="invalid"}};want(t,in,p,Denied)}
func TestUnverifiedSourceIdentityIndeterminate(t *testing.T){in,p:=base();in.Reports=in.Reports[0:0];want(t,in,p,Indeterminate)}
func TestDuplicateSourceSequenceDivergent(t *testing.T){
	in,p:=base();e:=in.Events[1];e.ID="other-event";e.RecordDigest=dig("other-event","z","event");in.Events=append(in.Events,e);addEventReport(&in,e);want(t,in,p,Divergent)
}
func TestCrossSourceWithoutWitnessIndeterminate(t *testing.T){
	in,p:=base();moveActivityBToDB(&in);want(t,in,p,Indeterminate)
}
func TestCrossSourceIndependentWitnessQualifies(t *testing.T){
	in,p:=base();moveActivityBToDB(&in)
	addWitnessReport(&in,OrderWitness{ID:"handoff-1",BeforeEventID:"gen-x",AfterEventID:"use-x",WitnessSourceID:"observer",EvidenceDigest:dig("handoff-1","g-before-u","order-witness")})
	want(t,in,p,Qualified)
}
func TestWitnessSharesEndpointRootIndeterminate(t *testing.T){
	in,p:=base();moveActivityBToDB(&in)
	addWitnessReport(&in,OrderWitness{ID:"w",BeforeEventID:"gen-x",AfterEventID:"use-x",WitnessSourceID:"db",EvidenceDigest:dig("w","h","order-witness")})
	want(t,in,p,Indeterminate)
}
func TestTimestampAloneCannotOrderCrossSource(t *testing.T){in,p:=base();moveActivityBToDB(&in);want(t,in,p,Indeterminate)}
func TestActivityBoundaryMissing(t *testing.T){in,p:=base();for i,e:=range in.Events{if e.ID=="start-b"{in.Events=append(in.Events[:i],in.Events[i+1:]...);break}};want(t,in,p,Indeterminate)}
func TestActivityEndBeforeUseDenied(t *testing.T){
	in,p:=base();setEventSequence(&in,"end-b",5);want(t,in,p,Denied)
}
func TestInvalidationAfterUseQualifies(t *testing.T){
	in,p:=base();e:=Event{ID:"invalidate-x",Kind:EntityInvalidated,SubjectID:"x",SourceID:"agent",Sequence:8,RecordDigest:dig("invalidate-x","i","event")}
	in.Events=append(in.Events,e);addEventReport(&in,e)
	want(t,in,p,Qualified)
}
func TestUseAfterInvalidationDenied(t *testing.T){
	in,p:=base();e:=Event{ID:"invalidate-x",Kind:EntityInvalidated,SubjectID:"x",SourceID:"agent",Sequence:5,RecordDigest:dig("invalidate-x","i","event")}
	in.Events=append(in.Events,e);addEventReport(&in,e)
	want(t,in,p,Denied)
}
func TestOrderWitnessCycleDivergent(t *testing.T){
	in,p:=base()
	moveActivityBToDB(&in)
	addWitnessReport(&in,OrderWitness{ID:"forward",BeforeEventID:"gen-x",AfterEventID:"use-x",WitnessSourceID:"observer",EvidenceDigest:dig("forward","payload","order-witness")})
	addWitnessReport(&in,OrderWitness{ID:"reverse",BeforeEventID:"use-x",AfterEventID:"gen-x",WitnessSourceID:"observer",EvidenceDigest:dig("reverse","payload","order-witness")})
	want(t,in,p,Divergent)
}
func TestSameSourceReverseWitnessDivergence(t *testing.T){
	in,p:=base()
	addWitnessReport(&in,OrderWitness{ID:"reverse",BeforeEventID:"use-x",AfterEventID:"gen-x",WitnessSourceID:"observer",EvidenceDigest:dig("reverse","payload","order-witness")})
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
	d := dig("handoff-1", "g-before-u", "order-witness")
	in.Witnesses = []OrderWitness{{ID: "handoff-1", BeforeEventID: "gen-x", AfterEventID: "use-x", WitnessSourceID: "observer", EvidenceDigest: d}}
	want(t, in, p, Indeterminate)
}

func TestUseWithoutGenerationIndeterminate(t *testing.T) {
	in,p:=base()
	for i,e:=range in.Events { if e.ID=="gen-x" { in.Events=append(in.Events[:i],in.Events[i+1:]...); break } }
	for i,r:=range in.Reports { if r.SubjectID=="event:gen-x" { in.Reports=append(in.Reports[:i],in.Reports[i+1:]...); break } }
	want(t,in,p,Indeterminate)
}
func TestEventDigestSubjectMismatchIndeterminate(t *testing.T) {
	in,p:=base()
	for i:=range in.Events { if in.Events[i].ID=="gen-x" { in.Events[i].RecordDigest.Subject="other-event" } }
	want(t,in,p,Indeterminate)
}
func TestControlRootEvidenceMissingIndeterminate(t *testing.T) {
	in,p:=base()
	for i,r:=range in.Reports { if r.SubjectID=="control-root:observer" { in.Reports=append(in.Reports[:i],in.Reports[i+1:]...); break } }
	want(t,in,p,Indeterminate)
}
