package causalgraph

import "testing"

func digest(id,value,purpose string) DigestRef {
	return DigestRef{"sha-256","test-profile","1","application/json","jcs",purpose,id,value}
}
func base() (Graph,map[string]VerifierPin) {
	in:=digest("input","abc","source-response")
	activity:=Node{ID:"tool-call",Kind:Activity,DeclaredInputs:[]string{"source-input"}}
	nodes:=[]Node{
		{ID:"source-input",Kind:Entity,Digest:&in,ExternalRoot:true,RequiresVerification:true},
		activity,
		{ID:"output",Kind:Entity,Digest:ptrDigest(digest("output","out","derived-output"))},
		{ID:"collector",Kind:Agent},
	}
	edges:=[]Edge{
		{ID:"use-1",Kind:Used,From:"source-input",To:"tool-call",EvidenceID:"source-input"},
		{ID:"gen-1",Kind:Generated,From:"tool-call",To:"output",EvidenceID:"source-input"},
		{ID:"assoc-1",Kind:AssociatedWith,From:"tool-call",To:"collector",EvidenceID:"source-input"},
	}
	m:=Manifest{ActivityID:"tool-call",CollectorID:"collector",SourceID:"source-a",SnapshotID:"s1",QueryID:"q1",StartCursor:"0",EndCursor:"1",TerminalWatermark:"1",ObservedInputs:[]string{"source-input"},Complete:true,ObservedAt:200}
	r:=VerifierReport{SubjectID:"source-input",InputDigest:in,Status:"verified",VerifierID:"v1",Version:"1",BuildDigest:"b1",TrustStoreDigest:"t1",VerifiedAt:100,FreshUntil:500}
	return Graph{Nodes:nodes,Edges:edges,Manifests:[]Manifest{m},Reports:[]VerifierReport{r}},map[string]VerifierPin{"v1":{"v1","1","b1","t1"}}
}
func ptrDigest(d DigestRef)*DigestRef{return &d}
func check(t *testing.T,g Graph,p map[string]VerifierPin,want Decision) {
	t.Helper(); got:=Evaluate(g,p)
	if got.Decision!=want { t.Fatalf("got %s (%s), want %s",got.Decision,got.Reason,want) }
}
func TestValidBoundedInputGraph(t *testing.T){g,p:=base();check(t,g,p,Qualified)}
func TestMissingNodeReference(t *testing.T){g,p:=base();g.Edges[0].From="missing";check(t,g,p,Denied)}
func TestCausalCycle(t *testing.T){g,p:=base();g.Edges=append(g.Edges,Edge{ID:"cycle",Kind:InformedBy,From:"tool-call",To:"tool-call",EvidenceID:"source-input"});check(t,g,p,Denied)}
func TestMissingEdgeEvidence(t *testing.T){g,p:=base();g.Edges[0].EvidenceID="absent";check(t,g,p,Indeterminate)}
func TestMissingActivityManifest(t *testing.T){g,p:=base();g.Manifests=nil;check(t,g,p,Indeterminate)}
func TestIncompleteCapture(t *testing.T){g,p:=base();g.Manifests[0].Complete=false;check(t,g,p,Indeterminate)}
func TestCaptureGap(t *testing.T){g,p:=base();g.Manifests[0].Gaps=[]string{"event-7"};check(t,g,p,Indeterminate)}
func TestMissingTerminalWatermark(t *testing.T){g,p:=base();g.Manifests[0].TerminalWatermark="";check(t,g,p,Indeterminate)}
func TestUndeclaredObservedToolInput(t *testing.T){g,p:=base();g.Manifests[0].ObservedInputs=append(g.Manifests[0].ObservedInputs,"hidden");check(t,g,p,Divergent)}
func TestGraphInputSetMismatch(t *testing.T){g,p:=base();g.Edges=append(g.Edges,Edge{ID:"use-hidden",Kind:Used,From:"output",To:"tool-call",EvidenceID:"source-input"});g.Manifests[0].ObservedInputs=append(g.Manifests[0].ObservedInputs,"output");g.Nodes[1].DeclaredInputs=append(g.Nodes[1].DeclaredInputs,"output");check(t,g,p,Denied)}
func TestUnverifiedSource(t *testing.T){g,p:=base();g.Reports[0].Status="unverified";check(t,g,p,Indeterminate)}
func TestInvalidSourceSignature(t *testing.T){g,p:=base();g.Reports[0].Status="invalid";check(t,g,p,Denied)}
func TestUnpinnedVerifier(t *testing.T){g,_:=base();check(t,g,map[string]VerifierPin{},Indeterminate)}
func TestWrongVerifierBuild(t *testing.T){g,p:=base();g.Reports[0].BuildDigest="other";check(t,g,p,Indeterminate)}
func TestStaleVerifierReport(t *testing.T){g,p:=base();g.Reports[0].FreshUntil=200;check(t,g,p,Indeterminate)}
func TestVerifierInputDigestMismatch(t *testing.T){g,p:=base();g.Reports[0].InputDigest.Value="substituted";check(t,g,p,Denied)}
func TestDigestProfileIncomparable(t *testing.T){g,p:=base();g.Reports[0].InputDigest.Purpose="other";check(t,g,p,Indeterminate)}
func TestDuplicateNodeConflict(t *testing.T){g,p:=base();n:=g.Nodes[0];n.ID="source-input";n.Digest=ptrDigest(digest("source-input","other","source-response"));g.Nodes=append(g.Nodes,n);check(t,g,p,Divergent)}
func TestDuplicateEdgeConflict(t *testing.T){g,p:=base();e:=g.Edges[0];e.To="output";g.Edges=append(g.Edges,e);check(t,g,p,Divergent)}
func TestMissingExternalOrigin(t *testing.T){g,p:=base();g.Nodes[0].ExternalRoot=false;check(t,g,p,Indeterminate)}
func TestGeneratedThenConsumedPath(t *testing.T){g,p:=base();g.Nodes=append(g.Nodes,Node{ID:"decision",Kind:Activity,DeclaredInputs:[]string{"output"}});g.Edges=append(g.Edges,Edge{ID:"use-output",Kind:Used,From:"output",To:"decision",EvidenceID:"source-input"});g.Manifests=append(g.Manifests,Manifest{ActivityID:"decision",CollectorID:"collector",SourceID:"source-a",SnapshotID:"s2",QueryID:"q2",StartCursor:"1",EndCursor:"2",TerminalWatermark:"2",ObservedInputs:[]string{"output"},Complete:true,ObservedAt:220});check(t,g,p,Qualified)}
func TestSupersessionNotInCausalDAG(t *testing.T){g,p:=base();g.Nodes=append(g.Nodes,Node{ID:"corrected",Kind:Entity,Digest:ptrDigest(digest("corrected","new","correction"))});g.Edges=append(g.Edges,Edge{ID:"supersedes-1",Kind:Supersedes,From:"corrected",To:"source-input",EvidenceID:"source-input"});check(t,g,p,Qualified)}
func TestDuplicateActivityManifestDivergent(t *testing.T){g,p:=base();g.Manifests=append(g.Manifests,g.Manifests[0]);check(t,g,p,Divergent)}
func TestTraceIdAloneCannotProveCompleteness(t *testing.T){g,p:=base();g.Manifests[0].CollectorID="";check(t,g,p,Indeterminate)}
func FuzzDigestProfileIsolation(f *testing.F){f.Add("response","decision","value");f.Fuzz(func(t *testing.T,a,b,v string){if a==b{return};x:=digest("s",v,a);y:=digest("s",v,b);eq,cmp:=x.Equal(y);if eq||cmp{t.Fatal("different digest purposes compared")}})}
