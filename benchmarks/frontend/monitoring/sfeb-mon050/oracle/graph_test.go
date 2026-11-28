package impact

import "testing"

func sampleGraph() Graph {
	return Graph{Profile: graphProfile, Nodes: []Node{
		{ID: "claim-a", Kind: ClaimNode}, {ID: "claim-b", Kind: ClaimNode}, {ID: "claim-c", Kind: ClaimNode},
		{ID: "evidence-a", Kind: EvidenceNode}, {ID: "evidence-b", Kind: EvidenceNode}, {ID: "evidence-c", Kind: EvidenceNode},
		{ID: "collector", Kind: CapabilityNode},
	}, Edges: []Edge{
		{From: "claim-a", To: "evidence-a", Mode: RequiredEdge}, {From: "evidence-a", To: "collector", Mode: RequiredEdge},
		{From: "claim-b", To: "evidence-b", Mode: RequiredEdge}, {From: "evidence-b", To: "collector", Mode: RequiredEdge},
		{From: "claim-c", To: "evidence-c", Mode: RequiredEdge}, {From: "claim-c", To: "evidence-a", Mode: ContextEdge},
	}}
}
func makeFixture(kind EventKind, target string, affected, unaffected []string) Fixture {
	return Fixture{Profile: fixtureProfile, ID: string(kind)+"-"+target, Graph: sampleGraph(), Event: Event{Profile: eventProfile, ID: "event-1", Kind: kind, Target: target}, ExpectedAffected: affected, ExpectedUnaffected: unaffected}
}
func TestTransitiveRequiredDependencyInvalidation(t *testing.T) {
	f := makeFixture(CapabilityGone, "collector", []string{"claim-a", "claim-b"}, []string{"claim-c"})
	r, err := Evaluate(f); if err != nil { t.Fatal(err) }
	if r.Verdict != "qualified" || len(r.PredictedAffected) != 2 || r.Precision.Rate != 1 || r.Recall.Rate != 1 { t.Fatalf("unexpected result: %+v", r) }
}
func TestOptionalContextDoesNotInvalidate(t *testing.T) {
	f := makeFixture(EvidenceRevoked, "evidence-a", []string{"claim-a"}, []string{"claim-b", "claim-c"})
	r, err := Evaluate(f); if err != nil { t.Fatal(err) }
	if r.Verdict != "qualified" || len(r.PredictedAffected) != 1 || r.PredictedAffected[0] != "claim-a" { t.Fatalf("optional edge polluted closure: %+v", r) }
}
func TestInvalidateAllAndNoneStrategiesFail(t *testing.T) {
	f := makeFixture(EvidenceRevoked, "evidence-a", []string{"claim-a"}, []string{"claim-b", "claim-c"})
	all := f; all.ID += "-all"; all.Graph = f.Graph
	all.Graph.Edges = append(append([]Edge(nil), f.Graph.Edges...), Edge{From:"claim-c", To:"evidence-a", Mode:RequiredEdge})
	ra, err := Evaluate(all); if err != nil { t.Fatal(err) }
	if ra.Verdict != "blocked" || ra.Precision.Rate >= 1 { t.Fatalf("invalidate-all should fail precision: %+v", ra) }
	none := f; none.ID += "-none"; none.Graph = f.Graph
	for i, edge := range none.Graph.Edges { if edge.From=="claim-a" && edge.To=="evidence-a" { none.Graph.Edges[i].Mode=ContextEdge } }
	rn, err := Evaluate(none); if err != nil { t.Fatal(err) }
	if rn.Verdict != "blocked" || rn.Recall.Rate >= 1 { t.Fatalf("invalidate-none should fail recall: %+v", rn) }
}
func TestGraphRejectsCyclesDanglingDuplicatesAndTypeConfusion(t *testing.T) {
	g:=sampleGraph(); g.Nodes=append(g.Nodes,Node{ID:"cap-a",Kind:CapabilityNode},Node{ID:"cap-b",Kind:CapabilityNode}); g.Edges=append(g.Edges,Edge{From:"cap-a",To:"cap-b",Mode:RequiredEdge},Edge{From:"cap-b",To:"cap-a",Mode:RequiredEdge})
	if err:=g.Validate(); err==nil { t.Fatal("cycle accepted") }
	g=sampleGraph(); g.Edges=append(g.Edges,Edge{From:"claim-a",To:"missing",Mode:RequiredEdge}); if err:=g.Validate(); err==nil { t.Fatal("dangling endpoint accepted") }
	g=sampleGraph(); g.Edges=append(g.Edges,g.Edges[0]); if err:=g.Validate(); err==nil { t.Fatal("duplicate edge accepted") }
	g=sampleGraph(); g.Edges=append(g.Edges,Edge{From:"claim-a",To:"claim-b",Mode:RequiredEdge}); if err:=g.Validate(); err==nil { t.Fatal("type-confused edge accepted") }
}
func TestEventRejectsWrongTargetKind(t *testing.T) {
	e:=Event{Profile:eventProfile,ID:"bad",Kind:CapabilityGone,Target:"evidence-a"}; if err:=e.Validate(sampleGraph()); err==nil { t.Fatal("wrong target kind accepted") }
}
func TestEventReplayAndEdgeOrderingAreDeterministic(t *testing.T) {
	g:=sampleGraph(); e:=Event{Profile:eventProfile,ID:"rev-1",Kind:EvidenceRevoked,Target:"evidence-a"}
	a,err:=ImpactedClaims(g,e); if err!=nil { t.Fatal(err) }; b,err:=ImpactedClaims(g,e); if err!=nil { t.Fatal(err) }
	if len(a)!=len(b) { t.Fatal("replay changed set") }; for i:=range a { if a[i]!=b[i] { t.Fatal("replay changed ordering") } }
	g2:=sampleGraph(); g2.Edges[0],g2.Edges[len(g2.Edges)-1]=g2.Edges[len(g2.Edges)-1],g2.Edges[0]
	c,err:=ImpactedClaims(g2,e); if err!=nil { t.Fatal(err) }; for i:=range a { if a[i]!=c[i] { t.Fatal("edge ordering changed result") } }
}
func TestRequiredToContextMutationMissesInvalidation(t *testing.T) {
	g:=sampleGraph(); for i,e:=range g.Edges { if e.From=="claim-a" && e.To=="evidence-a" { g.Edges[i].Mode=ContextEdge } }
	f:=Fixture{Profile:fixtureProfile,ID:"required-to-context",Graph:g,Event:Event{Profile:eventProfile,ID:"rev-1",Kind:EvidenceRevoked,Target:"evidence-a"},ExpectedAffected:[]string{"claim-a"},ExpectedUnaffected:[]string{"claim-b","claim-c"}}
	r,err:=Evaluate(f);if err!=nil{t.Fatal(err)};if r.Verdict!="blocked"||r.Recall.Rate!=0{t.Fatalf("required-to-context mutation not detected: %+v",r)}
}
func TestContextToRequiredMutationOverInvalidates(t *testing.T) {
	g:=sampleGraph();for i,e:=range g.Edges{if e.From=="claim-c"&&e.To=="evidence-a"{g.Edges[i].Mode=RequiredEdge}}
	f:=Fixture{Profile:fixtureProfile,ID:"context-to-required",Graph:g,Event:Event{Profile:eventProfile,ID:"rev-1",Kind:EvidenceRevoked,Target:"evidence-a"},ExpectedAffected:[]string{"claim-a"},ExpectedUnaffected:[]string{"claim-b","claim-c"}}
	r,err:=Evaluate(f);if err!=nil{t.Fatal(err)};if r.Verdict!="blocked"||r.FalseInvalidation.Numerator!=1{t.Fatalf("context-to-required mutation not detected: %+v",r)}
}
func TestRequiredEdgeAdditionIsMonotone(t *testing.T) {
	g:=sampleGraph();e:=Event{Profile:eventProfile,ID:"rev-1",Kind:EvidenceRevoked,Target:"evidence-a"};before,err:=ImpactedClaims(g,e);if err!=nil{t.Fatal(err)}
	for i,edge:=range g.Edges{if edge.From=="claim-c"&&edge.To=="evidence-a"{g.Edges[i].Mode=RequiredEdge}}
	after,err:=ImpactedClaims(g,e);if err!=nil{t.Fatal(err)};set:=map[string]bool{};for _,id:=range after{set[id]=true};for _,id:=range before{if !set[id]{t.Fatalf("adding required edge removed impact for %s",id)}}
}
