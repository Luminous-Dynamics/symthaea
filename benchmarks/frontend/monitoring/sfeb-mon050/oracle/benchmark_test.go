package impact

import (
	"fmt"
	"testing"
)
func scaleGraph(n int) Graph {
	g:=Graph{Profile:graphProfile,Nodes:make([]Node,0,n+1),Edges:make([]Edge,0,n)}
	g.Nodes=append(g.Nodes,Node{ID:"shared-evidence",Kind:EvidenceNode})
	for i:=0;i<n;i++{id:=fmt.Sprintf("claim-%06d",i);g.Nodes=append(g.Nodes,Node{ID:id,Kind:ClaimNode});g.Edges=append(g.Edges,Edge{From:id,To:"shared-evidence",Mode:RequiredEdge})};return g
}
func benchmarkImpact(b *testing.B,n int){g:=scaleGraph(n);e:=Event{Profile:eventProfile,ID:"benchmark-event",Kind:EvidenceRevoked,Target:"shared-evidence"};b.ReportAllocs();b.ResetTimer();for i:=0;i<b.N;i++{claims,err:=ImpactedClaims(g,e);if err!=nil{b.Fatal(err)};if len(claims)!=n{b.Fatalf("got %d impacted claims, want %d",len(claims),n)}}}
func BenchmarkImpactedClaims1000(b *testing.B){benchmarkImpact(b,1_000)}
func BenchmarkImpactedClaims10000(b *testing.B){benchmarkImpact(b,10_000)}
func BenchmarkImpactedClaims100000(b *testing.B){benchmarkImpact(b,100_000)}
