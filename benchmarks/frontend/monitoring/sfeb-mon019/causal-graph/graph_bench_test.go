
package causalgraph

import (
	"fmt"
	"testing"
)

func benchmarkGraph(inputCount int) (Graph,map[string]VerifierPin) {
	g,p:=base()
	nodes:=g.Nodes[:0]
	for _,n:=range g.Nodes { if n.ID!="source-input" { nodes=append(nodes,n) } }
	g.Nodes=nodes
	edges:=g.Edges[:0]
	for _,e:=range g.Edges { if e.ID!="use-1" { edges=append(edges,e) } }
	g.Edges=edges
	reports:=g.Reports[:0]
	for _,r:=range g.Reports { if r.SubjectID!="source-input" { reports=append(reports,r) } }
	g.Reports=reports
	declaredInputs:=make([]string,0,inputCount)
	m:=&g.Manifests[0]
	m.ObservedInputs=make([]string,0,inputCount)
	mk:=func(id string,d DigestRef)VerifierReport{
		return VerifierReport{SubjectID:id,InputDigest:d,Status:"verified",VerifierID:"v1",Version:"1",BuildDigest:"b1",TrustStoreDigest:"t1",VerifiedAt:100,FreshUntil:500}
	}
	for i:=0;i<inputCount;i++ {
		id:=fmt.Sprintf("input-%08d",i)
		d:=digest(id,fmt.Sprintf("value-%08d",i),"source-response")
		g.Nodes=append(g.Nodes,Node{ID:id,Kind:Entity,Digest:&d,ExternalRoot:true,RequiresVerification:true})
		g.Edges=append(g.Edges,Edge{ID:fmt.Sprintf("use-%08d",i),Kind:Used,From:id,To:"tool-call",EvidenceID:"invocation-record"})
		declaredInputs=append(declaredInputs,id)
		m.ObservedInputs=append(m.ObservedInputs,id)
		g.Reports=append(g.Reports,mk(id,d))
	}
	for i:=range g.Nodes { if g.Nodes[i].ID=="tool-call" { g.Nodes[i].DeclaredInputs=declaredInputs; break } }
	return g,p
}

func BenchmarkEvaluateValidGraph(b *testing.B) {
	for _,size:=range []int{10,100,1000,10000,50000} {
		b.Run(fmt.Sprintf("inputs_%d",size),func(b *testing.B){
			graph,pins:=benchmarkGraph(size)
			b.ReportAllocs()
			b.ResetTimer()
			for i:=0;i<b.N;i++ {
				r:=Evaluate(graph,pins)
				if r.Decision!=Qualified { b.Fatalf("unexpected %s: %s",r.Decision,r.Reason) }
			}
		})
	}
}
