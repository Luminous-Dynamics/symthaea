package surfacecoverage

import (
	"fmt"
	"testing"
)
func fixture(n int) Request {
	r:=Request{Scope:FullDatabaseState,SourceSystem:"src",TargetSystem:"dst",SourceControlRoot:"src-root",TargetControlRoot:"dst-root",InventoryDigest:"inventory",EvaluatedAt:100}
	for i:=0;i<n;i++ {
		id:=fmt.Sprintf("surface-%08d",i)
		scope:=fmt.Sprintf("scope-%08d",i)
		r.Surfaces=append(r.Surfaces,SurfaceSpec{ID:id,Class:"table_or_state_object",ScopeDigest:scope,RequiredForRowStream:true,RequiredForFullState:true})
		r.Attestations=append(r.Attestations,Attestation{ID:"att-"+id,SurfaceID:id,Mode:NativeReplication,SourceSystem:"src",TargetSystem:"dst",SourceControlRoot:"src-root",TargetControlRoot:"dst-root",InventoryDigest:"inventory",ScopeDigest:scope,SourceStateDigest:"state",TargetStateDigest:"state",StateDigestProfile:"state-v1",MethodProfileDigest:"method",EvidenceDigest:"evidence",EvidenceSubjectID:"att-"+id,EvidenceProfile:"mon022-surface-coverage-attestation-v1",Verified:true,Complete:true,ObservedAt:10,FreshUntil:1000})
	}
	return r
}
func BenchmarkEvaluateCoverage(b *testing.B) {
	for _,n:=range []int{12,100,1000,10000,100000} {
		b.Run(fmt.Sprintf("surfaces_%d",n),func(b *testing.B){
			r:=fixture(n);b.ReportAllocs();b.ResetTimer()
			for i:=0;i<b.N;i++ { result:=Evaluate(r);if result.Decision!=Qualified{b.Fatalf("%s: %s",result.Decision,result.Reason)} }
		})
	}
}
