package restoreassurance

import (
	"fmt"
	"testing"
)
func BenchmarkEvaluateServiceRecovery(b *testing.B){
	for _,n:=range []int{1,10,100,1000}{
		b.Run(fmt.Sprintf("dependencies_%d",n),func(b *testing.B){
			r:=base()
			for i:=2;i<n;i++{id:=fmt.Sprintf("dep-%d",i);r.Plan.RequiredDependencyIDs=append(r.Plan.RequiredDependencyIDs,id);r.Dependencies=append(r.Dependencies,Dependency{ID:id,VersionDigest:"v",ControlRoot:"dep-root",EvidenceDigest:"e",Ready:true,Required:true,Verified:true,ObservedAt:9_900_000_000_000,FreshUntil:10_400_000_000_000})}
			b.ReportAllocs();b.ResetTimer()
			for i:=0;i<b.N;i++{out:=Evaluate(r);if out.Decision!=Qualified{b.Fatalf("%s: %s",out.Decision,out.Reason)}}
		})
	}
}
