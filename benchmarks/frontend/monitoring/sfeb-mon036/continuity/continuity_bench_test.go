package continuity

import (
	"fmt"
	"testing"
	"time"
)
func benchmarkInput(events,claims int)Input{
	in:=baseInput();in.Claims=nil;in.CurrentStates=nil
	for i:=0;i<claims;i++{
		id:=fmt.Sprintf("dep-%04d",i)
		b:=DependencyBinding{DependencyID:id,Revision:1,Digest:digest('a')}
		in.CurrentStates=append(in.CurrentStates,DependencyState{DependencyID:id,Revision:1,Digest:digest('a')})
		in.Claims=append(in.Claims,Claim{ClaimID:fmt.Sprintf("claim-%04d",i),EvidenceID:fmt.Sprintf("bundle-%04d",i),EvidenceGrade:GradeRuntimeIndependent,ClosureComplete:true,ObservedAt:t0.Add(-time.Hour),ExpiresAt:t0.Add(time.Hour),Support:[]DependencyBinding{b}})
	}
	for i:=0;i<events;i++{
		appendEvent(&in,Event{EventID:fmt.Sprintf("unrelated-%06d",i),DependencyID:fmt.Sprintf("unrelated-dep-%06d",i),Kind:UnknownChange,OccurredAt:t0.Add(-time.Minute)})
	}
	in.Ledger.CapturedAt=time.Now().UTC();in.Ledger.ExpiresAt=time.Now().UTC().Add(time.Hour)
	return in
}
func BenchmarkEvaluate100Claims1000Events(b *testing.B){
	in:=benchmarkInput(1000,100);b.ReportAllocs();b.ResetTimer()
	for i:=0;i<b.N;i++{r:=Evaluate(in);if len(r.Claims)!=100{b.Fatal("claim count changed")}}
}
