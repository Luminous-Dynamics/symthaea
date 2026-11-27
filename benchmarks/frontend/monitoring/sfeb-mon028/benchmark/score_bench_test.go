package benchmark

import (
	"crypto/sha256"
	"encoding/hex"
	"testing"
	"time"
)

func BenchmarkScoreRun(b *testing.B) {
	digest:=func(v string)string{
		sum:=sha256.Sum256([]byte(v))
		return hex.EncodeToString(sum[:])
	}
	suite:=Suite{
		Profile:SuiteProfile,Version:Version,CampaignID:"benchmark-campaign",BenchmarkID:"benchmark-suite",
		FaultMatrixSHA256:digest("faults"),OracleSHA256:digest("oracle"),
		WorkloadSHA256:digest("workload"),EnvironmentSHA256:digest("environment"),
		ResourceBudgetSHA256:digest("budget"),
		Cases:[]CaseSpec{
			{CaseID:"positive",FaultID:"F-POS",Family:"positive",Expected:Qualified,Replicates:500},
			{CaseID:"blocked",FaultID:"F-BLOCK",Family:"data-loss",Expected:Blocked,Replicates:500},
			{CaseID:"unknown",FaultID:"F-UNK",Family:"trust",Expected:Indeterminate,Replicates:500},
		},
	}
	suiteDigest,err:=ComputeSuiteDigest(suite)
	if err!=nil { b.Fatal(err) }
	result:=CampaignResult{
		Profile:ResultProfile,Version:Version,CampaignID:suite.CampaignID,BenchmarkID:suite.BenchmarkID,
		SuiteSHA256:suiteDigest,FaultMatrixSHA256:suite.FaultMatrixSHA256,OracleSHA256:suite.OracleSHA256,
		WorkloadSHA256:suite.WorkloadSHA256,EnvironmentSHA256:suite.EnvironmentSHA256,
		ResourceBudgetSHA256:suite.ResourceBudgetSHA256,SystemID:"benchmark-sut",SystemVersion:"benchmark",
		ImplementationSHA256:digest("implementation"),CapturedAt:time.Date(2026,11,27,10,0,0,0,time.UTC),
	}
	for _,spec:=range suite.Cases {
		for replicate:=uint32(1);replicate<=spec.Replicates;replicate++ {
			result.Observations=append(result.Observations,Observation{
				CaseID:spec.CaseID,Replicate:replicate,Execution:Executed,Observed:spec.Expected,
				ElapsedNS:int64(replicate),PeakMemoryBytes:4096,
				ResultSHA256:digest(spec.CaseID+"-result"),EvidenceSHA256:digest(spec.CaseID+"-evidence"),
			})
		}
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i:=0;i<b.N;i++ {
		score,err:=ScoreRun(suite,result)
		if err!=nil { b.Fatal(err) }
		if !score.SafetyGate { b.Fatalf("valid benchmark campaign failed: %+v",score) }
	}
}
