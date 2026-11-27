package benchmark

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"testing"
	"time"
)

func testDigest(label string) string {
	sum:=sha256.Sum256([]byte(label))
	return hex.EncodeToString(sum[:])
}

func testSuite() Suite {
	return Suite{
		Profile:SuiteProfile,Version:Version,CampaignID:"campaign-001",BenchmarkID:"mon-recovery-v1",
		FaultMatrixSHA256:testDigest("fault-matrix"),OracleSHA256:testDigest("oracle"),
		WorkloadSHA256:testDigest("workload"),EnvironmentSHA256:testDigest("environment"),
		ResourceBudgetSHA256:testDigest("budget"),
		Cases:[]CaseSpec{
			{CaseID:"positive",FaultID:"F-POS",Family:"positive-control",Expected:Qualified,Replicates:2},
			{CaseID:"negative",FaultID:"F-NEG",Family:"continuity",Expected:Blocked,Replicates:2},
			{CaseID:"unknown",FaultID:"F-UNK",Family:"trust",Expected:Indeterminate,Replicates:2},
		},
	}
}

func testResult(t *testing.T,s Suite,system string) CampaignResult {
	t.Helper()
	digest,err:=ComputeSuiteDigest(s)
	if err!=nil { t.Fatal(err) }
	r:=CampaignResult{
		Profile:ResultProfile,Version:Version,CampaignID:s.CampaignID,BenchmarkID:s.BenchmarkID,
		SuiteSHA256:digest,FaultMatrixSHA256:s.FaultMatrixSHA256,OracleSHA256:s.OracleSHA256,
		WorkloadSHA256:s.WorkloadSHA256,EnvironmentSHA256:s.EnvironmentSHA256,
		ResourceBudgetSHA256:s.ResourceBudgetSHA256,SystemID:system,SystemVersion:"1.0.0",
		ImplementationSHA256:testDigest(system+"-implementation"),CapturedAt:time.Date(2026,11,27,10,0,0,0,time.UTC),
	}
	for _,spec:=range s.Cases {
		for rep:=uint32(1);rep<=spec.Replicates;rep++ {
			r.Observations=append(r.Observations,Observation{
				CaseID:spec.CaseID,Replicate:rep,Execution:Executed,Observed:spec.Expected,
				ElapsedNS:int64(rep)*1_000_000,PeakMemoryBytes:1024*rep,
				ResultSHA256:testDigest(spec.CaseID+"-result"),EvidenceSHA256:testDigest(spec.CaseID+"-evidence"),
			})
		}
	}
	return r
}

func TestAllPositiveAndNegativeControlsQualify(t *testing.T) {
	s:=testSuite()
	r:=testResult(t,s,"candidate")
	score,err:=ScoreRun(s,r)
	if err!=nil { t.Fatal(err) }
	if !score.Complete || !score.SafetyGate || !score.UsefulnessGate {
		t.Fatalf("expected exact complete benchmark to qualify: %+v",score)
	}
	if score.Planned!=6 || score.Executed!=6 || score.FalseQualified!=0 || score.PositiveAccepted!=2 {
		t.Fatalf("unexpected counts: %+v",score)
	}
	if score.P95LatencyNS!=2_000_000 || score.P99LatencyNS!=2_000_000 {
		t.Fatalf("unexpected nearest-rank percentile: %+v",score)
	}
}

func TestFalseQualificationFailsSafetyGate(t *testing.T) {
	s:=testSuite()
	r:=testResult(t,s,"unsafe-system")
	for i:=range r.Observations {
		if r.Observations[i].CaseID=="negative" && r.Observations[i].Replicate==1 {
			r.Observations[i].Observed=Qualified
		}
	}
	score,err:=ScoreRun(s,r)
	if err!=nil { t.Fatal(err) }
	if score.FalseQualified!=1 || score.SafetyGate {
		t.Fatalf("false qualification must fail safety gate: %+v",score)
	}
	if score.FalseQualifiedRate==nil || *score.FalseQualifiedRate<=0 {
		t.Fatalf("false qualification rate absent: %+v",score)
	}
}

func TestSkippedCaseCannotPassCampaign(t *testing.T) {
	s:=testSuite()
	r:=testResult(t,s,"incomplete")
	for i:=range r.Observations {
		if r.Observations[i].CaseID=="negative" && r.Observations[i].Replicate==2 {
			r.Observations[i].Execution=Skipped
			r.Observations[i].Observed=""
			r.Observations[i].ResultSHA256=""
			r.Observations[i].EvidenceSHA256=""
		}
	}
	score,err:=ScoreRun(s,r)
	if err!=nil { t.Fatal(err) }
	if score.Complete || score.SafetyGate || score.Skipped!=1 {
		t.Fatalf("skipped case must make campaign incomplete: %+v",score)
	}
}

func TestMissingReplicateCannotPassCampaign(t *testing.T) {
	s:=testSuite()
	r:=testResult(t,s,"missing")
	r.Observations=r.Observations[:len(r.Observations)-1]
	score,err:=ScoreRun(s,r)
	if err!=nil { t.Fatal(err) }
	if score.Complete || score.Missing!=1 || score.SafetyGate {
		t.Fatalf("missing replicate must be visible: %+v",score)
	}
}

func TestDuplicateObservationRejected(t *testing.T) {
	s:=testSuite()
	r:=testResult(t,s,"duplicate")
	r.Observations=append(r.Observations,r.Observations[0])
	_,err:=ScoreRun(s,r)
	if !errors.Is(err,ErrInvalidResult) {
		t.Fatalf("expected duplicate observation to be rejected, got %v",err)
	}
}

func TestUnknownCaseRejected(t *testing.T) {
	s:=testSuite()
	r:=testResult(t,s,"unknown-case")
	r.Observations[0].CaseID="candidate-invented-case"
	_,err:=ScoreRun(s,r)
	if !errors.Is(err,ErrInvalidResult) {
		t.Fatalf("expected unknown case to be rejected, got %v",err)
	}
}

func TestSuiteDigestIndependentOfCaseOrdering(t *testing.T) {
	s:=testSuite()
	first,err:=ComputeSuiteDigest(s)
	if err!=nil { t.Fatal(err) }
	s.Cases[0],s.Cases[2]=s.Cases[2],s.Cases[0]
	second,err:=ComputeSuiteDigest(s)
	if err!=nil { t.Fatal(err) }
	if first!=second { t.Fatalf("case order changed suite digest: %s != %s",first,second) }
}

func TestZeroFailuresStillHasNonzeroConfidenceBound(t *testing.T) {
	upper:=WilsonUpper95(0,10_000)
	if upper<=0 || upper>=0.0003 || upper<=0.00025 {
		t.Fatalf("unexpected one-sided Wilson upper bound: %.9f",upper)
	}
}

func TestUnsafeMismatchIsSafeButNotExact(t *testing.T) {
	s:=testSuite()
	r:=testResult(t,s,"conservative")
	for i:=range r.Observations {
		if r.Observations[i].CaseID=="negative" {
			r.Observations[i].Observed=Indeterminate
		}
	}
	score,err:=ScoreRun(s,r)
	if err!=nil { t.Fatal(err) }
	if !score.SafetyGate || score.UsefulnessGate || score.SafeOutcomeMismatches!=2 {
		t.Fatalf("safe but different outcomes must not be blended into exact accuracy: %+v",score)
	}
}

func TestMatchedComparisonRejectsEnvironmentMismatch(t *testing.T) {
	s:=testSuite()
	base:=testResult(t,s,"baseline")
	candidate:=testResult(t,s,"candidate")
	candidate.EnvironmentSHA256=testDigest("different-environment")
	_,err:=CompareMatched(s,base,candidate)
	if !errors.Is(err,ErrSuiteMismatch) {
		t.Fatalf("each run must first match the independent suite's environment: %v",err)
	}
}

func TestMatchedComparisonReportsParetoImprovement(t *testing.T) {
	s:=testSuite()
	base:=testResult(t,s,"baseline")
	candidate:=testResult(t,s,"candidate")
	for i:=range candidate.Observations {
		candidate.Observations[i].ElapsedNS/=2
		candidate.Observations[i].PeakMemoryBytes/=2
	}
	comparison,err:=CompareMatched(s,base,candidate)
	if err!=nil { t.Fatal(err) }
	if !comparison.CandidateParetoDominates || comparison.PairedObservations!=6 {
		t.Fatalf("matched lower-latency/memory candidate should dominate: %+v",comparison)
	}
}

func FuzzPairKeyDeterministic(f *testing.F) {
	f.Add("fault-A",uint32(1))
	f.Add("",uint32(0))
	f.Fuzz(func(t *testing.T,id string,rep uint32) {
		first:=pairKey(id,rep)
		second:=pairKey(id,rep)
		if first!=second { t.Fatalf("pair key not deterministic") }
	})
}
