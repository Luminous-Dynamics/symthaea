package benchmark

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"sort"
	"strings"
	"time"
)

const (
	SuiteProfile = "mon028-falsification-first-suite-v1"
	ResultProfile = "mon028-falsification-first-results-v1"
	Version = "1.0.0"
)

type Verdict string
const (
	Qualified Verdict = "qualified"
	Blocked Verdict = "blocked"
	Indeterminate Verdict = "indeterminate"
)

type ExecutionState string
const (
	Executed ExecutionState = "executed"
	Skipped ExecutionState = "skipped"
	TimedOut ExecutionState = "timed_out"
	InfrastructureError ExecutionState = "infrastructure_error"
)

type CaseSpec struct {
	CaseID string `json:"case_id"`
	FaultID string `json:"fault_id"`
	Family string `json:"family"`
	Expected Verdict `json:"expected"`
	Replicates uint32 `json:"replicates"`
}

type Suite struct {
	Profile string `json:"profile"`
	Version string `json:"version"`
	CampaignID string `json:"campaign_id"`
	BenchmarkID string `json:"benchmark_id"`
	FaultMatrixSHA256 string `json:"fault_matrix_sha256"`
	OracleSHA256 string `json:"oracle_sha256"`
	WorkloadSHA256 string `json:"workload_sha256"`
	EnvironmentSHA256 string `json:"environment_sha256"`
	ResourceBudgetSHA256 string `json:"resource_budget_sha256"`
	Cases []CaseSpec `json:"cases"`
}

type Observation struct {
	CaseID string `json:"case_id"`
	Replicate uint32 `json:"replicate"`
	Execution ExecutionState `json:"execution"`
	Observed Verdict `json:"observed,omitempty"`
	ElapsedNS int64 `json:"elapsed_ns,omitempty"`
	PeakMemoryBytes uint64 `json:"peak_memory_bytes,omitempty"`
	ResultSHA256 string `json:"result_sha256,omitempty"`
	EvidenceSHA256 string `json:"evidence_sha256,omitempty"`
	ErrorCode string `json:"error_code,omitempty"`
}

type CampaignResult struct {
	Profile string `json:"profile"`
	Version string `json:"version"`
	CampaignID string `json:"campaign_id"`
	BenchmarkID string `json:"benchmark_id"`
	SuiteSHA256 string `json:"suite_sha256"`
	FaultMatrixSHA256 string `json:"fault_matrix_sha256"`
	OracleSHA256 string `json:"oracle_sha256"`
	WorkloadSHA256 string `json:"workload_sha256"`
	EnvironmentSHA256 string `json:"environment_sha256"`
	ResourceBudgetSHA256 string `json:"resource_budget_sha256"`
	SystemID string `json:"system_id"`
	SystemVersion string `json:"system_version"`
	ImplementationSHA256 string `json:"implementation_sha256"`
	CapturedAt time.Time `json:"captured_at"`
	Observations []Observation `json:"observations"`
}

type ConfusionCell struct {
	Expected Verdict `json:"expected"`
	Observed Verdict `json:"observed"`
	Count int `json:"count"`
}

type Score struct {
	CampaignID string `json:"campaign_id"`
	BenchmarkID string `json:"benchmark_id"`
	SystemID string `json:"system_id"`
	SystemVersion string `json:"system_version"`
	SuiteSHA256 string `json:"suite_sha256"`
	Planned int `json:"planned"`
	Executed int `json:"executed"`
	Skipped int `json:"skipped"`
	TimedOut int `json:"timed_out"`
	InfrastructureErrors int `json:"infrastructure_errors"`
	Missing int `json:"missing"`
	Complete bool `json:"complete"`
	PositivePlanned int `json:"positive_planned"`
	PositiveAccepted int `json:"positive_accepted"`
	FalseRejected int `json:"false_rejected"`
	UnsafePlanned int `json:"unsafe_planned"`
	UnsafeExecuted int `json:"unsafe_executed"`
	FalseQualified int `json:"false_qualified"`
	FalseQualifiedRate *float64 `json:"false_qualified_rate,omitempty"`
	FalseQualifiedWilsonUpper95 *float64 `json:"false_qualified_wilson_upper_95,omitempty"`
	ExactMatches int `json:"exact_matches"`
	SafeOutcomeMismatches int `json:"safe_outcome_mismatches"`
	Confusion []ConfusionCell `json:"confusion"`
	P50LatencyNS int64 `json:"p50_latency_ns,omitempty"`
	P95LatencyNS int64 `json:"p95_latency_ns,omitempty"`
	P99LatencyNS int64 `json:"p99_latency_ns,omitempty"`
	PeakMemoryBytes uint64 `json:"peak_memory_bytes,omitempty"`
	SafetyGate bool `json:"safety_gate"`
	UsefulnessGate bool `json:"usefulness_gate"`
	OutcomeByPair map[string]Verdict `json:"-"`
}

type Comparison struct {
	CampaignID string `json:"campaign_id"`
	BaselineSystemID string `json:"baseline_system_id"`
	CandidateSystemID string `json:"candidate_system_id"`
	FalseQualifiedDelta int `json:"false_qualified_delta"`
	FalseRejectedDelta int `json:"false_rejected_delta"`
	P95LatencyDeltaNS int64 `json:"p95_latency_delta_ns"`
	PeakMemoryDeltaBytes int64 `json:"peak_memory_delta_bytes"`
	CandidateSafer bool `json:"candidate_safer"`
	CandidateParetoDominates bool `json:"candidate_pareto_dominates"`
	PairedObservations int `json:"paired_observations"`
}

var (
	ErrInvalidSuite = errors.New("invalid benchmark suite")
	ErrInvalidResult = errors.New("invalid campaign result")
	ErrSuiteMismatch = errors.New("campaign result does not match independent suite")
	ErrIncompletePair = errors.New("paired comparison requires complete campaigns")
	ErrUnmatchedComparison = errors.New("campaigns do not share matched benchmark conditions")
)

func validVerdict(v Verdict) bool {
	return v == Qualified || v == Blocked || v == Indeterminate
}

func validDigest(value string) bool {
	if len(value) != sha256.Size*2 { return false }
	_, err := hex.DecodeString(value)
	return err == nil
}

func ValidateSuite(s Suite) error {
	if s.Profile != SuiteProfile || s.Version != Version ||
		strings.TrimSpace(s.CampaignID) == "" || strings.TrimSpace(s.BenchmarkID) == "" ||
		!validDigest(s.FaultMatrixSHA256) || !validDigest(s.OracleSHA256) ||
		!validDigest(s.WorkloadSHA256) || !validDigest(s.EnvironmentSHA256) ||
		!validDigest(s.ResourceBudgetSHA256) || len(s.Cases) == 0 {
		return ErrInvalidSuite
	}
	seen := make(map[string]struct{}, len(s.Cases))
	positive, unsafe := false, false
	for _, c := range s.Cases {
		if strings.TrimSpace(c.CaseID) == "" || strings.TrimSpace(c.FaultID) == "" ||
			strings.TrimSpace(c.Family) == "" || !validVerdict(c.Expected) || c.Replicates == 0 {
			return ErrInvalidSuite
		}
		if _, ok := seen[c.CaseID]; ok {
			return fmt.Errorf("%w: duplicate case ID %q", ErrInvalidSuite, c.CaseID)
		}
		seen[c.CaseID] = struct{}{}
		if c.Expected == Qualified { positive = true } else { unsafe = true }
	}
	if !positive || !unsafe {
		return fmt.Errorf("%w: suite requires positive and unsafe controls", ErrInvalidSuite)
	}
	return nil
}

// ComputeSuiteDigest hashes a canonical struct encoding with cases sorted by CaseID.
// The suite should additionally be signed/custodied independently (MON-026/027).
func ComputeSuiteDigest(s Suite) (string, error) {
	if err := ValidateSuite(s); err != nil { return "", err }
	canonical := s
	canonical.Cases = append([]CaseSpec(nil), s.Cases...)
	sort.Slice(canonical.Cases, func(i, j int) bool { return canonical.Cases[i].CaseID < canonical.Cases[j].CaseID })
	encoded, err := json.Marshal(canonical)
	if err != nil { return "", err }
	sum := sha256.Sum256(encoded)
	return hex.EncodeToString(sum[:]), nil
}

func ScoreRun(s Suite, result CampaignResult) (Score, error) {
	if err := ValidateSuite(s); err != nil { return Score{}, err }
	if result.Profile != ResultProfile || result.Version != Version ||
		result.CampaignID != s.CampaignID || result.BenchmarkID != s.BenchmarkID ||
		result.FaultMatrixSHA256 != s.FaultMatrixSHA256 ||
		result.OracleSHA256 != s.OracleSHA256 ||
		result.WorkloadSHA256 != s.WorkloadSHA256 ||
		result.EnvironmentSHA256 != s.EnvironmentSHA256 ||
		result.ResourceBudgetSHA256 != s.ResourceBudgetSHA256 ||
		strings.TrimSpace(result.SystemID) == "" || strings.TrimSpace(result.SystemVersion) == "" ||
		!validDigest(result.ImplementationSHA256) || result.CapturedAt.IsZero() {
		return Score{}, ErrSuiteMismatch
	}
	suiteDigest, err := ComputeSuiteDigest(s)
	if err != nil { return Score{}, err }
	if !strings.EqualFold(result.SuiteSHA256, suiteDigest) {
		return Score{}, fmt.Errorf("%w: suite digest mismatch", ErrSuiteMismatch)
	}

	type expectedPair struct { verdict Verdict }
	expected := make(map[string]expectedPair)
	score := Score{
		CampaignID: result.CampaignID, BenchmarkID: result.BenchmarkID,
		SystemID: result.SystemID, SystemVersion: result.SystemVersion,
		SuiteSHA256: suiteDigest, OutcomeByPair: make(map[string]Verdict),
	}
	for _, c := range s.Cases {
		for rep := uint32(1); rep <= c.Replicates; rep++ {
			expected[pairKey(c.CaseID, rep)] = expectedPair{verdict:c.Expected}
			score.Planned++
			if c.Expected == Qualified { score.PositivePlanned++ } else { score.UnsafePlanned++ }
		}
	}
	seen := make(map[string]struct{}, len(result.Observations))
	latencies := make([]int64, 0, len(result.Observations))
	confusion := make(map[string]int)
	var peakMemory uint64
	for _, observation := range result.Observations {
		key := pairKey(observation.CaseID, observation.Replicate)
		exp, ok := expected[key]
		if !ok { return Score{}, fmt.Errorf("%w: unknown case/replicate %s", ErrInvalidResult, key) }
		if _, duplicate := seen[key]; duplicate { return Score{}, fmt.Errorf("%w: duplicate observation %s", ErrInvalidResult, key) }
		seen[key] = struct{}{}
		switch observation.Execution {
		case Executed:
			if !validVerdict(observation.Observed) || observation.ElapsedNS < 0 ||
				!validDigest(observation.ResultSHA256) || !validDigest(observation.EvidenceSHA256) {
				return Score{}, fmt.Errorf("%w: executed observation metadata invalid for %s", ErrInvalidResult, key)
			}
			score.Executed++
			score.OutcomeByPair[key] = observation.Observed
			latencies = append(latencies, observation.ElapsedNS)
			if observation.PeakMemoryBytes > peakMemory { peakMemory = observation.PeakMemoryBytes }
			confusion[string(exp.verdict)+"/"+string(observation.Observed)]++
			if exp.verdict == observation.Observed { score.ExactMatches++ } else if exp.verdict != Qualified && observation.Observed != Qualified { score.SafeOutcomeMismatches++ }
			if exp.verdict == Qualified {
				if observation.Observed == Qualified { score.PositiveAccepted++ } else { score.FalseRejected++ }
			} else {
				score.UnsafeExecuted++
				if observation.Observed == Qualified { score.FalseQualified++ }
			}
		case Skipped:
			if observation.Observed != "" { return Score{}, fmt.Errorf("%w: skipped observation has verdict", ErrInvalidResult) }
			score.Skipped++
		case TimedOut:
			if observation.Observed != "" { return Score{}, fmt.Errorf("%w: timed-out observation has verdict", ErrInvalidResult) }
			score.TimedOut++
		case InfrastructureError:
			if observation.Observed != "" { return Score{}, fmt.Errorf("%w: infrastructure-error observation has verdict", ErrInvalidResult) }
			score.InfrastructureErrors++
		default:
			return Score{}, fmt.Errorf("%w: unknown execution state", ErrInvalidResult)
		}
	}
	score.Missing = score.Planned - len(seen)
	score.Complete = score.Missing == 0 && score.Skipped == 0 && score.TimedOut == 0 &&
		score.InfrastructureErrors == 0 && score.Executed == score.Planned
	score.PeakMemoryBytes = peakMemory
	if score.UnsafeExecuted > 0 {
		rate := float64(score.FalseQualified)/float64(score.UnsafeExecuted)
		upper := WilsonUpper95(score.FalseQualified, score.UnsafeExecuted)
		score.FalseQualifiedRate = &rate
		score.FalseQualifiedWilsonUpper95 = &upper
	}
	for key, count := range confusion {
		parts := strings.SplitN(key, "/", 2)
		score.Confusion = append(score.Confusion, ConfusionCell{Expected:Verdict(parts[0]), Observed:Verdict(parts[1]), Count:count})
	}
	sort.Slice(score.Confusion, func(i,j int)bool {
		if score.Confusion[i].Expected == score.Confusion[j].Expected { return score.Confusion[i].Observed < score.Confusion[j].Observed }
		return score.Confusion[i].Expected < score.Confusion[j].Expected
	})
	score.P50LatencyNS = percentile(latencies, 0.50)
	score.P95LatencyNS = percentile(latencies, 0.95)
	score.P99LatencyNS = percentile(latencies, 0.99)
	score.SafetyGate = score.Complete && score.FalseQualified == 0 && score.PositiveAccepted == score.PositivePlanned
	score.UsefulnessGate = score.SafetyGate && score.ExactMatches == score.Executed
	return score, nil
}

// WilsonUpper95 returns a one-sided 95% Wilson score upper confidence bound.
// It is a screening interval; clustered trials require cluster-aware analysis.
func WilsonUpper95(successes, trials int) float64 {
	if trials <= 0 || successes < 0 || successes > trials { return math.NaN() }
	const z = 1.6448536269514722
	n := float64(trials)
	p := float64(successes)/n
	z2 := z*z
	center := p + z2/(2*n)
	spread := z*math.Sqrt((p*(1-p)/n)+(z2/(4*n*n)))
	return (center+spread)/(1+z2/n)
}

func percentile(values []int64, p float64) int64 {
	if len(values)==0 { return 0 }
	sorted := append([]int64(nil), values...)
	sort.Slice(sorted,func(i,j int)bool{return sorted[i]<sorted[j]})
	index := int(math.Ceil(p*float64(len(sorted))))-1
	if index<0 { index=0 }; if index>=len(sorted) { index=len(sorted)-1 }
	return sorted[index]
}

func pairKey(caseID string, replicate uint32) string { return fmt.Sprintf("%s#%010d",caseID,replicate) }

// CompareMatched requires the same suite, workload, fault matrix, environment
// and resource budget. Implementation identity is expected to differ.
func CompareMatched(s Suite, baseline, candidate CampaignResult) (Comparison,error) {
	baseScore,err:=ScoreRun(s,baseline); if err!=nil{return Comparison{},err}
	candidateScore,err:=ScoreRun(s,candidate); if err!=nil{return Comparison{},err}
	if !baseScore.Complete || !candidateScore.Complete { return Comparison{},ErrIncompletePair }
	if baseline.SystemID==candidate.SystemID ||
		baseline.EnvironmentSHA256!=candidate.EnvironmentSHA256 ||
		baseline.ResourceBudgetSHA256!=candidate.ResourceBudgetSHA256 ||
		baseline.WorkloadSHA256!=candidate.WorkloadSHA256 ||
		baseline.FaultMatrixSHA256!=candidate.FaultMatrixSHA256 ||
		baseline.OracleSHA256!=candidate.OracleSHA256 ||
		baseline.SuiteSHA256!=candidate.SuiteSHA256 || baseline.CampaignID!=candidate.CampaignID {
		return Comparison{},ErrUnmatchedComparison
	}
	paired:=0
	for key:=range baseScore.OutcomeByPair {
		if _,ok:=candidateScore.OutcomeByPair[key]; !ok { return Comparison{},ErrUnmatchedComparison }
		paired++
	}
	if paired!=baseScore.Executed || paired!=candidateScore.Executed { return Comparison{},ErrUnmatchedComparison }
	fqDelta:=candidateScore.FalseQualified-baseScore.FalseQualified
	frDelta:=candidateScore.FalseRejected-baseScore.FalseRejected
	p95Delta:=candidateScore.P95LatencyNS-baseScore.P95LatencyNS
	memoryDelta:=int64(candidateScore.PeakMemoryBytes)-int64(baseScore.PeakMemoryBytes)
	safer:=candidateScore.FalseQualified<baseScore.FalseQualified && candidateScore.FalseRejected<=baseScore.FalseRejected
	notWorse:=candidateScore.FalseQualified<=baseScore.FalseQualified &&
		candidateScore.FalseRejected<=baseScore.FalseRejected &&
		candidateScore.P95LatencyNS<=baseScore.P95LatencyNS &&
		candidateScore.PeakMemoryBytes<=baseScore.PeakMemoryBytes
	strictlyBetter:=candidateScore.FalseQualified<baseScore.FalseQualified ||
		candidateScore.FalseRejected<baseScore.FalseRejected ||
		candidateScore.P95LatencyNS<baseScore.P95LatencyNS ||
		candidateScore.PeakMemoryBytes<baseScore.PeakMemoryBytes
	return Comparison{
		CampaignID:baseline.CampaignID,BaselineSystemID:baseline.SystemID,CandidateSystemID:candidate.SystemID,
		FalseQualifiedDelta:fqDelta,FalseRejectedDelta:frDelta,P95LatencyDeltaNS:p95Delta,
		PeakMemoryDeltaBytes:memoryDelta,CandidateSafer:safer,
		CandidateParetoDominates:notWorse&&strictlyBetter,PairedObservations:paired,
	},nil
}
