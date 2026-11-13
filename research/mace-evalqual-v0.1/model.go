package evalqual

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strings"
)

const (
	SchemaVersion = "mace-evalqual/v1"
	ReportVersion = "mace-evalqual-report/v1"
)

type Status string

const (
	Pass          Status = "pass"
	Fail          Status = "fail"
	Indeterminate Status = "indeterminate"
	NotApplicable Status = "not_applicable"
)

type FixtureKind string

const (
	Positive           FixtureKind = "positive"
	Negative           FixtureKind = "negative"
	ConditionRemoved   FixtureKind = "condition_removed"
	RepresentationTwin FixtureKind = "representation_twin"
)

type Dimension string

const (
	TaskDimension      Dimension = "task"
	EffectDimension    Dimension = "effect_assurance"
	IntegrityDimension Dimension = "benchmark_integrity"
)

type BenchmarkPin struct {
	Name          string `json:"name"`
	Version       string `json:"version"`
	Revision      string `json:"revision"`
	Split         string `json:"split"`
	TaskSetSHA256 string `json:"task_set_sha256"`
}

type SamplingPlan struct {
	Design             string `json:"design"`
	Unit               string `json:"unit"`
	FrameSHA256        string `json:"frame_sha256,omitempty"`
	SelectionSeed      string `json:"selection_seed,omitempty"`
	SelectionMethod    string `json:"selection_method,omitempty"`
	IndependenceCaveat string `json:"independence_caveat"`
}

type ResultVector struct {
	Task               Status `json:"task_result"`
	EffectAssurance    Status `json:"effect_assurance_result"`
	BenchmarkIntegrity Status `json:"benchmark_integrity_result"`
}

type Fixture struct {
	ID                string       `json:"id"`
	TaskID            string       `json:"task_id"`
	TaskFamily        string       `json:"task_family"`
	Kind              FixtureKind  `json:"kind"`
	MutationFamily    string       `json:"mutation_family,omitempty"`
	TargetDimension   Dimension    `json:"target_dimension,omitempty"`
	BaselineFixtureID string       `json:"baseline_fixture_id,omitempty"`
	PairedFixtureID   string       `json:"paired_fixture_id,omitempty"`
	Expected          ResultVector `json:"expected"`
}

type Corpus struct {
	SchemaVersion string       `json:"schema_version"`
	Benchmark     BenchmarkPin `json:"benchmark"`
	Sampling      SamplingPlan `json:"sampling"`
	Fixtures      []Fixture    `json:"fixtures"`
}

type Budgets struct {
	MaxWallTimeMS  int64 `json:"max_wall_time_ms"`
	MaxTokens      int64 `json:"max_tokens"`
	MaxToolCalls   int64 `json:"max_tool_calls"`
	MaxRetries     int64 `json:"max_retries"`
	MaxConcurrency int64 `json:"max_concurrency"`
}

type RunManifest struct {
	SchemaVersion          string  `json:"schema_version"`
	RunID                  string  `json:"run_id"`
	BenchmarkName          string  `json:"benchmark_name"`
	BenchmarkVersion       string  `json:"benchmark_version"`
	BenchmarkRevision      string  `json:"benchmark_revision"`
	BenchmarkSplit         string  `json:"benchmark_split"`
	TaskSetSHA256          string  `json:"task_set_sha256"`
	ModelProvider          string  `json:"model_provider"`
	ModelVersion           string  `json:"model_version"`
	AgentScaffoldSHA256    string  `json:"agent_scaffold_sha256"`
	PromptPolicySHA256     string  `json:"prompt_policy_sha256"`
	MemoryStateSHA256      string  `json:"memory_state_sha256"`
	ToolSchemaSHA256       string  `json:"tool_schema_sha256"`
	RetryPolicySHA256      string  `json:"retry_policy_sha256"`
	NetworkPolicySHA256    string  `json:"network_policy_sha256"`
	EnvironmentImageSHA256 string  `json:"environment_image_sha256"`
	GraderSHA256           string  `json:"grader_sha256"`
	OracleSHA256           string  `json:"oracle_sha256"`
	ReporterSHA256         string  `json:"reporter_sha256"`
	HostPlatform           string  `json:"host_platform"`
	RuntimeVersion         string  `json:"runtime_version"`
	Seed                   int64   `json:"seed"`
	Budgets                Budgets `json:"budgets"`
}

type Observation struct {
	FixtureID string       `json:"fixture_id"`
	Observed  ResultVector `json:"observed"`
}

type ObservationSet struct {
	SchemaVersion  string        `json:"schema_version"`
	RunID          string        `json:"run_id"`
	CorpusSHA256   string        `json:"corpus_sha256"`
	ManifestSHA256 string        `json:"manifest_sha256"`
	Results        []Observation `json:"results"`
}

type DimensionMetrics struct {
	ExpectedPass          int `json:"expected_pass"`
	ExpectedFail          int `json:"expected_fail"`
	ExpectedIndeterminate int `json:"expected_indeterminate"`
	ObservedPass          int `json:"observed_pass"`
	ObservedFail          int `json:"observed_fail"`
	ObservedIndeterminate int `json:"observed_indeterminate"`
	FalseAccepts          int `json:"false_accepts"`
	FalseRejects          int `json:"false_rejects"`
	UnexpectedUnknowns    int `json:"unexpected_indeterminate"`
	Mismatches            int `json:"mismatches"`
}

type RelationMetrics struct {
	ConditionRemovedPairs int `json:"condition_removed_pairs"`
	ConditionRemovedPass  int `json:"condition_removed_pass"`
	RepresentationPairs   int `json:"representation_pairs"`
	RepresentationPass    int `json:"representation_pass"`
	Failures              int `json:"failures"`
}

type GeneralizationMetrics struct {
	SamplingDesign       string   `json:"sampling_design"`
	NegativeFixtures     int      `json:"negative_fixtures"`
	NegativeTargetMisses int      `json:"negative_target_misses"`
	FalseAccepts         int      `json:"false_accepts"`
	IIDZeroErrorUpper95  *float64 `json:"iid_zero_error_upper_95,omitempty"`
	InferencePermitted   bool     `json:"inference_permitted"`
	Interpretation       string   `json:"interpretation"`
}

type Report struct {
	SchemaVersion          string                `json:"schema_version"`
	RunID                  string                `json:"run_id"`
	CorpusSHA256           string                `json:"corpus_sha256"`
	ManifestSHA256         string                `json:"manifest_sha256"`
	ObservationsSHA256     string                `json:"observations_sha256"`
	FixtureCount           int                   `json:"fixture_count"`
	ExactFixtureMatches    int                   `json:"exact_fixture_matches"`
	Task                   DimensionMetrics      `json:"task"`
	EffectAssurance        DimensionMetrics      `json:"effect_assurance"`
	BenchmarkIntegrity     DimensionMetrics      `json:"benchmark_integrity"`
	Relations              RelationMetrics       `json:"relations"`
	Generalization         GeneralizationMetrics `json:"generalization"`
	MissingObservations    int                   `json:"missing_observations"`
	UnexpectedObservations int                   `json:"unexpected_observations"`
	DuplicateObservations  int                   `json:"duplicate_observations"`
	QualificationPassed    bool                  `json:"qualification_passed"`
	ReportSHA256           string                `json:"report_sha256,omitempty"`
	Limitations            []string              `json:"limitations"`
}

func ReadJSONFile(path string, dst any) ([]byte, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open %s: %w", path, err)
	}
	defer f.Close()
	data, err := io.ReadAll(f)
	if err != nil {
		return nil, fmt.Errorf("read %s: %w", path, err)
	}
	dec := json.NewDecoder(strings.NewReader(string(data)))
	dec.DisallowUnknownFields()
	if err := dec.Decode(dst); err != nil {
		return nil, fmt.Errorf("decode %s: %w", path, err)
	}
	if dec.Decode(&struct{}{}) != io.EOF {
		return nil, fmt.Errorf("decode %s: trailing JSON content", path)
	}
	return data, nil
}

func SHA256Hex(data []byte) string {
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

func IsSHA256(value string) bool {
	if len(value) != 64 || strings.ToLower(value) != value {
		return false
	}
	_, err := hex.DecodeString(value)
	return err == nil
}

func (v ResultVector) Equal(other ResultVector) bool {
	return v.Task == other.Task && v.EffectAssurance == other.EffectAssurance && v.BenchmarkIntegrity == other.BenchmarkIntegrity
}

func (v ResultVector) For(d Dimension) Status {
	switch d {
	case TaskDimension:
		return v.Task
	case EffectDimension:
		return v.EffectAssurance
	case IntegrityDimension:
		return v.BenchmarkIntegrity
	default:
		return ""
	}
}
