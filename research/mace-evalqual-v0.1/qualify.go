package evalqual

import (
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"strings"
)

func ValidateCorpus(c Corpus) error {
	if c.SchemaVersion != SchemaVersion {
		return fmt.Errorf("unsupported corpus schema_version %q", c.SchemaVersion)
	}
	if strings.TrimSpace(c.Benchmark.Name) == "" || strings.TrimSpace(c.Benchmark.Version) == "" || strings.TrimSpace(c.Benchmark.Revision) == "" || strings.TrimSpace(c.Benchmark.Split) == "" {
		return fmt.Errorf("benchmark name, version, immutable revision and split are required")
	}
	if !IsSHA256(c.Benchmark.TaskSetSHA256) {
		return fmt.Errorf("benchmark.task_set_sha256 must be 64 lowercase hex characters")
	}
	if c.Sampling.Design != "fixed_adversarial_corpus" && c.Sampling.Design != "iid_random" {
		return fmt.Errorf("sampling.design must be fixed_adversarial_corpus or iid_random")
	}
	if strings.TrimSpace(c.Sampling.Unit) == "" || strings.TrimSpace(c.Sampling.IndependenceCaveat) == "" {
		return fmt.Errorf("sampling.unit and sampling.independence_caveat are required")
	}
	if c.Sampling.Design == "iid_random" {
		if !IsSHA256(c.Sampling.FrameSHA256) || strings.TrimSpace(c.Sampling.SelectionSeed) == "" || strings.TrimSpace(c.Sampling.SelectionMethod) == "" {
			return fmt.Errorf("iid_random sampling requires frame_sha256, selection_seed and selection_method")
		}
	}
	if len(c.Fixtures) == 0 {
		return fmt.Errorf("corpus must contain at least one fixture")
	}

	byID := make(map[string]Fixture, len(c.Fixtures))
	families := map[string]map[FixtureKind]int{}
	for _, f := range c.Fixtures {
		if strings.TrimSpace(f.ID) == "" || strings.TrimSpace(f.TaskID) == "" || strings.TrimSpace(f.TaskFamily) == "" {
			return fmt.Errorf("fixture id, task_id and task_family are required")
		}
		if _, exists := byID[f.ID]; exists {
			return fmt.Errorf("duplicate fixture id %q", f.ID)
		}
		byID[f.ID] = f
		if families[f.TaskFamily] == nil {
			families[f.TaskFamily] = map[FixtureKind]int{}
		}
		families[f.TaskFamily][f.Kind]++
		if err := validateVector(f.Expected); err != nil {
			return fmt.Errorf("fixture %s: %w", f.ID, err)
		}
		switch f.Kind {
		case Positive:
			if f.Expected.Task != Pass || f.Expected.EffectAssurance != Pass || f.Expected.BenchmarkIntegrity != Pass {
				return fmt.Errorf("positive fixture %s must expect pass on all three dimensions", f.ID)
			}
			if f.BaselineFixtureID != "" || f.PairedFixtureID != "" || f.TargetDimension != "" {
				return fmt.Errorf("positive fixture %s cannot declare baseline, pair or target dimension", f.ID)
			}
		case Negative:
			if f.MutationFamily == "" || f.TargetDimension == "" || f.BaselineFixtureID == "" || f.PairedFixtureID == "" {
				return fmt.Errorf("negative fixture %s requires mutation_family, target_dimension, baseline_fixture_id and paired_fixture_id", f.ID)
			}
			if f.TargetDimension != TaskDimension && f.TargetDimension != EffectDimension && f.TargetDimension != IntegrityDimension {
				return fmt.Errorf("negative fixture %s has invalid target_dimension %q", f.ID, f.TargetDimension)
			}
			target := f.Expected.For(f.TargetDimension)
			if target == Pass || target == NotApplicable {
				return fmt.Errorf("negative fixture %s must expect fail or indeterminate on its target dimension", f.ID)
			}
		case ConditionRemoved:
			if f.BaselineFixtureID == "" || f.PairedFixtureID == "" {
				return fmt.Errorf("condition-removed fixture %s requires baseline_fixture_id and paired_fixture_id", f.ID)
			}
		case RepresentationTwin:
			if f.BaselineFixtureID == "" {
				return fmt.Errorf("representation twin %s requires baseline_fixture_id", f.ID)
			}
			if f.PairedFixtureID != "" || f.TargetDimension != "" {
				return fmt.Errorf("representation twin %s cannot declare paired_fixture_id or target_dimension", f.ID)
			}
		default:
			return fmt.Errorf("fixture %s has unknown kind %q", f.ID, f.Kind)
		}
	}

	for family, counts := range families {
		if counts[Positive] < 1 || counts[Negative] < 1 || counts[ConditionRemoved] < 1 {
			return fmt.Errorf("task family %q requires at least one positive, negative and condition_removed fixture", family)
		}
	}

	for _, f := range c.Fixtures {
		switch f.Kind {
		case Negative:
			base, ok := byID[f.BaselineFixtureID]
			if !ok || base.Kind != Positive {
				return fmt.Errorf("negative fixture %s baseline %q must reference a positive fixture", f.ID, f.BaselineFixtureID)
			}
			twin, ok := byID[f.PairedFixtureID]
			if !ok || twin.Kind != ConditionRemoved {
				return fmt.Errorf("negative fixture %s paired fixture %q must be condition_removed", f.ID, f.PairedFixtureID)
			}
			if twin.PairedFixtureID != f.ID || twin.BaselineFixtureID != f.BaselineFixtureID {
				return fmt.Errorf("negative fixture %s and its condition-removed twin do not cross-reference consistently", f.ID)
			}
			if f.TaskID != base.TaskID || f.TaskFamily != base.TaskFamily || twin.TaskID != base.TaskID || twin.TaskFamily != base.TaskFamily {
				return fmt.Errorf("negative fixture %s, condition-removed twin and baseline must share task_id and task_family", f.ID)
			}
			if !twin.Expected.Equal(base.Expected) {
				return fmt.Errorf("condition-removed fixture %s expected result must restore its positive baseline", twin.ID)
			}
		case ConditionRemoved:
			negative, ok := byID[f.PairedFixtureID]
			if !ok || negative.Kind != Negative || negative.PairedFixtureID != f.ID {
				return fmt.Errorf("condition-removed fixture %s must reference its negative fixture", f.ID)
			}
		case RepresentationTwin:
			base, ok := byID[f.BaselineFixtureID]
			if !ok || base.Kind != Positive {
				return fmt.Errorf("representation twin %s baseline %q must reference a positive fixture", f.ID, f.BaselineFixtureID)
			}
			if f.TaskID != base.TaskID || f.TaskFamily != base.TaskFamily || !f.Expected.Equal(base.Expected) {
				return fmt.Errorf("representation twin %s must preserve its positive baseline task and expected result", f.ID)
			}
		}
	}
	return nil
}

func validateVector(v ResultVector) error {
	if !validStatus(v.Task, false) || !validStatus(v.EffectAssurance, true) || !validStatus(v.BenchmarkIntegrity, false) {
		return fmt.Errorf("invalid result vector: task=%q effect_assurance=%q benchmark_integrity=%q", v.Task, v.EffectAssurance, v.BenchmarkIntegrity)
	}
	return nil
}

func validStatus(s Status, allowNA bool) bool {
	if s == Pass || s == Fail || s == Indeterminate {
		return true
	}
	return allowNA && s == NotApplicable
}

func ValidateManifest(m RunManifest, c Corpus) error {
	if m.SchemaVersion != SchemaVersion {
		return fmt.Errorf("unsupported run manifest schema_version %q", m.SchemaVersion)
	}
	if strings.TrimSpace(m.RunID) == "" || strings.TrimSpace(m.ModelProvider) == "" || strings.TrimSpace(m.ModelVersion) == "" || strings.TrimSpace(m.HostPlatform) == "" || strings.TrimSpace(m.RuntimeVersion) == "" {
		return fmt.Errorf("run_id, model identity, host_platform and runtime_version are required")
	}
	if m.BenchmarkName != c.Benchmark.Name || m.BenchmarkVersion != c.Benchmark.Version || m.BenchmarkRevision != c.Benchmark.Revision || m.BenchmarkSplit != c.Benchmark.Split || m.TaskSetSHA256 != c.Benchmark.TaskSetSHA256 {
		return fmt.Errorf("run manifest benchmark pins do not exactly match the corpus")
	}
	digests := map[string]string{
		"task_set_sha256":          m.TaskSetSHA256,
		"agent_scaffold_sha256":    m.AgentScaffoldSHA256,
		"prompt_policy_sha256":     m.PromptPolicySHA256,
		"memory_state_sha256":      m.MemoryStateSHA256,
		"tool_schema_sha256":       m.ToolSchemaSHA256,
		"retry_policy_sha256":      m.RetryPolicySHA256,
		"network_policy_sha256":    m.NetworkPolicySHA256,
		"environment_image_sha256": m.EnvironmentImageSHA256,
		"grader_sha256":            m.GraderSHA256,
		"oracle_sha256":            m.OracleSHA256,
		"reporter_sha256":          m.ReporterSHA256,
	}
	for name, value := range digests {
		if !IsSHA256(value) {
			return fmt.Errorf("%s must be 64 lowercase hex characters", name)
		}
	}
	if m.Budgets.MaxWallTimeMS <= 0 || m.Budgets.MaxTokens <= 0 || m.Budgets.MaxToolCalls <= 0 || m.Budgets.MaxRetries < 0 || m.Budgets.MaxConcurrency <= 0 {
		return fmt.Errorf("budgets must be positive except max_retries, which may be zero")
	}
	return nil
}

func ValidateObservations(o ObservationSet, c Corpus, m RunManifest, corpusSHA, manifestSHA string) error {
	if o.SchemaVersion != SchemaVersion {
		return fmt.Errorf("unsupported observations schema_version %q", o.SchemaVersion)
	}
	if o.RunID != m.RunID {
		return fmt.Errorf("observations run_id does not match run manifest")
	}
	if o.CorpusSHA256 != corpusSHA || o.ManifestSHA256 != manifestSHA {
		return fmt.Errorf("observations do not bind the exact corpus and run-manifest bytes")
	}
	if !IsSHA256(o.CorpusSHA256) || !IsSHA256(o.ManifestSHA256) {
		return fmt.Errorf("observation corpus_sha256 and manifest_sha256 must be lowercase SHA-256 hex")
	}
	known := make(map[string]Fixture, len(c.Fixtures))
	for _, f := range c.Fixtures {
		known[f.ID] = f
	}
	seen := map[string]bool{}
	for _, result := range o.Results {
		if _, ok := known[result.FixtureID]; !ok {
			return fmt.Errorf("observation references unknown fixture %q", result.FixtureID)
		}
		if seen[result.FixtureID] {
			return fmt.Errorf("duplicate observation for fixture %q", result.FixtureID)
		}
		seen[result.FixtureID] = true
		if err := validateVector(result.Observed); err != nil {
			return fmt.Errorf("observation %s: %w", result.FixtureID, err)
		}
	}
	if len(seen) != len(known) {
		missing := make([]string, 0)
		for id := range known {
			if !seen[id] {
				missing = append(missing, id)
			}
		}
		sort.Strings(missing)
		return fmt.Errorf("observations incomplete: have %d, need %d; missing %s", len(seen), len(known), strings.Join(missing, ","))
	}
	return nil
}

func Score(c Corpus, m RunManifest, o ObservationSet, corpusBytes, manifestBytes, observationBytes []byte) (Report, error) {
	if err := ValidateCorpus(c); err != nil {
		return Report{}, err
	}
	if err := ValidateManifest(m, c); err != nil {
		return Report{}, err
	}
	corpusSHA := SHA256Hex(corpusBytes)
	manifestSHA := SHA256Hex(manifestBytes)
	if err := ValidateObservations(o, c, m, corpusSHA, manifestSHA); err != nil {
		return Report{}, err
	}
	byID := make(map[string]ResultVector, len(o.Results))
	for _, r := range o.Results {
		byID[r.FixtureID] = r.Observed
	}

	report := Report{
		SchemaVersion:      ReportVersion,
		RunID:              m.RunID,
		CorpusSHA256:       corpusSHA,
		ManifestSHA256:     manifestSHA,
		ObservationsSHA256: SHA256Hex(observationBytes),
		FixtureCount:       len(c.Fixtures),
		Limitations: []string{
			"This tool checks declared vectors and exact input-byte binding; it does not execute or sandbox benchmark tasks or graders.",
			"SHA-256 digests identify bytes but do not authenticate their producer, prove semantic correctness, or establish administrative independence.",
			"A passing fixed adversarial corpus establishes only that its registered fixtures were classified as expected; it is not a universal error-rate guarantee.",
			"No probabilistic generalization bound is emitted for fixed or clustered adversarial fixtures.",
			"Observer completeness requires separate comparison against independently maintained event ground truth.",
		},
	}

	for _, f := range c.Fixtures {
		actual := byID[f.ID]
		if f.Expected.Equal(actual) {
			report.ExactFixtureMatches++
		}
		report.Task.add(f.Expected.Task, actual.Task)
		report.EffectAssurance.add(f.Expected.EffectAssurance, actual.EffectAssurance)
		report.BenchmarkIntegrity.add(f.Expected.BenchmarkIntegrity, actual.BenchmarkIntegrity)
		if f.Kind == Negative {
			report.Generalization.NegativeFixtures++
			targetExpected := f.Expected.For(f.TargetDimension)
			targetObserved := actual.For(f.TargetDimension)
			if targetObserved != targetExpected {
				report.Generalization.NegativeTargetMisses++
			}
			if (targetExpected == Fail || targetExpected == Indeterminate) && targetObserved == Pass {
				report.Generalization.FalseAccepts++
			}
		}
	}

	for _, f := range c.Fixtures {
		actual := byID[f.ID]
		switch f.Kind {
		case ConditionRemoved:
			report.Relations.ConditionRemovedPairs++
			if actual.Equal(byID[f.BaselineFixtureID]) && actual.Equal(f.Expected) {
				report.Relations.ConditionRemovedPass++
			} else {
				report.Relations.Failures++
			}
		case RepresentationTwin:
			report.Relations.RepresentationPairs++
			if actual.Equal(byID[f.BaselineFixtureID]) && actual.Equal(f.Expected) {
				report.Relations.RepresentationPass++
			} else {
				report.Relations.Failures++
			}
		}
	}

	report.Generalization.SamplingDesign = c.Sampling.Design
	report.Generalization.InferencePermitted = false
	report.Generalization.Interpretation = "No population-level inference: the corpus is a fixed or potentially clustered qualification set."
	if c.Sampling.Design == "iid_random" {
		report.Generalization.InferencePermitted = true
		report.Generalization.Interpretation = "The optional bound assumes the declared sample is genuinely IID from the pinned frame; this tool cannot verify that sampling claim or correct for clustering."
		if report.Generalization.FalseAccepts == 0 && report.Generalization.NegativeFixtures > 0 {
			upper := 1 - math.Pow(0.05, 1/float64(report.Generalization.NegativeFixtures))
			report.Generalization.IIDZeroErrorUpper95 = &upper
		}
	}

	report.QualificationPassed = report.ExactFixtureMatches == report.FixtureCount &&
		report.Relations.Failures == 0 &&
		report.Generalization.NegativeTargetMisses == 0 &&
		report.MissingObservations == 0 && report.UnexpectedObservations == 0 && report.DuplicateObservations == 0
	return report, nil
}

func (m *DimensionMetrics) add(expected, observed Status) {
	switch expected {
	case Pass:
		m.ExpectedPass++
	case Fail:
		m.ExpectedFail++
	case Indeterminate:
		m.ExpectedIndeterminate++
	}
	switch observed {
	case Pass:
		m.ObservedPass++
	case Fail:
		m.ObservedFail++
	case Indeterminate:
		m.ObservedIndeterminate++
	}
	if expected != observed {
		m.Mismatches++
	}
	if (expected == Fail || expected == Indeterminate) && observed == Pass {
		m.FalseAccepts++
	}
	if expected == Pass && observed == Fail {
		m.FalseRejects++
	}
	if (expected == Pass || expected == Fail) && observed == Indeterminate {
		m.UnexpectedUnknowns++
	}
}

// ReportBytes returns deterministic JSON with the report digest computed over
// the same report with report_sha256 omitted. The byte-level format is local to
// this tool version and is not a cross-language canonicalization profile.
func ReportBytes(report Report) ([]byte, error) {
	report.ReportSHA256 = ""
	unsigned, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		return nil, fmt.Errorf("marshal report: %w", err)
	}
	report.ReportSHA256 = SHA256Hex(unsigned)
	return json.MarshalIndent(report, "", "  ")
}
