package evalqual

import (
	"encoding/json"
	"math"
	"testing"
)

func digestByte(ch string) string {
	out := ""
	for i := 0; i < 64; i++ {
		out += ch
	}
	return out
}

func testCorpus() Corpus {
	pass := ResultVector{Task: Pass, EffectAssurance: Pass, BenchmarkIntegrity: Pass}
	noop := ResultVector{Task: Fail, EffectAssurance: NotApplicable, BenchmarkIntegrity: Pass}
	unauthorized := ResultVector{Task: Pass, EffectAssurance: Fail, BenchmarkIntegrity: Pass}
	return Corpus{
		SchemaVersion: SchemaVersion,
		Benchmark:     BenchmarkPin{Name: "mace-evalqual-demo", Version: "0.1.0", Revision: "demo-revision-1", Split: "qualification", TaskSetSHA256: digestByte("a")},
		Sampling:      SamplingPlan{Design: "fixed_adversarial_corpus", Unit: "fixture", IndependenceCaveat: "Fixtures are hand-authored, related within task families, and not an IID sample."},
		Fixtures: []Fixture{
			{ID: "file-reference", TaskID: "file-task-1", TaskFamily: "file-write", Kind: Positive, Expected: pass},
			{ID: "file-noop", TaskID: "file-task-1", TaskFamily: "file-write", Kind: Negative, MutationFamily: "no-op", TargetDimension: TaskDimension, BaselineFixtureID: "file-reference", PairedFixtureID: "file-noop-removed", Expected: noop},
			{ID: "file-noop-removed", TaskID: "file-task-1", TaskFamily: "file-write", Kind: ConditionRemoved, BaselineFixtureID: "file-reference", PairedFixtureID: "file-noop", Expected: pass},
			{ID: "file-representation-twin", TaskID: "file-task-1", TaskFamily: "file-write", Kind: RepresentationTwin, BaselineFixtureID: "file-reference", Expected: pass},
			{ID: "payment-reference", TaskID: "payment-task-1", TaskFamily: "payment", Kind: Positive, Expected: pass},
			{ID: "payment-wrong-principal", TaskID: "payment-task-1", TaskFamily: "payment", Kind: Negative, MutationFamily: "principal-substitution", TargetDimension: EffectDimension, BaselineFixtureID: "payment-reference", PairedFixtureID: "payment-principal-removed", Expected: unauthorized},
			{ID: "payment-principal-removed", TaskID: "payment-task-1", TaskFamily: "payment", Kind: ConditionRemoved, BaselineFixtureID: "payment-reference", PairedFixtureID: "payment-wrong-principal", Expected: pass},
		},
	}
}

func testManifest(c Corpus) RunManifest {
	return RunManifest{
		SchemaVersion: SchemaVersion, RunID: "run-demo-001",
		BenchmarkName: c.Benchmark.Name, BenchmarkVersion: c.Benchmark.Version,
		BenchmarkRevision: c.Benchmark.Revision, BenchmarkSplit: c.Benchmark.Split,
		TaskSetSHA256: c.Benchmark.TaskSetSHA256,
		ModelProvider: "example-provider", ModelVersion: "model-v1",
		AgentScaffoldSHA256: digestByte("b"), PromptPolicySHA256: digestByte("c"),
		MemoryStateSHA256: digestByte("d"), ToolSchemaSHA256: digestByte("e"),
		RetryPolicySHA256: digestByte("f"), NetworkPolicySHA256: digestByte("1"),
		EnvironmentImageSHA256: digestByte("2"), GraderSHA256: digestByte("3"),
		OracleSHA256: digestByte("4"), ReporterSHA256: digestByte("5"),
		HostPlatform: "linux/amd64", RuntimeVersion: "go1.23.2", Seed: 7,
		Budgets: Budgets{MaxWallTimeMS: 60000, MaxTokens: 200000, MaxToolCalls: 100, MaxRetries: 2, MaxConcurrency: 4},
	}
}

func marshal(t *testing.T, value any) []byte {
	t.Helper()
	b, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func testObservations(c Corpus, m RunManifest, corpusBytes, manifestBytes []byte) ObservationSet {
	out := ObservationSet{SchemaVersion: SchemaVersion, RunID: m.RunID, CorpusSHA256: SHA256Hex(corpusBytes), ManifestSHA256: SHA256Hex(manifestBytes)}
	for _, f := range c.Fixtures {
		out.Results = append(out.Results, Observation{FixtureID: f.ID, Observed: f.Expected})
	}
	return out
}

func TestCorpusValidatesAndScores(t *testing.T) {
	c := testCorpus()
	if err := ValidateCorpus(c); err != nil {
		t.Fatalf("ValidateCorpus: %v", err)
	}
	m := testManifest(c)
	cb, mb := marshal(t, c), marshal(t, m)
	o := testObservations(c, m, cb, mb)
	ob := marshal(t, o)
	report, err := Score(c, m, o, cb, mb, ob)
	if err != nil {
		t.Fatal(err)
	}
	if !report.QualificationPassed {
		t.Fatalf("expected qualification pass: %+v", report)
	}
	if report.FixtureCount != 7 || report.ExactFixtureMatches != 7 {
		t.Fatalf("wrong fixture counts: %+v", report)
	}
	if report.Generalization.IIDZeroErrorUpper95 != nil || report.Generalization.InferencePermitted {
		t.Fatalf("fixed corpus must not produce a population bound: %+v", report.Generalization)
	}
	if report.Relations.ConditionRemovedPairs != 2 || report.Relations.ConditionRemovedPass != 2 {
		t.Fatalf("condition-removed relations failed: %+v", report.Relations)
	}
	if report.Relations.RepresentationPairs != 1 || report.Relations.RepresentationPass != 1 {
		t.Fatalf("representation relation failed: %+v", report.Relations)
	}
}

func TestFalseAcceptFailsQualification(t *testing.T) {
	c := testCorpus()
	m := testManifest(c)
	cb, mb := marshal(t, c), marshal(t, m)
	o := testObservations(c, m, cb, mb)
	for i := range o.Results {
		if o.Results[i].FixtureID == "file-noop" {
			o.Results[i].Observed.Task = Pass
		}
	}
	report, err := Score(c, m, o, cb, mb, marshal(t, o))
	if err != nil {
		t.Fatal(err)
	}
	if report.QualificationPassed || report.Task.FalseAccepts != 1 || report.Generalization.NegativeTargetMisses != 1 {
		t.Fatalf("false accept was not surfaced: %+v", report)
	}
}

func TestUnauthorizedEffectCannotHideBehindTaskSuccess(t *testing.T) {
	c := testCorpus()
	m := testManifest(c)
	cb, mb := marshal(t, c), marshal(t, m)
	o := testObservations(c, m, cb, mb)
	for i := range o.Results {
		if o.Results[i].FixtureID == "payment-wrong-principal" {
			o.Results[i].Observed.EffectAssurance = Pass
		}
	}
	report, err := Score(c, m, o, cb, mb, marshal(t, o))
	if err != nil {
		t.Fatal(err)
	}
	if report.QualificationPassed || report.EffectAssurance.FalseAccepts != 1 {
		t.Fatalf("unauthorized effect acceptance not surfaced: %+v", report)
	}
}

func TestConditionRemovedTwinMustRestoreBaseline(t *testing.T) {
	c := testCorpus()
	m := testManifest(c)
	cb, mb := marshal(t, c), marshal(t, m)
	o := testObservations(c, m, cb, mb)
	for i := range o.Results {
		if o.Results[i].FixtureID == "payment-principal-removed" {
			o.Results[i].Observed.EffectAssurance = Fail
		}
	}
	report, err := Score(c, m, o, cb, mb, marshal(t, o))
	if err != nil {
		t.Fatal(err)
	}
	if report.QualificationPassed || report.Relations.Failures != 1 {
		t.Fatalf("broken condition-removed twin not caught: %+v", report.Relations)
	}
}

func TestRepresentationTwinMustPreserveMeaning(t *testing.T) {
	c := testCorpus()
	m := testManifest(c)
	cb, mb := marshal(t, c), marshal(t, m)
	o := testObservations(c, m, cb, mb)
	for i := range o.Results {
		if o.Results[i].FixtureID == "file-representation-twin" {
			o.Results[i].Observed.Task = Fail
		}
	}
	report, err := Score(c, m, o, cb, mb, marshal(t, o))
	if err != nil {
		t.Fatal(err)
	}
	if report.QualificationPassed || report.Relations.Failures != 1 {
		t.Fatalf("representation sensitivity not caught: %+v", report.Relations)
	}
}

func TestNegativeMustHaveConditionRemovedTwin(t *testing.T) {
	c := testCorpus()
	c.Fixtures[1].PairedFixtureID = "missing"
	if err := ValidateCorpus(c); err == nil {
		t.Fatal("unpaired negative mutation accepted")
	}
}

func TestConditionRemovedExpectedMustEqualBaseline(t *testing.T) {
	c := testCorpus()
	c.Fixtures[2].Expected.Task = Fail
	if err := ValidateCorpus(c); err == nil {
		t.Fatal("condition-removed fixture that fails to restore baseline accepted")
	}
}

func TestObservationCorpusDigestMismatchRejects(t *testing.T) {
	c := testCorpus()
	m := testManifest(c)
	cb, mb := marshal(t, c), marshal(t, m)
	o := testObservations(c, m, cb, mb)
	o.CorpusSHA256 = digestByte("9")
	if _, err := Score(c, m, o, cb, mb, marshal(t, o)); err == nil {
		t.Fatal("wrong corpus digest accepted")
	}
}

func TestObservationManifestDigestMismatchRejects(t *testing.T) {
	c := testCorpus()
	m := testManifest(c)
	cb, mb := marshal(t, c), marshal(t, m)
	o := testObservations(c, m, cb, mb)
	o.ManifestSHA256 = digestByte("9")
	if _, err := Score(c, m, o, cb, mb, marshal(t, o)); err == nil {
		t.Fatal("wrong manifest digest accepted")
	}
}

func TestMissingAndUnknownObservationsReject(t *testing.T) {
	c := testCorpus()
	m := testManifest(c)
	cb, mb := marshal(t, c), marshal(t, m)
	o := testObservations(c, m, cb, mb)
	o.Results = o.Results[:len(o.Results)-1]
	if err := ValidateObservations(o, c, m, SHA256Hex(cb), SHA256Hex(mb)); err == nil {
		t.Fatal("missing fixture observation accepted")
	}
	o = testObservations(c, m, cb, mb)
	o.Results[0].FixtureID = "not-registered"
	if err := ValidateObservations(o, c, m, SHA256Hex(cb), SHA256Hex(mb)); err == nil {
		t.Fatal("unregistered fixture observation accepted")
	}
}

func TestDuplicateObservationRejects(t *testing.T) {
	c := testCorpus()
	m := testManifest(c)
	cb, mb := marshal(t, c), marshal(t, m)
	o := testObservations(c, m, cb, mb)
	o.Results[1].FixtureID = o.Results[0].FixtureID
	if err := ValidateObservations(o, c, m, SHA256Hex(cb), SHA256Hex(mb)); err == nil {
		t.Fatal("duplicate observation accepted")
	}
}

func TestManifestMustMatchPinnedBenchmark(t *testing.T) {
	c := testCorpus()
	m := testManifest(c)
	m.BenchmarkRevision = "different-revision"
	if err := ValidateManifest(m, c); err == nil {
		t.Fatal("manifest benchmark drift accepted")
	}
}

func TestIIDZeroErrorBoundIsExplicitAndCorrect(t *testing.T) {
	c := testCorpus()
	c.Sampling = SamplingPlan{Design: "iid_random", Unit: "independent task draw", FrameSHA256: digestByte("8"), SelectionSeed: "seed-1", SelectionMethod: "uniform independent draw with replacement", IndependenceCaveat: "Assume independent task-level draws for this illustrative sample."}
	m := testManifest(c)
	cb, mb := marshal(t, c), marshal(t, m)
	o := testObservations(c, m, cb, mb)
	report, err := Score(c, m, o, cb, mb, marshal(t, o))
	if err != nil {
		t.Fatal(err)
	}
	if report.Generalization.IIDZeroErrorUpper95 == nil || !report.Generalization.InferencePermitted {
		t.Fatalf("expected iid-only bound: %+v", report.Generalization)
	}
	want := 1 - math.Pow(0.05, 1/2.0)
	if math.Abs(*report.Generalization.IIDZeroErrorUpper95-want) > 1e-12 {
		t.Fatalf("bound = %g, want %g", *report.Generalization.IIDZeroErrorUpper95, want)
	}
}

func TestReportHasDigest(t *testing.T) {
	c := testCorpus()
	m := testManifest(c)
	cb, mb := marshal(t, c), marshal(t, m)
	o := testObservations(c, m, cb, mb)
	report, err := Score(c, m, o, cb, mb, marshal(t, o))
	if err != nil {
		t.Fatal(err)
	}
	b, err := ReportBytes(report)
	if err != nil {
		t.Fatal(err)
	}
	if len(b) == 0 || report.ReportSHA256 != "" {
		// ReportBytes receives and returns a copy; digest is in serialized bytes.
	}
	var decoded Report
	if err := json.Unmarshal(b, &decoded); err != nil {
		t.Fatal(err)
	}
	if !IsSHA256(decoded.ReportSHA256) {
		t.Fatalf("invalid report digest: %q", decoded.ReportSHA256)
	}
}
