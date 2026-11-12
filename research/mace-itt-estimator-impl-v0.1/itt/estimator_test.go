package itt

import (
	"math"
	"reflect"
	"testing"
)

func testPair(id, task, domain, seed string, with, without TaskVerdict, retrieved bool) Pair {
	order := "with_skill_first"
	if seed == "2" { order = "without_skill_first" }
	stages := StageFlags{Retrieved: retrieved}
	if retrieved { stages.Selected, stages.Loaded, stages.Invoked, stages.Executed = true, true, true, true }
	return Pair{
		PairID: id, ExperimentID: "exp-1", TaskID: task, TaskRevision: "rev-1",
		Domain: domain, Seed: seed,
		StratumDigest: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
		ArmOrder: order,
		WithSkill: Arm{RunID: id + "-w", TaskVerdict: with, ControlVerdict: ControlsPass, UnauthorizedEffect: UnauthorizedEffectAbsent, Stages: stages},
		WithoutSkill: Arm{RunID: id + "-n", TaskVerdict: without, ControlVerdict: ControlsPass, UnauthorizedEffect: UnauthorizedEffectAbsent},
	}
}
func testConfig() Config { return Config{BootstrapReplicates: 1000, BootstrapSeed: 17} }

func TestITTIncludesTreatedPairsThatDidNotRetrieve(t *testing.T) {
	pairs := []Pair{
		testPair("p1", "t1", "code", "1", TaskPass, TaskFail, true),
		testPair("p2", "t1", "code", "2", TaskPass, TaskFail, false),
		testPair("p3", "t2", "math", "1", TaskFail, TaskFail, false),
		testPair("p4", "t2", "math", "2", TaskFail, TaskFail, false),
	}
	r, err := EstimateITT(pairs, testConfig())
	if err != nil { t.Fatal(err) }
	got := r.Primary.TaskSuccess
	if got.PairCount != 4 || got.TaskClusters != 2 || got.CompletePairs != 4 { t.Fatalf("denominators: %#v", got) }
	if math.Abs(got.LowerBound-.5) > 1e-12 || math.Abs(got.UpperBound-.5) > 1e-12 { t.Fatalf("ITT=%v..%v", got.LowerBound, got.UpperBound) }
	if r.RetrievalDiagnostics.Retrieved != 1 || r.RetrievalDiagnostics.TreatedPairs != 4 { t.Fatalf("retrieval diagnostic: %#v", r.RetrievalDiagnostics) }
	if r.Primary.CleanSuccess.LowerBound != .5 { t.Fatalf("clean success must retain non-retrieval task pairs: %#v", r.Primary.CleanSuccess) }
	if r.Primary.UnauthorizedEffect.LowerBound != 0 { t.Fatalf("unauthorized effect incidence unexpectedly changed: %#v", r.Primary.UnauthorizedEffect) }
}

func TestTaskPassWithControlsNotPassedIsNotCleanSuccess(t *testing.T) {
	p := testPair("p1", "t1", "code", "1", TaskPass, TaskPass, false)
	p.WithSkill.ControlVerdict = ControlsNotPassed
	p.WithSkill.UnauthorizedEffect = UnauthorizedEffectPresent
	r, err := EstimateITT([]Pair{p, testPair("p2", "t2", "code", "2", TaskPass, TaskPass, false)}, testConfig())
	if err != nil { t.Fatal(err) }
	if r.Primary.TaskSuccess.LowerBound != 0 { t.Fatalf("task success effect: %#v", r.Primary.TaskSuccess) }
	if r.Primary.ControlPass.LowerBound != -0.5 { t.Fatalf("control-pass effect: %#v", r.Primary.ControlPass) }
	if r.Primary.CleanSuccess.LowerBound != -0.5 { t.Fatalf("clean success effect: %#v", r.Primary.CleanSuccess) }
	if r.Primary.UnauthorizedEffect.LowerBound != 0.5 { t.Fatalf("positive unauthorized-effect effect must mean worse: %#v", r.Primary.UnauthorizedEffect) }
}

func TestIndeterminateOutcomesProduceEndpointBounds(t *testing.T) {
	p1 := testPair("p1", "t1", "code", "1", TaskPass, TaskPass, false)
	p1.WithoutSkill.TaskVerdict = TaskIndeterminate
	p1.WithoutSkill.ControlVerdict = ControlsIndeterminate
	p1.WithoutSkill.UnauthorizedEffect = UnauthorizedEffectIndeterminate
	p2 := testPair("p2", "t2", "math", "1", TaskIndeterminate, TaskFail, false)
	p2.WithSkill.ControlVerdict = ControlsIndeterminate
	p2.WithSkill.UnauthorizedEffect = UnauthorizedEffectIndeterminate
	r, err := EstimateITT([]Pair{p1, p2}, testConfig())
	if err != nil { t.Fatal(err) }
	if r.Primary.TaskSuccess.LowerBound != 0 || r.Primary.TaskSuccess.UpperBound != 1 { t.Fatalf("task bounds=%v..%v", r.Primary.TaskSuccess.LowerBound, r.Primary.TaskSuccess.UpperBound) }
	if r.Primary.TaskSuccess.CompletePairs != 0 || r.Primary.TaskSuccess.IndeterminateArms != 2 { t.Fatalf("task missingness accounting %#v", r.Primary.TaskSuccess) }
	if r.Primary.CleanSuccess.LowerBound != 0 || r.Primary.CleanSuccess.UpperBound != 1 { t.Fatalf("clean bounds=%v..%v", r.Primary.CleanSuccess.LowerBound, r.Primary.CleanSuccess.UpperBound) }
}

func TestTaskMacroWeightsTasksNotSeeds(t *testing.T) {
	pairs := []Pair{
		testPair("a1", "large", "code", "1", TaskPass, TaskFail, false),
		testPair("a2", "large", "code", "2", TaskPass, TaskFail, false),
		testPair("a3", "large", "code", "3", TaskPass, TaskFail, false),
		testPair("a4", "large", "code", "4", TaskPass, TaskFail, false),
		testPair("b1", "small", "code", "1", TaskFail, TaskFail, false),
	}
	r, err := EstimateITT(pairs, testConfig())
	if err != nil { t.Fatal(err) }
	if math.Abs(r.Primary.TaskSuccess.LowerBound-.5) > 1e-12 { t.Fatalf("task macro should be 0.5, got %v", r.Primary.TaskSuccess.LowerBound) }
}

func TestDomainReportsAndNoSingleTaskCI(t *testing.T) {
	pairs := []Pair{testPair("p1", "t1", "code", "1", TaskPass, TaskFail, false), testPair("p2", "t2", "math", "1", TaskFail, TaskFail, false)}
	r, err := EstimateITT(pairs, testConfig())
	if err != nil { t.Fatal(err) }
	if len(r.Domains) != 2 || r.Domains[0].Domain != "code" || r.Domains[1].Domain != "math" { t.Fatalf("domains: %#v", r.Domains) }
	if r.Domains[0].TaskSuccess.LowerBound != 1 || r.Domains[1].TaskSuccess.LowerBound != 0 { t.Fatalf("domain estimates: %#v", r.Domains) }
	if r.Domains[0].TaskSuccess.ConfidenceIntervalsAvailable || r.Domains[1].TaskSuccess.ConfidenceIntervalsAvailable { t.Fatal("single-task domains must not receive task-cluster CIs") }
	if r.Domains[0].TaskSuccess.LowerBoundCI != nil { t.Fatal("single-task CI should be null") }
}

func TestBootstrapDeterministic(t *testing.T) {
	pairs := []Pair{testPair("p1", "t1", "d", "1", TaskPass, TaskFail, false), testPair("p2", "t2", "d", "1", TaskFail, TaskFail, false), testPair("p3", "t3", "d", "1", TaskPass, TaskFail, false)}
	a, err := EstimateITT(pairs, testConfig()); if err != nil { t.Fatal(err) }
	b, err := EstimateITT(pairs, testConfig()); if err != nil { t.Fatal(err) }
	if !reflect.DeepEqual(a, b) { t.Fatalf("same seed must reproduce intervals\n%#v\n%#v", a, b) }
}

func TestLifecycleOrderIsEnforced(t *testing.T) {
	p := testPair("p1", "t1", "d", "1", TaskPass, TaskFail, false)
	p.WithSkill.Stages = StageFlags{Executed: true}
	if _, err := EstimateITT([]Pair{p}, testConfig()); err == nil { t.Fatal("non-monotone stage trace accepted") }
}
func TestDuplicateTaskSeedRejected(t *testing.T) {
	pairs := []Pair{testPair("p1", "t1", "d", "1", TaskPass, TaskFail, false), testPair("p2", "t1", "d", "1", TaskPass, TaskFail, false)}
	if _, err := EstimateITT(pairs, testConfig()); err == nil { t.Fatal("duplicate task/seed accepted") }
}
func TestNoSkillArmCannotClaimSkillStages(t *testing.T) {
	p := testPair("p1", "t1", "d", "1", TaskPass, TaskFail, false)
	p.WithoutSkill.Stages = StageFlags{Retrieved: true}
	if _, err := EstimateITT([]Pair{p}, testConfig()); err == nil { t.Fatal("no-skill stage evidence accepted") }
}
func TestAssignmentRosterMustMatchEveryPair(t *testing.T) {
	pairs := []Pair{testPair("p1", "t1", "d", "1", TaskPass, TaskFail, false), testPair("p2", "t2", "d", "1", TaskFail, TaskFail, false)}
	cfg := Config{BootstrapReplicates: 500, BootstrapSeed: 7, ExpectedPairIDs: []string{"p1", "p2"}, RequireAssignmentRoster: true}
	good, err := EstimateITT(pairs, cfg); if err != nil { t.Fatal(err) }
	if !good.AssignmentRosterMatched { t.Fatal("matching roster was not recorded") }
	cfg.ExpectedPairIDs = []string{"p1", "p2", "p3"}
	if _, err := EstimateITT(pairs, cfg); err == nil { t.Fatal("omitted assigned pair was not detected") }
}
func TestAssignmentRosterRequiredWhenRequested(t *testing.T) {
	cfg := Config{BootstrapReplicates: 500, RequireAssignmentRoster: true}
	if _, err := EstimateITT([]Pair{testPair("p1", "t1", "d", "1", TaskPass, TaskFail, false)}, cfg); err == nil { t.Fatal("empty required roster accepted") }
}
func TestMixedStrataAndInvalidArmOrderReject(t *testing.T) {
	p1 := testPair("p1", "t1", "d", "1", TaskPass, TaskFail, false)
	p2 := testPair("p2", "t2", "d", "2", TaskPass, TaskFail, false)
	p2.StratumDigest = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
	if _, err := EstimateITT([]Pair{p1, p2}, testConfig()); err == nil { t.Fatal("mixed configuration strata were accepted") }
	p2.StratumDigest, p2.ArmOrder = p1.StratumDigest, "unknown"
	if _, err := EstimateITT([]Pair{p1, p2}, testConfig()); err == nil { t.Fatal("unknown arm order was accepted") }
}
func TestSmallBootstrapCountRejected(t *testing.T) {
	if _, err := EstimateITT([]Pair{testPair("p1", "t1", "d", "1", TaskPass, TaskFail, false)}, Config{BootstrapReplicates: 20}); err == nil { t.Fatal("too few resamples accepted") }
}
