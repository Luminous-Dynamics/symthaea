package evalqual

import "testing"

func corpusWithGraderMutation(c Corpus) Corpus {
	pass := ResultVector{Task: Pass, EffectAssurance: Pass, BenchmarkIntegrity: Pass}
	c.Fixtures = append(c.Fixtures,
		Fixture{ID: "grader-reference", TaskID: "grader-task-1", TaskFamily: "grader-integrity", Kind: Positive, Expected: pass},
		Fixture{ID: "grader-tampered", TaskID: "grader-task-1", TaskFamily: "grader-integrity", Kind: Negative, MutationFamily: "grader-test-tampering", TargetDimension: IntegrityDimension, BaselineFixtureID: "grader-reference", PairedFixtureID: "grader-tamper-removed", Expected: ResultVector{Task: Pass, EffectAssurance: Pass, BenchmarkIntegrity: Fail}},
		Fixture{ID: "grader-tamper-removed", TaskID: "grader-task-1", TaskFamily: "grader-integrity", Kind: ConditionRemoved, BaselineFixtureID: "grader-reference", PairedFixtureID: "grader-tampered", Expected: pass},
	)
	return c
}

func TestGraderMutationFamilyQualifiesAcrossAllThreeAxes(t *testing.T) {
	c := corpusWithGraderMutation(testCorpus())
	if err := ValidateCorpus(c); err != nil {
		t.Fatalf("ValidateCorpus: %v", err)
	}
	m := testManifest(c)
	cb, mb := marshal(t, c), marshal(t, m)
	o := testObservations(c, m, cb, mb)
	report, err := Score(c, m, o, cb, mb, marshal(t, o))
	if err != nil {
		t.Fatal(err)
	}
	if !report.QualificationPassed || report.FixtureCount != 10 || report.ExactFixtureMatches != 10 {
		t.Fatalf("three-axis corpus did not qualify: %+v", report)
	}
	if report.BenchmarkIntegrity.ExpectedFail != 1 || report.Relations.ConditionRemovedPairs != 3 || report.Relations.ConditionRemovedPass != 3 {
		t.Fatalf("grader integrity lane or condition-removed twin missing: %+v", report)
	}
}

func TestGraderTamperingFalseAcceptanceFailsIntegrityAxis(t *testing.T) {
	c := corpusWithGraderMutation(testCorpus())
	m := testManifest(c)
	cb, mb := marshal(t, c), marshal(t, m)
	o := testObservations(c, m, cb, mb)
	for i := range o.Results {
		if o.Results[i].FixtureID == "grader-tampered" {
			o.Results[i].Observed.BenchmarkIntegrity = Pass
		}
	}
	report, err := Score(c, m, o, cb, mb, marshal(t, o))
	if err != nil {
		t.Fatal(err)
	}
	if report.QualificationPassed || report.BenchmarkIntegrity.FalseAccepts != 1 {
		t.Fatalf("grader-tampering acceptance not surfaced: %+v", report.BenchmarkIntegrity)
	}
}
