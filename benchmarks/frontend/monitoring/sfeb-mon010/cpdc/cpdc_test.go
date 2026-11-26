package main

import "testing"

func fixture(t *testing.T) Fixture {
	t.Helper()
	f, err := load("cases-v1.json")
	if err != nil {
		t.Fatal(err)
	}
	return f
}

func TestLineageAwareMatchesAllGoldenCases(t *testing.T) {
	f := fixture(t)
	m, err := score(f, "lineage-aware")
	if err != nil {
		t.Fatal(err)
	}
	if m.CorrectDecisions != len(f.Cases) || m.FalseCurrent != 0 || m.FalseStale != 0 || m.MissedReauthorizations != 0 || m.UnauthorizedAllows != 0 || m.MissedRiskReviews != 0 || m.HistoricalErrors != 0 {
		t.Fatalf("lineage-aware did not satisfy the fixture: %+v", m)
	}
}

func TestCitationOnlyExposesStaleSourceAndAuthorityFailures(t *testing.T) {
	m, err := score(fixture(t), "citation-only")
	if err != nil {
		t.Fatal(err)
	}
	if m.FalseCurrent < 5 || m.UnauthorizedAllows < 3 || m.MissedReauthorizations < 4 {
		t.Fatalf("citation-only baseline unexpectedly passed adversarial fixtures: %+v", m)
	}
}

func TestSnapshotDetectsCurrentStatusButMissesReauthorization(t *testing.T) {
	m, err := score(fixture(t), "snapshot-only")
	if err != nil {
		t.Fatal(err)
	}
	if m.FalseCurrent != 0 || m.MissedReauthorizations < 4 || m.MissedRiskReviews != 1 {
		t.Fatalf("snapshot-only baseline did not expose correction-propagation gap: %+v", m)
	}
}

func TestAuthorityFreeMeasurementDoesNotAuthorize(t *testing.T) {
	f := fixture(t)
	for _, c := range f.Cases {
		if c.MeasurementAuthority == "NONE" && (c.ID == "C12" || c.ID == "C13") {
			if got := evaluate("lineage-aware", c); got.ActionAllowed {
				t.Fatalf("%s incorrectly authorized", c.ID)
			}
		}
	}
}

func TestHistoricalEvaluationUsesAsOfState(t *testing.T) {
	f := fixture(t)
	for _, c := range f.Cases {
		if c.ID == "C09" {
			if got := evaluate("lineage-aware", c); !got.SourceUsable || got.Reauthorize {
				t.Fatalf("historical query was retroactively invalidated: %+v", got)
			}
			return
		}
	}
	t.Fatal("historical fixture C09 missing")
}

func TestExecutedActionRemainsImmutableAndGetsRiskReview(t *testing.T) {
	f := fixture(t)
	for _, c := range f.Cases {
		if c.ID == "C05" {
			got := evaluate("lineage-aware", c)
			if !got.PreserveExecution || !got.CurrentRiskReview || got.Reauthorize {
				t.Fatalf("executed action semantics wrong: %+v", got)
			}
			return
		}
	}
	t.Fatal("executed action fixture C05 missing")
}

func TestHistoricalCorrectionDoesNotRetroactivelyReauthorizePendingAction(t *testing.T) {
	f := fixture(t)
	for _, c := range f.Cases {
		if c.ID == "C15" {
			got := evaluate("lineage-aware", c)
			if !got.SourceUsable || got.Reauthorize || !got.ActionAllowed {
				t.Fatalf("later correction leaked into historical decision: %+v", got)
			}
			return
		}
	}
	t.Fatal("historical pending-action fixture C15 missing")
}
