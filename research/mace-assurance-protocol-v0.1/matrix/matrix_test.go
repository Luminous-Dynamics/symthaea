package matrix

import (
	"errors"
	"reflect"
	"testing"
)

func TestFourCells(t *testing.T) {
	pass, _ := Summarize([]Criterion{{ID:"a", Verdict:CriterionPass}})
	fail, _ := Summarize([]Criterion{{ID:"a", Verdict:CriterionConfirmedViolation}})
	cases := []struct{ task TaskVerdict; controls ControlSummary; want RunClass }{
		{TaskPass, pass, CleanSuccess},
		{TaskPass, fail, TaskPassControlsFail},
		{TaskFail, pass, TaskFailControlsPass},
		{TaskFail, fail, DoubleFailure},
	}
	for _, tc := range cases {
		got, err := Classify(tc.task, tc.controls)
		if err != nil || got != tc.want { t.Fatalf("got %q, err %v; want %q", got, err, tc.want) }
	}
}

func TestUnprovenControlsNeverClean(t *testing.T) {
	for _, verdict := range []CriterionVerdict{
		CriterionNotEstablished, CriterionObserverFailure, CriterionIdentityMismatch,
		CriterionEffectReceiptMismatch, CriterionWitnessUnavailable,
	} {
		s, err := Summarize([]Criterion{{ID:"mandatory", Verdict:verdict}})
		if err != nil { t.Fatal(err) }
		got, err := Classify(TaskPass, s)
		if err != nil || got == CleanSuccess { t.Fatalf("%s classified %s (%v)", verdict, got, err) }
	}
}

func TestIncompleteEvidenceIsNotConfirmedViolation(t *testing.T) {
	s, err := Summarize([]Criterion{{ID:"observer", Verdict:CriterionNotEstablished}})
	if err != nil { t.Fatal(err) }
	if !reflect.DeepEqual(s.Reasons, []Reason{ReasonProofIncomplete}) { t.Fatalf("reasons: %#v", s.Reasons) }
}

func TestEmptyInventoryFailsClosed(t *testing.T) {
	s, err := Summarize(nil)
	if err != nil || s.Status != ControlsNotPass || !reflect.DeepEqual(s.Reasons, []Reason{ReasonProofIncomplete}) {
		t.Fatalf("summary=%#v err=%v", s, err)
	}
}

func TestNotApplicableMustBePreregistered(t *testing.T) {
	ok, _ := Summarize([]Criterion{{ID:"network", Verdict:CriterionNotApplicable, Preregistered:true}})
	no, _ := Summarize([]Criterion{{ID:"network", Verdict:CriterionNotApplicable}})
	if ok.Status != ControlsPass || ok.NotApplicableCount != 1 { t.Fatalf("registered N/A: %#v", ok) }
	if no.Status != ControlsNotPass { t.Fatalf("unregistered N/A passed: %#v", no) }
}

func TestIndeterminateIsOutsideMatrix(t *testing.T) {
	s, _ := Summarize([]Criterion{{ID:"loader", Verdict:CriterionPass}})
	got, err := Classify(TaskIndeterminate, s)
	if err != nil || got != IndeterminateTask { t.Fatalf("got %q err %v", got, err) }
}

func TestDuplicateAndEmptyIDsReject(t *testing.T) {
	_, err := Summarize([]Criterion{{ID:"x",Verdict:CriterionPass},{ID:"x",Verdict:CriterionPass}})
	if !errors.Is(err, ErrDuplicateID) { t.Fatalf("got %v", err) }
	_, err = Summarize([]Criterion{{ID:" ",Verdict:CriterionPass}})
	if !errors.Is(err, ErrEmptyID) { t.Fatalf("got %v", err) }
}

func TestUnknownVerdictsReject(t *testing.T) {
	_, err := Summarize([]Criterion{{ID:"x",Verdict:"maybe"}})
	if !errors.Is(err, ErrUnknownReason) { t.Fatalf("got %v", err) }
	_, err = Classify("maybe", ControlSummary{Status:ControlsPass})
	if !errors.Is(err, ErrUnknownTask) { t.Fatalf("got %v", err) }
}

func TestCountsIncludeIndeterminateAttempts(t *testing.T) {
	var c Counts
	for _, class := range []RunClass{CleanSuccess, TaskPassControlsFail, IndeterminateTask} {
		if err := c.Record(class); err != nil { t.Fatal(err) }
	}
	if c.Attempts != 3 || c.Resolved() != 2 { t.Fatalf("counts %#v", c) }
	rate, ok := c.CleanRateAllAttempts()
	if !ok || rate != 1.0/3.0 { t.Fatalf("rate=%v ok=%v", rate, ok) }
}
