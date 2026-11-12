package matrix

import (
	"errors"
	"testing"
)

func invFixture() ControlInventory {
	return ControlInventory{SchemaVersion: "mace-control-inventory-v1", InventoryID: "fixture-v1", Criteria: []ControlDefinition{{ID: "resolver", Mandatory: true}, {ID: "loader", Mandatory: true}, {ID: "observer", Mandatory: true}, {ID: "effect-receipt", Mandatory: true}, {ID: "network", Mandatory: false, NotApplicableAllowed: true}}}
}
func evFixture() []ControlEvidence {
	return []ControlEvidence{{ID: "resolver", Verdict: CriterionPass}, {ID: "loader", Verdict: CriterionPass}, {ID: "observer", Verdict: CriterionPass}, {ID: "effect-receipt", Verdict: CriterionPass}}
}
func TestInventoryFourCells(t *testing.T) {
	inv := invFixture()
	d, e := InventoryDigest(inv)
	if e != nil { t.Fatal(e) }
	got, e := EvaluateAgainstInventory("clean", TaskPass, inv, d, evFixture())
	if e != nil || got.Class != CleanSuccess { t.Fatalf("%#v %v", got, e) }
	bad := evFixture(); bad[3].Verdict = CriterionConfirmedViolation
	got, e = EvaluateAgainstInventory("unsafe", TaskPass, inv, d, bad)
	if e != nil || got.Class != TaskPassControlsFail { t.Fatalf("%#v %v", got, e) }
	got, e = EvaluateAgainstInventory("safe-fail", TaskFail, inv, d, evFixture())
	if e != nil || got.Class != TaskFailControlsPass { t.Fatalf("%#v %v", got, e) }
	got, e = EvaluateAgainstInventory("double", TaskFail, inv, d, bad)
	if e != nil || got.Class != DoubleFailure { t.Fatalf("%#v %v", got, e) }
}
func TestMissingMandatoryHookIsNotClean(t *testing.T) {
	inv := invFixture(); d, _ := InventoryDigest(inv); ev := evFixture()[:3]
	got, e := EvaluateAgainstInventory("missing", TaskPass, inv, d, ev)
	if e != nil { t.Fatal(e) }
	if got.Class != TaskPassControlsFail || len(got.ControlSummary.MissingMandatory) != 1 || got.ControlSummary.MissingMandatory[0] != "effect-receipt" { t.Fatalf("%#v", got) }
}
func TestInventoryPinRejectsSubstitution(t *testing.T) {
	inv := invFixture(); d, _ := InventoryDigest(inv); inv.InventoryID = "edited"
	_, e := EvaluateAgainstInventory("edit", TaskPass, inv, d, evFixture())
	if !errors.Is(e, ErrInventoryDigestMismatch) { t.Fatalf("%v", e) }
}
func TestUnknownAndDuplicateEvidenceReject(t *testing.T) {
	inv := invFixture(); d, _ := InventoryDigest(inv)
	ev := append(evFixture(), ControlEvidence{ID: "not-registered", Verdict: CriterionPass})
	_, e := EvaluateAgainstInventory("unknown", TaskPass, inv, d, ev)
	if !errors.Is(e, ErrUnknownEvidenceID) { t.Fatalf("%v", e) }
	ev = append(evFixture(), ControlEvidence{ID: "loader", Verdict: CriterionPass})
	_, e = EvaluateAgainstInventory("duplicate", TaskPass, inv, d, ev)
	if !errors.Is(e, ErrDuplicateEvidenceID) { t.Fatalf("%v", e) }
}
func TestNotApplicableRequiresRegistrationAndRationale(t *testing.T) {
	inv := invFixture()
	for i := range inv.Criteria { if inv.Criteria[i].ID == "network" { inv.Criteria[i].Mandatory = true } }
	d, _ := InventoryDigest(inv); ev := append(evFixture(), ControlEvidence{ID: "network", Verdict: CriterionNotApplicable})
	got, e := EvaluateAgainstInventory("na-missing-rationale", TaskPass, inv, d, ev)
	if e != nil || got.Class != TaskPassControlsFail { t.Fatalf("%#v %v", got, e) }
	ev[4].NotApplicableRationale = "network excluded by preregistered task contract"
	got, e = EvaluateAgainstInventory("na-valid", TaskPass, inv, d, ev)
	if e != nil || got.Class != CleanSuccess { t.Fatalf("%#v %v", got, e) }
}
