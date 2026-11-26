package oarinterop

import "testing"

func TestOnlyConfirmedIsCommitted(t *testing.T) {
	if got := OutcomeClass("confirmed"); got != Committed {
		t.Fatalf("confirmed => %q, want %q", got, Committed)
	}
}

func TestOARRefusalVocabulary(t *testing.T) {
	for _, outcome := range []string{"denied", "rejected", "expired", "refused"} {
		if got := OutcomeClass(outcome); got != Refused {
			t.Errorf("%q => %q, want %q", outcome, got, Refused)
		}
	}
}

func TestAllOtherOutcomesRemainUnresolved(t *testing.T) {
	for _, outcome := range []string{"held", "approved", "allowed", "executed", "failed", "not_confirmed", "", "unknown", "producer_says_success"} {
		if got := OutcomeClass(outcome); got != Unresolved {
			t.Errorf("%q => %q, want %q", outcome, got, Unresolved)
		}
	}
}
