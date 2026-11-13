package observerlab

import "testing"

func fixture() (GroundTruth, ObservationLog) {
	runID := "observerlab-demo-001"
	gt := GroundTruth{SchemaVersion: "mace-observerlab/v1", RunID: runID, Events: []Event{
		{EventID: "evt-auth", SourceSequence: 1, Kind: "authorization", Principal: "user:alice", Target: "payment:42", PayloadSHA256: PayloadDigest([]byte("approved-payment")), ParentEventIDs: []string{}, Outcome: "approved"},
		{EventID: "evt-dispatch", SourceSequence: 2, Kind: "dispatch", Principal: "agent:runtime", Target: "gateway:payments", PayloadSHA256: PayloadDigest([]byte("payment-request")), ParentEventIDs: []string{"evt-auth"}, Outcome: "accepted"},
		{EventID: "evt-effect", SourceSequence: 3, Kind: "external-effect", Principal: "service:payments", Target: "payment:42", PayloadSHA256: PayloadDigest([]byte("payment-completed")), ParentEventIDs: []string{"evt-auth", "evt-dispatch"}, Outcome: "success"},
		{EventID: "evt-reconcile", SourceSequence: 4, Kind: "reconciliation", Principal: "agent:runtime", Target: "payment:42", PayloadSHA256: PayloadDigest([]byte("remote-success-receipt")), ParentEventIDs: []string{"evt-effect"}, Outcome: "reconciled"},
	}}
	log := ObservationLog{SchemaVersion: "mace-observerlab/v1", RunID: runID}
	for i, event := range gt.Events {
		log.Observations = append(log.Observations, Observation{DeliverySequence: uint64(i + 1), Event: event})
	}
	return gt, log
}

func TestCleanObserverAccepted(t *testing.T) {
	gt, log := fixture()
	report, err := Compare(gt, log)
	if err != nil { t.Fatal(err) }
	if !report.Accepted || report.MatchedEvents != 4 || !report.ObservationComplete {
		t.Fatalf("clean observer rejected: %+v", report)
	}
}

func TestDroppedEventDetected(t *testing.T) {
	gt, log := fixture()
	log.Observations = append(log.Observations[:2], log.Observations[3:]...)
	report, err := Compare(gt, log)
	if err != nil { t.Fatal(err) }
	if report.Accepted || report.MissingEvents != 1 || report.MissingEventIDs[0] != "evt-effect" {
		t.Fatalf("dropped event not detected: %+v", report)
	}
}

func TestFabricatedEventDetected(t *testing.T) {
	gt, log := fixture()
	fake := Event{EventID: "evt-fake", SourceSequence: 5, Kind: "external-effect", Principal: "service:payments", Target: "payment:99", PayloadSHA256: PayloadDigest([]byte("fabricated")), Outcome: "success"}
	log.Observations = append(log.Observations, Observation{DeliverySequence: 5, Event: fake})
	report, err := Compare(gt, log)
	if err != nil { t.Fatal(err) }
	if report.Accepted || report.ExtraEvents != 1 || report.ExtraEventIDs[0] != "evt-fake" {
		t.Fatalf("fabricated event not detected: %+v", report)
	}
}

func TestDuplicateReplayDetected(t *testing.T) {
	gt, log := fixture()
	log.Observations = append(log.Observations, Observation{DeliverySequence: 5, Event: gt.Events[2]})
	report, err := Compare(gt, log)
	if err != nil { t.Fatal(err) }
	if report.Accepted || report.DuplicateEvents != 1 || report.DuplicateEventIDs[0] != "evt-effect" {
		t.Fatalf("duplicate replay not detected: %+v", report)
	}
}

func TestPrincipalAndPayloadSubstitutionDetected(t *testing.T) {
	gt, log := fixture()
	log.Observations[2].Event.Principal = "service:attacker"
	log.Observations[2].Event.PayloadSHA256 = PayloadDigest([]byte("different-payment"))
	report, err := Compare(gt, log)
	if err != nil { t.Fatal(err) }
	if report.Accepted || report.IdentityMismatches != 1 || report.PayloadMismatches != 1 {
		t.Fatalf("principal/payload substitution not detected: %+v", report)
	}
}

func TestCausalParentSubstitutionDetected(t *testing.T) {
	gt, log := fixture()
	log.Observations[2].Event.ParentEventIDs = []string{"evt-dispatch"}
	report, err := Compare(gt, log)
	if err != nil { t.Fatal(err) }
	if report.Accepted || report.ParentMismatches != 1 || report.CausalGraphValid {
		t.Fatalf("causal parent substitution not detected: %+v", report)
	}
}

func TestDeliveryReorderingDoesNotRewriteCausality(t *testing.T) {
	gt, log := fixture()
	log.Observations[0].DeliverySequence = 3
	log.Observations[1].DeliverySequence = 1
	log.Observations[2].DeliverySequence = 4
	log.Observations[3].DeliverySequence = 2
	report, err := Compare(gt, log)
	if err != nil { t.Fatal(err) }
	if !report.Accepted || report.DeliveryReorders == 0 || report.SequenceMismatches != 0 {
		t.Fatalf("transport reorder should be recorded without falsifying source causality: %+v", report)
	}
}

func TestSourceSequenceSubstitutionRejected(t *testing.T) {
	gt, log := fixture()
	log.Observations[2].Event.SourceSequence = 1
	report, err := Compare(gt, log)
	if err != nil { t.Fatal(err) }
	if report.Accepted || report.SequenceMismatches != 1 || report.CausalGraphValid {
		t.Fatalf("source sequence substitution not detected: %+v", report)
	}
}

func TestGroundTruthRejectsCycleAndUnknownParent(t *testing.T) {
	gt, _ := fixture()
	gt.Events[0].ParentEventIDs = []string{"evt-reconcile"}
	if err := ValidateGroundTruth(gt); err == nil { t.Fatal("causal cycle/noncausal parent accepted") }
	gt, _ = fixture()
	gt.Events[1].ParentEventIDs = []string{"not-present"}
	if err := ValidateGroundTruth(gt); err == nil { t.Fatal("unknown parent accepted") }
}
