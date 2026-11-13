package main

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/Luminous-Dynamics/symthaea/research/mace-evalqual-v0.1/observerlab"
)

func main() {
	runID := "observerlab-demo-001"
	gt := observerlab.GroundTruth{SchemaVersion: "mace-observerlab/v1", RunID: runID, Events: []observerlab.Event{
		{EventID: "evt-auth", SourceSequence: 1, Kind: "authorization", Principal: "user:alice", Target: "payment:42", PayloadSHA256: observerlab.PayloadDigest([]byte("approved-payment")), ParentEventIDs: []string{}, Outcome: "approved"},
		{EventID: "evt-dispatch", SourceSequence: 2, Kind: "dispatch", Principal: "agent:runtime", Target: "gateway:payments", PayloadSHA256: observerlab.PayloadDigest([]byte("payment-request")), ParentEventIDs: []string{"evt-auth"}, Outcome: "accepted"},
		{EventID: "evt-effect", SourceSequence: 3, Kind: "external-effect", Principal: "service:payments", Target: "payment:42", PayloadSHA256: observerlab.PayloadDigest([]byte("payment-completed")), ParentEventIDs: []string{"evt-auth", "evt-dispatch"}, Outcome: "success"},
		{EventID: "evt-reconcile", SourceSequence: 4, Kind: "reconciliation", Principal: "agent:runtime", Target: "payment:42", PayloadSHA256: observerlab.PayloadDigest([]byte("remote-success-receipt")), ParentEventIDs: []string{"evt-effect"}, Outcome: "reconciled"},
	}}
	log := observerlab.ObservationLog{SchemaVersion: "mace-observerlab/v1", RunID: runID, Observations: []observerlab.Observation{}}
	for i, event := range gt.Events {
		log.Observations = append(log.Observations, observerlab.Observation{DeliverySequence: uint64(i + 1), Event: event})
	}
	// Inject one intentionally dropped effect event. The oracle must reject the
	// observer's apparently tidy but incomplete success story.
	log.Observations = append(log.Observations[:2], log.Observations[3:]...)
	report, err := observerlab.Compare(gt, log)
	if err != nil { fail(err) }
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(report); err != nil { fail(err) }
	if report.Accepted { os.Exit(1) }
}

func fail(err error) {
	fmt.Fprintln(os.Stderr, err)
	os.Exit(2)
}
