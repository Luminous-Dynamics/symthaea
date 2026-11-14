package observerlab

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"reflect"
	"sort"
)

type Event struct {
	EventID        string   `json:"event_id"`
	SourceSequence uint64   `json:"source_sequence"`
	Kind           string   `json:"kind"`
	Principal      string   `json:"principal"`
	Target         string   `json:"target"`
	PayloadSHA256  string   `json:"payload_sha256"`
	ParentEventIDs []string `json:"parent_event_ids"`
	Outcome        string   `json:"outcome"`
}

type GroundTruth struct {
	SchemaVersion string  `json:"schema_version"`
	RunID         string  `json:"run_id"`
	Events        []Event `json:"events"`
}

type Observation struct {
	DeliverySequence uint64 `json:"delivery_sequence"`
	Event            Event  `json:"event"`
}

type ObservationLog struct {
	SchemaVersion string        `json:"schema_version"`
	RunID         string        `json:"run_id"`
	Observations  []Observation `json:"observations"`
}

type Report struct {
	SchemaVersion       string   `json:"schema_version"`
	RunID               string   `json:"run_id"`
	GroundTruthEvents   int      `json:"ground_truth_events"`
	ObservedRecords     int      `json:"observed_records"`
	MatchedEvents       int      `json:"matched_events"`
	MissingEvents       int      `json:"missing_events"`
	ExtraEvents         int      `json:"extra_events"`
	DuplicateEvents     int      `json:"duplicate_events"`
	SequenceMismatches  int      `json:"sequence_mismatches"`
	IdentityMismatches  int      `json:"identity_mismatches"`
	PayloadMismatches   int      `json:"payload_mismatches"`
	ParentMismatches    int      `json:"parent_mismatches"`
	OutcomeMismatches   int      `json:"outcome_mismatches"`
	DeliveryReorders    int      `json:"delivery_reorders"`
	MissingEventIDs     []string `json:"missing_event_ids"`
	ExtraEventIDs       []string `json:"extra_event_ids"`
	DuplicateEventIDs   []string `json:"duplicate_event_ids"`
	CausalGraphValid    bool     `json:"causal_graph_valid"`
	ObservationComplete bool     `json:"observation_complete"`
	Accepted            bool     `json:"accepted"`
	Limitations         []string `json:"limitations"`
}

func PayloadDigest(payload []byte) string {
	sum := sha256.Sum256(payload)
	return hex.EncodeToString(sum[:])
}

// ValidateGroundTruth checks internal consistency of a simulator-authored
// reference ledger. It does not independently observe or prove every real-world effect.
func ValidateGroundTruth(gt GroundTruth) error {
	if gt.SchemaVersion != "mace-observerlab/v1" || gt.RunID == "" {
		return fmt.Errorf("unsupported ground-truth schema or empty run_id")
	}
	if len(gt.Events) == 0 {
		return fmt.Errorf("ground-truth event ledger is empty")
	}
	byID := make(map[string]Event, len(gt.Events))
	bySequence := make(map[uint64]string, len(gt.Events))
	for _, event := range gt.Events {
		if event.EventID == "" || event.Kind == "" || event.Principal == "" || event.Target == "" || event.Outcome == "" {
			return fmt.Errorf("event identity, kind, principal, target and outcome are required")
		}
		if len(event.PayloadSHA256) != 64 || event.PayloadSHA256 != lowerHex(event.PayloadSHA256) {
			return fmt.Errorf("event %s has invalid payload SHA-256", event.EventID)
		}
		if _, err := hex.DecodeString(event.PayloadSHA256); err != nil {
			return fmt.Errorf("event %s has invalid payload SHA-256: %w", event.EventID, err)
		}
		if _, exists := byID[event.EventID]; exists {
			return fmt.Errorf("duplicate ground-truth event_id %q", event.EventID)
		}
		if previous, exists := bySequence[event.SourceSequence]; exists {
			return fmt.Errorf("duplicate source_sequence %d for %s and %s", event.SourceSequence, previous, event.EventID)
		}
		byID[event.EventID] = event
		bySequence[event.SourceSequence] = event.EventID
	}
	for _, event := range gt.Events {
		seenParents := map[string]bool{}
		for _, parentID := range event.ParentEventIDs {
			parent, ok := byID[parentID]
			if !ok {
				return fmt.Errorf("event %s references missing parent %s", event.EventID, parentID)
			}
			if parent.SourceSequence >= event.SourceSequence {
				return fmt.Errorf("event %s has non-causal parent %s", event.EventID, parentID)
			}
			if seenParents[parentID] {
				return fmt.Errorf("event %s repeats parent %s", event.EventID, parentID)
			}
			seenParents[parentID] = true
		}
	}
	return nil
}

func Compare(gt GroundTruth, log ObservationLog) (Report, error) {
	if err := ValidateGroundTruth(gt); err != nil {
		return Report{}, fmt.Errorf("invalid ground truth: %w", err)
	}
	if log.SchemaVersion != "mace-observerlab/v1" || log.RunID != gt.RunID {
		return Report{}, fmt.Errorf("observer log schema or run_id mismatch")
	}
	report := Report{
		SchemaVersion:     "mace-observerlab-report/v1",
		RunID:             gt.RunID,
		GroundTruthEvents: len(gt.Events),
		ObservedRecords:   len(log.Observations),
		CausalGraphValid:  true,
		Limitations: []string{
			"The ground-truth ledger is simulator-authored; it is not an independent observation of a production operating system or external service.",
			"This comparator checks event identity, selected fields, source sequence and explicit parent IDs; it does not prove observer completeness outside the declared simulator boundary.",
			"Delivery reordering is diagnostic only when source sequence and exact parent references remain correct; wall-clock timestamps are not used to infer causality.",
		},
	}

	truth := make(map[string]Event, len(gt.Events))
	for _, event := range gt.Events {
		truth[event.EventID] = event
	}
	seen := map[string]bool{}
	firstObserved := map[string]Event{}
	firstDelivery := map[string]uint64{}
	for _, observation := range log.Observations {
		event := observation.Event
		if _, exists := truth[event.EventID]; !exists {
			report.ExtraEvents++
			report.ExtraEventIDs = append(report.ExtraEventIDs, event.EventID)
			continue
		}
		if seen[event.EventID] {
			report.DuplicateEvents++
			report.DuplicateEventIDs = append(report.DuplicateEventIDs, event.EventID)
			continue
		}
		seen[event.EventID] = true
		firstObserved[event.EventID] = event
		firstDelivery[event.EventID] = observation.DeliverySequence
	}

	for _, expected := range gt.Events {
		observed, ok := firstObserved[expected.EventID]
		if !ok {
			report.MissingEvents++
			report.MissingEventIDs = append(report.MissingEventIDs, expected.EventID)
			continue
		}
		report.MatchedEvents++
		if observed.SourceSequence != expected.SourceSequence {
			report.SequenceMismatches++
			report.CausalGraphValid = false
		}
		if observed.Kind != expected.Kind || observed.Principal != expected.Principal || observed.Target != expected.Target {
			report.IdentityMismatches++
		}
		if observed.PayloadSHA256 != expected.PayloadSHA256 {
			report.PayloadMismatches++
		}
		if observed.Outcome != expected.Outcome {
			report.OutcomeMismatches++
		}
		if !reflect.DeepEqual(observed.ParentEventIDs, expected.ParentEventIDs) {
			report.ParentMismatches++
			report.CausalGraphValid = false
		}
	}

	// Delivery order is not causal order. Count regressions among known, unique
	// observations but do not fail solely because transport batches were reordered.
	ordered := append([]Observation(nil), log.Observations...)
	sort.SliceStable(ordered, func(i, j int) bool {
		return ordered[i].DeliverySequence < ordered[j].DeliverySequence
	})
	var lastSourceSequence uint64
	hasLast := false
	for _, observation := range ordered {
		if _, known := truth[observation.Event.EventID]; !known || !seen[observation.Event.EventID] || firstDelivery[observation.Event.EventID] != observation.DeliverySequence {
			continue
		}
		if hasLast && observation.Event.SourceSequence < lastSourceSequence {
			report.DeliveryReorders++
		}
		lastSourceSequence = observation.Event.SourceSequence
		hasLast = true
	}

	sort.Strings(report.MissingEventIDs)
	sort.Strings(report.ExtraEventIDs)
	sort.Strings(report.DuplicateEventIDs)
	report.ObservationComplete = report.MissingEvents == 0 && report.ExtraEvents == 0 && report.DuplicateEvents == 0
	report.Accepted = report.ObservationComplete && report.SequenceMismatches == 0 && report.IdentityMismatches == 0 && report.PayloadMismatches == 0 && report.ParentMismatches == 0 && report.OutcomeMismatches == 0 && report.CausalGraphValid
	return report, nil
}

func lowerHex(value string) string {
	out := []byte(value)
	for i := range out {
		if out[i] >= 'A' && out[i] <= 'F' {
			out[i] += 'a' - 'A'
		}
	}
	return string(out)
}
