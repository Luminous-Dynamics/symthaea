package matrix

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strings"
)

type ControlDefinition struct {
	ID                   string `json:"id"`
	Mandatory            bool   `json:"mandatory"`
	NotApplicableAllowed bool   `json:"not_applicable_allowed,omitempty"`
}

type ControlInventory struct {
	SchemaVersion string              `json:"schema_version"`
	InventoryID   string              `json:"inventory_id"`
	Criteria      []ControlDefinition `json:"criteria"`
}

type ControlEvidence struct {
	ID                     string           `json:"id"`
	Verdict                CriterionVerdict `json:"verdict"`
	NotApplicableRationale string           `json:"not_applicable_rationale,omitempty"`
}

type InventorySummary struct {
	Summary          ControlSummary `json:"summary"`
	MissingMandatory []string       `json:"missing_mandatory,omitempty"`
	InventorySHA256  string         `json:"inventory_sha256"`
}

type InventoryRunResult struct {
	RunID          string           `json:"run_id"`
	TaskVerdict    TaskVerdict      `json:"task_verdict"`
	ControlSummary InventorySummary `json:"control_summary"`
	Class          RunClass         `json:"class"`
}

var (
	ErrInvalidInventory        = errors.New("invalid control inventory")
	ErrInventoryDigestMismatch = errors.New("control inventory digest mismatch")
	ErrUnknownEvidenceID       = errors.New("evidence ID is outside the registered inventory")
	ErrDuplicateEvidenceID     = errors.New("duplicate control evidence ID")
)

// InventoryDigest computes a deterministic SHA-256 over the typed inventory.
// The expected digest must be pinned by an independent preregistration
// authority; a digest alone is not a signature or proof of preregistration.
func InventoryDigest(inventory ControlInventory) (string, error) {
	if inventory.SchemaVersion != "mace-control-inventory-v1" || !portableID(inventory.InventoryID) || len(inventory.Criteria) == 0 {
		return "", fmt.Errorf("%w: schema, canonical inventory ID and criteria are required", ErrInvalidInventory)
	}
	seen := make(map[string]bool, len(inventory.Criteria))
	mandatory := 0
	canonical := ControlInventory{SchemaVersion: inventory.SchemaVersion, InventoryID: inventory.InventoryID, Criteria: append([]ControlDefinition(nil), inventory.Criteria...)}
	for _, criterion := range canonical.Criteria {
		if !portableID(criterion.ID) {
			return "", fmt.Errorf("%w: noncanonical criterion ID %q", ErrInvalidInventory, criterion.ID)
		}
		if seen[criterion.ID] {
			return "", fmt.Errorf("%w: duplicate criterion ID %q", ErrInvalidInventory, criterion.ID)
		}
		seen[criterion.ID] = true
		if criterion.Mandatory {
			mandatory++
		}
	}
	if mandatory == 0 {
		return "", fmt.Errorf("%w: at least one mandatory criterion is required", ErrInvalidInventory)
	}
	sort.Slice(canonical.Criteria, func(i, j int) bool { return canonical.Criteria[i].ID < canonical.Criteria[j].ID })
	encoded, err := json.Marshal(canonical)
	if err != nil {
		return "", fmt.Errorf("marshal inventory: %w", err)
	}
	digest := sha256.Sum256(encoded)
	return hex.EncodeToString(digest[:]), nil
}

// SummarizeAgainstInventory proves evidence-set closure relative to the
// supplied inventory. Omitted mandatory criteria become proof-incomplete.
func SummarizeAgainstInventory(inventory ControlInventory, expectedDigest string, evidence []ControlEvidence) (InventorySummary, error) {
	actualDigest, err := InventoryDigest(inventory)
	if err != nil {
		return InventorySummary{}, err
	}
	if len(expectedDigest) != 64 || strings.ToLower(expectedDigest) != expectedDigest {
		return InventorySummary{}, fmt.Errorf("%w: lower-case SHA-256 pin required", ErrInventoryDigestMismatch)
	}
	if _, err := hex.DecodeString(expectedDigest); err != nil || actualDigest != expectedDigest {
		return InventorySummary{}, fmt.Errorf("%w: expected %s computed %s", ErrInventoryDigestMismatch, expectedDigest, actualDigest)
	}
	definitions := make(map[string]ControlDefinition, len(inventory.Criteria))
	for _, definition := range inventory.Criteria {
		definitions[definition.ID] = definition
	}
	observed := make(map[string]ControlEvidence, len(evidence))
	for _, item := range evidence {
		if !portableID(item.ID) {
			return InventorySummary{}, fmt.Errorf("%w: invalid evidence ID %q", ErrInvalidInventory, item.ID)
		}
		if _, ok := definitions[item.ID]; !ok {
			return InventorySummary{}, fmt.Errorf("%w: %s", ErrUnknownEvidenceID, item.ID)
		}
		if _, ok := observed[item.ID]; ok {
			return InventorySummary{}, fmt.Errorf("%w: %s", ErrDuplicateEvidenceID, item.ID)
		}
		if !knownVerdict(item.Verdict) {
			return InventorySummary{}, fmt.Errorf("%w: %q for %s", ErrUnknownReason, item.Verdict, item.ID)
		}
		if item.Verdict != CriterionNotApplicable && strings.TrimSpace(item.NotApplicableRationale) != "" {
			return InventorySummary{}, fmt.Errorf("%w: N/A rationale on non-N/A criterion %s", ErrInvalidInventory, item.ID)
		}
		observed[item.ID] = item
	}

	var reasons []Reason
	reasonSet := make(map[Reason]bool)
	missing := make([]string, 0)
	naCount := 0
	for _, definition := range inventory.Criteria {
		if !definition.Mandatory {
			continue
		}
		item, ok := observed[definition.ID]
		if !ok {
			missing = append(missing, definition.ID)
			continue
		}
		var reason Reason
		switch item.Verdict {
		case CriterionPass:
			continue
		case CriterionConfirmedViolation:
			reason = ReasonViolation
		case CriterionNotEstablished:
			reason = ReasonProofIncomplete
		case CriterionObserverFailure:
			reason = ReasonObserverFailure
		case CriterionIdentityMismatch:
			reason = ReasonIdentityMismatch
		case CriterionEffectReceiptMismatch:
			reason = ReasonEffectReceiptMismatch
		case CriterionWitnessUnavailable:
			reason = ReasonWitnessUnavailable
		case CriterionNotApplicable:
			if definition.NotApplicableAllowed && strings.TrimSpace(item.NotApplicableRationale) != "" {
				naCount++
				continue
			}
			reason = ReasonProofIncomplete
		}
		if !reasonSet[reason] {
			reasons = append(reasons, reason)
			reasonSet[reason] = true
		}
	}
	if len(missing) > 0 {
		sort.Strings(missing)
		if !reasonSet[ReasonProofIncomplete] {
			reasons = append(reasons, ReasonProofIncomplete)
		}
	}
	sort.Slice(reasons, func(i, j int) bool { return reasons[i] < reasons[j] })
	status := ControlsPass
	if len(reasons) > 0 {
		status = ControlsNotPass
	}
	return InventorySummary{Summary: ControlSummary{Status: status, Reasons: reasons, NotApplicableCount: naCount}, MissingMandatory: missing, InventorySHA256: actualDigest}, nil
}

// EvaluateAgainstInventory is the safe high-level entry point. The caller must
// supply the digest from harness-controlled preregistration, not agent output.
func EvaluateAgainstInventory(runID string, task TaskVerdict, inventory ControlInventory, expectedDigest string, evidence []ControlEvidence) (InventoryRunResult, error) {
	if strings.TrimSpace(runID) == "" {
		return InventoryRunResult{}, errors.New("run ID is empty")
	}
	summary, err := SummarizeAgainstInventory(inventory, expectedDigest, evidence)
	if err != nil {
		return InventoryRunResult{}, err
	}
	class, err := Classify(task, summary.Summary)
	if err != nil {
		return InventoryRunResult{}, err
	}
	return InventoryRunResult{RunID: runID, TaskVerdict: task, ControlSummary: summary, Class: class}, nil
}

func knownVerdict(verdict CriterionVerdict) bool {
	switch verdict {
	case CriterionPass, CriterionConfirmedViolation, CriterionNotEstablished, CriterionObserverFailure,
		CriterionIdentityMismatch, CriterionEffectReceiptMismatch, CriterionWitnessUnavailable, CriterionNotApplicable:
		return true
	default:
		return false
	}
}

func portableID(value string) bool {
	if len(value) == 0 || len(value) > 128 || strings.TrimSpace(value) != value {
		return false
	}
	for i := 0; i < len(value); i++ {
		c := value[i]
		if !((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || (i > 0 && (c == '-' || c == '_' || c == '.' || c == ':'))) {
			return false
		}
	}
	return true
}
