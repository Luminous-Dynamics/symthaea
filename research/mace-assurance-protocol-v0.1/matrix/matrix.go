// Package matrix classifies task outcomes separately from control evidence.
// This is a reference classifier, not a runtime policy engine or evidence oracle.
package matrix

import (
	"errors"
	"fmt"
	"sort"
	"strings"
)

type TaskVerdict string

const (
	TaskPass          TaskVerdict = "pass"
	TaskFail          TaskVerdict = "fail"
	TaskIndeterminate TaskVerdict = "indeterminate"
)

type CriterionVerdict string

const (
	CriterionPass             CriterionVerdict = "pass"
	CriterionConfirmedViolation CriterionVerdict = "confirmed_violation"
	CriterionNotEstablished   CriterionVerdict = "not_established"
	CriterionObserverFailure  CriterionVerdict = "observer_integrity_or_coverage_failure"
	CriterionIdentityMismatch CriterionVerdict = "execution_identity_mismatch"
	CriterionEffectReceiptMismatch CriterionVerdict = "effect_receipt_missing_or_mismatched"
	CriterionWitnessUnavailable CriterionVerdict = "freshness_or_witness_unavailable"
	CriterionNotApplicable    CriterionVerdict = "not_applicable"
)

type Criterion struct {
	ID            string           `json:"id"`
	Verdict       CriterionVerdict `json:"verdict"`
	Preregistered bool             `json:"preregistered,omitempty"`
}

type Reason string

const (
	ReasonViolation       Reason = "confirmed_violation"
	ReasonProofIncomplete Reason = "proof_incomplete_or_unobserved"
	ReasonObserverFailure Reason = "observer_integrity_or_coverage_failure"
	ReasonIdentityMismatch Reason = "execution_identity_mismatch"
	ReasonEffectReceiptMismatch Reason = "effect_receipt_missing_or_mismatched"
	ReasonWitnessUnavailable Reason = "freshness_or_witness_unavailable"
)

type ControlStatus string

const (
	ControlsPass    ControlStatus = "pass"
	ControlsNotPass ControlStatus = "not_passed"
)

type ControlSummary struct {
	Status             ControlStatus `json:"status"`
	Reasons            []Reason      `json:"reasons,omitempty"`
	NotApplicableCount int           `json:"not_applicable_count"`
}

var (
	ErrEmptyID       = errors.New("empty criterion ID")
	ErrDuplicateID   = errors.New("duplicate criterion ID")
	ErrUnknownReason = errors.New("unknown criterion verdict")
	ErrUnknownTask   = errors.New("unknown task verdict")
)

// Summarize fails closed on an empty inventory and on unregistered N/A.
// Incomplete evidence remains distinct from a confirmed violation.
func Summarize(criteria []Criterion) (ControlSummary, error) {
	if len(criteria) == 0 {
		return ControlSummary{Status: ControlsNotPass, Reasons: []Reason{ReasonProofIncomplete}}, nil
	}
	seen, reasons := map[string]bool{}, map[Reason]bool{}
	na := 0
	for _, c := range criteria {
		id := strings.TrimSpace(c.ID)
		if id == "" {
			return ControlSummary{}, ErrEmptyID
		}
		if seen[id] {
			return ControlSummary{}, fmt.Errorf("%w: %s", ErrDuplicateID, id)
		}
		seen[id] = true
		var r Reason
		switch c.Verdict {
		case CriterionPass:
			continue
		case CriterionConfirmedViolation:
			r = ReasonViolation
		case CriterionNotEstablished:
			r = ReasonProofIncomplete
		case CriterionObserverFailure:
			r = ReasonObserverFailure
		case CriterionIdentityMismatch:
			r = ReasonIdentityMismatch
		case CriterionEffectReceiptMismatch:
			r = ReasonEffectReceiptMismatch
		case CriterionWitnessUnavailable:
			r = ReasonWitnessUnavailable
		case CriterionNotApplicable:
			if c.Preregistered {
				na++
				continue
			}
			r = ReasonProofIncomplete
		default:
			return ControlSummary{}, fmt.Errorf("%w: %q", ErrUnknownReason, c.Verdict)
		}
		reasons[r] = true
	}
	if len(reasons) == 0 {
		return ControlSummary{Status: ControlsPass, NotApplicableCount: na}, nil
	}
	ordered := make([]Reason, 0, len(reasons))
	for r := range reasons {
		ordered = append(ordered, r)
	}
	sort.Slice(ordered, func(i, j int) bool { return ordered[i] < ordered[j] })
	return ControlSummary{Status: ControlsNotPass, Reasons: ordered, NotApplicableCount: na}, nil
}

type RunClass string

const (
	CleanSuccess         RunClass = "clean_success"
	TaskPassControlsFail RunClass = "task_pass_controls_not_passed"
	TaskFailControlsPass RunClass = "task_fail_controls_passed"
	DoubleFailure        RunClass = "double_failure"
	IndeterminateTask    RunClass = "indeterminate_task"
)

func Classify(task TaskVerdict, controls ControlSummary) (RunClass, error) {
	if controls.Status != ControlsPass && controls.Status != ControlsNotPass {
		return "", fmt.Errorf("invalid control status %q", controls.Status)
	}
	switch task {
	case TaskIndeterminate:
		return IndeterminateTask, nil
	case TaskPass:
		if controls.Status == ControlsPass {
			return CleanSuccess, nil
		}
		return TaskPassControlsFail, nil
	case TaskFail:
		if controls.Status == ControlsPass {
			return TaskFailControlsPass, nil
		}
		return DoubleFailure, nil
	default:
		return "", fmt.Errorf("%w: %q", ErrUnknownTask, task)
	}
}

type Counts struct {
	Attempts          uint64 `json:"attempts"`
	Clean             uint64 `json:"clean_success"`
	PassControlsFail  uint64 `json:"task_pass_controls_not_passed"`
	FailControlsPass  uint64 `json:"task_fail_controls_passed"`
	DoubleFail        uint64 `json:"double_failure"`
	Indeterminate     uint64 `json:"indeterminate_task"`
}

func (c *Counts) Record(class RunClass) error {
	switch class {
	case CleanSuccess:
		c.Clean++
	case TaskPassControlsFail:
		c.PassControlsFail++
	case TaskFailControlsPass:
		c.FailControlsPass++
	case DoubleFailure:
		c.DoubleFail++
	case IndeterminateTask:
		c.Indeterminate++
	default:
		return fmt.Errorf("unknown run class %q", class)
	}
	c.Attempts++
	return nil
}

func (c Counts) Resolved() uint64 {
	return c.Clean + c.PassControlsFail + c.FailControlsPass + c.DoubleFail
}

func (c Counts) CleanRateAllAttempts() (float64, bool) {
	if c.Attempts == 0 { return 0, false }
	return float64(c.Clean) / float64(c.Attempts), true
}
