package oraclelab

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"sort"
	"strings"
)

const SchemaVersion = "mace-oraclelab/v1"

type Status string

const (
	Pass          Status = "pass"
	Fail          Status = "fail"
	Indeterminate Status = "indeterminate"
)

type Verdict string

const (
	VerdictPass          Verdict = "pass"
	VerdictFail          Verdict = "fail"
	VerdictIndeterminate Verdict = "indeterminate"
)

type Snapshot struct {
	ScopeID  string `json:"scope_id"`
	Revision string `json:"revision"`
	Digest   string `json:"digest"`
}

type Coverage struct {
	SourceID         string   `json:"source_id"`
	ScopeID          string   `json:"scope_id"`
	Complete         bool     `json:"complete"`
	Uninstrumented   []string `json:"uninstrumented_paths,omitempty"`
	KnownLimitations []string `json:"known_limitations,omitempty"`
}

type Event struct {
	Sequence     uint64 `json:"sequence"`
	InvocationID string `json:"invocation_id"`
	Operation    string `json:"operation"`
	Resource     string `json:"resource"`
	Mutation     bool   `json:"mutation"`
	BeforeDigest string `json:"before_digest,omitempty"`
	AfterDigest  string `json:"after_digest,omitempty"`
}

type StateReader interface {
	Read(key string) ([]byte, bool, error)
}

// World is the trusted harness boundary. Evaluators receive only StateReader.
// Reset must restore the exact immutable snapshot and start a fresh event window.
// EventCursor/EventsSince must describe every event in the declared scope.
type World interface {
	StateReader
	Snapshot() (Snapshot, error)
	Reset(Snapshot) error
	EventCursor() (uint64, error)
	EventsSince(uint64) ([]Event, error)
	Coverage() (Coverage, error)
	BeginInvocation(string)
	EndInvocation()
}

type Task struct {
	ID          string `json:"id"`
	InputSHA256 string `json:"input_sha256"`
}

type GradeResult struct {
	Verdict      Verdict `json:"verdict"`
	OutputSHA256 string  `json:"output_sha256"`
}

type Evaluator interface {
	Grade(context.Context, Task, StateReader) (GradeResult, error)
}

type EvaluatorFactory func() Evaluator

type InvocationReport struct {
	InvocationID     string      `json:"invocation_id"`
	TaskID           string      `json:"task_id"`
	Grade            GradeResult `json:"grade"`
	NonInterference  Status      `json:"non_interference"`
	StateChanged     bool        `json:"state_changed"`
	MutationEvents   int         `json:"mutation_events"`
	EventGap         bool        `json:"event_gap"`
	Before           Snapshot    `json:"before"`
	After            Snapshot    `json:"after"`
	EventCursorStart uint64      `json:"event_cursor_start"`
	EventCursorEnd   uint64      `json:"event_cursor_end"`
	Error            string      `json:"error,omitempty"`
	Reasons          []string    `json:"reasons,omitempty"`
}

type RepeatabilityReport struct {
	TaskID      string              `json:"task_id"`
	Repetitions int                 `json:"repetitions"`
	Stable      bool                `json:"stable"`
	Qualified   bool                `json:"qualified"`
	Reference   GradeResult         `json:"reference"`
	Invocations []InvocationReport  `json:"invocations"`
	Reasons     []string            `json:"reasons,omitempty"`
}

type OrderReport struct {
	FirstOrder   []string            `json:"first_order"`
	ReverseOrder []string            `json:"reverse_order"`
	Invariant    bool                `json:"invariant"`
	Qualified    bool                `json:"qualified"`
	FirstArm     []InvocationReport  `json:"first_arm"`
	ReverseArm   []InvocationReport  `json:"reverse_arm"`
	Reasons      []string            `json:"reasons,omitempty"`
}

type Harness struct {
	World           World
	ExpectedScopeID string
}

func Digest(data []byte) string {
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

func validDigest(value string) bool {
	if len(value) != 64 || strings.ToLower(value) != value {
		return false
	}
	_, err := hex.DecodeString(value)
	return err == nil
}

func (h Harness) Invoke(ctx context.Context, invocationID string, task Task, evaluator Evaluator) InvocationReport {
	report := InvocationReport{
		InvocationID:    invocationID,
		TaskID:          task.ID,
		NonInterference: Indeterminate,
		Grade:           GradeResult{Verdict: VerdictIndeterminate},
	}
	if h.World == nil || evaluator == nil || h.ExpectedScopeID == "" || invocationID == "" || task.ID == "" || !validDigest(task.InputSHA256) {
		report.Reasons = append(report.Reasons, "missing world/evaluator/identity or invalid task input digest")
		return report
	}

	coverageBefore, err := h.World.Coverage()
	if err != nil {
		report.Reasons = append(report.Reasons, "coverage read failed: "+err.Error())
		return report
	}
	if coverageBefore.SourceID == "" || coverageBefore.ScopeID != h.ExpectedScopeID || !coverageBefore.Complete || len(coverageBefore.Uninstrumented) != 0 {
		report.Reasons = append(report.Reasons, "observer coverage is incomplete or has undeclared paths")
		return report
	}

	before, err := h.World.Snapshot()
	if err != nil {
		report.Reasons = append(report.Reasons, "pre-grade snapshot failed: "+err.Error())
		return report
	}
	report.Before = before
	if before.ScopeID != coverageBefore.ScopeID || !validDigest(before.Digest) {
		report.Reasons = append(report.Reasons, "snapshot scope does not match observer coverage or digest is invalid")
		return report
	}
	start, err := h.World.EventCursor()
	if err != nil {
		report.Reasons = append(report.Reasons, "pre-grade event cursor failed: "+err.Error())
		return report
	}
	report.EventCursorStart = start

	h.World.BeginInvocation(invocationID)
	grade, gradeErr := callEvaluator(ctx, evaluator, task, h.World)
	h.World.EndInvocation()
	if gradeErr != nil {
		report.Error = gradeErr.Error()
		report.Reasons = append(report.Reasons, "grader returned an error; outcome is not a pass")
	}
	report.Grade = grade
	if gradeErr != nil {
		report.Grade.Verdict = VerdictIndeterminate
	} else if !validVerdict(grade.Verdict) || !validDigest(grade.OutputSHA256) {
		report.Grade.Verdict = VerdictIndeterminate
		report.Reasons = append(report.Reasons, "grader result has invalid verdict or output digest")
	}

	after, afterErr := h.World.Snapshot()
	end, cursorErr := h.World.EventCursor()
	coverageAfter, coverageErr := h.World.Coverage()
	if afterErr != nil || cursorErr != nil || coverageErr != nil {
		report.Reasons = append(report.Reasons, "post-grade snapshot/cursor/coverage unavailable")
		return report
	}
	report.After = after
	report.EventCursorEnd = end
	if after.ScopeID != before.ScopeID || coverageAfter.ScopeID != coverageBefore.ScopeID || coverageAfter.SourceID != coverageBefore.SourceID || !coverageAfter.Complete || len(coverageAfter.Uninstrumented) != 0 {
		report.Reasons = append(report.Reasons, "observer or snapshot scope changed during evaluation")
		return report
	}
	if end < start {
		report.Reasons = append(report.Reasons, "observer event cursor moved backwards")
		return report
	}
	events, eventsErr := h.World.EventsSince(start)
	if eventsErr != nil {
		report.Reasons = append(report.Reasons, "event retrieval failed: "+eventsErr.Error())
		return report
	}
	if uint64(len(events)) != end-start {
		report.EventGap = true
		report.Reasons = append(report.Reasons, "observer event stream has missing or extra records")
		return report
	}
	for i, event := range events {
		expectedSequence := start + uint64(i) + 1
		if event.Sequence != expectedSequence || event.InvocationID != invocationID {
			report.EventGap = true
			report.Reasons = append(report.Reasons, "observer sequence or invocation attribution mismatch")
			return report
		}
		if event.Mutation {
			report.MutationEvents++
		}
	}

	report.StateChanged = before.Digest != after.Digest || before.Revision != after.Revision
	if report.StateChanged || report.MutationEvents > 0 {
		report.NonInterference = Fail
		report.Reasons = append(report.Reasons, "grader changed protected state or emitted a mutation event")
		return report
	}
	if gradeErr != nil || report.Grade.Verdict == VerdictIndeterminate {
		report.NonInterference = Pass
		return report
	}
	report.NonInterference = Pass
	return report
}

func (h Harness) CheckRepeatability(ctx context.Context, base Snapshot, task Task, factory EvaluatorFactory, repetitions int) RepeatabilityReport {
	report := RepeatabilityReport{TaskID: task.ID, Repetitions: repetitions, Stable: true, Qualified: true}
	if h.World == nil || h.ExpectedScopeID == "" || factory == nil || repetitions < 2 {
		report.Qualified = false
		report.Stable = false
		report.Reasons = append(report.Reasons, "repeatability requires a world, evaluator factory and at least two repetitions")
		return report
	}
	evaluator := factory()
	if evaluator == nil {
		report.Qualified = false
		report.Stable = false
		report.Reasons = append(report.Reasons, "evaluator factory returned nil")
		return report
	}
	for i := 0; i < repetitions; i++ {
		if err := h.World.Reset(base); err != nil {
			report.Qualified = false
			report.Reasons = append(report.Reasons, "reset failed: "+err.Error())
			return report
		}
		pre, err := h.World.Snapshot()
		if err != nil || pre != base {
			report.Qualified = false
			report.Reasons = append(report.Reasons, "reset did not restore exact baseline snapshot")
			return report
		}
		inv := h.Invoke(ctx, fmt.Sprintf("repeat-%s-%d", task.ID, i+1), task, evaluator)
		report.Invocations = append(report.Invocations, inv)
		if inv.NonInterference != Pass {
			report.Qualified = false
			report.Reasons = append(report.Reasons, fmt.Sprintf("repetition %d did not establish non-interference", i+1))
		}
		if inv.Grade.Verdict == VerdictIndeterminate {
			report.Qualified = false
			report.Reasons = append(report.Reasons, fmt.Sprintf("repetition %d did not produce a resolved grade", i+1))
		}
		if i == 0 {
			report.Reference = inv.Grade
		} else if !sameGrade(report.Reference, inv.Grade) {
			report.Stable = false
			report.Qualified = false
			report.Reasons = append(report.Reasons, fmt.Sprintf("repetition %d changed the grade result", i+1))
		}
	}
	return report
}

func (h Harness) CheckOrderInvariance(ctx context.Context, base Snapshot, first, second Task, factory EvaluatorFactory) OrderReport {
	report := OrderReport{
		FirstOrder:   []string{first.ID, second.ID},
		ReverseOrder: []string{second.ID, first.ID},
		Invariant:    true,
		Qualified:    true,
	}
	if h.World == nil || h.ExpectedScopeID == "" || factory == nil || first.ID == "" || second.ID == "" || first.ID == second.ID || !validDigest(first.InputSHA256) || !validDigest(second.InputSHA256) {
		report.Invariant = false
		report.Qualified = false
		report.Reasons = append(report.Reasons, "order test requires distinct valid tasks, world and evaluator factory")
		return report
	}

	runArm := func(order []Task, arm string) ([]InvocationReport, error) {
		if err := h.World.Reset(base); err != nil {
			return nil, err
		}
		pre, err := h.World.Snapshot()
		if err != nil || pre != base {
			return nil, fmt.Errorf("%s arm did not restore exact baseline", arm)
		}
		evaluator := factory()
		if evaluator == nil {
			return nil, fmt.Errorf("%s arm evaluator factory returned nil", arm)
		}
		out := make([]InvocationReport, 0, len(order))
		for index, task := range order {
			out = append(out, h.Invoke(ctx, fmt.Sprintf("order-%s-%d-%s", arm, index+1, task.ID), task, evaluator))
		}
		return out, nil
	}

	var err error
	report.FirstArm, err = runArm([]Task{first, second}, "forward")
	if err != nil {
		report.Invariant = false
		report.Qualified = false
		report.Reasons = append(report.Reasons, err.Error())
		return report
	}
	report.ReverseArm, err = runArm([]Task{second, first}, "reverse")
	if err != nil {
		report.Invariant = false
		report.Qualified = false
		report.Reasons = append(report.Reasons, err.Error())
		return report
	}
	for _, inv := range append(append([]InvocationReport{}, report.FirstArm...), report.ReverseArm...) {
		if inv.NonInterference != Pass {
			report.Qualified = false
			report.Reasons = append(report.Reasons, "an invocation failed the non-interference gate")
		}
		if inv.Grade.Verdict == VerdictIndeterminate {
			report.Qualified = false
			report.Reasons = append(report.Reasons, "an invocation did not produce a resolved grade")
		}
	}
	forward := map[string]GradeResult{report.FirstArm[0].TaskID: report.FirstArm[0].Grade, report.FirstArm[1].TaskID: report.FirstArm[1].Grade}
	reverse := map[string]GradeResult{report.ReverseArm[0].TaskID: report.ReverseArm[0].Grade, report.ReverseArm[1].TaskID: report.ReverseArm[1].Grade}
	for taskID, result := range forward {
		if !sameGrade(result, reverse[taskID]) {
			report.Invariant = false
			report.Qualified = false
			report.Reasons = append(report.Reasons, "task "+taskID+" changed result under evaluation-order permutation")
		}
	}
	return report
}

func callEvaluator(ctx context.Context, evaluator Evaluator, task Task, reader StateReader) (result GradeResult, err error) {
	defer func() {
		if recovered := recover(); recovered != nil {
			err = fmt.Errorf("grader panic: %v", recovered)
			result = GradeResult{Verdict: VerdictIndeterminate}
		}
	}()
	return evaluator.Grade(ctx, task, reader)
}

func validVerdict(v Verdict) bool {
	return v == VerdictPass || v == VerdictFail || v == VerdictIndeterminate
}

func sameGrade(a, b GradeResult) bool {
	return a.Verdict == b.Verdict && a.OutputSHA256 == b.OutputSHA256
}

// CanonicalStateDigest is provided for small reference adapters. Production
// adapters should use a domain-native immutable snapshot or transaction-log digest.
func CanonicalStateDigest(values map[string][]byte) string {
	keys := make([]string, 0, len(values))
	for key := range values {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	var b strings.Builder
	for _, key := range keys {
		fmt.Fprintf(&b, "%d:%s:%d:", len(key), key, len(values[key]))
		b.Write(values[key])
	}
	return Digest([]byte(b.String()))
}
