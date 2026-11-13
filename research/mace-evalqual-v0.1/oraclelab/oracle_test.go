package oraclelab

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"sync"
	"testing"
)

type memoryWorld struct {
	mu         sync.Mutex
	data       map[string][]byte
	snapshots  map[string]map[string][]byte
	revision   uint64
	events     []Event
	cursor     uint64
	invocation string
	coverage   Coverage
	dropNext   bool
}

func newMemoryWorld() *memoryWorld {
	w := &memoryWorld{
		data:      map[string][]byte{"account/42": []byte("approved")},
		snapshots: map[string]map[string][]byte{},
		coverage:  Coverage{SourceID: "test-kernel-audit", ScopeID: "benchmark-state", Complete: true},
	}
	_, _ = w.Snapshot()
	return w
}

func cloneMap(in map[string][]byte) map[string][]byte {
	out := make(map[string][]byte, len(in))
	for k, v := range in {
		out[k] = append([]byte(nil), v...)
	}
	return out
}

func (w *memoryWorld) Read(key string) ([]byte, bool, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	v, ok := w.data[key]
	return append([]byte(nil), v...), ok, nil
}

func (w *memoryWorld) Snapshot() (Snapshot, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	digest := CanonicalStateDigest(w.data)
	w.snapshots[digest] = cloneMap(w.data)
	return Snapshot{ScopeID: w.coverage.ScopeID, Revision: fmt.Sprint(w.revision), Digest: digest}, nil
}

func (w *memoryWorld) Reset(snapshot Snapshot) error {
	w.mu.Lock()
	defer w.mu.Unlock()
	stored, ok := w.snapshots[snapshot.Digest]
	if !ok {
		return errors.New("unknown snapshot")
	}
	w.data = cloneMap(stored)
	w.revision, _ = parseUint(snapshot.Revision)
	w.events = nil
	w.cursor = 0
	w.invocation = ""
	return nil
}

func parseUint(s string) (uint64, error) {
	var n uint64
	_, err := fmt.Sscan(s, &n)
	return n, err
}

func (w *memoryWorld) EventCursor() (uint64, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.cursor, nil
}

func (w *memoryWorld) EventsSince(cursor uint64) ([]Event, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	out := []Event{}
	for _, event := range w.events {
		if event.Sequence > cursor {
			out = append(out, event)
		}
	}
	return out, nil
}

func (w *memoryWorld) Coverage() (Coverage, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.coverage, nil
}

func (w *memoryWorld) BeginInvocation(id string) { w.mu.Lock(); w.invocation = id; w.mu.Unlock() }
func (w *memoryWorld) EndInvocation()            { w.mu.Lock(); w.invocation = ""; w.mu.Unlock() }

func (w *memoryWorld) Write(key string, value []byte) {
	w.mu.Lock()
	defer w.mu.Unlock()
	before := append([]byte(nil), w.data[key]...)
	w.data[key] = append([]byte(nil), value...)
	w.revision++
	w.cursor++
	beforeDigest := Digest(before)
	afterDigest := Digest(value)
	if w.dropNext {
		w.dropNext = false
		return
	}
	w.events = append(w.events, Event{Sequence: w.cursor, InvocationID: w.invocation, Operation: "write", Resource: key, Mutation: true, BeforeDigest: beforeDigest, AfterDigest: afterDigest})
}

func task(id, input string) Task { return Task{ID: id, InputSHA256: Digest([]byte(input))} }
func result(verdict Verdict, value string) GradeResult {
	return GradeResult{Verdict: verdict, OutputSHA256: Digest([]byte(value))}
}

type gradeFunc func(context.Context, Task, StateReader) (GradeResult, error)

func (f gradeFunc) Grade(ctx context.Context, task Task, reader StateReader) (GradeResult, error) {
	return f(ctx, task, reader)
}

func TestReadOnlyOraclePasses(t *testing.T) {
	w := newMemoryWorld()
	base, _ := w.Snapshot()
	h := Harness{World: w, ExpectedScopeID: "benchmark-state"}
	report := h.Invoke(context.Background(), "read-only-1", task("a", "input-a"), gradeFunc(func(_ context.Context, _ Task, reader StateReader) (GradeResult, error) {
		_, ok, err := reader.Read("account/42")
		if err != nil || !ok {
			return GradeResult{}, errors.New("missing state")
		}
		return result(VerdictPass, "ok"), nil
	}))
	if report.NonInterference != Pass || report.StateChanged || report.MutationEvents != 0 {
		t.Fatalf("pure grader rejected: %+v", report)
	}
	after, _ := w.Snapshot()
	if base.Digest != after.Digest {
		t.Fatal("world changed")
	}
}

func TestPersistentMutationFails(t *testing.T) {
	w := newMemoryWorld()
	h := Harness{World: w, ExpectedScopeID: "benchmark-state"}
	report := h.Invoke(context.Background(), "mutating-1", task("a", "input-a"), gradeFunc(func(_ context.Context, _ Task, _ StateReader) (GradeResult, error) {
		w.Write("account/42", []byte("corrupted"))
		return result(VerdictPass, "apparently-ok"), nil
	}))
	if report.NonInterference != Fail || !report.StateChanged || report.MutationEvents != 1 {
		t.Fatalf("persistent mutation missed: %+v", report)
	}
}

func TestTransientMutationFailsEvenWhenFinalSnapshotMatches(t *testing.T) {
	w := newMemoryWorld()
	base, _ := w.Snapshot()
	h := Harness{World: w, ExpectedScopeID: "benchmark-state"}
	report := h.Invoke(context.Background(), "transient-1", task("a", "input-a"), gradeFunc(func(_ context.Context, _ Task, _ StateReader) (GradeResult, error) {
		w.Write("account/42", []byte("temporary-corruption"))
		w.Write("account/42", []byte("approved"))
		return result(VerdictPass, "ok"), nil
	}))
	if report.NonInterference != Fail || !report.StateChanged || report.MutationEvents != 2 {
		t.Fatalf("transient write not detected: %+v", report)
	}
	after, _ := w.Snapshot()
	if after.Digest != base.Digest {
		t.Fatal("test fixture should restore final state")
	}
}

func TestIncompleteObserverCoverageIsIndeterminateAndDoesNotInvokeGrader(t *testing.T) {
	w := newMemoryWorld()
	w.coverage.Complete = false
	h := Harness{World: w, ExpectedScopeID: "benchmark-state"}
	called := false
	report := h.Invoke(context.Background(), "uncovered-1", task("a", "input-a"), gradeFunc(func(context.Context, Task, StateReader) (GradeResult, error) {
		called = true
		return result(VerdictPass, "ok"), nil
	}))
	if report.NonInterference != Indeterminate || called {
		t.Fatalf("incomplete coverage should block evaluation: %+v called=%v", report, called)
	}
}

func TestObserverEventGapIsIndeterminate(t *testing.T) {
	w := newMemoryWorld()
	w.dropNext = true
	h := Harness{World: w, ExpectedScopeID: "benchmark-state"}
	report := h.Invoke(context.Background(), "gap-1", task("a", "input-a"), gradeFunc(func(_ context.Context, _ Task, _ StateReader) (GradeResult, error) {
		w.Write("account/42", []byte("changed"))
		return result(VerdictPass, "ok"), nil
	}))
	if report.NonInterference != Indeterminate || !report.EventGap {
		t.Fatalf("event gap not fail-closed: %+v", report)
	}
}

func TestGraderErrorCannotBecomePass(t *testing.T) {
	w := newMemoryWorld()
	h := Harness{World: w, ExpectedScopeID: "benchmark-state"}
	report := h.Invoke(context.Background(), "error-1", task("a", "input-a"), gradeFunc(func(context.Context, Task, StateReader) (GradeResult, error) {
		return result(VerdictPass, "ok"), errors.New("grader crashed")
	}))
	if report.Grade.Verdict != VerdictIndeterminate || report.NonInterference != Pass {
		t.Fatalf("grader error incorrectly promoted or non-interference lost: %+v", report)
	}
}

func TestGraderErrorAfterMutationStillFailsNonInterference(t *testing.T) {
	w := newMemoryWorld()
	h := Harness{World: w, ExpectedScopeID: "benchmark-state"}
	report := h.Invoke(context.Background(), "error-mutate-1", task("a", "input-a"), gradeFunc(func(context.Context, Task, StateReader) (GradeResult, error) {
		w.Write("account/42", []byte("changed"))
		return result(VerdictPass, "ok"), errors.New("crashed after write")
	}))
	if report.Grade.Verdict != VerdictIndeterminate || report.NonInterference != Fail {
		t.Fatalf("mutating exception not detected: %+v", report)
	}
}

func TestRepeatabilityPassesForDeterministicOracle(t *testing.T) {
	w := newMemoryWorld()
	base, _ := w.Snapshot()
	h := Harness{World: w, ExpectedScopeID: "benchmark-state"}
	report := h.CheckRepeatability(context.Background(), base, task("a", "input-a"), func() Evaluator {
		return gradeFunc(func(context.Context, Task, StateReader) (GradeResult, error) {
			return result(VerdictPass, "stable-answer"), nil
		})
	}, 3)
	if !report.Qualified || !report.Stable || len(report.Invocations) != 3 {
		t.Fatalf("repeatability should pass: %+v", report)
	}
}

func TestRepeatabilityDetectsStatefulOracleOutput(t *testing.T) {
	w := newMemoryWorld()
	base, _ := w.Snapshot()
	h := Harness{World: w, ExpectedScopeID: "benchmark-state"}
	count := 0
	report := h.CheckRepeatability(context.Background(), base, task("a", "input-a"), func() Evaluator {
		return gradeFunc(func(context.Context, Task, StateReader) (GradeResult, error) {
			count++
			return result(VerdictPass, fmt.Sprintf("answer-%d", count)), nil
		})
	}, 3)
	if report.Qualified || report.Stable {
		t.Fatalf("stateful result variation was not detected: %+v", report)
	}
}

type orderSensitiveGrader struct{ count int }

func (g *orderSensitiveGrader) Grade(_ context.Context, task Task, _ StateReader) (GradeResult, error) {
	g.count++
	return result(VerdictPass, fmt.Sprintf("%s-after-%d", task.ID, g.count)), nil
}

func TestOrderInvariancePassesForPureOracle(t *testing.T) {
	w := newMemoryWorld()
	base, _ := w.Snapshot()
	h := Harness{World: w, ExpectedScopeID: "benchmark-state"}
	factory := func() Evaluator {
		return gradeFunc(func(_ context.Context, task Task, _ StateReader) (GradeResult, error) {
			return result(VerdictPass, "result-for-"+task.ID), nil
		})
	}
	report := h.CheckOrderInvariance(context.Background(), base, task("a", "input-a"), task("b", "input-b"), factory)
	if !report.Qualified || !report.Invariant {
		t.Fatalf("pure oracle should be order invariant: %+v", report)
	}
}

func TestOrderInvarianceDetectsSharedGraderState(t *testing.T) {
	w := newMemoryWorld()
	base, _ := w.Snapshot()
	h := Harness{World: w, ExpectedScopeID: "benchmark-state"}
	report := h.CheckOrderInvariance(context.Background(), base, task("a", "input-a"), task("b", "input-b"), func() Evaluator { return &orderSensitiveGrader{} })
	if report.Qualified || report.Invariant {
		t.Fatalf("evaluation order dependence not detected: %+v", report)
	}
}

func TestSnapshotScopeMismatchBlocksOracle(t *testing.T) {
	w := newMemoryWorld()
	w.coverage.ScopeID = "wrong-scope"
	h := Harness{World: w, ExpectedScopeID: "benchmark-state"}
	called := false
	report := h.Invoke(context.Background(), "scope-1", task("a", "input-a"), gradeFunc(func(context.Context, Task, StateReader) (GradeResult, error) {
		called = true
		return result(VerdictPass, "ok"), nil
	}))
	if report.NonInterference != Indeterminate || called {
		t.Fatalf("scope mismatch failed open: %+v", report)
	}
}

func TestMutationEventAttributionMustMatchInvocation(t *testing.T) {
	w := newMemoryWorld()
	h := Harness{World: w, ExpectedScopeID: "benchmark-state"}
	report := h.Invoke(context.Background(), "attribution-1", task("a", "input-a"), gradeFunc(func(context.Context, Task, StateReader) (GradeResult, error) {
		w.mu.Lock()
		w.invocation = "spoofed"
		w.mu.Unlock()
		w.Write("account/42", []byte("bad"))
		return result(VerdictPass, "ok"), nil
	}))
	if report.NonInterference != Indeterminate || !report.EventGap {
		t.Fatalf("incorrect event attribution did not fail closed: %+v", report)
	}
}

func TestCanonicalDigestIgnoresMapIterationOrder(t *testing.T) {
	a := map[string][]byte{"b": []byte("2"), "a": []byte("1")}
	b := map[string][]byte{"a": []byte("1"), "b": []byte("2")}
	if CanonicalStateDigest(a) != CanonicalStateDigest(b) {
		t.Fatal("digest changed with map insertion order")
	}
}

func TestMultipleReadOperationsRemainNonMutating(t *testing.T) {
	w := newMemoryWorld()
	h := Harness{World: w, ExpectedScopeID: "benchmark-state"}
	report := h.Invoke(context.Background(), "reads-1", task("a", "input-a"), gradeFunc(func(_ context.Context, _ Task, reader StateReader) (GradeResult, error) {
		keys := []string{"account/42", "missing", "account/42"}
		sort.Strings(keys)
		for _, key := range keys {
			_, _, err := reader.Read(key)
			if err != nil {
				return GradeResult{}, err
			}
		}
		return result(VerdictPass, "read-only"), nil
	}))
	if report.NonInterference != Pass {
		t.Fatalf("read-only operations caused non-interference failure: %+v", report)
	}
}

func TestGraderPanicIsCapturedAsIndeterminate(t *testing.T) {
	w := newMemoryWorld()
	h := Harness{World: w, ExpectedScopeID: "benchmark-state"}
	report := h.Invoke(context.Background(), "panic-1", task("a", "input-a"), gradeFunc(func(context.Context, Task, StateReader) (GradeResult, error) {
		panic("injected grader panic")
	}))
	if report.Grade.Verdict != VerdictIndeterminate || report.NonInterference != Pass || report.Error == "" {
		t.Fatalf("grader panic was not preserved as indeterminate: %+v", report)
	}
}

func TestUninstrumentedPathBlocksOracle(t *testing.T) {
	w := newMemoryWorld()
	w.coverage.Uninstrumented = []string{"database-trigger"}
	h := Harness{World: w, ExpectedScopeID: "benchmark-state"}
	called := false
	report := h.Invoke(context.Background(), "uninstrumented-1", task("a", "input-a"), gradeFunc(func(context.Context, Task, StateReader) (GradeResult, error) {
		called = true
		return result(VerdictPass, "ok"), nil
	}))
	if report.NonInterference != Indeterminate || called {
		t.Fatalf("uninstrumented path failed open: %+v called=%v", report, called)
	}
}

func TestIndeterminateGradeCannotQualifyRepeatability(t *testing.T) {
	w := newMemoryWorld()
	base, _ := w.Snapshot()
	h := Harness{World: w, ExpectedScopeID: "benchmark-state"}
	report := h.CheckRepeatability(context.Background(), base, task("a", "input-a"), func() Evaluator {
		return gradeFunc(func(context.Context, Task, StateReader) (GradeResult, error) {
			return GradeResult{Verdict: VerdictIndeterminate}, nil
		})
	}, 2)
	if report.Qualified || !report.Stable {
		t.Fatalf("stable indeterminate results must not qualify: %+v", report)
	}
}

func TestStateRevisionChangeDetectedEvenWhenContentDigestSame(t *testing.T) {
	w := newMemoryWorld()
	h := Harness{World: w, ExpectedScopeID: "benchmark-state"}
	report := h.Invoke(context.Background(), "revision-1", task("a", "input-a"), gradeFunc(func(context.Context, Task, StateReader) (GradeResult, error) {
		w.mu.Lock()
		w.revision++
		w.mu.Unlock()
		return result(VerdictPass, "ok"), nil
	}))
	if report.NonInterference != Fail || !report.StateChanged || report.Before.Digest != report.After.Digest {
		t.Fatalf("metadata/revision mutation was not detected: %+v", report)
	}
}
