package effectbridge

import (
	"errors"
	"fmt"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
)

type efxHistoryKind uint8

const (
	efxHistoryAppend efxHistoryKind = iota + 1
	efxHistoryFence
	efxHistoryLookup
)

type efxHistoryResult struct {
	recordKind string
	duplicate  bool
	errKind    string
	status     string
}

type efxHistoryOperation struct {
	kind       efxHistoryKind
	req        Request
	invokedAt  int64
	returnedAt int64
	result     efxHistoryResult
}

type efxModelDecision struct {
	kind   string
	intent string
}

func efxHistoryRequest(key, payload string) Request {
	return Request{
		IdempotencyKey: key,
		Principal:      "agent:linearizability-test",
		Target:         "test-destination:object/1",
		PayloadSHA256:  Digest([]byte(payload)),
	}
}

func efxHistoryErrorKind(err error) string {
	switch {
	case err == nil:
		return ""
	case errors.Is(err, ErrIntentConflict):
		return "intent_conflict"
	case errors.Is(err, ErrFenced):
		return "fenced"
	case errors.Is(err, ErrJournalPoisoned):
		return "journal_poisoned"
	default:
		return fmt.Sprintf("unexpected:%T:%v", err, err)
	}
}

func efxInvokeHistoryOperation(j *Journal, op *efxHistoryOperation, clock *atomic.Int64) {
	op.invokedAt = clock.Add(1)
	switch op.kind {
	case efxHistoryAppend:
		receipt, duplicate, err := j.Append(op.req)
		op.result = efxHistoryResult{recordKind: receipt.RecordKind, duplicate: duplicate, errKind: efxHistoryErrorKind(err)}
	case efxHistoryFence:
		receipt, duplicate, err := j.Fence(op.req)
		op.result = efxHistoryResult{recordKind: receipt.RecordKind, duplicate: duplicate, errKind: efxHistoryErrorKind(err)}
	case efxHistoryLookup:
		op.result = efxHistoryResult{status: j.Lookup(op.req.IdempotencyKey).Status}
	default:
		op.result = efxHistoryResult{errKind: "unexpected:unknown_operation"}
	}
	op.returnedAt = clock.Add(1)
}

// efxModelStep applies one operation to the sequential specification. The
// model intentionally excludes storage mechanics: it specifies only the
// externally visible key/intent terminal-state contract.
func efxModelStep(state map[string]efxModelDecision, op efxHistoryOperation) (map[string]efxModelDecision, efxHistoryResult) {
	next := make(map[string]efxModelDecision, len(state)+1)
	for key, value := range state {
		next[key] = value
	}

	decision, exists := next[op.req.IdempotencyKey]
	if op.kind == efxHistoryLookup {
		if !exists {
			return next, efxHistoryResult{status: "not_found_unfenced"}
		}
		if decision.kind == "effect_committed" {
			return next, efxHistoryResult{status: "found"}
		}
		return next, efxHistoryResult{status: "not_executed_fenced"}
	}

	intent, err := IntentDigest(op.req)
	if err != nil {
		return next, efxHistoryResult{errKind: "invalid_model_input"}
	}
	if exists && decision.intent != intent {
		return next, efxHistoryResult{errKind: "intent_conflict"}
	}

	if !exists {
		if op.kind == efxHistoryAppend {
			next[op.req.IdempotencyKey] = efxModelDecision{kind: "effect_committed", intent: intent}
			return next, efxHistoryResult{recordKind: "effect_committed"}
		}
		next[op.req.IdempotencyKey] = efxModelDecision{kind: "fence_committed", intent: intent}
		return next, efxHistoryResult{recordKind: "fence_committed"}
	}

	if op.kind == efxHistoryAppend && decision.kind == "fence_committed" {
		return next, efxHistoryResult{errKind: "fenced"}
	}
	return next, efxHistoryResult{recordKind: decision.kind, duplicate: true}
}

// efxHistoryLinearizable searches all sequential orders consistent with the
// real-time constraints in the recorded concurrent history. A completed
// operation must precede every operation invoked after it returned.
func efxHistoryLinearizable(history []efxHistoryOperation) bool {
	remaining := make([]int, len(history))
	for i := range history {
		remaining[i] = i
	}
	return efxHistorySearch(history, remaining, map[string]efxModelDecision{})
}

func efxHistorySearch(history []efxHistoryOperation, remaining []int, state map[string]efxModelDecision) bool {
	if len(remaining) == 0 {
		return true
	}

	for position, candidate := range remaining {
		canLinearizeNext := true
		for _, other := range remaining {
			if other == candidate {
				continue
			}
			if history[other].returnedAt < history[candidate].invokedAt {
				canLinearizeNext = false
				break
			}
		}
		if !canLinearizeNext {
			continue
		}

		nextState, expected := efxModelStep(state, history[candidate])
		if expected != history[candidate].result {
			continue
		}
		nextRemaining := make([]int, 0, len(remaining)-1)
		nextRemaining = append(nextRemaining, remaining[:position]...)
		nextRemaining = append(nextRemaining, remaining[position+1:]...)
		if efxHistorySearch(history, nextRemaining, nextState) {
			return true
		}
	}
	return false
}

func TestConcurrentEffectFenceLookupHistoriesAreLinearizable(t *testing.T) {
	const repetitions = 128
	for iteration := 0; iteration < repetitions; iteration++ {
		path := filepath.Join(t.TempDir(), "effects.jsonl")
		journal, err := OpenJournal(path)
		if err != nil {
			t.Fatalf("iteration %d: open journal: %v", iteration, err)
		}

		requestA := efxHistoryRequest("linear-key", "intent-a")
		requestB := efxHistoryRequest("linear-key", "intent-b")
		history := []efxHistoryOperation{
			{kind: efxHistoryAppend, req: requestA},
			{kind: efxHistoryFence, req: requestA},
			{kind: efxHistoryLookup, req: requestA},
			{kind: efxHistoryAppend, req: requestB},
		}

		start := make(chan struct{})
		var ready sync.WaitGroup
		var workers sync.WaitGroup
		var clock atomic.Int64
		ready.Add(len(history))
		workers.Add(len(history))
		for i := range history {
			op := &history[i]
			go func() {
				defer workers.Done()
				ready.Done()
				<-start
				efxInvokeHistoryOperation(journal, op, &clock)
			}()
		}
		ready.Wait()
		close(start)
		workers.Wait()

		if !efxHistoryLinearizable(history) {
			t.Fatalf("iteration %d: concurrent history has no legal linearization: %+v", iteration, history)
		}
		beforeRestart := journal.Lookup("linear-key")
		if beforeRestart.Journal.Sequence != 1 || beforeRestart.Status == "not_found_unfenced" {
			t.Fatalf("iteration %d: expected one durable terminal decision, got %+v", iteration, beforeRestart)
		}
		if err := journal.Close(); err != nil {
			t.Fatalf("iteration %d: close journal: %v", iteration, err)
		}

		reopened, err := OpenJournal(path)
		if err != nil {
			t.Fatalf("iteration %d: reopen journal: %v", iteration, err)
		}
		afterRestart := reopened.Lookup("linear-key")
		if afterRestart.Status != beforeRestart.Status || afterRestart.Journal.Sequence != 1 {
			t.Fatalf("iteration %d: terminal decision changed after restart: before=%+v after=%+v", iteration, beforeRestart, afterRestart)
		}
		if err := reopened.Close(); err != nil {
			t.Fatalf("iteration %d: close reopened journal: %v", iteration, err)
		}
	}
}
