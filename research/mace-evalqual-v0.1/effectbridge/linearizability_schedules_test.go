package effectbridge

import (
	"math/rand"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
)

func TestGeneratedMultiKeyEffectFenceHistoriesAreLinearizable(t *testing.T) {
	const (
		historyCount        = 256
		operationsPerHistory = 6
	)
	rng := rand.New(rand.NewSource(0x4d414345))
	keys := []string{"generated-key-a", "generated-key-b"}
	intents := []string{"intent-a", "intent-b"}

	for iteration := 0; iteration < historyCount; iteration++ {
		journalPath := filepath.Join(t.TempDir(), "effects.jsonl")
		journal, err := OpenJournal(journalPath)
		if err != nil {
			t.Fatalf("history %d: open journal: %v", iteration, err)
		}

		history := make([]efxHistoryOperation, operationsPerHistory)
		for index := range history {
			key := keys[rng.Intn(len(keys))]
			intent := intents[rng.Intn(len(intents))]
			history[index] = efxHistoryOperation{
				kind: efxHistoryKind(1 + rng.Intn(3)),
				req:  efxHistoryRequest(key, intent),
			}
		}

		start := make(chan struct{})
		var ready sync.WaitGroup
		var workers sync.WaitGroup
		var clock atomic.Int64
		ready.Add(len(history))
		workers.Add(len(history))
		for index := range history {
			op := &history[index]
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
			t.Fatalf("history %d: no legal real-time-respecting sequential history: %+v", iteration, history)
		}

		beforeRestart := make(map[string]string, len(keys))
		for _, key := range keys {
			beforeRestart[key] = journal.Lookup(key).Status
		}
		if err := journal.Close(); err != nil {
			t.Fatalf("history %d: close journal: %v", iteration, err)
		}
		// Reopen the exact journal path: terminal decisions must survive
		// recovery, not just the in-memory serialization schedule.
		reopened, err := OpenJournal(journalPath)
		if err != nil {
			t.Fatalf("history %d: reopen journal: %v", iteration, err)
		}
		for key, expected := range beforeRestart {
			actual := reopened.Lookup(key).Status
			if actual != expected {
				_ = reopened.Close()
				t.Fatalf("history %d: key %s changed after restart: before=%s after=%s", iteration, key, expected, actual)
			}
		}
		if err := reopened.Close(); err != nil {
			t.Fatalf("history %d: close reopened journal: %v", iteration, err)
		}
	}
}
