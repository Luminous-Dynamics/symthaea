package witness

import (
	"bytes"
	"crypto/ed25519"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/Luminous-Dynamics/symthaea/research/mace-evalqual-v0.1/checkpoints"
)

func newTestWitness(t *testing.T, path string, clock Clock) (*Witness, ed25519.PublicKey, ed25519.PrivateKey) {
	t.Helper()
	pub, priv, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	w, err := Open(Options{Path: path, KeyID: "witness-test-1", DatabaseID: "db-test-1", PrivateKey: priv, Clock: clock})
	if err != nil {
		t.Fatal(err)
	}
	return w, pub, priv
}

func observed(sequence uint64, label string) checkpoints.ObservedHead {
	digest := sha256.Sum256([]byte(label))
	return checkpoints.ObservedHead{DatabaseID: "db-test-1", JournalSequence: sequence, JournalHeadSHA256: hex.EncodeToString(digest[:])}
}

func TestWitnessIssuesSignedGenesisAndMonotonicSuccessor(t *testing.T) {
	path := filepath.Join(t.TempDir(), "witness.jsonl")
	now := time.Unix(1000, 0)
	w, pub, _ := newTestWitness(t, path, func() time.Time { return now })
	defer w.Close()
	first, duplicate, err := w.Issue(observed(1, "head-1"))
	if err != nil || duplicate {
		t.Fatalf("genesis issue: duplicate=%v err=%v", duplicate, err)
	}
	if err := first.Verify(pub); err != nil {
		t.Fatal(err)
	}
	if first.Checkpoint.CheckpointSequence != 1 || first.Checkpoint.JournalSequence != 1 {
		t.Fatalf("unexpected genesis: %+v", first.Checkpoint)
	}
	now = time.Unix(1001, 0)
	second, duplicate, err := w.Issue(observed(2, "head-2"))
	if err != nil || duplicate {
		t.Fatalf("successor issue: duplicate=%v err=%v", duplicate, err)
	}
	if err := second.Verify(pub); err != nil {
		t.Fatal(err)
	}
	if second.Checkpoint.CheckpointSequence != 2 || second.Checkpoint.PreviousCheckpointSHA256 == "" {
		t.Fatalf("unexpected successor: %+v", second.Checkpoint)
	}
	state, err := checkpoints.Bootstrap(first, pub, "witness-test-1", "db-test-1")
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := checkpoints.AcceptNext(second, pub, state); err != nil {
		t.Fatalf("public verifier rejected witness chain: %v", err)
	}
}

func TestWitnessExactHeadRetryReturnsIdenticalCheckpoint(t *testing.T) {
	w, pub, _ := newTestWitness(t, filepath.Join(t.TempDir(), "witness.jsonl"), nil)
	defer w.Close()
	first, duplicate, err := w.Issue(observed(7, "head-7"))
	if err != nil || duplicate {
		t.Fatalf("first issue: %v %v", duplicate, err)
	}
	retry, duplicate, err := w.Issue(observed(7, "head-7"))
	if err != nil || !duplicate {
		t.Fatalf("retry: %v %v", duplicate, err)
	}
	firstDigest, _ := first.DigestSHA256()
	retryDigest, _ := retry.DigestSHA256()
	if firstDigest != retryDigest {
		t.Fatalf("lost-ack retry changed signed checkpoint: %s != %s", firstDigest, retryDigest)
	}
	if err := retry.Verify(pub); err != nil {
		t.Fatal(err)
	}
}

func TestWitnessRejectsSameProgressEquivocationAndNoProgress(t *testing.T) {
	w, _, _ := newTestWitness(t, filepath.Join(t.TempDir(), "witness.jsonl"), nil)
	defer w.Close()
	if _, _, err := w.Issue(observed(5, "head-5")); err != nil {
		t.Fatal(err)
	}
	if _, _, err := w.Issue(observed(5, "forked-head-5")); !errors.Is(err, ErrEquivocation) {
		t.Fatalf("expected same-sequence equivocation rejection, got %v", err)
	}
	if _, _, err := w.Issue(observed(6, "head-5")); !errors.Is(err, ErrEquivocation) {
		t.Fatalf("expected sequence-only head replay rejection, got %v", err)
	}
	if _, _, err := w.Issue(observed(4, "old-head")); !errors.Is(err, checkpoints.ErrRollback) {
		t.Fatalf("expected journal rollback rejection, got %v", err)
	}
}

func TestWitnessRejectsWrongDatabaseAndInvalidHead(t *testing.T) {
	w, _, _ := newTestWitness(t, filepath.Join(t.TempDir(), "witness.jsonl"), nil)
	defer w.Close()
	if _, _, err := w.Issue(checkpoints.ObservedHead{DatabaseID: "other-db", JournalSequence: 1, JournalHeadSHA256: observed(1, "h").JournalHeadSHA256}); !errors.Is(err, ErrWrongDatabase) {
		t.Fatalf("expected database mismatch, got %v", err)
	}
	if _, _, err := w.Issue(checkpoints.ObservedHead{DatabaseID: "db-test-1", JournalSequence: 1, JournalHeadSHA256: "bad"}); !errors.Is(err, ErrInvalidObservation) {
		t.Fatalf("expected invalid observation, got %v", err)
	}
}

func TestWitnessRestartReplaysAndRecoversLostAcknowledgement(t *testing.T) {
	path := filepath.Join(t.TempDir(), "witness.jsonl")
	w, pub, priv := newTestWitness(t, path, nil)
	first, _, err := w.Issue(observed(2, "head-2"))
	if err != nil {
		t.Fatal(err)
	}
	firstDigest, _ := first.DigestSHA256()
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
	reopened, err := Open(Options{Path: path, KeyID: "witness-test-1", DatabaseID: "db-test-1", PrivateKey: priv})
	if err != nil {
		t.Fatal(err)
	}
	defer reopened.Close()
	retried, duplicate, err := reopened.Issue(observed(2, "head-2"))
	if err != nil || !duplicate {
		t.Fatalf("reopened retry: duplicate=%v err=%v", duplicate, err)
	}
	retryDigest, _ := retried.DigestSHA256()
	if firstDigest != retryDigest {
		t.Fatalf("restart changed signed checkpoint: %s != %s", firstDigest, retryDigest)
	}
	if err := retried.Verify(pub); err != nil {
		t.Fatal(err)
	}
}

func TestWitnessRejectsTamperedAndTornJournal(t *testing.T) {
	t.Run("tampered signature", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "witness.jsonl")
		w, _, priv := newTestWitness(t, path, nil)
		if _, _, err := w.Issue(observed(1, "head-1")); err != nil {
			t.Fatal(err)
		}
		if err := w.Close(); err != nil {
			t.Fatal(err)
		}
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		var record journalRecord
		if err := json.Unmarshal(bytes.TrimSpace(data), &record); err != nil {
			t.Fatal(err)
		}
		record.Checkpoint.Signature[0] ^= 0xff
		data, err = json.Marshal(record)
		if err != nil {
			t.Fatal(err)
		}
		data = append(data, '\n')
		if err := os.WriteFile(path, data, 0o600); err != nil {
			t.Fatal(err)
		}
		if _, err := Open(Options{Path: path, KeyID: "witness-test-1", DatabaseID: "db-test-1", PrivateKey: priv}); !errors.Is(err, ErrCorruptJournal) {
			t.Fatalf("expected corrupt journal rejection, got %v", err)
		}
	})
	t.Run("torn tail", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "witness.jsonl")
		w, _, priv := newTestWitness(t, path, nil)
		if _, _, err := w.Issue(observed(1, "head-1")); err != nil {
			t.Fatal(err)
		}
		if err := w.Close(); err != nil {
			t.Fatal(err)
		}
		file, err := os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0o600)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := file.Write([]byte(`{"partial":`)); err != nil {
			t.Fatal(err)
		}
		_ = file.Close()
		if _, err := Open(Options{Path: path, KeyID: "witness-test-1", DatabaseID: "db-test-1", PrivateKey: priv}); !errors.Is(err, ErrCorruptJournal) {
			t.Fatalf("expected torn-journal rejection, got %v", err)
		}
	})
}

func TestWitnessSerializesConcurrentIdenticalRequests(t *testing.T) {
	w, _, _ := newTestWitness(t, filepath.Join(t.TempDir(), "witness.jsonl"), nil)
	defer w.Close()
	const callers = 32
	var wg sync.WaitGroup
	wg.Add(callers)
	errs := make(chan error, callers)
	digests := make(chan string, callers)
	for i := 0; i < callers; i++ {
		go func() {
			defer wg.Done()
			cp, _, err := w.Issue(observed(1, "head-1"))
			if err != nil {
				errs <- err
				return
			}
			digest, err := cp.DigestSHA256()
			if err != nil {
				errs <- err
				return
			}
			digests <- digest
		}()
	}
	wg.Wait()
	close(errs)
	close(digests)
	for err := range errs {
		t.Fatal(err)
	}
	var expected string
	for digest := range digests {
		if expected == "" {
			expected = digest
		} else if digest != expected {
			t.Fatalf("concurrent identical observations produced different checkpoint digests: %s != %s", digest, expected)
		}
	}
}

func TestWitnessSingleWriterLock(t *testing.T) {
	path := filepath.Join(t.TempDir(), "witness.jsonl")
	w, _, priv := newTestWitness(t, path, nil)
	defer w.Close()
	if _, err := Open(Options{Path: path, KeyID: "witness-test-1", DatabaseID: "db-test-1", PrivateKey: priv}); err == nil {
		t.Fatal("second writer unexpectedly acquired witness journal")
	}
}

func TestWitnessRejectsClockRollbackBeforeAppending(t *testing.T) {
	path := filepath.Join(t.TempDir(), "witness.jsonl")
	now := time.Unix(1000, 0)
	w, _, priv := newTestWitness(t, path, func() time.Time { return now })
	if _, _, err := w.Issue(observed(1, "head-1")); err != nil {
		t.Fatal(err)
	}
	now = time.Unix(999, 0)
	if _, _, err := w.Issue(observed(2, "head-2")); !errors.Is(err, ErrClockRollback) {
		t.Fatalf("expected clock rollback rejection, got %v", err)
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
	reopened, err := Open(Options{Path: path, KeyID: "witness-test-1", DatabaseID: "db-test-1", PrivateKey: priv})
	if err != nil {
		t.Fatalf("clock rollback should not poison persisted history: %v", err)
	}
	defer reopened.Close()
	if _, exists, err := reopened.Current(); err != nil || !exists {
		t.Fatalf("previous checkpoint should remain valid after rejected issuance: exists=%v err=%v", exists, err)
	}
}
