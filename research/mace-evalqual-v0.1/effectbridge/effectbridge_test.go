package effectbridge

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"math/rand"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

func testRequest(key, payload string) Request {
	return Request{IdempotencyKey: key, Principal: "agent:test", Target: "payments:test/42", PayloadSHA256: Digest([]byte(payload))}
}

func TestServiceHelper(t *testing.T) {
	if os.Getenv("MACE_EFFECT_HELPER_MODE") != "service" {
		return
	}
	server, err := NewServer(os.Getenv("MACE_EFFECT_JOURNAL"), true)
	if err != nil {
		t.Fatal(err)
	}
	listener, err := net.Listen("tcp", os.Getenv("MACE_EFFECT_LISTEN"))
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(os.Getenv("MACE_EFFECT_READY"), []byte(listener.Addr().String()), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := server.Serve(listener); err != nil && err != http.ErrServerClosed {
		t.Fatal(err)
	}
	_ = server.Close()
}

func TestClientHelper(t *testing.T) {
	if os.Getenv("MACE_EFFECT_HELPER_MODE") != "client" {
		return
	}
	client, err := NewClient(os.Getenv("MACE_EFFECT_URL"), 5*time.Second)
	if err != nil {
		t.Fatal(err)
	}
	result := client.Dispatch(context.Background(), testRequest(os.Getenv("MACE_EFFECT_KEY"), os.Getenv("MACE_EFFECT_PAYLOAD")), false)
	if result.Status != DispatchCompleted {
		t.Fatalf("child dispatch: %+v", result)
	}
	if err := os.WriteFile(os.Getenv("MACE_EFFECT_RESULT"), []byte("completed"), 0o600); err != nil {
		t.Fatal(err)
	}
}

func TestLockProbeHelper(t *testing.T) {
	if os.Getenv("MACE_EFFECT_HELPER_MODE") != "lock" {
		return
	}
	_, err := OpenJournal(os.Getenv("MACE_EFFECT_JOURNAL"))
	if err == nil || !strings.Contains(err.Error(), "another process owns") {
		t.Fatalf("second writer should be rejected, got %v", err)
	}
}

type runningService struct {
	cmd     *exec.Cmd
	url     string
	journal string
}

func startService(t *testing.T, journal string) *runningService {
	t.Helper()
	probe, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	addr := probe.Addr().String()
	_ = probe.Close()
	ready := filepath.Join(t.TempDir(), "ready")
	cmd := exec.Command(os.Args[0], "-test.run=^TestServiceHelper$")
	cmd.Env = append(os.Environ(),
		"MACE_EFFECT_HELPER_MODE=service",
		"MACE_EFFECT_JOURNAL="+journal,
		"MACE_EFFECT_LISTEN="+addr,
		"MACE_EFFECT_READY="+ready)
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	service := &runningService{cmd: cmd, url: "http://" + addr, journal: journal}
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		if _, err := os.Stat(ready); err == nil {
			resp, err := http.Get(service.url + "/health")
			if err == nil {
				resp.Body.Close()
				if resp.StatusCode == http.StatusOK {
					return service
				}
			}
		}
		time.Sleep(20 * time.Millisecond)
	}
	_ = cmd.Process.Kill()
	_ = cmd.Wait()
	t.Fatalf("service process did not start")
	return nil
}

func (s *runningService) stop(t *testing.T) {
	t.Helper()
	if s == nil || s.cmd == nil || s.cmd.Process == nil {
		return
	}
	_ = s.cmd.Process.Kill()
	_ = s.cmd.Wait()
	s.cmd = nil
}

func TestJournalRejectsTornTailAndTamperedRecord(t *testing.T) {
	path := filepath.Join(t.TempDir(), "events.jsonl")
	j, err := OpenJournal(path)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := j.Append(testRequest("k1", "payload")); err != nil {
		t.Fatal(err)
	}
	if err := j.Close(); err != nil {
		t.Fatal(err)
	}
	file, err := os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0o600)
	if err != nil {
		t.Fatal(err)
	}
	_, _ = file.Write([]byte("{\"schema_version\":\"torn"))
	_ = file.Close()
	if _, err := OpenJournal(path); err == nil {
		t.Fatal("truncated journal tail accepted")
	}
	if err := os.Truncate(path, 0); err != nil {
		t.Fatal(err)
	}
	j, err = OpenJournal(path)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := j.Append(testRequest("k1", "payload")); err != nil {
		t.Fatal(err)
	}
	_ = j.Close()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	data = []byte(strings.Replace(string(data), "payments:test/42", "payments:attacker/42", 1))
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := OpenJournal(path); err == nil {
		t.Fatal("altered journal record accepted")
	}
}

func TestIdempotencyAndIntentConflict(t *testing.T) {
	j, err := OpenJournal(filepath.Join(t.TempDir(), "events.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	defer j.Close()
	req := testRequest("same-key", "payload-a")
	first, duplicate, err := j.Append(req)
	if err != nil || duplicate {
		t.Fatalf("first append: duplicate=%v err=%v", duplicate, err)
	}
	second, duplicate, err := j.Append(req)
	if err != nil || !duplicate || first != second || j.sequence != 1 {
		t.Fatalf("duplicate did not return same stable receipt: dup=%v err=%v seq=%d", duplicate, err, j.sequence)
	}
	if _, _, err := j.Append(testRequest("same-key", "payload-b")); !errors.Is(err, ErrIntentConflict) {
		t.Fatalf("changed intent did not conflict: %v", err)
	}
}

func TestJournalWriterLockIsExclusiveAcrossProcesses(t *testing.T) {
	path := filepath.Join(t.TempDir(), "journal", "events.jsonl")
	owner, err := OpenJournal(path)
	if err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command(os.Args[0], "-test.run=^TestLockProbeHelper$")
	cmd.Env = append(os.Environ(), "MACE_EFFECT_HELPER_MODE=lock", "MACE_EFFECT_JOURNAL="+path)
	if output, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("lock helper: %v: %s", err, output)
	}
	if err := owner.Close(); err != nil {
		t.Fatal(err)
	}
	reopened, err := OpenJournal(path)
	if err != nil {
		t.Fatalf("lock not released at close: %v", err)
	}
	_ = reopened.Close()
}

func TestRemoteSuccessLostAckRecoversAfterServiceRestart(t *testing.T) {
	journal := filepath.Join(t.TempDir(), "service", "effects.jsonl")
	service := startService(t, journal)
	client, err := NewClient(service.url, 2*time.Second)
	if err != nil {
		t.Fatal(err)
	}
	req := testRequest("lost-ack", "transfer-10")
	dispatch := client.Dispatch(context.Background(), req, true)
	if dispatch.Status != DispatchUnknown {
		t.Fatalf("lost acknowledgement should be unknown: %+v", dispatch)
	}
	first := client.Reconcile(context.Background(), req)
	if first.Status != ReconcileCompleted || first.Evidence.Receipt == nil || first.Evidence.Journal.Sequence != 1 {
		t.Fatalf("remote effect not reconciled: %+v", first)
	}
	service.stop(t)
	service = startService(t, journal)
	defer service.stop(t)
	client, err = NewClient(service.url, time.Second)
	if err != nil {
		t.Fatal(err)
	}
	afterRestart := client.Reconcile(context.Background(), req)
	if afterRestart.Status != ReconcileCompleted || afterRestart.Evidence.Receipt == nil || afterRestart.Evidence.Receipt.Sequence != 1 {
		t.Fatalf("restart did not recover original receipt: %+v", afterRestart)
	}
	records, summary, err := ReadJournal(journal)
	if err != nil || summary.Sequence != 1 || len(records) != 1 {
		t.Fatalf("effect duplicated after restart: records=%d summary=%+v err=%v", len(records), summary, err)
	}
}

func TestNotFoundIsIndeterminateUntilDestinationFence(t *testing.T) {
	journal := filepath.Join(t.TempDir(), "effects.jsonl")
	service := startService(t, journal)
	defer service.stop(t)
	client, err := NewClient(service.url, time.Second)
	if err != nil {
		t.Fatal(err)
	}
	req := testRequest("fence-me", "payload")
	absence := client.Reconcile(context.Background(), req)
	if absence.Status != ReconcileIndeterminate || absence.Evidence.Status != "not_found_unfenced" {
		t.Fatalf("not found promoted to non-execution: %+v", absence)
	}
	fenced := client.FenceAndReconcile(context.Background(), req)
	if fenced.Status != ReconcileNotExecuted || fenced.Evidence.Receipt == nil || fenced.Evidence.Receipt.RecordKind != "fence_committed" {
		t.Fatalf("destination fence failed: %+v", fenced)
	}
	late := client.Dispatch(context.Background(), req, false)
	if late.Status != DispatchRejected {
		t.Fatalf("late request passed committed fence: %+v", late)
	}
	records, summary, err := ReadJournal(journal)
	if err != nil || summary.Sequence != 1 || len(records) != 1 || records[0].RecordKind != "fence_committed" {
		t.Fatalf("fence history incorrect: records=%+v summary=%+v err=%v", records, summary, err)
	}
}

func TestLostFenceAckRecoversByRepeatingExactFence(t *testing.T) {
	service := startService(t, filepath.Join(t.TempDir(), "effects.jsonl"))
	defer service.stop(t)
	client, err := NewClient(service.url, time.Second)
	if err != nil {
		t.Fatal(err)
	}
	req := testRequest("fence-lost-ack", "payload")
	client.WithFenceAckLossForTest(true)
	first := client.FenceAndReconcile(context.Background(), req)
	if first.Status != ReconcileIndeterminate {
		t.Fatalf("lost fence ack must stay indeterminate: %+v", first)
	}
	client.WithFenceAckLossForTest(false)
	second := client.FenceAndReconcile(context.Background(), req)
	if second.Status != ReconcileNotExecuted || second.Evidence.Receipt == nil {
		t.Fatalf("repeated exact fence did not recover tombstone: %+v", second)
	}
}

func TestEffectAndFenceRaceHasOneTerminalRecord(t *testing.T) {
	j, err := OpenJournal(filepath.Join(t.TempDir(), "race.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	server := &Server{journal: j}
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/effects", server.handleEffect)
	mux.HandleFunc("/v1/fence/", server.handleFence)
	httpServer := httptest.NewServer(mux)
	defer httpServer.Close()
	defer j.Close()
	client, err := NewClient(httpServer.URL, 3*time.Second)
	if err != nil {
		t.Fatal(err)
	}
	req := testRequest("race-key", "payload")
	start := make(chan struct{})
	var dispatch DispatchResult
	var fence ReconcileResult
	var wg sync.WaitGroup
	wg.Add(2)
	go func() { defer wg.Done(); <-start; dispatch = client.Dispatch(context.Background(), req, false) }()
	go func() { defer wg.Done(); <-start; fence = client.FenceAndReconcile(context.Background(), req) }()
	close(start)
	wg.Wait()
	if j.sequence != 1 {
		t.Fatalf("race created %d terminal journal records, want one", j.sequence)
	}
	if dispatch.Status == DispatchCompleted {
		if fence.Status != ReconcileCompleted {
			t.Fatalf("effect should win both race outcomes: dispatch=%+v fence=%+v", dispatch, fence)
		}
	} else if dispatch.Status == DispatchRejected {
		if fence.Status != ReconcileNotExecuted {
			t.Fatalf("fence should block dispatch: dispatch=%+v fence=%+v", dispatch, fence)
		}
	} else {
		t.Fatalf("unexpected race outcome: dispatch=%+v fence=%+v", dispatch, fence)
	}
}

func TestEightClientProcessesShareOneIdempotencyDecision(t *testing.T) {
	journal := filepath.Join(t.TempDir(), "multiprocess", "events.jsonl")
	service := startService(t, journal)
	defer service.stop(t)
	const clients = 8
	type child struct {
		cmd *exec.Cmd
		output *bytes.Buffer
	}
	children := make([]child, 0, clients)
	for i := 0; i < clients; i++ {
		resultPath := filepath.Join(t.TempDir(), fmt.Sprintf("client-%d", i))
		cmd := exec.Command(os.Args[0], "-test.run=^TestClientHelper$")
		cmd.Env = append(os.Environ(),
			"MACE_EFFECT_HELPER_MODE=client",
			"MACE_EFFECT_URL="+service.url,
			"MACE_EFFECT_RESULT="+resultPath,
			"MACE_EFFECT_KEY=shared-idempotency",
			"MACE_EFFECT_PAYLOAD=identical")
		if err := cmd.Start(); err != nil {
			t.Fatal(err)
		}
		children = append(children, child{cmd: cmd, output: &output})
	}
	for i, child := range children {
		if err := child.cmd.Wait(); err != nil {
			t.Fatalf("client %d failed: %v: %s", i, err, child.output.String())
		}
	}
	records, summary, err := ReadJournal(journal)
	if err != nil || len(records) != 1 || summary.Sequence != 1 || records[0].RecordKind != "effect_committed" {
		t.Fatalf("multiprocess clients created duplicate effects: records=%+v summary=%+v err=%v", records, summary, err)
	}
}

func TestTenThousandSeededOperationsMatchFiniteStateModel(t *testing.T) {
	j, err := OpenJournal(filepath.Join(t.TempDir(), "model.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	defer j.Close()
	type entry struct {
		intent  string
		kind    string
		receipt Receipt
	}
	model := map[string]entry{}
	rng := rand.New(rand.NewSource(20261114))
	for i := 0; i < 10_000; i++ {
		key := fmt.Sprintf("key-%03d", rng.Intn(128))
		payload := fmt.Sprintf("payload-%d", rng.Intn(4))
		req := testRequest(key, payload)
		intent, _ := IntentDigest(req)
		want, exists := model[key]
		if rng.Intn(2) == 0 {
			got, duplicate, appendErr := j.Append(req)
			if !exists {
				if appendErr != nil || duplicate || got.RecordKind != "effect_committed" {
					t.Fatalf("step %d new effect mismatch: dup=%v err=%v", i, duplicate, appendErr)
				}
				model[key] = entry{intent: intent, kind: "effect_committed", receipt: got}
			} else if want.intent != intent {
				if !errors.Is(appendErr, ErrIntentConflict) {
					t.Fatalf("step %d expected intent conflict: %v", i, appendErr)
				}
			} else if want.kind == "fence_committed" {
				if !errors.Is(appendErr, ErrFenced) {
					t.Fatalf("step %d expected fenced reject: %v", i, appendErr)
				}
			} else if appendErr != nil || !duplicate || got != want.receipt {
				t.Fatalf("step %d duplicate mismatch: dup=%v err=%v", i, duplicate, appendErr)
			}
		} else {
			got, duplicate, fenceErr := j.Fence(req)
			if !exists {
				if fenceErr != nil || duplicate || got.RecordKind != "fence_committed" {
					t.Fatalf("step %d new fence mismatch: dup=%v err=%v", i, duplicate, fenceErr)
				}
				model[key] = entry{intent: intent, kind: "fence_committed", receipt: got}
			} else if want.intent != intent {
				if !errors.Is(fenceErr, ErrIntentConflict) {
					t.Fatalf("step %d expected fence intent conflict: %v", i, fenceErr)
				}
			} else if fenceErr != nil || !duplicate || got != want.receipt {
				t.Fatalf("step %d fence retry mismatch: dup=%v err=%v", i, duplicate, fenceErr)
			}
		}
	}
	if j.sequence != uint64(len(model)) {
		t.Fatalf("journal sequence %d != model terminal keys %d", j.sequence, len(model))
	}
}

func TestUnavailableReconciliationStaysIndeterminate(t *testing.T) {
	client, err := NewClient("http://127.0.0.1:1", 100*time.Millisecond)
	if err != nil {
		t.Fatal(err)
	}
	req := testRequest("unreachable", "payload")
	if got := client.Dispatch(context.Background(), req, false); got.Status != DispatchUnknown {
		t.Fatalf("unreachable dispatch was not unknown: %+v", got)
	}
	if got := client.Reconcile(context.Background(), req); got.Status != ReconcileIndeterminate {
		t.Fatalf("unreachable reconciliation did not remain indeterminate: %+v", got)
	}
}

func TestFenceDoesNotApplyToDifferentIntent(t *testing.T) {
	j, err := OpenJournal(filepath.Join(t.TempDir(), "intent.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	defer j.Close()
	if _, _, err := j.Fence(testRequest("key", "payload-a")); err != nil {
		t.Fatal(err)
	}
	if _, _, err := j.Fence(testRequest("key", "payload-b")); !errors.Is(err, ErrIntentConflict) {
		t.Fatalf("different intent reused fence key: %v", err)
	}
}

func TestMultipleEffectsRetainOrderedReceiptChain(t *testing.T) {
	j, err := OpenJournal(filepath.Join(t.TempDir(), "ordered.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	defer j.Close()
	for i := 0; i < 5; i++ {
		if _, _, err := j.Append(testRequest(fmt.Sprintf("key-%d", i), fmt.Sprintf("payload-%d", i))); err != nil {
			t.Fatal(err)
		}
	}
	records, summary, err := ReadJournal(j.path)
	if err != nil || len(records) != 5 || summary.Sequence != 5 {
		t.Fatalf("ordered journal read failed: records=%d summary=%+v err=%v", len(records), summary, err)
	}
	for i, record := range records {
		if record.Sequence != uint64(i+1) {
			t.Fatalf("record %d sequence=%d", i, record.Sequence)
		}
	}
}
