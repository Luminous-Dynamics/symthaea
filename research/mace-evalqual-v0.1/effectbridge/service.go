package effectbridge

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

const (
	SchemaVersion   = "mace-effectbridge/v1"
	DurabilityMode  = "fsync_per_record"
	maxRequestBytes = 16 * 1024
	maxJournalFrame = 1 << 20
)

var (
	ErrIntentConflict  = errors.New("idempotency key already used for a different intent")
	ErrJournalPoisoned = errors.New("effect journal is poisoned; restart and reconcile")
	ErrFenced          = errors.New("idempotency key has been durably fenced against late execution")
)

type Request struct {
	IdempotencyKey string `json:"idempotency_key"`
	Principal      string `json:"principal"`
	Target         string `json:"target"`
	PayloadSHA256  string `json:"payload_sha256"`
}

type Receipt struct {
	SchemaVersion  string `json:"schema_version"`
	RecordKind     string `json:"record_kind"`
	DurabilityMode string `json:"durability_mode"`
	Sequence       uint64 `json:"sequence"`
	IdempotencyKey string `json:"idempotency_key"`
	IntentSHA256   string `json:"intent_sha256"`
	Principal      string `json:"principal"`
	Target         string `json:"target"`
	PayloadSHA256  string `json:"payload_sha256"`
	EffectID       string `json:"effect_id"`
	PreviousSHA256 string `json:"previous_sha256"`
	RecordSHA256   string `json:"record_sha256"`
}

type JournalSummary struct {
	DurabilityMode string `json:"durability_mode"`
	Sequence       uint64 `json:"sequence"`
	HeadSHA256     string `json:"head_sha256"`
	Complete       bool   `json:"complete"`
}

type Reconciliation struct {
	SchemaVersion  string         `json:"schema_version"`
	Status         string         `json:"status"`
	IdempotencyKey string         `json:"idempotency_key"`
	Journal        JournalSummary `json:"journal"`
	Receipt        *Receipt       `json:"receipt,omitempty"`
}

type Journal struct {
	mu       sync.Mutex
	path     string
	file     *os.File
	records  map[string]Receipt
	sequence uint64
	head     string
	poisoned bool
}

func Digest(data []byte) string {
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

func IntentDigest(req Request) (string, error) {
	if err := validateRequest(req); err != nil {
		return "", err
	}
	// Struct field order is fixed and the wire schema is versioned.
	intent := struct {
		SchemaVersion string `json:"schema_version"`
		Principal     string `json:"principal"`
		Target        string `json:"target"`
		PayloadSHA256 string `json:"payload_sha256"`
	}{SchemaVersion, req.Principal, req.Target, req.PayloadSHA256}
	data, err := json.Marshal(intent)
	if err != nil {
		return "", err
	}
	return Digest(data), nil
}

func validateRequest(req Request) error {
	if strings.TrimSpace(req.IdempotencyKey) == "" || strings.Contains(req.IdempotencyKey, "/") || strings.Contains(req.IdempotencyKey, "\\") {
		return errors.New("idempotency_key must be non-empty and path-safe")
	}
	if strings.TrimSpace(req.Principal) == "" || strings.TrimSpace(req.Target) == "" {
		return errors.New("principal and target are required")
	}
	if !isDigest(req.PayloadSHA256) {
		return errors.New("payload_sha256 must be lowercase SHA-256 hex")
	}
	return nil
}

func isDigest(value string) bool {
	if len(value) != 64 || strings.ToLower(value) != value {
		return false
	}
	_, err := hex.DecodeString(value)
	return err == nil
}

func OpenJournal(path string) (*Journal, error) {
	if strings.TrimSpace(path) == "" {
		return nil, errors.New("journal path is required")
	}
	abs, err := filepath.Abs(path)
	if err != nil {
		return nil, err
	}
	if err := os.MkdirAll(filepath.Dir(abs), 0o700); err != nil {
		return nil, err
	}
	file, err := os.OpenFile(abs, os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		return nil, err
	}
	if err := lockJournal(file); err != nil {
		_ = file.Close()
		return nil, err
	}
	// Persist the directory entry as well as later journal frames. A successful
	// file fsync alone does not guarantee that a newly created filename survives
	// a power-loss model on filesystems that require directory fsync.
	dir, err := os.Open(filepath.Dir(abs))
	if err != nil {
		_ = unlockJournal(file)
		_ = file.Close()
		return nil, err
	}
	dirSyncErr := dir.Sync()
	_ = dir.Close()
	if dirSyncErr != nil {
		_ = unlockJournal(file)
		_ = file.Close()
		return nil, fmt.Errorf("sync journal parent directory: %w", dirSyncErr)
	}
	j := &Journal{path: abs, file: file, records: make(map[string]Receipt)}
	if err := j.replay(); err != nil {
		_ = unlockJournal(file)
		_ = file.Close()
		return nil, fmt.Errorf("journal recovery rejected: %w", err)
	}
	if _, err := file.Seek(0, io.SeekEnd); err != nil {
		_ = unlockJournal(file)
		_ = file.Close()
		return nil, err
	}
	return j, nil
}

func (j *Journal) replay() error {
	if _, err := j.file.Seek(0, io.SeekStart); err != nil {
		return err
	}
	data, err := io.ReadAll(j.file)
	if err != nil {
		return err
	}
	if len(data) > 0 && data[len(data)-1] != '\n' {
		return errors.New("truncated final journal frame; no automatic tail truncation")
	}
	scanner := bufio.NewScanner(bytes.NewReader(data))
	scanner.Buffer(make([]byte, 4096), maxJournalFrame)
	line := 0
	for scanner.Scan() {
		line++
		frame := append([]byte(nil), scanner.Bytes()...)
		if len(frame) == 0 {
			return fmt.Errorf("empty journal frame at line %d", line)
		}
		var receipt Receipt
		dec := json.NewDecoder(bytes.NewReader(frame))
		dec.DisallowUnknownFields()
		if err := dec.Decode(&receipt); err != nil {
			return fmt.Errorf("decode line %d: %w", line, err)
		}
		if err := ensureJSONEOF(dec); err != nil {
			return fmt.Errorf("trailing JSON on line %d: %w", line, err)
		}
		if err := validateReceipt(receipt, j.sequence+1, j.head); err != nil {
			return fmt.Errorf("validate line %d: %w", line, err)
		}
		if prior, exists := j.records[receipt.IdempotencyKey]; exists {
			if prior.IntentSHA256 != receipt.IntentSHA256 {
				return fmt.Errorf("idempotency key conflict at line %d", line)
			}
			return fmt.Errorf("duplicate durable effect record at line %d", line)
		}
		j.records[receipt.IdempotencyKey] = receipt
		j.sequence = receipt.Sequence
		j.head = receipt.RecordSHA256
	}
	if err := scanner.Err(); err != nil {
		return err
	}
	return nil
}

func ensureJSONEOF(dec *json.Decoder) error {
	var extra any
	if err := dec.Decode(&extra); err != io.EOF {
		if err == nil {
			return errors.New("multiple JSON values")
		}
		return err
	}
	return nil
}

func VerifyReceiptForRequest(receipt Receipt, req Request) error {
	intent, err := IntentDigest(req)
	if err != nil {
		return err
	}
	if receipt.IdempotencyKey != req.IdempotencyKey || receipt.IntentSHA256 != intent {
		return errors.New("receipt does not bind exact request")
	}
	return validateReceipt(receipt, receipt.Sequence, receipt.PreviousSHA256)
}

func validateReceipt(receipt Receipt, expectedSequence uint64, expectedPrevious string) error {
	if receipt.SchemaVersion != SchemaVersion || receipt.DurabilityMode != DurabilityMode || receipt.Sequence == 0 || receipt.Sequence != expectedSequence {
		return errors.New("schema or contiguous sequence mismatch")
	}
	if receipt.IdempotencyKey == "" || receipt.Principal == "" || receipt.Target == "" {
		return errors.New("required receipt identity missing")
	}
	switch receipt.RecordKind {
	case "effect_committed":
		if receipt.EffectID == "" {
			return errors.New("effect record requires effect_id")
		}
	case "fence_committed":
		if receipt.EffectID != "" {
			return errors.New("fence record must not claim an effect_id")
		}
	default:
		return errors.New("unknown journal record_kind")
	}
	if !isDigest(receipt.IntentSHA256) || !isDigest(receipt.PayloadSHA256) || receipt.PreviousSHA256 != expectedPrevious || !isDigest(receipt.RecordSHA256) {
		return errors.New("digest or previous-head mismatch")
	}
	copy := receipt
	copy.RecordSHA256 = strings.Repeat("0", 64)
	data, err := json.Marshal(copy)
	if err != nil {
		return err
	}
	if Digest(data) != receipt.RecordSHA256 {
		return errors.New("record digest mismatch")
	}
	req := Request{IdempotencyKey: receipt.IdempotencyKey, Principal: receipt.Principal, Target: receipt.Target, PayloadSHA256: receipt.PayloadSHA256}
	intent, err := IntentDigest(req)
	if err != nil {
		return err
	}
	if intent != receipt.IntentSHA256 {
		return errors.New("intent digest does not bind receipt fields")
	}
	return nil
}

func (j *Journal) Append(req Request) (Receipt, bool, error) {
	if err := validateRequest(req); err != nil {
		return Receipt{}, false, err
	}
	intent, err := IntentDigest(req)
	if err != nil {
		return Receipt{}, false, err
	}
	j.mu.Lock()
	defer j.mu.Unlock()
	if j.poisoned {
		return Receipt{}, false, ErrJournalPoisoned
	}
	if existing, ok := j.records[req.IdempotencyKey]; ok {
		if existing.IntentSHA256 != intent {
			return Receipt{}, false, ErrIntentConflict
		}
		if existing.RecordKind == "fence_committed" {
			return Receipt{}, false, ErrFenced
		}
		return existing, true, nil
	}
	receipt := Receipt{
		SchemaVersion:  SchemaVersion,
		RecordKind:     "effect_committed",
		DurabilityMode: DurabilityMode,
		Sequence:       j.sequence + 1,
		IdempotencyKey: req.IdempotencyKey,
		IntentSHA256:   intent,
		Principal:      req.Principal,
		Target:         req.Target,
		PayloadSHA256:  req.PayloadSHA256,
		EffectID:       fmt.Sprintf("effect-%020d", j.sequence+1),
		PreviousSHA256: j.head,
		RecordSHA256:   strings.Repeat("0", 64),
	}
	unsigned, err := json.Marshal(receipt)
	if err != nil {
		return Receipt{}, false, err
	}
	receipt.RecordSHA256 = Digest(unsigned)
	frame, err := json.Marshal(receipt)
	if err != nil {
		return Receipt{}, false, err
	}
	frame = append(frame, '\n')
	n, writeErr := j.file.Write(frame)
	if writeErr != nil || n != len(frame) {
		j.poisoned = true
		if writeErr != nil {
			return Receipt{}, false, fmt.Errorf("journal write uncertain after %d/%d bytes: %w", n, len(frame), writeErr)
		}
		return Receipt{}, false, fmt.Errorf("journal short write: %d/%d; journal poisoned", n, len(frame))
	}
	if err := j.file.Sync(); err != nil {
		j.poisoned = true
		return Receipt{}, false, fmt.Errorf("journal fsync outcome uncertain; journal poisoned: %w", err)
	}
	j.records[req.IdempotencyKey] = receipt
	j.sequence = receipt.Sequence
	j.head = receipt.RecordSHA256
	return receipt, false, nil
}

func (j *Journal) Fence(req Request) (Receipt, bool, error) {
	if err := validateRequest(req); err != nil {
		return Receipt{}, false, err
	}
	intent, err := IntentDigest(req)
	if err != nil {
		return Receipt{}, false, err
	}
	j.mu.Lock()
	defer j.mu.Unlock()
	if j.poisoned {
		return Receipt{}, false, ErrJournalPoisoned
	}
	if existing, ok := j.records[req.IdempotencyKey]; ok {
		if existing.IntentSHA256 != intent {
			return Receipt{}, false, ErrIntentConflict
		}
		// If the effect already committed, the fence resolves to that effect.
		// If a fence already committed, return the same stable fence receipt.
		return existing, true, nil
	}
	receipt := Receipt{
		SchemaVersion:  SchemaVersion,
		RecordKind:     "fence_committed",
		DurabilityMode: DurabilityMode,
		Sequence:       j.sequence + 1,
		IdempotencyKey: req.IdempotencyKey,
		IntentSHA256:   intent,
		Principal:      req.Principal,
		Target:         req.Target,
		PayloadSHA256:  req.PayloadSHA256,
		EffectID:       "",
		PreviousSHA256: j.head,
		RecordSHA256:   strings.Repeat("0", 64),
	}
	unsigned, err := json.Marshal(receipt)
	if err != nil {
		return Receipt{}, false, err
	}
	receipt.RecordSHA256 = Digest(unsigned)
	frame, err := json.Marshal(receipt)
	if err != nil {
		return Receipt{}, false, err
	}
	frame = append(frame, '\n')
	n, writeErr := j.file.Write(frame)
	if writeErr != nil || n != len(frame) {
		j.poisoned = true
		if writeErr != nil {
			return Receipt{}, false, fmt.Errorf("fence journal write uncertain after %d/%d bytes: %w", n, len(frame), writeErr)
		}
		return Receipt{}, false, fmt.Errorf("fence journal short write: %d/%d; journal poisoned", n, len(frame))
	}
	if err := j.file.Sync(); err != nil {
		j.poisoned = true
		return Receipt{}, false, fmt.Errorf("fence fsync outcome uncertain; journal poisoned: %w", err)
	}
	j.records[req.IdempotencyKey] = receipt
	j.sequence = receipt.Sequence
	j.head = receipt.RecordSHA256
	return receipt, false, nil
}

func (j *Journal) Lookup(key string) Reconciliation {
	j.mu.Lock()
	defer j.mu.Unlock()
	result := Reconciliation{SchemaVersion: SchemaVersion, IdempotencyKey: key, Journal: JournalSummary{DurabilityMode: DurabilityMode, Sequence: j.sequence, HeadSHA256: j.head, Complete: !j.poisoned}}
	if j.poisoned {
		result.Status = "indeterminate"
		result.Journal.Complete = false
		return result
	}
	if receipt, ok := j.records[key]; ok {
		result.Receipt = &receipt
		if receipt.RecordKind == "effect_committed" {
			result.Status = "found"
		} else {
			result.Status = "not_executed_fenced"
		}
		return result
	}
	// Absence is deliberately not non-execution: an old request may still be
	// in flight and arrive after this read. Only an atomic durable fence closes it.
	result.Status = "not_found_unfenced"
	return result
}

func (j *Journal) Close() error {
	j.mu.Lock()
	defer j.mu.Unlock()
	if j.file == nil {
		return nil
	}
	unlockErr := unlockJournal(j.file)
	closeErr := j.file.Close()
	j.file = nil
	return errors.Join(unlockErr, closeErr)
}

type Server struct {
	journal             *Journal
	allowFaultInjection bool
	http                *http.Server
}

func NewServer(journalPath string, allowFaultInjection bool) (*Server, error) {
	journal, err := OpenJournal(journalPath)
	if err != nil {
		return nil, err
	}
	server := &Server{journal: journal, allowFaultInjection: allowFaultInjection}
	mux := http.NewServeMux()
	mux.HandleFunc("/health", server.handleHealth)
	mux.HandleFunc("/v1/effects", server.handleEffect)
	mux.HandleFunc("/v1/reconcile/", server.handleReconcile)
	mux.HandleFunc("/v1/fence/", server.handleFence)
	server.http = &http.Server{Handler: mux, ReadHeaderTimeout: 2 * time.Second, MaxHeaderBytes: 16 * 1024}
	return server, nil
}

func (s *Server) Serve(listener net.Listener) error {
	return s.http.Serve(listener)
}

func (s *Server) Close() error {
	_ = s.http.Close()
	return s.journal.Close()
}

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"schema_version": SchemaVersion, "status": "ready"})
}

func (s *Server) handleEffect(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, maxRequestBytes))
	if err != nil {
		http.Error(w, "invalid request body", http.StatusBadRequest)
		return
	}
	var req Request
	dec := json.NewDecoder(bytes.NewReader(body))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&req); err != nil {
		http.Error(w, "invalid request JSON", http.StatusBadRequest)
		return
	}
	if err := ensureJSONEOF(dec); err != nil {
		http.Error(w, "trailing request JSON", http.StatusBadRequest)
		return
	}
	if err := validateRequest(req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	if r.Header.Get("X-MACE-Test-Drop-Ack") != "" && (!s.allowFaultInjection || r.Header.Get("X-MACE-Test-Drop-Ack") != "after-commit") {
		http.Error(w, "fault injection not permitted", http.StatusForbidden)
		return
	}
	receipt, duplicate, err := s.journal.Append(req)
	if errors.Is(err, ErrIntentConflict) || errors.Is(err, ErrFenced) {
		http.Error(w, err.Error(), http.StatusConflict)
		return
	}
	if err != nil {
		http.Error(w, "journal commit uncertain; reconcile before any retry", http.StatusServiceUnavailable)
		return
	}
	if r.Header.Get("X-MACE-Test-Drop-Ack") == "after-commit" {
		if hijacker, ok := w.(http.Hijacker); ok {
			conn, _, err := hijacker.Hijack()
			if err == nil {
				_ = conn.Close()
				return
			}
		}
		// If hijacking is unavailable, do not return a false success for the fault test.
		http.Error(w, "unable to inject post-commit acknowledgement loss", http.StatusNotImplemented)
		return
	}
	status := http.StatusCreated
	if duplicate {
		status = http.StatusOK
	}
	writeJSON(w, status, receipt)
}

func (s *Server) handleReconcile(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	key := strings.TrimPrefix(r.URL.Path, "/v1/reconcile/")
	if key == "" || strings.Contains(key, "/") {
		http.Error(w, "invalid idempotency key", http.StatusBadRequest)
		return
	}
	result := s.journal.Lookup(key)
	if result.Status == "indeterminate" {
		writeJSON(w, http.StatusServiceUnavailable, result)
		return
	}
	writeJSON(w, http.StatusOK, result)
}

func (s *Server) handleFence(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPut {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	key := strings.TrimPrefix(r.URL.Path, "/v1/fence/")
	if key == "" || strings.Contains(key, "/") {
		http.Error(w, "invalid idempotency key", http.StatusBadRequest)
		return
	}
	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, maxRequestBytes))
	if err != nil {
		http.Error(w, "invalid request body", http.StatusBadRequest)
		return
	}
	var req Request
	dec := json.NewDecoder(bytes.NewReader(body))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&req); err != nil {
		http.Error(w, "invalid request JSON", http.StatusBadRequest)
		return
	}
	if err := ensureJSONEOF(dec); err != nil {
		http.Error(w, "trailing request JSON", http.StatusBadRequest)
		return
	}
	if req.IdempotencyKey != key {
		http.Error(w, "path and body idempotency key mismatch", http.StatusBadRequest)
		return
	}
	if r.Header.Get("X-MACE-Test-Drop-Fence-Ack") != "" && (!s.allowFaultInjection || r.Header.Get("X-MACE-Test-Drop-Fence-Ack") != "after-commit") {
		http.Error(w, "fault injection not permitted", http.StatusForbidden)
		return
	}
	_, duplicate, err := s.journal.Fence(req)
	if errors.Is(err, ErrIntentConflict) {
		http.Error(w, err.Error(), http.StatusConflict)
		return
	}
	if err != nil {
		http.Error(w, "fence commit uncertain; reconcile before any further action", http.StatusServiceUnavailable)
		return
	}
	if r.Header.Get("X-MACE-Test-Drop-Fence-Ack") == "after-commit" {
		if hijacker, ok := w.(http.Hijacker); ok {
			conn, _, err := hijacker.Hijack()
			if err == nil {
				_ = conn.Close()
				return
			}
		}
		http.Error(w, "unable to inject post-fence acknowledgement loss", http.StatusNotImplemented)
		return
	}
	status := http.StatusCreated
	if duplicate {
		status = http.StatusOK
	}
	writeJSON(w, status, s.journal.Lookup(key))
}

func writeJSON(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(value)
}

// ReadJournal independently reopens and validates a closed or quiescent journal.
// It is a second reader of the same service-authored evidence, not an independent
// physical observation of the service's effect boundary.
func ReadJournal(path string) ([]Receipt, JournalSummary, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, JournalSummary{}, err
	}
	defer file.Close()
	j := &Journal{path: path, file: file, records: make(map[string]Receipt)}
	if err := j.replay(); err != nil {
		return nil, JournalSummary{Complete: false}, err
	}
	out := make([]Receipt, 0, len(j.records))
	for _, receipt := range j.records {
		out = append(out, receipt)
	}
	// Sequence order, rather than map iteration, is the durable event order.
	sort.Slice(out, func(i, k int) bool { return out[i].Sequence < out[k].Sequence })
	return out, JournalSummary{DurabilityMode: DurabilityMode, Sequence: j.sequence, HeadSHA256: j.head, Complete: true}, nil
}
