package witness

import (
	"bufio"
	"bytes"
	"crypto/ed25519"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/Luminous-Dynamics/symthaea/research/mace-evalqual-v0.1/checkpoints"
)

const schemaVersion = "mace-witness-journal/v1"
const maxRecordBytes = 1 << 20

var (
	ErrEquivocation       = errors.New("witness refuses a conflicting checkpoint for the same journal progress")
	ErrWrongDatabase      = errors.New("observed database identity does not match witness configuration")
	ErrInvalidObservation = errors.New("invalid observed journal head")
	ErrCorruptJournal     = errors.New("witness journal is corrupt")
	ErrPoisoned           = errors.New("witness journal outcome is uncertain; restart and reconcile")
	ErrClockRollback      = errors.New("witness clock moved backward")
	ErrSequenceExhausted  = errors.New("witness checkpoint sequence exhausted")
)

type Clock func() time.Time

type Options struct {
	Path       string
	KeyID      string
	DatabaseID string
	PrivateKey ed25519.PrivateKey
	Clock      Clock
}

type journalRecord struct {
	SchemaVersion string                       `json:"schema_version"`
	Checkpoint    checkpoints.SignedCheckpoint `json:"checkpoint"`
	RecordSHA256  string                       `json:"record_sha256"`
}

type Witness struct {
	mu         sync.Mutex
	file       *os.File
	path       string
	keyID      string
	databaseID string
	publicKey  ed25519.PublicKey
	privateKey ed25519.PrivateKey
	clock      Clock
	last       *checkpoints.SignedCheckpoint
	poisoned   bool
	closed     bool
}

func Open(options Options) (*Witness, error) {
	if options.Path == "" || options.KeyID == "" || options.DatabaseID == "" {
		return nil, errors.New("witness path, key ID and database ID are required")
	}
	if len(options.PrivateKey) != ed25519.PrivateKeySize {
		return nil, errors.New("invalid Ed25519 private key size")
	}
	if options.Clock == nil {
		options.Clock = time.Now
	}
	if err := os.MkdirAll(filepath.Dir(options.Path), 0o700); err != nil {
		return nil, fmt.Errorf("create witness journal directory: %w", err)
	}
	file, err := os.OpenFile(options.Path, os.O_RDWR|os.O_CREATE, 0o600)
	if err != nil {
		return nil, fmt.Errorf("open witness journal: %w", err)
	}
	if err := lockFile(file); err != nil {
		_ = file.Close()
		return nil, err
	}
	w := &Witness{
		file: file, path: options.Path, keyID: options.KeyID,
		databaseID: options.DatabaseID,
		publicKey:  append(ed25519.PublicKey(nil), options.PrivateKey.Public().(ed25519.PublicKey)...),
		privateKey: append(ed25519.PrivateKey(nil), options.PrivateKey...),
		clock:      options.Clock,
	}
	if err := w.replay(); err != nil {
		_ = unlockFile(file)
		_ = file.Close()
		return nil, err
	}
	return w, nil
}

func (w *Witness) replay() error {
	if _, err := w.file.Seek(0, io.SeekStart); err != nil {
		return fmt.Errorf("seek witness journal: %w", err)
	}
	info, err := w.file.Stat()
	if err != nil {
		return fmt.Errorf("stat witness journal: %w", err)
	}
	if info.Size() == 0 {
		return nil
	}
	if _, err := w.file.Seek(-1, io.SeekEnd); err != nil {
		return fmt.Errorf("seek witness journal tail: %w", err)
	}
	var lastByte [1]byte
	if _, err := w.file.Read(lastByte[:]); err != nil {
		return fmt.Errorf("read witness journal tail: %w", err)
	}
	if lastByte[0] != '\n' {
		return fmt.Errorf("%w: unterminated final record", ErrCorruptJournal)
	}
	if _, err := w.file.Seek(0, io.SeekStart); err != nil {
		return err
	}
	scanner := bufio.NewScanner(w.file)
	scanner.Buffer(make([]byte, 64*1024), maxRecordBytes)
	line := 0
	for scanner.Scan() {
		line++
		payload := bytes.TrimSpace(scanner.Bytes())
		if len(payload) == 0 {
			return fmt.Errorf("%w: empty record at line %d", ErrCorruptJournal, line)
		}
		var record journalRecord
		decoder := json.NewDecoder(bytes.NewReader(payload))
		decoder.DisallowUnknownFields()
		if err := decoder.Decode(&record); err != nil {
			return fmt.Errorf("%w: decode line %d: %v", ErrCorruptJournal, line, err)
		}
		if err := ensureEOF(decoder); err != nil {
			return fmt.Errorf("%w: trailing data at line %d", ErrCorruptJournal, line)
		}
		if err := w.validateNext(record); err != nil {
			return fmt.Errorf("%w: line %d: %v", ErrCorruptJournal, line, err)
		}
		cp := record.Checkpoint
		w.last = &cp
	}
	if err := scanner.Err(); err != nil {
		return fmt.Errorf("%w: scan journal: %v", ErrCorruptJournal, err)
	}
	_, err = w.file.Seek(0, io.SeekEnd)
	return err
}

func (w *Witness) validateNext(record journalRecord) error {
	if record.SchemaVersion != schemaVersion {
		return errors.New("unsupported witness journal schema")
	}
	if err := record.Checkpoint.Verify(w.publicKey); err != nil {
		return fmt.Errorf("checkpoint signature: %w", err)
	}
	cp := record.Checkpoint.Checkpoint
	if cp.KeyID != w.keyID || cp.DatabaseID != w.databaseID {
		return errors.New("checkpoint identity mismatch")
	}
	digest, err := record.Checkpoint.DigestSHA256()
	if err != nil {
		return err
	}
	if record.RecordSHA256 != digest {
		return errors.New("witness record digest mismatch")
	}
	if w.last == nil {
		if cp.CheckpointSequence != 1 || cp.PreviousCheckpointSHA256 != "" {
			return errors.New("invalid witness genesis checkpoint")
		}
		return nil
	}
	previousDigest, err := w.last.DigestSHA256()
	if err != nil {
		return err
	}
	previous := w.last.Checkpoint
	if cp.CheckpointSequence != previous.CheckpointSequence+1 {
		return errors.New("witness checkpoint sequence is not contiguous")
	}
	if cp.PreviousCheckpointSHA256 != previousDigest {
		return errors.New("witness checkpoint predecessor mismatch")
	}
	if cp.JournalSequence <= previous.JournalSequence || cp.JournalHeadSHA256 == previous.JournalHeadSHA256 {
		return errors.New("witness journal head did not advance")
	}
	if cp.IssuedAtUnix < previous.IssuedAtUnix {
		return errors.New("witness issue time regressed")
	}
	return nil
}

func (w *Witness) Issue(observed checkpoints.ObservedHead) (checkpoints.SignedCheckpoint, bool, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.closed {
		return checkpoints.SignedCheckpoint{}, false, errors.New("witness is closed")
	}
	if w.poisoned {
		return checkpoints.SignedCheckpoint{}, false, ErrPoisoned
	}
	if observed.DatabaseID != w.databaseID {
		return checkpoints.SignedCheckpoint{}, false, ErrWrongDatabase
	}
	if observed.JournalSequence == 0 || !validDigest(observed.JournalHeadSHA256) {
		return checkpoints.SignedCheckpoint{}, false, ErrInvalidObservation
	}

	var sequence uint64 = 1
	var predecessor string
	if w.last != nil {
		previous := w.last.Checkpoint
		previousDigest, err := w.last.DigestSHA256()
		if err != nil {
			w.poisoned = true
			return checkpoints.SignedCheckpoint{}, false, ErrPoisoned
		}
		if observed.JournalSequence < previous.JournalSequence {
			return checkpoints.SignedCheckpoint{}, false, checkpoints.ErrRollback
		}
		if observed.JournalSequence == previous.JournalSequence {
			if observed.JournalHeadSHA256 == previous.JournalHeadSHA256 {
				return *w.last, true, nil
			}
			return checkpoints.SignedCheckpoint{}, false, ErrEquivocation
		}
		if observed.JournalHeadSHA256 == previous.JournalHeadSHA256 {
			return checkpoints.SignedCheckpoint{}, false, ErrEquivocation
		}
		if previous.CheckpointSequence == ^uint64(0) {
			return checkpoints.SignedCheckpoint{}, false, ErrSequenceExhausted
		}
		sequence = previous.CheckpointSequence + 1
		predecessor = previousDigest
	}

	issuedAt := w.clock().Unix()
	if w.last != nil && issuedAt < w.last.Checkpoint.IssuedAtUnix {
		return checkpoints.SignedCheckpoint{}, false, ErrClockRollback
	}
	checkpoint := checkpoints.Checkpoint{
		SchemaVersion: checkpoints.SchemaVersion,
		KeyID: w.keyID, DatabaseID: w.databaseID,
		CheckpointSequence: sequence,
		JournalSequence: observed.JournalSequence,
		JournalHeadSHA256: observed.JournalHeadSHA256,
		PreviousCheckpointSHA256: predecessor,
		IssuedAtUnix: issuedAt,
	}
	signed, err := checkpoints.Sign(checkpoint, w.privateKey)
	if err != nil {
		return checkpoints.SignedCheckpoint{}, false, err
	}
	digest, err := signed.DigestSHA256()
	if err != nil {
		return checkpoints.SignedCheckpoint{}, false, err
	}
	record := journalRecord{SchemaVersion: schemaVersion, Checkpoint: signed, RecordSHA256: digest}
	encoded, err := json.Marshal(record)
	if err != nil {
		return checkpoints.SignedCheckpoint{}, false, err
	}
	encoded = append(encoded, '\n')
	if err := writeAll(w.file, encoded); err != nil {
		w.poisoned = true
		return checkpoints.SignedCheckpoint{}, false, ErrPoisoned
	}
	if err := w.file.Sync(); err != nil {
		w.poisoned = true
		return checkpoints.SignedCheckpoint{}, false, ErrPoisoned
	}
	if w.last == nil {
		if err := syncDirectory(filepath.Dir(w.path)); err != nil {
			w.poisoned = true
			return checkpoints.SignedCheckpoint{}, false, ErrPoisoned
		}
	}
	cp := signed
	w.last = &cp
	return signed, false, nil
}

func (w *Witness) Current() (checkpoints.SignedCheckpoint, bool, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.closed {
		return checkpoints.SignedCheckpoint{}, false, errors.New("witness is closed")
	}
	if w.poisoned {
		return checkpoints.SignedCheckpoint{}, false, ErrPoisoned
	}
	if w.last == nil {
		return checkpoints.SignedCheckpoint{}, false, nil
	}
	return *w.last, true, nil
}

func (w *Witness) Close() error {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.closed {
		return nil
	}
	w.closed = true
	clear(w.privateKey)
	unlockErr := unlockFile(w.file)
	closeErr := w.file.Close()
	return errors.Join(unlockErr, closeErr)
}

func validDigest(value string) bool {
	decoded, err := hex.DecodeString(value)
	return err == nil && len(decoded) == sha256.Size && strings.ToLower(value) == value
}

func ensureEOF(decoder *json.Decoder) error {
	var extra any
	if err := decoder.Decode(&extra); err != io.EOF {
		if err == nil {
			return errors.New("trailing JSON value")
		}
		return err
	}
	return nil
}

func writeAll(file *os.File, data []byte) error {
	for len(data) > 0 {
		written, err := file.Write(data)
		if err != nil {
			return err
		}
		if written == 0 {
			return io.ErrShortWrite
		}
		data = data[written:]
	}
	return nil
}

func syncDirectory(path string) error {
	dir, err := os.Open(path)
	if err != nil {
		return err
	}
	defer dir.Close()
	return dir.Sync()
}
