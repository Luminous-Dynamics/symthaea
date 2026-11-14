package checkpoints

import (
	"crypto/ed25519"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"strings"
)

const SchemaVersion = "mace-checkpoint/v1"

var (
	ErrInvalidCheckpoint   = errors.New("invalid checkpoint")
	ErrInvalidSignature    = errors.New("checkpoint signature invalid")
	ErrWrongDatabase       = errors.New("checkpoint database identity mismatch")
	ErrWrongKey            = errors.New("checkpoint signing key identity mismatch")
	ErrRollback            = errors.New("checkpoint rollback detected")
	ErrFork                = errors.New("checkpoint sequence fork detected")
	ErrGap                 = errors.New("checkpoint sequence gap detected")
	ErrBrokenChain         = errors.New("checkpoint predecessor digest mismatch")
	ErrNoProgress          = errors.New("checkpoint does not advance journal head")
	ErrInvalidTrustedState = errors.New("invalid trusted checkpoint state")
	ErrObservedHeadMismatch = errors.New("observed journal head does not match signed checkpoint")
)

const (
	checkpointDomain = "LUMINOUS-MACE-CHECKPOINT\x00v1\x00"
	digestDomain     = "LUMINOUS-MACE-CHECKPOINT-DIGEST\x00v1\x00"
)

// Checkpoint is a detached statement about a journal head. The private key
// must be held by a witness outside the database writer's trust domain.
// CheckpointSequence tracks the witness's append-only sequence; JournalSequence
// binds the statement to the destination journal's current terminal-record head.
type Checkpoint struct {
	SchemaVersion            string `json:"schema_version"`
	KeyID                    string `json:"key_id"`
	DatabaseID               string `json:"database_id"`
	CheckpointSequence       uint64 `json:"checkpoint_sequence"`
	JournalSequence          uint64 `json:"journal_sequence"`
	JournalHeadSHA256        string `json:"journal_head_sha256"`
	PreviousCheckpointSHA256 string `json:"previous_checkpoint_sha256,omitempty"`
	IssuedAtUnix              int64  `json:"issued_at_unix"`
}

// SignedCheckpoint is the transport envelope. encoding/json represents the
// signature as base64; the signature is over SigningBytes, not serialized JSON.
type SignedCheckpoint struct {
	Checkpoint Checkpoint `json:"checkpoint"`
	Signature  []byte     `json:"signature"`
}

// TrustedState is the consumer's monotonic checkpoint cursor. It MUST be
// persisted outside the database and outside any administrator domain able to
// rewrite the database. Keeping it beside the database defeats rollback checks.
type TrustedState struct {
	KeyID                  string `json:"key_id"`
	DatabaseID             string `json:"database_id"`
	CheckpointSequence     uint64 `json:"checkpoint_sequence"`
	JournalSequence        uint64 `json:"journal_sequence"`
	JournalHeadSHA256      string `json:"journal_head_sha256"`
	CheckpointDigestSHA256 string `json:"checkpoint_digest_sha256"`
	IssuedAtUnix           int64  `json:"issued_at_unix"`
}

func (s TrustedState) Validate() error {
	if s.KeyID == "" || len(s.KeyID) > 128 || strings.TrimSpace(s.KeyID) != s.KeyID {
		return fmt.Errorf("%w: invalid key identity", ErrInvalidTrustedState)
	}
	if s.DatabaseID == "" || len(s.DatabaseID) > 256 || strings.TrimSpace(s.DatabaseID) != s.DatabaseID {
		return fmt.Errorf("%w: invalid database identity", ErrInvalidTrustedState)
	}
	if s.CheckpointSequence == 0 || s.JournalSequence == 0 || s.IssuedAtUnix <= 0 {
		return fmt.Errorf("%w: sequence and issue time must be positive", ErrInvalidTrustedState)
	}
	for name, value := range map[string]string{
		"journal head":      s.JournalHeadSHA256,
		"checkpoint digest": s.CheckpointDigestSHA256,
	} {
		if len(value) != sha256.Size*2 || strings.ToLower(value) != value {
			return fmt.Errorf("%w: %s must be a lowercase SHA-256 digest", ErrInvalidTrustedState, name)
		}
		decoded, err := hex.DecodeString(value)
		if err != nil || len(decoded) != sha256.Size {
			return fmt.Errorf("%w: malformed %s", ErrInvalidTrustedState, name)
		}
	}
	return nil
}

// ObservedHead is read from the destination by an observer independent of the
// checkpoint signer. All fields must come from one consistent database snapshot.
type ObservedHead struct {
	DatabaseID        string
	JournalSequence   uint64
	JournalHeadSHA256 string
}

type AcceptResult uint8

const (
	AcceptedAdvanced AcceptResult = iota + 1
	AcceptedAlreadyCurrent
)

func (r AcceptResult) String() string {
	switch r {
	case AcceptedAdvanced:
		return "advanced"
	case AcceptedAlreadyCurrent:
		return "already_current"
	default:
		return "unknown"
	}
}

func validateHexDigest(name, value string, optional bool) error {
	if optional && value == "" {
		return nil
	}
	if len(value) != sha256.Size*2 || strings.ToLower(value) != value {
		return fmt.Errorf("%w: %s must be a lowercase SHA-256 hex digest", ErrInvalidCheckpoint, name)
	}
	decoded, err := hex.DecodeString(value)
	if err != nil || len(decoded) != sha256.Size {
		return fmt.Errorf("%w: malformed %s", ErrInvalidCheckpoint, name)
	}
	return nil
}

func (c Checkpoint) Validate() error {
	if c.SchemaVersion != SchemaVersion {
		return fmt.Errorf("%w: unsupported schema version %q", ErrInvalidCheckpoint, c.SchemaVersion)
	}
	if c.KeyID == "" || len(c.KeyID) > 128 || strings.TrimSpace(c.KeyID) != c.KeyID {
		return fmt.Errorf("%w: key_id must be 1..128 trimmed bytes", ErrInvalidCheckpoint)
	}
	if c.DatabaseID == "" || len(c.DatabaseID) > 256 || strings.TrimSpace(c.DatabaseID) != c.DatabaseID {
		return fmt.Errorf("%w: database_id must be 1..256 trimmed bytes", ErrInvalidCheckpoint)
	}
	if c.CheckpointSequence == 0 || c.JournalSequence == 0 {
		return fmt.Errorf("%w: checkpoint and journal sequences must be positive", ErrInvalidCheckpoint)
	}
	if c.IssuedAtUnix <= 0 {
		return fmt.Errorf("%w: issued_at_unix must be positive", ErrInvalidCheckpoint)
	}
	if err := validateHexDigest("journal_head_sha256", c.JournalHeadSHA256, false); err != nil {
		return err
	}
	if err := validateHexDigest("previous_checkpoint_sha256", c.PreviousCheckpointSHA256, true); err != nil {
		return err
	}
	if c.CheckpointSequence == 1 && c.PreviousCheckpointSHA256 != "" {
		return fmt.Errorf("%w: genesis checkpoint must not have a predecessor", ErrInvalidCheckpoint)
	}
	if c.CheckpointSequence > 1 && c.PreviousCheckpointSHA256 == "" {
		return fmt.Errorf("%w: non-genesis checkpoint requires a predecessor", ErrInvalidCheckpoint)
	}
	return nil
}

func writeLengthPrefixed(dst []byte, value string) []byte {
	var size [4]byte
	binary.BigEndian.PutUint32(size[:], uint32(len(value)))
	dst = append(dst, size[:]...)
	return append(dst, value...)
}

// SigningBytes is a versioned, language-independent binary representation.
// It avoids relying on map ordering or a particular JSON implementation.
func (c Checkpoint) SigningBytes() ([]byte, error) {
	if err := c.Validate(); err != nil {
		return nil, err
	}
	out := make([]byte, 0, 256)
	out = append(out, checkpointDomain...)
	out = writeLengthPrefixed(out, c.SchemaVersion)
	out = writeLengthPrefixed(out, c.KeyID)
	out = writeLengthPrefixed(out, c.DatabaseID)
	var number [8]byte
	binary.BigEndian.PutUint64(number[:], c.CheckpointSequence)
	out = append(out, number[:]...)
	binary.BigEndian.PutUint64(number[:], c.JournalSequence)
	out = append(out, number[:]...)
	head, _ := hex.DecodeString(c.JournalHeadSHA256)
	out = append(out, head...)
	previous := []byte{}
	if c.PreviousCheckpointSHA256 != "" {
		previous, _ = hex.DecodeString(c.PreviousCheckpointSHA256)
	}
	out = append(out, byte(len(previous))) // 0 for genesis, 32 otherwise
	out = append(out, previous...)
	binary.BigEndian.PutUint64(number[:], uint64(c.IssuedAtUnix))
	out = append(out, number[:]...)
	return out, nil
}

func Sign(c Checkpoint, privateKey ed25519.PrivateKey) (SignedCheckpoint, error) {
	if len(privateKey) != ed25519.PrivateKeySize {
		return SignedCheckpoint{}, errors.New("invalid Ed25519 private key size")
	}
	message, err := c.SigningBytes()
	if err != nil {
		return SignedCheckpoint{}, err
	}
	signature := ed25519.Sign(privateKey, message)
	return SignedCheckpoint{Checkpoint: c, Signature: signature}, nil
}

func (s SignedCheckpoint) Verify(publicKey ed25519.PublicKey) error {
	if len(publicKey) != ed25519.PublicKeySize || len(s.Signature) != ed25519.SignatureSize {
		return ErrInvalidSignature
	}
	message, err := s.Checkpoint.SigningBytes()
	if err != nil {
		return err
	}
	if !ed25519.Verify(publicKey, message, s.Signature) {
		return ErrInvalidSignature
	}
	return nil
}

// DigestSHA256 binds both the canonical signed fields and detached signature.
func (s SignedCheckpoint) DigestSHA256() (string, error) {
	message, err := s.Checkpoint.SigningBytes()
	if err != nil {
		return "", err
	}
	if len(s.Signature) != ed25519.SignatureSize {
		return "", ErrInvalidSignature
	}
	h := sha256.New()
	_, _ = h.Write([]byte(digestDomain))
	_, _ = h.Write(message)
	_, _ = h.Write(s.Signature)
	return hex.EncodeToString(h.Sum(nil)), nil
}

// Bootstrap verifies an initial trusted checkpoint against an out-of-band
// public key and expected identities. Bootstrap does not prove freshness:
// provisioning must obtain this initial anchor from a trusted process.
func Bootstrap(s SignedCheckpoint, publicKey ed25519.PublicKey, expectedKeyID, expectedDatabaseID string) (TrustedState, error) {
	if err := s.Verify(publicKey); err != nil {
		return TrustedState{}, err
	}
	c := s.Checkpoint
	if c.KeyID != expectedKeyID {
		return TrustedState{}, ErrWrongKey
	}
	if c.DatabaseID != expectedDatabaseID {
		return TrustedState{}, ErrWrongDatabase
	}
	digest, err := s.DigestSHA256()
	if err != nil {
		return TrustedState{}, err
	}
	return TrustedState{
		KeyID: c.KeyID, DatabaseID: c.DatabaseID,
		CheckpointSequence: c.CheckpointSequence, JournalSequence: c.JournalSequence,
		JournalHeadSHA256: c.JournalHeadSHA256,
		CheckpointDigestSHA256: digest, IssuedAtUnix: c.IssuedAtUnix,
	}, nil
}

// AcceptNext verifies a strictly next signed checkpoint. The caller persists
// the returned TrustedState atomically in a separate trust domain. Replaying
// the exact current envelope is idempotent; older checkpoints, forks, skipped
// checkpoint numbers, changed identities and non-advancing journal heads fail.
func AcceptNext(s SignedCheckpoint, publicKey ed25519.PublicKey, current TrustedState) (TrustedState, AcceptResult, error) {
	if err := current.Validate(); err != nil {
		return TrustedState{}, 0, err
	}
	if err := s.Verify(publicKey); err != nil {
		return TrustedState{}, 0, err
	}
	c := s.Checkpoint
	if c.KeyID != current.KeyID {
		return TrustedState{}, 0, ErrWrongKey
	}
	if c.DatabaseID != current.DatabaseID {
		return TrustedState{}, 0, ErrWrongDatabase
	}
	digest, err := s.DigestSHA256()
	if err != nil {
		return TrustedState{}, 0, err
	}

	if c.CheckpointSequence < current.CheckpointSequence {
		return TrustedState{}, 0, ErrRollback
	}
	if c.CheckpointSequence == current.CheckpointSequence {
		if digest == current.CheckpointDigestSHA256 {
			return current, AcceptedAlreadyCurrent, nil
		}
		return TrustedState{}, 0, ErrFork
	}
	if c.CheckpointSequence != current.CheckpointSequence+1 {
		return TrustedState{}, 0, ErrGap
	}
	if c.PreviousCheckpointSHA256 != current.CheckpointDigestSHA256 {
		return TrustedState{}, 0, ErrBrokenChain
	}
	if c.JournalSequence <= current.JournalSequence || c.JournalHeadSHA256 == current.JournalHeadSHA256 {
		return TrustedState{}, 0, ErrNoProgress
	}
	if c.IssuedAtUnix < current.IssuedAtUnix {
		return TrustedState{}, 0, ErrRollback
	}

	return TrustedState{
		KeyID: c.KeyID, DatabaseID: c.DatabaseID,
		CheckpointSequence: c.CheckpointSequence, JournalSequence: c.JournalSequence,
		JournalHeadSHA256: c.JournalHeadSHA256,
		CheckpointDigestSHA256: digest, IssuedAtUnix: c.IssuedAtUnix,
	}, AcceptedAdvanced, nil
}


// AcceptObserved first joins a signed checkpoint to an independently observed
// destination snapshot, then applies the monotonic checkpoint-chain rules. It
// prevents callers from accepting a validly signed checkpoint for a different
// database head by forgetting to compare the observed sequence and digest. The
// observer must obtain all three fields from one coherent snapshot.
func AcceptObserved(s SignedCheckpoint, publicKey ed25519.PublicKey, current TrustedState, observed ObservedHead) (TrustedState, AcceptResult, error) {
	if observed.DatabaseID == "" || observed.DatabaseID != s.Checkpoint.DatabaseID {
		return TrustedState{}, 0, ErrWrongDatabase
	}
	if observed.JournalSequence == 0 || validateHexDigest("observed journal head", observed.JournalHeadSHA256, false) != nil {
		return TrustedState{}, 0, ErrObservedHeadMismatch
	}
	if observed.JournalSequence != s.Checkpoint.JournalSequence || observed.JournalHeadSHA256 != s.Checkpoint.JournalHeadSHA256 {
		return TrustedState{}, 0, ErrObservedHeadMismatch
	}
	return AcceptNext(s, publicKey, current)
}
