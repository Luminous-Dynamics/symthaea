package checkpoints

import (
	"bytes"
	"crypto/ed25519"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"os"
	"testing"
)

func testKeys(t *testing.T) (ed25519.PublicKey, ed25519.PrivateKey) {
	t.Helper()
	pub, priv, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	return pub, priv
}

func hashLabel(s string) string {
	d := sha256.Sum256([]byte(s))
	return hex.EncodeToString(d[:])
}

func makeCheckpoint(sequence, journalSequence uint64, previous string, label string, issued int64) Checkpoint {
	return Checkpoint{
		SchemaVersion:            SchemaVersion,
		KeyID:                    "witness-key-1",
		DatabaseID:               "payments-db-prod-1",
		CheckpointSequence:       sequence,
		JournalSequence:          journalSequence,
		JournalHeadSHA256:        hashLabel(label),
		PreviousCheckpointSHA256: previous,
		IssuedAtUnix:             issued,
	}
}

func signCheckpoint(t *testing.T, c Checkpoint, priv ed25519.PrivateKey) SignedCheckpoint {
	t.Helper()
	s, err := Sign(c, priv)
	if err != nil {
		t.Fatal(err)
	}
	return s
}

func bootstrapTest(t *testing.T, priv ed25519.PrivateKey, pub ed25519.PublicKey) (SignedCheckpoint, TrustedState) {
	t.Helper()
	genesis := signCheckpoint(t, makeCheckpoint(1, 4, "", "genesis", 100), priv)
	state, err := Bootstrap(genesis, pub, "witness-key-1", "payments-db-prod-1")
	if err != nil {
		t.Fatal(err)
	}
	return genesis, state
}

func TestCheckpointSignatureAndMonotonicAdvance(t *testing.T) {
	pub, priv := testKeys(t)
	_, state := bootstrapTest(t, priv, pub)
	next := signCheckpoint(t, makeCheckpoint(2, 9, state.CheckpointDigestSHA256, "head-9", 101), priv)
	advanced, result, err := AcceptNext(next, pub, state)
	if err != nil {
		t.Fatal(err)
	}
	if result != AcceptedAdvanced || advanced.CheckpointSequence != 2 || advanced.JournalSequence != 9 {
		t.Fatalf("unexpected advance: result=%s state=%+v", result, advanced)
	}
}

func TestExactCheckpointReplayIsIdempotent(t *testing.T) {
	pub, priv := testKeys(t)
	genesis, state := bootstrapTest(t, priv, pub)
	replayed, result, err := AcceptNext(genesis, pub, state)
	if err != nil {
		t.Fatal(err)
	}
	if result != AcceptedAlreadyCurrent || replayed.CheckpointDigestSHA256 != state.CheckpointDigestSHA256 {
		t.Fatalf("unexpected exact-replay result: %s %+v", result, replayed)
	}
}

func TestOlderSignedCheckpointIsRollback(t *testing.T) {
	pub, priv := testKeys(t)
	genesis, state := bootstrapTest(t, priv, pub)
	next := signCheckpoint(t, makeCheckpoint(2, 9, state.CheckpointDigestSHA256, "head-9", 101), priv)
	advanced, _, err := AcceptNext(next, pub, state)
	if err != nil {
		t.Fatal(err)
	}
	_, _, err = AcceptNext(genesis, pub, advanced)
	if !errors.Is(err, ErrRollback) {
		t.Fatalf("expected rollback, got %v", err)
	}
}

func TestSameSequenceDifferentSignedHeadIsFork(t *testing.T) {
	pub, priv := testKeys(t)
	_, state := bootstrapTest(t, priv, pub)
	fork := signCheckpoint(t, makeCheckpoint(1, 4, "", "rewritten-genesis", 100), priv)
	_, _, err := AcceptNext(fork, pub, state)
	if !errors.Is(err, ErrFork) {
		t.Fatalf("expected fork, got %v", err)
	}
}

func TestTamperedCheckpointFieldFailsSignature(t *testing.T) {
	pub, priv := testKeys(t)
	_, state := bootstrapTest(t, priv, pub)
	next := signCheckpoint(t, makeCheckpoint(2, 9, state.CheckpointDigestSHA256, "head-9", 101), priv)
	next.Checkpoint.JournalHeadSHA256 = hashLabel("tampered")
	_, _, err := AcceptNext(next, pub, state)
	if !errors.Is(err, ErrInvalidSignature) {
		t.Fatalf("expected signature failure, got %v", err)
	}
}

func TestTamperedSignatureFails(t *testing.T) {
	pub, priv := testKeys(t)
	genesis, _ := bootstrapTest(t, priv, pub)
	genesis.Signature[0] ^= 0xff
	_, err := Bootstrap(genesis, pub, "witness-key-1", "payments-db-prod-1")
	if !errors.Is(err, ErrInvalidSignature) {
		t.Fatalf("expected signature failure, got %v", err)
	}
}

func TestWrongDatabaseAndKeyAreRejected(t *testing.T) {
	pub, priv := testKeys(t)
	genesis := signCheckpoint(t, makeCheckpoint(1, 4, "", "genesis", 100), priv)
	if _, err := Bootstrap(genesis, pub, "witness-key-1", "other-db"); !errors.Is(err, ErrWrongDatabase) {
		t.Fatalf("expected database mismatch, got %v", err)
	}
	if _, err := Bootstrap(genesis, pub, "other-key", "payments-db-prod-1"); !errors.Is(err, ErrWrongKey) {
		t.Fatalf("expected key mismatch, got %v", err)
	}
}

func TestBrokenPredecessorIsRejected(t *testing.T) {
	pub, priv := testKeys(t)
	_, state := bootstrapTest(t, priv, pub)
	broken := signCheckpoint(t, makeCheckpoint(2, 9, hashLabel("wrong predecessor"), "head-9", 101), priv)
	_, _, err := AcceptNext(broken, pub, state)
	if !errors.Is(err, ErrBrokenChain) {
		t.Fatalf("expected predecessor mismatch, got %v", err)
	}
}

func TestCheckpointGapIsRejected(t *testing.T) {
	pub, priv := testKeys(t)
	_, state := bootstrapTest(t, priv, pub)
	gap := signCheckpoint(t, makeCheckpoint(3, 12, state.CheckpointDigestSHA256, "head-12", 102), priv)
	_, _, err := AcceptNext(gap, pub, state)
	if !errors.Is(err, ErrGap) {
		t.Fatalf("expected checkpoint gap, got %v", err)
	}
}

func TestJournalSequenceMustAdvance(t *testing.T) {
	pub, priv := testKeys(t)
	_, state := bootstrapTest(t, priv, pub)
	stale := signCheckpoint(t, makeCheckpoint(2, 4, state.CheckpointDigestSHA256, "same-journal-head", 101), priv)
	_, _, err := AcceptNext(stale, pub, state)
	if !errors.Is(err, ErrNoProgress) {
		t.Fatalf("expected no-progress error, got %v", err)
	}
}

func TestIssuedTimeCannotMoveBackward(t *testing.T) {
	pub, priv := testKeys(t)
	_, state := bootstrapTest(t, priv, pub)
	staleTime := signCheckpoint(t, makeCheckpoint(2, 9, state.CheckpointDigestSHA256, "head-9", 99), priv)
	_, _, err := AcceptNext(staleTime, pub, state)
	if !errors.Is(err, ErrRollback) {
		t.Fatalf("expected monotonic-time rollback, got %v", err)
	}
}

func TestGenesisAndDigestValidation(t *testing.T) {
	_, priv := testKeys(t)
	bad := makeCheckpoint(1, 4, hashLabel("should not exist"), "genesis", 100)
	if _, err := Sign(bad, priv); !errors.Is(err, ErrInvalidCheckpoint) {
		t.Fatalf("expected invalid genesis rejection, got %v", err)
	}
	good := signCheckpoint(t, makeCheckpoint(1, 4, "", "genesis", 100), priv)
	if _, err := good.DigestSHA256(); err != nil {
		t.Fatal(err)
	}
}

func TestInvalidPrivateKeyRejected(t *testing.T) {
	_, err := Sign(makeCheckpoint(1, 4, "", "genesis", 100), []byte("bad"))
	if err == nil {
		t.Fatal("expected invalid private-key error")
	}
}

func TestSignedCheckpointJSONRoundTrip(t *testing.T) {
	pub, priv := testKeys(t)
	genesis := signCheckpoint(t, makeCheckpoint(1, 4, "", "genesis", 100), priv)
	wire, err := json.Marshal(genesis)
	if err != nil {
		t.Fatal(err)
	}
	var decoded SignedCheckpoint
	if err := json.Unmarshal(wire, &decoded); err != nil {
		t.Fatal(err)
	}
	if err := decoded.Verify(pub); err != nil {
		t.Fatalf("round-tripped checkpoint did not verify: %v", err)
	}
	before, err := genesis.DigestSHA256()
	if err != nil {
		t.Fatal(err)
	}
	after, err := decoded.DigestSHA256()
	if err != nil {
		t.Fatal(err)
	}
	if before != after {
		t.Fatalf("JSON round trip changed signed checkpoint digest: before=%s after=%s", before, after)
	}
}

func TestPublishedCheckpointVector(t *testing.T) {
	type vector struct {
		Checkpoint           Checkpoint `json:"checkpoint"`
		PublicKeyHex         string     `json:"public_key_hex"`
		SigningBytesHex      string     `json:"signing_bytes_hex"`
		SignatureHex         string     `json:"signature_hex"`
		SignatureBase64      string     `json:"signature_base64"`
		EnvelopeDigestSHA256 string     `json:"envelope_digest_sha256"`
	}
	wire, err := os.ReadFile("testdata/checkpoint-v1.json")
	if err != nil {
		t.Fatal(err)
	}
	var v vector
	if err := json.Unmarshal(wire, &v); err != nil {
		t.Fatal(err)
	}
	publicKey, err := hex.DecodeString(v.PublicKeyHex)
	if err != nil {
		t.Fatal(err)
	}
	signature, err := hex.DecodeString(v.SignatureHex)
	if err != nil {
		t.Fatal(err)
	}
	wantSigningBytes, err := hex.DecodeString(v.SigningBytesHex)
	if err != nil {
		t.Fatal(err)
	}
	gotSigningBytes, err := v.Checkpoint.SigningBytes()
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(gotSigningBytes, wantSigningBytes) {
		t.Fatalf("signing bytes diverged from published vector")
	}
	if !ed25519.Verify(ed25519.PublicKey(publicKey), gotSigningBytes, signature) {
		t.Fatal("published Ed25519 signature did not verify")
	}
	signed := SignedCheckpoint{Checkpoint: v.Checkpoint, Signature: signature}
	digest, err := signed.DigestSHA256()
	if err != nil {
		t.Fatal(err)
	}
	if digest != v.EnvelopeDigestSHA256 {
		t.Fatalf("envelope digest mismatch: got %s want %s", digest, v.EnvelopeDigestSHA256)
	}
	jsonEnvelope, err := json.Marshal(signed)
	if err != nil {
		t.Fatal(err)
	}
	var envelope map[string]json.RawMessage
	if err := json.Unmarshal(jsonEnvelope, &envelope); err != nil {
		t.Fatal(err)
	}
	var encodedSignature string
	if err := json.Unmarshal(envelope["signature"], &encodedSignature); err != nil {
		t.Fatal(err)
	}
	if encodedSignature != v.SignatureBase64 {
		t.Fatalf("base64 signature mismatch: got %s want %s", encodedSignature, v.SignatureBase64)
	}
}

func TestCorruptTrustedCursorIsRejected(t *testing.T) {
	pub, priv := testKeys(t)
	_, state := bootstrapTest(t, priv, pub)
	state.CheckpointDigestSHA256 = "not-a-digest"
	next := signCheckpoint(t, makeCheckpoint(2, 9, hashLabel("trusted"), "head-9", 101), priv)
	_, _, err := AcceptNext(next, pub, state)
	if !errors.Is(err, ErrInvalidTrustedState) {
		t.Fatalf("expected corrupt trust cursor rejection, got %v", err)
	}
}


func TestAcceptObservedBindsCheckpointToExactSnapshot(t *testing.T) {
	pub, priv := testKeys(t)
	_, state := bootstrapTest(t, priv, pub)
	next := signCheckpoint(t, makeCheckpoint(2, 9, state.CheckpointDigestSHA256, "head-9", 101), priv)
	observed := ObservedHead{DatabaseID: "payments-db-prod-1", JournalSequence: 9, JournalHeadSHA256: hashLabel("head-9")}
	advanced, result, err := AcceptObserved(next, pub, state, observed)
	if err != nil {
		t.Fatal(err)
	}
	if result != AcceptedAdvanced || advanced.CheckpointSequence != 2 {
		t.Fatalf("unexpected accepted observed checkpoint: %s %+v", result, advanced)
	}
}

func TestAcceptObservedRejectsWrongJournalHead(t *testing.T) {
	pub, priv := testKeys(t)
	_, state := bootstrapTest(t, priv, pub)
	next := signCheckpoint(t, makeCheckpoint(2, 9, state.CheckpointDigestSHA256, "head-9", 101), priv)
	observed := ObservedHead{DatabaseID: "payments-db-prod-1", JournalSequence: 9, JournalHeadSHA256: hashLabel("rewritten-head")}
	_, _, err := AcceptObserved(next, pub, state, observed)
	if !errors.Is(err, ErrObservedHeadMismatch) {
		t.Fatalf("expected observed head mismatch, got %v", err)
	}
}

func TestAcceptObservedRejectsWrongJournalSequence(t *testing.T) {
	pub, priv := testKeys(t)
	_, state := bootstrapTest(t, priv, pub)
	next := signCheckpoint(t, makeCheckpoint(2, 9, state.CheckpointDigestSHA256, "head-9", 101), priv)
	observed := ObservedHead{DatabaseID: "payments-db-prod-1", JournalSequence: 10, JournalHeadSHA256: hashLabel("head-9")}
	_, _, err := AcceptObserved(next, pub, state, observed)
	if !errors.Is(err, ErrObservedHeadMismatch) {
		t.Fatalf("expected observed sequence mismatch, got %v", err)
	}
}

func TestAcceptObservedRejectsWrongDatabase(t *testing.T) {
	pub, priv := testKeys(t)
	_, state := bootstrapTest(t, priv, pub)
	next := signCheckpoint(t, makeCheckpoint(2, 9, state.CheckpointDigestSHA256, "head-9", 101), priv)
	observed := ObservedHead{DatabaseID: "other-database", JournalSequence: 9, JournalHeadSHA256: hashLabel("head-9")}
	_, _, err := AcceptObserved(next, pub, state, observed)
	if !errors.Is(err, ErrWrongDatabase) {
		t.Fatalf("expected database mismatch, got %v", err)
	}
}


func TestJournalHeadMustChangeWhenSequenceAdvances(t *testing.T) {
	pub, priv := testKeys(t)
	_, state := bootstrapTest(t, priv, pub)
	staleHead := signCheckpoint(t, makeCheckpoint(2, 9, state.CheckpointDigestSHA256, "genesis", 101), priv)
	_, _, err := AcceptNext(staleHead, pub, state)
	if !errors.Is(err, ErrNoProgress) {
		t.Fatalf("expected repeated journal head to be rejected, got %v", err)
	}
}
