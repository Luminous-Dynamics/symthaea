package packager

import (
	"crypto/ed25519"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"os"
	"path/filepath"
	"testing"
	"time"

	custody "luminousdynamics/mon026-custody"
)

func testIndex(now time.Time) InputIndex {
	scope := sha256.Sum256([]byte("scope"))
	inventory := sha256.Sum256([]byte("inventory"))
	return InputIndex{
		Profile: InputProfile,
		Version: "1.0.0",
		RunID: "mon025-run-001",
		Sequence: 1,
		CapturedAt: now,
		ClockDomain: "lab-clock",
		ClockUncertaintyNS: int64(5*time.Millisecond),
		Source: custody.SourceBoundary{
			IncidentID: "incident-001",
			SourceClusterID: "pg-cluster-001",
			Database: "ledger",
			SystemIdentifier: "7438291029384756102",
			TimelineID: "1",
			SnapshotID: "snapshot-001",
			CaptureLSN: "0/ABCDEF0",
			ScopeDigest: hex.EncodeToString(scope[:]),
			InventoryDigest: hex.EncodeToString(inventory[:]),
		},
		SignerKeyID: "custody-key-2026-01",
		Artifacts: []ArtifactInput{
			{
				Path: "run-manifest.json", MediaType: "application/json",
				Role: "run-metadata", Producer: "mon025", CommandID: "capture-run",
				CollectedAt: now.Add(-time.Second), ClockDomain: "lab-clock",
				ClockUncertaintyNS: int64(5*time.Millisecond),
			},
			{
				Path: "state/source.json", MediaType: "application/json",
				Role: "source-truth", Producer: "mon025", CommandID: "capture-source-state",
				CollectedAt: now.Add(-time.Second), ClockDomain: "lab-clock",
				ClockUncertaintyNS: int64(5*time.Millisecond),
			},
		},
	}
}

func writeFixture(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	for name, value := range map[string]string{
		"run-manifest.json": "{\"run_id\":\"mon025-run-001\"}\n",
		"state/source.json": "{\"ledger_digest\":\"abc123\"}\n",
	} {
		full := filepath.Join(root, filepath.FromSlash(name))
		if err := os.MkdirAll(filepath.Dir(full), 0o700); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(full, []byte(value), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	return root
}

func TestBuildManifestAndSigningRequest(t *testing.T) {
	now := time.Date(2026, 11, 27, 10, 0, 0, 0, time.UTC)
	root := writeFixture(t)
	result, err := BuildManifest(root, testIndex(now))
	if err != nil {
		t.Fatal(err)
	}
	if result.Manifest.RootSHA256 == "" || len(result.Manifest.Artifacts) != 2 {
		t.Fatalf("manifest root or artifact inventory missing: %+v", result.Manifest)
	}
	if result.Manifest.Signature.Value != "" {
		t.Fatal("assembler must not create a signature")
	}
	if result.SigningRequest.RequestID == "" || result.SigningRequest.RootSHA256 != result.Manifest.RootSHA256 {
		t.Fatalf("signing request is not bound to manifest root: %+v", result.SigningRequest)
	}
	message, err := base64.StdEncoding.DecodeString(result.SigningRequest.MessageBase64)
	if err != nil || len(message) == 0 {
		t.Fatalf("invalid signing message: %v", err)
	}
}

func TestBuildRejectsUnlistedFile(t *testing.T) {
	now := time.Date(2026, 11, 27, 10, 0, 0, 0, time.UTC)
	root := writeFixture(t)
	if err := os.WriteFile(filepath.Join(root, "extra.log"), []byte("unlisted"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := BuildManifest(root, testIndex(now)); err == nil {
		t.Fatal("unlisted artifact file must be rejected")
	}
}

func TestBuildRejectsTraversal(t *testing.T) {
	now := time.Date(2026, 11, 27, 10, 0, 0, 0, time.UTC)
	index := testIndex(now)
	index.Artifacts[0].Path = "../escape.json"
	if _, err := BuildManifest(t.TempDir(), index); err == nil {
		t.Fatal("path traversal must be rejected")
	}
}

func TestBuildRejectsDuplicateAllowlistPath(t *testing.T) {
	now := time.Date(2026, 11, 27, 10, 0, 0, 0, time.UTC)
	root := writeFixture(t)
	index := testIndex(now)
	index.Artifacts[1].Path = index.Artifacts[0].Path
	if _, err := BuildManifest(root, index); err == nil {
		t.Fatal("duplicate artifact paths must be rejected")
	}
}

func TestAttachExternalSignatureAndVerifyBundle(t *testing.T) {
	now := time.Date(2026, 11, 27, 10, 0, 0, 0, time.UTC)
	root := writeFixture(t)
	built, err := BuildManifest(root, testIndex(now))
	if err != nil {
		t.Fatal(err)
	}
	publicKey, privateKey, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	message, err := base64.StdEncoding.DecodeString(built.SigningRequest.MessageBase64)
	if err != nil {
		t.Fatal(err)
	}
	signature := ed25519.Sign(privateKey, message)
	envelope := SignatureEnvelope{
		Profile: SignatureEnvelopeProfile,
		Version: "1.0.0",
		RequestID: built.SigningRequest.RequestID,
		RunID: built.Manifest.RunID,
		Sequence: built.Manifest.Sequence,
		RootSHA256: built.Manifest.RootSHA256,
		Algorithm: built.SigningRequest.Algorithm,
		KeyID: built.SigningRequest.KeyID,
		Value: base64.StdEncoding.EncodeToString(signature),
	}
	policy := custody.Policy{
		Now: now.Add(time.Minute),
		MaxAge: time.Hour,
		ClockDomain: "lab-clock",
		MaxClockUncertaintyNS: int64(100*time.Millisecond),
		ExpectedRunID: built.Manifest.RunID,
		ExpectedSequence: 1,
		TrustedKeys: map[string]ed25519.PublicKey{
			envelope.KeyID: publicKey,
		},
	}
	manifest, result := AttachSignature(root, built.Manifest, envelope, policy)
	if result.Verdict != custody.VerdictQualified {
		t.Fatalf("detached signature should qualify against independent trust: %+v", result)
	}
	if manifest.Signature.Value == "" {
		t.Fatal("signed manifest must retain detached signature value")
	}
}

func TestAttachBlocksAfterArtifactMutation(t *testing.T) {
	now := time.Date(2026, 11, 27, 10, 0, 0, 0, time.UTC)
	root := writeFixture(t)
	built, err := BuildManifest(root, testIndex(now))
	if err != nil {
		t.Fatal(err)
	}
	publicKey, privateKey, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	message, _ := base64.StdEncoding.DecodeString(built.SigningRequest.MessageBase64)
	signature := ed25519.Sign(privateKey, message)
	envelope := SignatureEnvelope{
		Profile: SignatureEnvelopeProfile, Version: "1.0.0",
		RequestID: built.SigningRequest.RequestID,
		RunID: built.Manifest.RunID, Sequence: built.Manifest.Sequence,
		RootSHA256: built.Manifest.RootSHA256, Algorithm: "Ed25519",
		KeyID: built.SigningRequest.KeyID, Value: base64.StdEncoding.EncodeToString(signature),
	}
	if err := os.WriteFile(filepath.Join(root, "state", "source.json"), []byte("changed after assemble"), 0o600); err != nil {
		t.Fatal(err)
	}
	policy := custody.Policy{
		Now: now.Add(time.Minute), MaxAge: time.Hour, ClockDomain: "lab-clock",
		MaxClockUncertaintyNS: int64(100*time.Millisecond),
		ExpectedRunID: built.Manifest.RunID, ExpectedSequence: 1,
		TrustedKeys: map[string]ed25519.PublicKey{envelope.KeyID: publicKey},
	}
	_, result := AttachSignature(root, built.Manifest, envelope, policy)
	if result.Verdict != custody.VerdictBlocked {
		t.Fatalf("post-assembly artifact mutation must block signature attachment: %+v", result)
	}
}

func TestAttachRejectsMismatchedSigningRequestID(t *testing.T) {
	now := time.Date(2026, 11, 27, 10, 0, 0, 0, time.UTC)
	root := writeFixture(t)
	built, err := BuildManifest(root, testIndex(now))
	if err != nil {
		t.Fatal(err)
	}
	envelope := SignatureEnvelope{
		Profile: SignatureEnvelopeProfile, Version: "1.0.0",
		RequestID: "wrong-request-id",
		RunID: built.Manifest.RunID, Sequence: built.Manifest.Sequence,
		RootSHA256: built.Manifest.RootSHA256, Algorithm: "Ed25519",
		KeyID: built.Manifest.Signature.KeyID, Value: "not-used",
	}
	_, result := AttachSignature(root, built.Manifest, envelope, custody.Policy{})
	if result.Verdict != custody.VerdictBlocked {
		t.Fatalf("detached signature from another request must block: %+v", result)
	}
}
