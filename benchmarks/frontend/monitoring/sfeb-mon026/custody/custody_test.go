package custody

import (
	"crypto/ed25519"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func digestBytes(b []byte) string {
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

func testFixture(t *testing.T) (string, Manifest, ed25519.PublicKey, ed25519.PrivateKey, Policy) {
	t.Helper()
	root := t.TempDir()
	now := time.Date(2026, 11, 27, 9, 0, 0, 0, time.UTC)
	files := map[string][]byte{
		"run-manifest.json": []byte("{\"run_id\":\"run-001\"}\n"),
		"state/source.json": []byte("{\"ledger_digest\":\"abc\"}\n"),
	}
	for name, data := range files {
		full := filepath.Join(root, filepath.FromSlash(name))
		if err := os.MkdirAll(filepath.Dir(full), 0o700); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(full, data, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	pub, private, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	m := Manifest{
		Profile: ProfileV1, Version: VersionV1, RunID: "run-001", Sequence: 1,
		CapturedAt: now, ClockDomain: "lab-clock", ClockUncertaintyNS: int64(5 * time.Millisecond),
		Source: SourceBoundary{
			IncidentID: "incident-001", SourceClusterID: "pg-cluster-001", Database: "ledger",
			SystemIdentifier: "7438291029384756102", TimelineID: "1", SnapshotID: "snapshot-001",
			CaptureLSN: "0/ABCDEF0", ScopeDigest: digestBytes([]byte("scope")),
			InventoryDigest: digestBytes([]byte("inventory")),
		},
	}
	for name, data := range files {
		m.Artifacts = append(m.Artifacts, Artifact{
			Path: name, MediaType: "application/json", Role: "truth-observation",
			Producer: "mon025-lab", CommandID: "cmd-" + filepath.Base(name),
			SizeBytes: int64(len(data)), SHA256: digestBytes(data),
			CollectedAt: now.Add(-time.Second), ClockDomain: "lab-clock",
			ClockUncertaintyNS: int64(5 * time.Millisecond),
		})
	}
	if err := SignManifest(&m, "custody-key-1", private); err != nil {
		t.Fatal(err)
	}
	policy := Policy{
		Now: now.Add(time.Minute), MaxAge: time.Hour, ClockDomain: "lab-clock",
		MaxClockUncertaintyNS: int64(50 * time.Millisecond),
		ExpectedRunID: "run-001", ExpectedSequence: 1,
		TrustedKeys: map[string]ed25519.PublicKey{"custody-key-1": pub},
	}
	return root, m, pub, private, policy
}

func hasReason(r Result, expected string) bool {
	for _, reason := range r.ReasonCodes {
		if reason == expected {
			return true
		}
	}
	return false
}

func TestValidSignedBundleQualifies(t *testing.T) {
	root, m, _, _, policy := testFixture(t)
	r := VerifyBundle(root, m, policy)
	if r.Verdict != VerdictQualified {
		t.Fatalf("expected qualified bundle, got %+v", r)
	}
	if r.VerifiedArtifacts != 2 || r.ArtifactCount != 2 {
		t.Fatalf("unexpected artifact counts: %+v", r)
	}
}

func TestManifestRootIgnoresJSONArtifactOrder(t *testing.T) {
	_, m, _, _, _ := testFixture(t)
	first, err := ComputeRoot(m)
	if err != nil {
		t.Fatal(err)
	}
	m.Artifacts[0], m.Artifacts[1] = m.Artifacts[1], m.Artifacts[0]
	second, err := ComputeRoot(m)
	if err != nil {
		t.Fatal(err)
	}
	if first != second {
		t.Fatalf("artifact JSON ordering changed semantic root: %s != %s", first, second)
	}
}

func TestArtifactMutationBlocks(t *testing.T) {
	root, m, _, _, policy := testFixture(t)
	if err := os.WriteFile(filepath.Join(root, "state", "source.json"), []byte("tampered"), 0o600); err != nil {
		t.Fatal(err)
	}
	r := VerifyBundle(root, m, policy)
	if r.Verdict != VerdictBlocked || !hasReason(r, "artifact_size_mismatch") && !hasReason(r, "artifact_digest_mismatch") {
		t.Fatalf("expected tampered artifact to block, got %+v", r)
	}
}

func TestMissingArtifactBlocks(t *testing.T) {
	root, m, _, _, policy := testFixture(t)
	if err := os.Remove(filepath.Join(root, "run-manifest.json")); err != nil {
		t.Fatal(err)
	}
	r := VerifyBundle(root, m, policy)
	if r.Verdict != VerdictBlocked || !hasReason(r, "artifact_missing") {
		t.Fatalf("expected missing artifact to block, got %+v", r)
	}
}

func TestPathTraversalBlocksEvenWithValidSignature(t *testing.T) {
	root, m, _, private, policy := testFixture(t)
	m.Artifacts[0].Path = "../outside.json"
	if err := SignManifest(&m, "custody-key-1", private); err != nil {
		t.Fatal(err)
	}
	r := VerifyBundle(root, m, policy)
	if r.Verdict != VerdictBlocked || !hasReason(r, "artifact_path_unsafe") {
		t.Fatalf("expected unsafe path to block, got %+v", r)
	}
}

func TestSymlinkArtifactBlocks(t *testing.T) {
	root, m, _, _, policy := testFixture(t)
	target := filepath.Join(t.TempDir(), "external.json")
	if err := os.WriteFile(target, []byte("{\"outside\":true}"), 0o600); err != nil {
		t.Fatal(err)
	}
	artifact := filepath.Join(root, "state", "source.json")
	if err := os.Remove(artifact); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(target, artifact); err != nil {
		t.Skipf("symlinks unavailable: %v", err)
	}
	r := VerifyBundle(root, m, policy)
	if r.Verdict != VerdictBlocked || !hasReason(r, "artifact_path_unsafe") {
		t.Fatalf("expected symlink artifact to block, got %+v", r)
	}
}

func TestUnknownSignerIsIndeterminate(t *testing.T) {
	root, m, _, _, policy := testFixture(t)
	policy.TrustedKeys = map[string]ed25519.PublicKey{}
	r := VerifyBundle(root, m, policy)
	if r.Verdict != VerdictIndeterminate || !hasReason(r, "signer_key_unknown") {
		t.Fatalf("expected unknown signer to be indeterminate, got %+v", r)
	}
}

func TestInvalidSignatureBlocks(t *testing.T) {
	root, m, _, _, policy := testFixture(t)
	m.Signature.Value = "AAAA"
	r := VerifyBundle(root, m, policy)
	if r.Verdict != VerdictBlocked || !hasReason(r, "signature_encoding_invalid") {
		t.Fatalf("expected invalid signature to block, got %+v", r)
	}
}

func TestMissingSignatureIsIndeterminate(t *testing.T) {
	root, m, _, _, policy := testFixture(t)
	m.Signature = Signature{}
	r := VerifyBundle(root, m, policy)
	if r.Verdict != VerdictIndeterminate || !hasReason(r, "signature_missing") {
		t.Fatalf("expected missing signature to be indeterminate, got %+v", r)
	}
}

func TestPreviousRootMustMatchExpectedChainHead(t *testing.T) {
	root, m, _, private, policy := testFixture(t)
	m.Sequence = 2
	m.PreviousRootSHA256 = digestBytes([]byte("wrong previous root"))
	if err := SignManifest(&m, "custody-key-1", private); err != nil {
		t.Fatal(err)
	}
	policy.ExpectedSequence = 2
	policy.ExpectedPreviousRoot = digestBytes([]byte("expected previous root"))
	r := VerifyBundle(root, m, policy)
	if r.Verdict != VerdictBlocked || !hasReason(r, "previous_root_mismatch") {
		t.Fatalf("expected chain mismatch to block, got %+v", r)
	}
}

func TestMissingPreviousRootExpectationIsIndeterminate(t *testing.T) {
	root, m, _, private, policy := testFixture(t)
	m.Sequence = 2
	m.PreviousRootSHA256 = digestBytes([]byte("previous"))
	policy.ExpectedSequence = 2
	if err := SignManifest(&m, "custody-key-1", private); err != nil {
		t.Fatal(err)
	}
	r := VerifyBundle(root, m, policy)
	if r.Verdict != VerdictIndeterminate || !hasReason(r, "expected_previous_root_unavailable") {
		t.Fatalf("expected unavailable chain head to be indeterminate, got %+v", r)
	}
}

func TestExpiredManifestIsIndeterminate(t *testing.T) {
	root, m, _, _, policy := testFixture(t)
	policy.Now = m.CapturedAt.Add(2 * time.Hour)
	r := VerifyBundle(root, m, policy)
	if r.Verdict != VerdictIndeterminate || !hasReason(r, "manifest_expired") {
		t.Fatalf("expected expiry to be indeterminate, got %+v", r)
	}
}

func TestClockUncertaintyOverPolicyIsIndeterminate(t *testing.T) {
	root, m, _, private, policy := testFixture(t)
	m.ClockUncertaintyNS = int64(time.Second)
	if err := SignManifest(&m, "custody-key-1", private); err != nil {
		t.Fatal(err)
	}
	r := VerifyBundle(root, m, policy)
	if r.Verdict != VerdictIndeterminate || !hasReason(r, "manifest_clock_uncertainty_exceeds_policy") {
		t.Fatalf("expected clock bound failure to be indeterminate, got %+v", r)
	}
}

func TestDuplicateArtifactPathBlocks(t *testing.T) {
	root, m, _, private, policy := testFixture(t)
	m.Artifacts = append(m.Artifacts, m.Artifacts[0])
	if err := SignManifest(&m, "custody-key-1", private); err != nil {
		t.Fatal(err)
	}
	r := VerifyBundle(root, m, policy)
	if r.Verdict != VerdictBlocked || !hasReason(r, "artifact_path_duplicated") {
		t.Fatalf("expected duplicate artifact path to block, got %+v", r)
	}
}

func TestManifestFieldMutationBreaksRootAndSignature(t *testing.T) {
	root, m, _, _, policy := testFixture(t)
	m.Source.TimelineID = "2"
	r := VerifyBundle(root, m, policy)
	if r.Verdict != VerdictBlocked || !hasReason(r, "manifest_root_mismatch") {
		t.Fatalf("expected signed source-boundary mutation to block, got %+v", r)
	}
}

func TestArtifactClockDomainMismatchIsIndeterminate(t *testing.T) {
	root, m, _, private, policy := testFixture(t)
	m.Artifacts[0].ClockDomain = "different-clock"
	if err := SignManifest(&m, "custody-key-1", private); err != nil {
		t.Fatal(err)
	}
	r := VerifyBundle(root, m, policy)
	if r.Verdict != VerdictIndeterminate || !hasReason(r, "artifact_clock_unqualified") {
		t.Fatalf("expected incomparable artifact clock to be indeterminate, got %+v", r)
	}
}

func FuzzValidateRelativePathNeverPanics(f *testing.F) {
	for _, seed := range []string{"artifact.json", "../escape", "/absolute", "a/../b", "a\\b", "", ".", "nested/state.json"} {
		f.Add(seed)
	}
	f.Fuzz(func(t *testing.T, candidate string) {
		_ = validateRelativePath(candidate)
	})
}

func TestUnlistedArtifactFileBlocks(t *testing.T) {
	root, m, _, _, policy := testFixture(t)
	if err := os.WriteFile(filepath.Join(root, "unlisted.log"), []byte("not in manifest"), 0o600); err != nil {
		t.Fatal(err)
	}
	r := VerifyBundle(root, m, policy)
	if r.Verdict != VerdictBlocked || !hasReason(r, "unlisted_artifact_file") {
		t.Fatalf("expected unlisted file to block exact inventory, got %+v", r)
	}
}

func TestExpectedRunIDMismatchBlocks(t *testing.T) {
	root, m, _, _, policy := testFixture(t)
	policy.ExpectedRunID = "another-run"
	r := VerifyBundle(root, m, policy)
	if r.Verdict != VerdictBlocked || !hasReason(r, "run_id_mismatch") {
		t.Fatalf("expected run replay to block, got %+v", r)
	}
}

func TestExpectedSequenceMismatchBlocks(t *testing.T) {
	root, m, _, _, policy := testFixture(t)
	policy.ExpectedSequence = 2
	r := VerifyBundle(root, m, policy)
	if r.Verdict != VerdictBlocked || !hasReason(r, "sequence_mismatch") {
		t.Fatalf("expected stale sequence to block, got %+v", r)
	}
}

func TestSignatureBindsKeyIdentity(t *testing.T) {
	root, m, _, _, policy := testFixture(t)
	otherPublic, _, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	policy.TrustedKeys["custody-key-2"] = otherPublic
	m.Signature.KeyID = "custody-key-2"
	r := VerifyBundle(root, m, policy)
	if r.Verdict != VerdictBlocked || !hasReason(r, "signature_invalid") {
		t.Fatalf("changing signer identity must invalidate signature: %+v", r)
	}
}

func TestValidChainedBundleQualifies(t *testing.T) {
	root, m, _, private, policy := testFixture(t)
	m.Sequence = 2
	previousRoot := digestBytes([]byte("previous-manifest-root"))
	m.PreviousRootSHA256 = previousRoot
	if err := SignManifest(&m, "custody-key-1", private); err != nil {
		t.Fatal(err)
	}
	policy.ExpectedSequence = 2
	policy.ExpectedPreviousRoot = previousRoot
	r := VerifyBundle(root, m, policy)
	if r.Verdict != VerdictQualified {
		t.Fatalf("expected valid chained bundle, got %+v", r)
	}
}

func TestOldArtifactIsIndeterminateEvenWhenBundleIsFresh(t *testing.T) {
	root, m, _, private, policy := testFixture(t)
	m.Artifacts[0].CollectedAt = policy.Now.Add(-2 * time.Hour)
	if err := SignManifest(&m, "custody-key-1", private); err != nil {
		t.Fatal(err)
	}
	r := VerifyBundle(root, m, policy)
	if r.Verdict != VerdictIndeterminate || !hasReason(r, "artifact_expired") {
		t.Fatalf("old artifact must not inherit freshness from new manifest: %+v", r)
	}
}

func TestFutureArtifactCollectionTimeIsIndeterminate(t *testing.T) {
	root, m, _, private, policy := testFixture(t)
	m.Artifacts[0].CollectedAt = policy.Now.Add(time.Hour)
	if err := SignManifest(&m, "custody-key-1", private); err != nil {
		t.Fatal(err)
	}
	r := VerifyBundle(root, m, policy)
	if r.Verdict != VerdictIndeterminate || !hasReason(r, "artifact_collection_time_in_future") {
		t.Fatalf("future artifact time must remain indeterminate: %+v", r)
	}
}
