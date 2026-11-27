package reader

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"os"
	"path/filepath"
	"testing"

	custody "luminousdynamics/mon026-custody"
)

type testDocument struct {
	Message string `json:"message"`
}

func digest(data []byte) string {
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

func manifestFor(path string, data []byte) custody.Manifest {
	return custody.Manifest{Artifacts: []custody.Artifact{{
		Path: path, SizeBytes: int64(len(data)), SHA256: digest(data),
	}}}
}

func writeArtifact(t *testing.T, root, relative string, data []byte) {
	t.Helper()
	full := filepath.Join(root, filepath.FromSlash(relative))
	if err := os.MkdirAll(filepath.Dir(full), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(full, data, 0o600); err != nil {
		t.Fatal(err)
	}
}

func TestReadArtifactMatchingManifestDigest(t *testing.T) {
	root := t.TempDir()
	data := []byte("{\"message\":\"immutable\"}\n")
	writeArtifact(t, root, "nested/result.json", data)
	got, err := ReadArtifactMatchingManifestDigest[testDocument](root, "nested/result.json", manifestFor("nested/result.json", data))
	if err != nil {
		t.Fatal(err)
	}
	if got.Message != "immutable" {
		t.Fatalf("unexpected parsed content: %+v", got)
	}
}

func TestRejectsReplacementAfterCustodyVerification(t *testing.T) {
	root := t.TempDir()
	signedBytes := []byte("{\"message\":\"immutable\"}")
	writeArtifact(t, root, "result.json", signedBytes)
	manifest := manifestFor("result.json", signedBytes)

	// Model a same-size replacement after custody verification and before use.
	replacement := []byte("{\"message\":\"tampered!\"}")
	if len(replacement) != len(signedBytes) {
		t.Fatalf("test replacement must preserve size: %d != %d", len(replacement), len(signedBytes))
	}
	writeArtifact(t, root, "result.json", replacement)

	_, err := ReadArtifactMatchingManifestDigest[testDocument](root, "result.json", manifest)
	if !errors.Is(err, ErrArtifactDigestMismatch) {
		t.Fatalf("replaced artifact must fail the signed digest check, got %v", err)
	}
}

func TestRejectsSizeMismatch(t *testing.T) {
	root := t.TempDir()
	data := []byte("{\"message\":\"immutable\"}")
	writeArtifact(t, root, "result.json", data)
	manifest := manifestFor("result.json", data)
	manifest.Artifacts[0].SizeBytes--
	_, err := ReadArtifactMatchingManifestDigest[testDocument](root, "result.json", manifest)
	if !errors.Is(err, ErrArtifactSizeMismatch) {
		t.Fatalf("expected size mismatch, got %v", err)
	}
}

func TestRejectsUndeclaredAndDuplicatePaths(t *testing.T) {
	root := t.TempDir()
	data := []byte("{\"message\":\"immutable\"}")
	writeArtifact(t, root, "result.json", data)
	manifest := manifestFor("result.json", data)
	if _, err := ReadArtifactMatchingManifestDigest[testDocument](root, "other.json", manifest); !errors.Is(err, ErrArtifactNotDeclared) {
		t.Fatalf("undeclared path must fail: %v", err)
	}
	manifest.Artifacts = append(manifest.Artifacts, manifest.Artifacts[0])
	if _, err := ReadArtifactMatchingManifestDigest[testDocument](root, "result.json", manifest); !errors.Is(err, ErrArtifactNotDeclared) {
		t.Fatalf("duplicate manifest path must fail: %v", err)
	}
}

func TestRejectsTraversalAndSymlinks(t *testing.T) {
	root := t.TempDir()
	data := []byte("{\"message\":\"immutable\"}")
	writeArtifact(t, root, "result.json", data)
	if _, err := ReadArtifactMatchingManifestDigest[testDocument](root, "../result.json", manifestFor("../result.json", data)); !errors.Is(err, ErrUnsafeArtifactPath) {
		t.Fatalf("path traversal must fail: %v", err)
	}

	external := filepath.Join(t.TempDir(), "external.json")
	if err := os.WriteFile(external, data, 0o600); err != nil {
		t.Fatal(err)
	}
	linked := filepath.Join(root, "linked.json")
	if err := os.Symlink(external, linked); err != nil {
		t.Skipf("symlink creation unavailable: %v", err)
	}
	if _, err := ReadArtifactMatchingManifestDigest[testDocument](root, "linked.json", manifestFor("linked.json", data)); !errors.Is(err, ErrUnsafeArtifactPath) {
		t.Fatalf("symlink artifact must fail: %v", err)
	}
}

func TestRejectsTrailingJSONAndUnknownFields(t *testing.T) {
	root := t.TempDir()
	trailing := []byte("{\"message\":\"one\"} {\"message\":\"two\"}")
	writeArtifact(t, root, "trailing.json", trailing)
	if _, err := ReadArtifactMatchingManifestDigest[testDocument](root, "trailing.json", manifestFor("trailing.json", trailing)); !errors.Is(err, ErrInvalidJSON) {
		t.Fatalf("multiple JSON values must fail: %v", err)
	}
	unknown := []byte("{\"message\":\"one\",\"unexpected\":true}")
	writeArtifact(t, root, "unknown.json", unknown)
	if _, err := ReadArtifactMatchingManifestDigest[testDocument](root, "unknown.json", manifestFor("unknown.json", unknown)); !errors.Is(err, ErrInvalidJSON) {
		t.Fatalf("unknown JSON fields must fail: %v", err)
	}
}

func TestRejectsOversizedStructuredArtifactBeforeOpening(t *testing.T) {
	manifest := custody.Manifest{Artifacts: []custody.Artifact{{
		Path: "large.json", SizeBytes: MaxStructuredArtifactBytes + 1, SHA256: digest([]byte("unused")),
	}}}
	_, err := ReadArtifactMatchingManifestDigest[testDocument](t.TempDir(), "large.json", manifest)
	if !errors.Is(err, ErrArtifactTooLarge) {
		t.Fatalf("oversized artifact must fail before read: %v", err)
	}
}

func TestRejectsSymlinkRoot(t *testing.T) {
	parent := t.TempDir()
	actual := filepath.Join(parent, "actual")
	if err := os.Mkdir(actual, 0o700); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(parent, "link")
	if err := os.Symlink(actual, link); err != nil {
		t.Skipf("symlink creation unavailable: %v", err)
	}
	data := []byte("{\"message\":\"immutable\"}")
	writeArtifact(t, actual, "result.json", data)
	_, err := ReadArtifactMatchingManifestDigest[testDocument](link, "result.json", manifestFor("result.json", data))
	if !errors.Is(err, ErrUnsafeArtifactPath) {
		t.Fatalf("symlink root must fail: %v", err)
	}
}
