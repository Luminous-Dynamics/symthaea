// Package reader binds downstream interpretation to the exact bytes
// authenticated by a MON-026 artifact manifest.
package reader

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"strings"

	custody "luminousdynamics/mon026-custody"
)

const MaxStructuredArtifactBytes int64 = 256 << 20

var (
	ErrArtifactNotDeclared = errors.New("artifact is not declared exactly once in manifest")
	ErrUnsafeArtifactPath = errors.New("unsafe artifact path")
	ErrArtifactTooLarge = errors.New("structured artifact exceeds read limit")
	ErrArtifactSizeMismatch = errors.New("artifact size does not match signed manifest")
	ErrArtifactDigestMismatch = errors.New("artifact bytes do not match signed manifest digest")
	ErrArtifactChanged = errors.New("artifact changed while being read")
	ErrInvalidJSON = errors.New("artifact is not exactly one valid JSON document")
)

// ReadArtifactMatchingManifestDigest opens the selected artifact, verifies its
// size and SHA-256 against the manifest, then decodes that same in-memory byte
// slice. It closes the verify-then-reopen gap in downstream consumers.
//
// This function does not authenticate the manifest or verify its signature.
// Callers must first obtain a qualified result from custody.VerifyBundle using
// an out-of-band trust policy, then pass that exact verified manifest here.
func ReadArtifactMatchingManifestDigest[T any](rootDir, relativePath string, manifest custody.Manifest) (T, error) {
	var zero T
	artifact, err := findDeclaredArtifact(relativePath, manifest)
	if err != nil {
		return zero, err
	}
	data, err := readAndVerifyBytes(rootDir, relativePath, artifact)
	if err != nil {
		return zero, err
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	var value T
	if err := decoder.Decode(&value); err != nil {
		return zero, fmt.Errorf("%w: %v", ErrInvalidJSON, err)
	}
	var trailing any
	if err := decoder.Decode(&trailing); err != io.EOF {
		return zero, ErrInvalidJSON
	}
	return value, nil
}

func findDeclaredArtifact(relativePath string, manifest custody.Manifest) (custody.Artifact, error) {
	if !safeRelativePath(relativePath) {
		return custody.Artifact{}, ErrUnsafeArtifactPath
	}
	var match custody.Artifact
	matches := 0
	for _, artifact := range manifest.Artifacts {
		if artifact.Path == relativePath {
			match = artifact
			matches++
		}
	}
	if matches != 1 || match.SizeBytes < 0 || !validSHA256(match.SHA256) {
		return custody.Artifact{}, ErrArtifactNotDeclared
	}
	if match.SizeBytes > MaxStructuredArtifactBytes {
		return custody.Artifact{}, ErrArtifactTooLarge
	}
	return match, nil
}

func readAndVerifyBytes(rootDir, relativePath string, expected custody.Artifact) ([]byte, error) {
	root, err := filepath.Abs(rootDir)
	if err != nil {
		return nil, err
	}
	rootInfo, err := os.Lstat(root)
	if err != nil {
		return nil, err
	}
	if !rootInfo.IsDir() || rootInfo.Mode()&os.ModeSymlink != 0 {
		return nil, ErrUnsafeArtifactPath
	}

	current := root
	parts := strings.Split(relativePath, "/")
	var pathInfo os.FileInfo
	for index, part := range parts {
		current = filepath.Join(current, part)
		info, statErr := os.Lstat(current)
		if statErr != nil {
			return nil, statErr
		}
		if info.Mode()&os.ModeSymlink != 0 {
			return nil, ErrUnsafeArtifactPath
		}
		if index < len(parts)-1 && !info.IsDir() {
			return nil, ErrUnsafeArtifactPath
		}
		if index == len(parts)-1 {
			if !info.Mode().IsRegular() {
				return nil, ErrUnsafeArtifactPath
			}
			pathInfo = info
		}
	}

	file, err := os.Open(current)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	before, err := file.Stat()
	if err != nil {
		return nil, err
	}
	if !before.Mode().IsRegular() || pathInfo == nil || !os.SameFile(pathInfo, before) {
		return nil, ErrUnsafeArtifactPath
	}
	if before.Size() != expected.SizeBytes {
		return nil, ErrArtifactSizeMismatch
	}

	data, err := io.ReadAll(io.LimitReader(file, expected.SizeBytes+1))
	if err != nil {
		return nil, err
	}
	after, err := file.Stat()
	if err != nil {
		return nil, err
	}
	if int64(len(data)) != expected.SizeBytes || after.Size() != before.Size() {
		return nil, ErrArtifactChanged
	}

	digest := sha256.Sum256(data)
	if !strings.EqualFold(hex.EncodeToString(digest[:]), expected.SHA256) {
		return nil, ErrArtifactDigestMismatch
	}
	return data, nil
}

func safeRelativePath(value string) bool {
	if value == "" || value == "." || !fs.ValidPath(value) || strings.ContainsRune(value, 0) ||
		strings.Contains(value, "\\") || strings.Contains(strings.SplitN(value, "/", 2)[0], ":") {
		return false
	}
	return true
}

func validSHA256(value string) bool {
	if len(value) != sha256.Size*2 {
		return false
	}
	_, err := hex.DecodeString(value)
	return err == nil
}
