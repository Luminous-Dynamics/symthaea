// Package packager builds MON-026 custody manifests from an explicit artifact allowlist.
// It never reads or accepts a private signing key.
package packager

import (
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"os"
	"path"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	custody "luminousdynamics/mon026-custody"
)

const (
	InputProfile = "mon027-evidence-bundle-input-v1"
	SigningRequestProfile = "mon027-detached-signing-request-v1"
	SignatureEnvelopeProfile = "mon027-detached-signature-v1"
)

var (
	ErrInputInvalid = errors.New("invalid MON-027 input index")
	ErrInventoryMismatch = errors.New("artifact directory does not match allowlist")
	ErrUnsafeArtifactPath = errors.New("unsafe artifact path")
)

type ArtifactInput struct {
	Path string `json:"path"`
	MediaType string `json:"media_type"`
	Role string `json:"role"`
	Producer string `json:"producer"`
	CommandID string `json:"command_id"`
	CollectedAt time.Time `json:"collected_at"`
	ClockDomain string `json:"clock_domain"`
	ClockUncertaintyNS int64 `json:"clock_uncertainty_ns"`
}

type InputIndex struct {
	Profile string `json:"profile"`
	Version string `json:"version"`
	RunID string `json:"run_id"`
	Sequence uint64 `json:"sequence"`
	PreviousRootSHA256 string `json:"previous_root_sha256,omitempty"`
	CapturedAt time.Time `json:"captured_at"`
	ClockDomain string `json:"clock_domain"`
	ClockUncertaintyNS int64 `json:"clock_uncertainty_ns"`
	Source custody.SourceBoundary `json:"source"`
	SignerKeyID string `json:"signer_key_id"`
	Artifacts []ArtifactInput `json:"artifacts"`
}

type SigningRequest struct {
	Profile string `json:"profile"`
	Version string `json:"version"`
	RequestID string `json:"request_id"`
	RunID string `json:"run_id"`
	Sequence uint64 `json:"sequence"`
	RootSHA256 string `json:"root_sha256"`
	Algorithm string `json:"algorithm"`
	KeyID string `json:"key_id"`
	MessageBase64 string `json:"message_base64"`
	CreatedAt time.Time `json:"created_at"`
}

type SignatureEnvelope struct {
	Profile string `json:"profile"`
	Version string `json:"version"`
	RequestID string `json:"request_id"`
	RunID string `json:"run_id"`
	Sequence uint64 `json:"sequence"`
	RootSHA256 string `json:"root_sha256"`
	Algorithm string `json:"algorithm"`
	KeyID string `json:"key_id"`
	Value string `json:"value"`
}

type BuildResult struct {
	Manifest custody.Manifest `json:"manifest"`
	SigningRequest SigningRequest `json:"signing_request"`
}

// BuildManifest hashes only explicitly allowlisted regular files and requires
// the artifact root's file inventory to match that allowlist exactly.
func BuildManifest(rootDir string, index InputIndex) (BuildResult, error) {
	if err := validateIndex(index); err != nil {
		return BuildResult{}, err
	}
	declared := make(map[string]struct{}, len(index.Artifacts))
	artifacts := make([]custody.Artifact, 0, len(index.Artifacts))
	inputs := append([]ArtifactInput(nil), index.Artifacts...)
	sort.Slice(inputs, func(i, j int) bool { return inputs[i].Path < inputs[j].Path })
	for _, input := range inputs {
		if err := validateRelativePath(input.Path); err != nil {
			return BuildResult{}, err
		}
		if _, exists := declared[input.Path]; exists {
			return BuildResult{}, ErrInputInvalid
		}
		declared[input.Path] = struct{}{}
		size, digest, err := hashArtifact(rootDir, input.Path)
		if err != nil {
			return BuildResult{}, err
		}
		artifacts = append(artifacts, custody.Artifact{
			Path: input.Path,
			MediaType: input.MediaType,
			Role: input.Role,
			Producer: input.Producer,
			CommandID: input.CommandID,
			SizeBytes: size,
			SHA256: digest,
			CollectedAt: input.CollectedAt,
			ClockDomain: input.ClockDomain,
			ClockUncertaintyNS: input.ClockUncertaintyNS,
		})
	}
	if err := verifyExactInventory(rootDir, declared); err != nil {
		return BuildResult{}, err
	}

	manifest := custody.Manifest{
		Profile: custody.ProfileV1,
		Version: custody.VersionV1,
		RunID: index.RunID,
		Sequence: index.Sequence,
		PreviousRootSHA256: index.PreviousRootSHA256,
		CapturedAt: index.CapturedAt,
		ClockDomain: index.ClockDomain,
		ClockUncertaintyNS: index.ClockUncertaintyNS,
		Source: index.Source,
		Artifacts: artifacts,
		Signature: custody.Signature{Algorithm: "Ed25519", KeyID: index.SignerKeyID},
	}
	root, err := custody.ComputeRoot(manifest)
	if err != nil {
		return BuildResult{}, err
	}
	manifest.RootSHA256 = root
	message := []byte("MON026-EVIDENCE-CUSTODY-V1\x00" + "Ed25519" + "\x00" + index.SignerKeyID + "\x00" + root)
	request := SigningRequest{
		Profile: SigningRequestProfile,
		Version: "1.0.0",
		RequestID: signingRequestID(index.RunID, index.Sequence, root, index.SignerKeyID),
		RunID: index.RunID,
		Sequence: index.Sequence,
		RootSHA256: root,
		Algorithm: "Ed25519",
		KeyID: index.SignerKeyID,
		MessageBase64: base64.StdEncoding.EncodeToString(message),
		CreatedAt: index.CapturedAt,
	}
	return BuildResult{Manifest: manifest, SigningRequest: request}, nil
}

// AttachSignature inserts a detached signer response and verifies all bytes,
// trust and chain policy using MON-026 before returning a signed manifest.
func AttachSignature(rootDir string, manifest custody.Manifest, envelope SignatureEnvelope, policy custody.Policy) (custody.Manifest, custody.Result) {
	if envelope.Profile != SignatureEnvelopeProfile || envelope.Version != "1.0.0" ||
		envelope.RequestID != signingRequestID(manifest.RunID, manifest.Sequence, manifest.RootSHA256, manifest.Signature.KeyID) ||
		envelope.RunID != manifest.RunID || envelope.Sequence != manifest.Sequence ||
		!strings.EqualFold(envelope.RootSHA256, manifest.RootSHA256) ||
		envelope.Algorithm != manifest.Signature.Algorithm ||
		envelope.KeyID != manifest.Signature.KeyID || envelope.Value == "" {
		return manifest, custody.Result{
			Verdict: custody.VerdictBlocked,
			ReasonCodes: []string{"detached_signature_envelope_mismatch"},
			RootSHA256: manifest.RootSHA256,
			SignerKeyID: envelope.KeyID,
			ArtifactCount: len(manifest.Artifacts),
		}
	}
	manifest.Signature.Value = envelope.Value
	result := custody.VerifyBundle(rootDir, manifest, policy)
	return manifest, result
}


func signingRequestID(runID string, sequence uint64, root, keyID string) string {
	message := "MON027-SIGNING-REQUEST-V1\\x00" + runID + "\\x00" +
		strconv.FormatUint(sequence, 10) + "\\x00" + strings.ToLower(root) + "\\x00" + keyID
	sum := sha256.Sum256([]byte(message))
	return hex.EncodeToString(sum[:])
}

func validateIndex(index InputIndex) error {
	if index.Profile != InputProfile || index.Version != "1.0.0" ||
		strings.TrimSpace(index.RunID) == "" || index.Sequence == 0 ||
		index.CapturedAt.IsZero() || index.ClockDomain == "" || index.ClockUncertaintyNS < 0 ||
		strings.TrimSpace(index.SignerKeyID) == "" || len(index.Artifacts) == 0 ||
		!completeBoundary(index.Source) {
		return ErrInputInvalid
	}
	if index.Sequence == 1 && index.PreviousRootSHA256 != "" {
		return ErrInputInvalid
	}
	if index.Sequence > 1 && !validSHA256(index.PreviousRootSHA256) {
		return ErrInputInvalid
	}
	for _, artifact := range index.Artifacts {
		if strings.TrimSpace(artifact.MediaType) == "" || strings.TrimSpace(artifact.Role) == "" ||
			strings.TrimSpace(artifact.Producer) == "" || strings.TrimSpace(artifact.CommandID) == "" ||
			artifact.CollectedAt.IsZero() || artifact.ClockDomain != index.ClockDomain ||
			artifact.ClockUncertaintyNS < 0 {
			return ErrInputInvalid
		}
	}
	return nil
}

func hashArtifact(rootDir, relative string) (int64, string, error) {
	full, err := safeArtifactPath(rootDir, relative)
	if err != nil {
		return 0, "", err
	}
	file, err := os.Open(full)
	if err != nil {
		return 0, "", err
	}
	defer file.Close()
	before, err := file.Stat()
	if err != nil || !before.Mode().IsRegular() {
		return 0, "", ErrInventoryMismatch
	}
	hash := sha256.New()
	n, err := io.Copy(hash, file)
	if err != nil {
		return 0, "", err
	}
	after, err := file.Stat()
	if err != nil || n != before.Size() || after.Size() != before.Size() ||
		!after.ModTime().Equal(before.ModTime()) {
		return 0, "", ErrInventoryMismatch
	}
	return n, hex.EncodeToString(hash.Sum(nil)), nil
}

func safeArtifactPath(rootDir, relative string) (string, error) {
	if err := validateRelativePath(relative); err != nil {
		return "", err
	}
	root, err := filepath.Abs(rootDir)
	if err != nil {
		return "", err
	}
	rootInfo, err := os.Lstat(root)
	if err != nil || !rootInfo.IsDir() || rootInfo.Mode()&os.ModeSymlink != 0 {
		return "", ErrUnsafeArtifactPath
	}
	current := root
	parts := strings.Split(relative, "/")
	for i, part := range parts {
		current = filepath.Join(current, part)
		info, statErr := os.Lstat(current)
		if statErr != nil {
			return "", statErr
		}
		if info.Mode()&os.ModeSymlink != 0 {
			return "", ErrUnsafeArtifactPath
		}
		if i < len(parts)-1 && !info.IsDir() {
			return "", ErrUnsafeArtifactPath
		}
		if i == len(parts)-1 && !info.Mode().IsRegular() {
			return "", ErrInventoryMismatch
		}
	}
	return current, nil
}

func verifyExactInventory(rootDir string, declared map[string]struct{}) error {
	root, err := filepath.Abs(rootDir)
	if err != nil {
		return err
	}
	info, err := os.Lstat(root)
	if err != nil || !info.IsDir() || info.Mode()&os.ModeSymlink != 0 {
		return ErrUnsafeArtifactPath
	}
	return filepath.WalkDir(root, func(current string, entry os.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if current == root {
			return nil
		}
		if entry.Type()&os.ModeSymlink != 0 {
			return ErrUnsafeArtifactPath
		}
		if entry.IsDir() {
			return nil
		}
		fileInfo, err := entry.Info()
		if err != nil {
			return err
		}
		if !fileInfo.Mode().IsRegular() {
			return ErrInventoryMismatch
		}
		rel, err := filepath.Rel(root, current)
		if err != nil {
			return err
		}
		if _, ok := declared[filepath.ToSlash(rel)]; !ok {
			return ErrInventoryMismatch
		}
		return nil
	})
}

func validateRelativePath(value string) error {
	if value == "" || strings.ContainsRune(value, 0) || strings.Contains(value, "\\") ||
		path.IsAbs(value) || strings.Contains(strings.SplitN(value, "/", 2)[0], ":") {
		return ErrUnsafeArtifactPath
	}
	clean := path.Clean(value)
	if clean == "." || clean == ".." || strings.HasPrefix(clean, "../") || clean != value {
		return ErrUnsafeArtifactPath
	}
	return nil
}

func completeBoundary(s custody.SourceBoundary) bool {
	return s.IncidentID != "" && s.SourceClusterID != "" && s.Database != "" &&
		s.SystemIdentifier != "" && s.TimelineID != "" && s.SnapshotID != "" &&
		s.CaptureLSN != "" && validSHA256(s.ScopeDigest) && validSHA256(s.InventoryDigest)
}

func validSHA256(value string) bool {
	if len(value) != sha256.Size*2 {
		return false
	}
	_, err := hex.DecodeString(value)
	return err == nil
}

// DecodeStrict parses exactly one JSON value and rejects unknown fields.
func DecodeStrict[T any](reader io.Reader) (T, error) {
	var value T
	decoder := json.NewDecoder(reader)
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&value); err != nil {
		return value, err
	}
	var trailing any
	if err := decoder.Decode(&trailing); err != io.EOF {
		return value, ErrInputInvalid
	}
	return value, nil
}
