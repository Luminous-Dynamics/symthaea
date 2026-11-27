// Package custody verifies portable evidence bundles without relying on their producer.
// A valid signature authenticates the signed manifest; it does not prove that signed
// source observations or recovery conclusions are true.
package custody

import (
	"crypto/ed25519"
	"crypto/sha256"
	"encoding/base64"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"hash"
	"io"
	"os"
	pathpkg "path"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

const (
	ProfileV1 = "mon026-evidence-custody-v1"
	VersionV1 = "1.0.0"
	signatureDomain = "MON026-EVIDENCE-CUSTODY-V1\x00"
)

type SourceBoundary struct {
	IncidentID       string `json:"incident_id"`
	SourceClusterID  string `json:"source_cluster_id"`
	Database         string `json:"database"`
	SystemIdentifier string `json:"system_identifier"`
	TimelineID       string `json:"timeline_id"`
	SnapshotID       string `json:"snapshot_id"`
	CaptureLSN       string `json:"capture_lsn"`
	ScopeDigest      string `json:"scope_digest"`
	InventoryDigest  string `json:"inventory_digest"`
}

type Artifact struct {
	Path               string    `json:"path"`
	MediaType          string    `json:"media_type"`
	Role               string    `json:"role"`
	Producer           string    `json:"producer"`
	CommandID          string    `json:"command_id"`
	SizeBytes          int64     `json:"size_bytes"`
	SHA256             string    `json:"sha256"`
	CollectedAt        time.Time `json:"collected_at"`
	ClockDomain        string    `json:"clock_domain"`
	ClockUncertaintyNS int64     `json:"clock_uncertainty_ns"`
}

type Signature struct {
	Algorithm string `json:"algorithm"`
	KeyID     string `json:"key_id"`
	Value     string `json:"value"`
}

type Manifest struct {
	Profile            string         `json:"profile"`
	Version            string         `json:"version"`
	RunID              string         `json:"run_id"`
	Sequence           uint64         `json:"sequence"`
	PreviousRootSHA256 string         `json:"previous_root_sha256,omitempty"`
	CapturedAt         time.Time      `json:"captured_at"`
	ClockDomain        string         `json:"clock_domain"`
	ClockUncertaintyNS int64          `json:"clock_uncertainty_ns"`
	Source             SourceBoundary `json:"source"`
	Artifacts          []Artifact     `json:"artifacts"`
	RootSHA256         string         `json:"root_sha256"`
	Signature          Signature      `json:"signature"`
}

type Policy struct {
	Now                   time.Time
	MaxAge                time.Duration
	ClockDomain           string
	MaxClockUncertaintyNS int64
	ExpectedPreviousRoot  string
	ExpectedRunID         string
	ExpectedSequence      uint64
	TrustedKeys           map[string]ed25519.PublicKey
}

type Verdict string

const (
	VerdictQualified     Verdict = "qualified"
	VerdictBlocked       Verdict = "blocked"
	VerdictIndeterminate Verdict = "indeterminate"
)

type Result struct {
	Verdict           Verdict  `json:"verdict"`
	ReasonCodes       []string `json:"reason_codes"`
	RootSHA256        string   `json:"root_sha256,omitempty"`
	SignerKeyID       string   `json:"signer_key_id,omitempty"`
	VerifiedArtifacts int      `json:"verified_artifacts"`
	ArtifactCount     int      `json:"artifact_count"`
}

var (
	ErrUnsafePath       = errors.New("unsafe artifact path")
	ErrDuplicatePath    = errors.New("duplicate artifact path")
	ErrInvalidManifest  = errors.New("invalid manifest")
	ErrArtifactMissing  = errors.New("artifact missing")
	ErrArtifactNotFile  = errors.New("artifact is not a regular file")
	ErrArtifactSize     = errors.New("artifact size mismatch")
	ErrArtifactDigest   = errors.New("artifact digest mismatch")
)

// ComputeRoot returns the deterministic SHA-256 semantic root of a manifest.
// Signature and RootSHA256 are excluded to avoid self-reference. Artifact ordering
// in the JSON document does not affect the result; artifact paths are sorted here.
func ComputeRoot(m Manifest) (string, error) {
	artifacts := append([]Artifact(nil), m.Artifacts...)
	sort.Slice(artifacts, func(i, j int) bool {
		return artifacts[i].Path < artifacts[j].Path
	})

	h := sha256.New()
	writeString(h, signatureDomain)
	writeString(h, m.Profile)
	writeString(h, m.Version)
	writeString(h, m.RunID)
	writeUint64(h, m.Sequence)
	writeString(h, m.PreviousRootSHA256)
	writeTime(h, m.CapturedAt)
	writeString(h, m.ClockDomain)
	writeInt64(h, m.ClockUncertaintyNS)

	writeString(h, m.Source.IncidentID)
	writeString(h, m.Source.SourceClusterID)
	writeString(h, m.Source.Database)
	writeString(h, m.Source.SystemIdentifier)
	writeString(h, m.Source.TimelineID)
	writeString(h, m.Source.SnapshotID)
	writeString(h, m.Source.CaptureLSN)
	writeString(h, m.Source.ScopeDigest)
	writeString(h, m.Source.InventoryDigest)

	writeUint64(h, uint64(len(artifacts)))
	for _, a := range artifacts {
		writeString(h, a.Path)
		writeString(h, a.MediaType)
		writeString(h, a.Role)
		writeString(h, a.Producer)
		writeString(h, a.CommandID)
		writeInt64(h, a.SizeBytes)
		writeString(h, strings.ToLower(a.SHA256))
		writeTime(h, a.CollectedAt)
		writeString(h, a.ClockDomain)
		writeInt64(h, a.ClockUncertaintyNS)
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

// SignManifest computes the root and signs it with Ed25519. The private key must
// be held outside the bundle by a separate signing authority.
func SignManifest(m *Manifest, keyID string, privateKey ed25519.PrivateKey) error {
	if m == nil || len(privateKey) != ed25519.PrivateKeySize || strings.TrimSpace(keyID) == "" {
		return ErrInvalidManifest
	}
	root, err := ComputeRoot(*m)
	if err != nil {
		return err
	}
	m.RootSHA256 = root
	m.Signature = Signature{Algorithm: "Ed25519", KeyID: keyID}
	msg := signatureMessage(root, m.Signature.Algorithm, keyID)
	sig := ed25519.Sign(privateKey, msg)
	m.Signature.Value = base64.StdEncoding.EncodeToString(sig)
	return nil
}

// VerifyBundle validates manifest structure, the out-of-band trust policy,
// artifact bytes, signature, freshness and the expected previous chain root.
// It has no network, database, subprocess or mutation side effects.
func VerifyBundle(rootDir string, m Manifest, policy Policy) Result {
	result := Result{
		Verdict:       VerdictQualified,
		ArtifactCount: len(m.Artifacts),
	}
	reasons := make(map[string]struct{})
	blocked := false
	indeterminate := false

	addBlocked := func(code string) {
		blocked = true
		reasons[code] = struct{}{}
	}
	addIndeterminate := func(code string) {
		indeterminate = true
		reasons[code] = struct{}{}
	}

	if m.Profile != ProfileV1 || m.Version != VersionV1 {
		addBlocked("unsupported_manifest_profile_or_version")
	}
	if strings.TrimSpace(m.RunID) == "" {
		addIndeterminate("run_id_missing")
	}
	if policy.ExpectedRunID == "" {
		addIndeterminate("expected_run_id_unavailable")
	} else if m.RunID != policy.ExpectedRunID {
		addBlocked("run_id_mismatch")
	}
	if policy.ExpectedSequence == 0 {
		addIndeterminate("expected_sequence_unavailable")
	} else if m.Sequence != policy.ExpectedSequence {
		addBlocked("sequence_mismatch")
	}
	if m.Sequence == 0 {
		addBlocked("sequence_zero")
	}
	if len(m.Artifacts) == 0 {
		addIndeterminate("artifact_inventory_empty")
	}
	if !completeBoundary(m.Source) {
		addIndeterminate("source_boundary_incomplete")
	}

	if policy.Now.IsZero() || policy.MaxAge <= 0 {
		addIndeterminate("freshness_policy_missing")
	}
	if policy.ClockDomain == "" || policy.MaxClockUncertaintyNS < 0 {
		addIndeterminate("clock_policy_missing")
	}
	if m.ClockDomain == "" || m.ClockUncertaintyNS < 0 {
		addIndeterminate("manifest_clock_invalid")
	} else {
		if policy.ClockDomain != "" && m.ClockDomain != policy.ClockDomain {
			addIndeterminate("manifest_clock_domain_mismatch")
		}
		if policy.MaxClockUncertaintyNS > 0 && m.ClockUncertaintyNS > policy.MaxClockUncertaintyNS {
			addIndeterminate("manifest_clock_uncertainty_exceeds_policy")
		}
	}
	if m.CapturedAt.IsZero() {
		addIndeterminate("manifest_capture_time_missing")
	} else if !policy.Now.IsZero() && m.ClockUncertaintyNS >= 0 {
		uncertainty := time.Duration(m.ClockUncertaintyNS)
		if m.CapturedAt.After(policy.Now.Add(uncertainty)) {
			addIndeterminate("manifest_capture_time_in_future")
		}
		if policy.MaxAge > 0 && policy.Now.Sub(m.CapturedAt) > policy.MaxAge+uncertainty {
			addIndeterminate("manifest_expired")
		}
	}

	root, err := ComputeRoot(m)
	if err != nil {
		addBlocked("manifest_root_cannot_be_computed")
	} else {
		result.RootSHA256 = root
		if !validSHA256(m.RootSHA256) {
			addIndeterminate("manifest_root_missing_or_malformed")
		} else if !strings.EqualFold(root, m.RootSHA256) {
			addBlocked("manifest_root_mismatch")
		}
	}

	declaredPaths := make(map[string]struct{}, len(m.Artifacts))
	validateArtifacts := func() {
		seen := make(map[string]struct{}, len(m.Artifacts))
		for _, a := range m.Artifacts {
			if err := validateRelativePath(a.Path); err != nil {
				addBlocked("artifact_path_unsafe")
				continue
			}
			if _, exists := seen[a.Path]; exists {
				addBlocked("artifact_path_duplicated")
				continue
			}
			seen[a.Path] = struct{}{}
			declaredPaths[a.Path] = struct{}{}

			if strings.TrimSpace(a.MediaType) == "" || strings.TrimSpace(a.Role) == "" ||
				strings.TrimSpace(a.Producer) == "" || strings.TrimSpace(a.CommandID) == "" ||
				a.SizeBytes < 0 || !validSHA256(a.SHA256) {
				addBlocked("artifact_metadata_invalid")
				continue
			}
			if a.ClockDomain == "" || a.ClockDomain != m.ClockDomain || a.ClockUncertaintyNS < 0 {
				addIndeterminate("artifact_clock_unqualified")
			}
			if policy.MaxClockUncertaintyNS > 0 && a.ClockUncertaintyNS > policy.MaxClockUncertaintyNS {
				addIndeterminate("artifact_clock_uncertainty_exceeds_policy")
			}
			if a.CollectedAt.IsZero() {
				addIndeterminate("artifact_collection_time_missing")
			} else {
				artifactUncertainty := time.Duration(a.ClockUncertaintyNS)
				if !policy.Now.IsZero() && a.ClockUncertaintyNS >= 0 {
					if a.CollectedAt.After(policy.Now.Add(artifactUncertainty)) {
						addIndeterminate("artifact_collection_time_in_future")
					}
					if policy.MaxAge > 0 && policy.Now.Sub(a.CollectedAt) > policy.MaxAge+artifactUncertainty {
						addIndeterminate("artifact_expired")
					}
				}
				if !m.CapturedAt.IsZero() && a.ClockUncertaintyNS >= 0 {
				combined := boundedCombinedUncertainty(a.ClockUncertaintyNS, m.ClockUncertaintyNS)
					if a.CollectedAt.After(m.CapturedAt.Add(combined)) {
						addIndeterminate("artifact_collected_after_manifest_capture")
					}
				}
			}

			code, reason := verifyArtifact(rootDir, a)
			switch code {
			case "ok":
				result.VerifiedArtifacts++
			case "unreadable":
				addIndeterminate(reason)
			default:
				addBlocked(reason)
			}
		}
	}
	validateArtifacts()
	if reason, isBlocked := verifyExactInventory(rootDir, declaredPaths); reason != "" {
		if isBlocked {
			addBlocked(reason)
		} else {
			addIndeterminate(reason)
		}
	}

	if m.Sequence == 1 {
		if m.PreviousRootSHA256 != "" {
			addBlocked("first_bundle_has_previous_root")
		}
		if policy.ExpectedPreviousRoot != "" {
			addBlocked("unexpected_previous_root_policy_for_first_bundle")
		}
	} else {
		if !validSHA256(m.PreviousRootSHA256) {
			addIndeterminate("previous_root_missing_or_malformed")
		} else if policy.ExpectedPreviousRoot == "" {
			addIndeterminate("expected_previous_root_unavailable")
		} else if !strings.EqualFold(m.PreviousRootSHA256, policy.ExpectedPreviousRoot) {
			addBlocked("previous_root_mismatch")
		}
	}

	result.SignerKeyID = m.Signature.KeyID
	if m.Signature.Algorithm == "" || m.Signature.Value == "" || m.Signature.KeyID == "" {
		addIndeterminate("signature_missing")
	} else if m.Signature.Algorithm != "Ed25519" {
		addBlocked("unsupported_signature_algorithm")
	} else {
		pub, ok := policy.TrustedKeys[m.Signature.KeyID]
		if !ok {
			addIndeterminate("signer_key_unknown")
		} else if len(pub) != ed25519.PublicKeySize {
			addBlocked("trusted_public_key_malformed")
		} else {
			sig, decodeErr := base64.StdEncoding.DecodeString(m.Signature.Value)
			if decodeErr != nil || len(sig) != ed25519.SignatureSize {
				addBlocked("signature_encoding_invalid")
			} else if root == "" || !ed25519.Verify(pub, signatureMessage(root, m.Signature.Algorithm, m.Signature.KeyID), sig) {
				addBlocked("signature_invalid")
			}
		}
	}

	switch {
	case blocked:
		result.Verdict = VerdictBlocked
	case indeterminate:
		result.Verdict = VerdictIndeterminate
	default:
		result.Verdict = VerdictQualified
	}
	for reason := range reasons {
		result.ReasonCodes = append(result.ReasonCodes, reason)
	}
	sort.Strings(result.ReasonCodes)
	return result
}

func verifyArtifact(rootDir string, a Artifact) (string, string) {
	fullPath, err := safeArtifactPath(rootDir, a.Path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return "blocked", "artifact_missing"
		}
		if errors.Is(err, ErrArtifactNotFile) {
			return "blocked", "artifact_not_regular_file"
		}
		return "blocked", "artifact_path_unsafe"
	}
	f, err := os.Open(fullPath)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return "blocked", "artifact_missing"
		}
		return "unreadable", "artifact_unreadable"
	}
	defer f.Close()

	before, err := f.Stat()
	if err != nil {
		return "unreadable", "artifact_stat_failed"
	}
	if !before.Mode().IsRegular() {
		return "blocked", "artifact_not_regular_file"
	}
	if before.Size() != a.SizeBytes {
		return "blocked", "artifact_size_mismatch"
	}

	h := sha256.New()
	n, err := io.Copy(h, f)
	if err != nil {
		return "unreadable", "artifact_read_failed"
	}
	after, err := f.Stat()
	if err != nil {
		return "unreadable", "artifact_post_read_stat_failed"
	}
	if n != a.SizeBytes || after.Size() != before.Size() || !after.ModTime().Equal(before.ModTime()) {
		return "blocked", "artifact_changed_during_verification"
	}
	if !strings.EqualFold(hex.EncodeToString(h.Sum(nil)), a.SHA256) {
		return "blocked", "artifact_digest_mismatch"
	}
	return "ok", ""
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
		return "", ErrUnsafePath
	}

	parts := strings.Split(relative, "/")
	current := root
	for i, part := range parts {
		current = filepath.Join(current, part)
		info, statErr := os.Lstat(current)
		if statErr != nil {
			return "", statErr
		}
		if info.Mode()&os.ModeSymlink != 0 {
			return "", ErrUnsafePath
		}
		if i < len(parts)-1 && !info.IsDir() {
			return "", ErrUnsafePath
		}
		if i == len(parts)-1 && !info.Mode().IsRegular() {
			return "", ErrArtifactNotFile
		}
	}
	rel, err := filepath.Rel(root, current)
	if err != nil || rel == "." || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		return "", ErrUnsafePath
	}
	return current, nil
}

func validateRelativePath(p string) error {
	if p == "" || strings.ContainsRune(p, 0) || strings.Contains(p, "\\") || pathpkg.IsAbs(p) {
		return ErrUnsafePath
	}
	clean := pathpkg.Clean(p)
	if clean == "." || clean == ".." || strings.HasPrefix(clean, "../") || clean != p {
		return ErrUnsafePath
	}
	return nil
}


func verifyExactInventory(rootDir string, declared map[string]struct{}) (string, bool) {
	root, err := filepath.Abs(rootDir)
	if err != nil {
		return "bundle_root_unreadable", false
	}
	rootInfo, err := os.Lstat(root)
	if err != nil || !rootInfo.IsDir() || rootInfo.Mode()&os.ModeSymlink != 0 {
		return "bundle_root_invalid", true
	}
	var reason string
	blocked := false
	walkErr := filepath.WalkDir(root, func(current string, entry os.DirEntry, err error) error {
		if err != nil {
			reason = "bundle_inventory_unreadable"
			return err
		}
		if current == root {
			return nil
		}
		if entry.Type()&os.ModeSymlink != 0 {
			reason = "unlisted_or_indirected_symlink"
			blocked = true
			return nil
		}
		if entry.IsDir() {
			return nil
		}
		info, statErr := entry.Info()
		if statErr != nil {
			reason = "bundle_inventory_unreadable"
			return statErr
		}
		if !info.Mode().IsRegular() {
			reason = "bundle_contains_nonregular_file"
			blocked = true
			return nil
		}
		relative, relErr := filepath.Rel(root, current)
		if relErr != nil {
			reason = "bundle_inventory_unreadable"
			return relErr
		}
		relative = filepath.ToSlash(relative)
		if _, ok := declared[relative]; !ok {
			reason = "unlisted_artifact_file"
			blocked = true
			return nil
		}
		return nil
	})
	if walkErr != nil && reason == "" {
		return "bundle_inventory_unreadable", false
	}
	return reason, blocked
}

func completeBoundary(s SourceBoundary) bool {
	return s.IncidentID != "" && s.SourceClusterID != "" && s.Database != "" &&
		s.SystemIdentifier != "" && s.TimelineID != "" && s.SnapshotID != "" &&
		s.CaptureLSN != "" && validSHA256(s.ScopeDigest) && validSHA256(s.InventoryDigest)
}

func validSHA256(s string) bool {
	if len(s) != sha256.Size*2 {
		return false
	}
	_, err := hex.DecodeString(s)
	return err == nil
}

func boundedCombinedUncertainty(a, b int64) time.Duration {
	if a < 0 || b < 0 {
		return time.Duration(1<<63 - 1)
	}
	maxDuration := time.Duration(1<<63 - 1)
	left := time.Duration(a)
	right := time.Duration(b)
	if left > maxDuration-right {
		return maxDuration
	}
	return left + right
}

func signatureMessage(root, algorithm, keyID string) []byte {
	return []byte(signatureDomain + algorithm + "\x00" + keyID + "\x00" + strings.ToLower(root))
}

func writeString(h hash.Hash, value string) {
	writeUint64(h, uint64(len(value)))
	_, _ = io.WriteString(h, value)
}

func writeUint64(h hash.Hash, value uint64) {
	var b [8]byte
	binary.BigEndian.PutUint64(b[:], value)
	_, _ = h.Write(b[:])
}

func writeInt64(h hash.Hash, value int64) {
	writeUint64(h, uint64(value))
}

func writeTime(h hash.Hash, value time.Time) {
	if value.IsZero() {
		writeString(h, "")
		return
	}
	writeString(h, value.UTC().Format(time.RFC3339Nano))
}

// DescribeRootMismatch is a small diagnostic helper for operators.
func DescribeRootMismatch(expected, observed string) error {
	if !validSHA256(expected) || !validSHA256(observed) {
		return fmt.Errorf("%w: malformed SHA-256 value", ErrInvalidManifest)
	}
	if !strings.EqualFold(expected, observed) {
		return fmt.Errorf("%w: expected %s, observed %s", ErrInvalidManifest, expected, observed)
	}
	return nil
}
