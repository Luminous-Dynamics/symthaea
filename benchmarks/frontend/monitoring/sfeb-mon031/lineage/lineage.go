// Package lineage qualifies a physical backup candidate against an explicit
// PostgreSQL version-transition and parent-chain policy. It does not execute
// backup or restore commands and does not replace pgBackRest verification.
package lineage

import (
	"errors"
	"fmt"
	"sort"
	"strconv"
	"strings"
	"time"
)

type BackupType string

const (
	BackupFull        BackupType = "full"
	BackupIncremental BackupType = "incremental"
)

type EvidenceStatus string

const (
	EvidenceVerified EvidenceStatus = "verified"
	EvidenceFailed   EvidenceStatus = "failed"
	EvidenceUnknown  EvidenceStatus = "unknown"
)

type Verdict string

const (
	Qualified     Verdict = "qualified"
	Blocked       Verdict = "blocked"
	Indeterminate Verdict = "indeterminate"
)

type BackupRecord struct {
	ID               string         `json:"id"`
	Type             BackupType     `json:"type"`
	ServerVersion    string         `json:"server_version"`
	SystemIdentifier string         `json:"system_identifier"`
	CreatedAt        time.Time      `json:"created_at"`
	ParentID         string         `json:"parent_id,omitempty"`
	IntegrityStatus  EvidenceStatus `json:"integrity_status"`
	WALArchiveStatus EvidenceStatus `json:"wal_archive_status"`
}

type Policy struct {
	CandidateID              string `json:"candidate_id"`
	CurrentServerVersion     string `json:"current_server_version"`
	ExpectedSystemIdentifier string `json:"expected_system_identifier"`
	MinimumFullBackupVersion string `json:"minimum_full_backup_version,omitempty"`
}

type Result struct {
	Verdict        Verdict  `json:"verdict"`
	ReasonCodes    []string `json:"reason_codes"`
	CandidateID    string   `json:"candidate_id"`
	RootFullID     string   `json:"root_full_id,omitempty"`
	ChainBackupIDs []string `json:"chain_backup_ids,omitempty"`
}

var (
	ErrInvalidPolicy = errors.New("invalid backup-lineage policy")
	ErrDuplicateID   = errors.New("duplicate backup ID")
)

type version struct{ major, minor, patch int }

func parseVersion(value string) (version, error) {
	var v version
	parts := strings.Split(strings.TrimSpace(value), ".")
	if len(parts) < 2 || len(parts) > 3 {
		return v, fmt.Errorf("invalid PostgreSQL version %q", value)
	}
	parsePart := func(part string) (int, error) {
		if part == "" {
			return 0, errors.New("empty version component")
		}
		for _, char := range part {
			if char < '0' || char > '9' {
				return 0, errors.New("non-numeric version component")
			}
		}
		return strconv.Atoi(part)
	}
	var err error
	if v.major, err = parsePart(parts[0]); err != nil || v.major < 10 {
		return version{}, fmt.Errorf("invalid PostgreSQL major version %q", value)
	}
	if v.minor, err = parsePart(parts[1]); err != nil {
		return version{}, fmt.Errorf("invalid PostgreSQL minor version %q", value)
	}
	if len(parts) == 3 {
		if v.patch, err = parsePart(parts[2]); err != nil {
			return version{}, fmt.Errorf("invalid PostgreSQL patch version %q", value)
		}
	}
	return v, nil
}

func compareVersion(a, b version) int {
	if a.major != b.major {
		if a.major < b.major {
			return -1
		}
		return 1
	}
	if a.minor != b.minor {
		if a.minor < b.minor {
			return -1
		}
		return 1
	}
	if a.patch != b.patch {
		if a.patch < b.patch {
			return -1
		}
		return 1
	}
	return 0
}

// Evaluate requires a unique, causal chain terminating in a full backup.
// Every record must have verified integrity; the selected candidate must
// additionally have verified WAL archive coverage for its declared boundary.
func Evaluate(policy Policy, records []BackupRecord) Result {
	result := Result{
		Verdict:     Qualified,
		ReasonCodes: []string{},
		CandidateID: policy.CandidateID,
	}
	reasons := map[string]struct{}{}
	blocked, indeterminate := false, false
	addBlocked := func(code string) {
		blocked = true
		reasons[code] = struct{}{}
	}
	addUnknown := func(code string) {
		indeterminate = true
		reasons[code] = struct{}{}
	}

	current, currentErr := parseVersion(policy.CurrentServerVersion)
	if currentErr != nil || strings.TrimSpace(policy.CandidateID) == "" || strings.TrimSpace(policy.ExpectedSystemIdentifier) == "" {
		addUnknown("policy_incomplete_or_version_invalid")
	}
	var minimumFull *version
	if policy.MinimumFullBackupVersion != "" {
		parsed, err := parseVersion(policy.MinimumFullBackupVersion)
		if err != nil {
			addUnknown("minimum_full_backup_version_invalid")
		} else {
			minimumFull = &parsed
			if currentErr == nil && parsed.major != current.major {
				addBlocked("minimum_full_backup_version_major_mismatch")
			}
		}
	}

	byID := make(map[string]BackupRecord, len(records))
	for _, record := range records {
		if strings.TrimSpace(record.ID) == "" {
			addBlocked("backup_id_missing")
			continue
		}
		if _, exists := byID[record.ID]; exists {
			addBlocked("backup_id_duplicated")
			continue
		}
		byID[record.ID] = record
	}
	candidate, ok := byID[policy.CandidateID]
	if !ok {
		addUnknown("candidate_backup_missing")
	}

	if ok {
		visited := map[string]struct{}{}
		chain := make([]BackupRecord, 0, 8)
		currentID := candidate.ID
		for currentID != "" {
			if _, exists := visited[currentID]; exists {
				addBlocked("backup_parent_cycle")
				break
			}
			visited[currentID] = struct{}{}
			record, exists := byID[currentID]
			if !exists {
				addUnknown("backup_parent_missing")
				break
			}
			chain = append(chain, record)
			if record.CreatedAt.IsZero() {
				addUnknown("backup_timestamp_missing")
			}
			if record.SystemIdentifier == "" {
				addUnknown("backup_system_identifier_missing")
			} else if policy.ExpectedSystemIdentifier != "" && record.SystemIdentifier != policy.ExpectedSystemIdentifier {
				addBlocked("backup_system_identifier_mismatch")
			}
			if record.IntegrityStatus == EvidenceFailed {
				addBlocked("backup_integrity_failed")
			} else if record.IntegrityStatus != EvidenceVerified {
				addUnknown("backup_integrity_unverified")
			}
			v, err := parseVersion(record.ServerVersion)
			if err != nil {
				addUnknown("backup_server_version_invalid")
			} else if currentErr == nil && v.major != current.major {
				addBlocked("backup_major_version_mismatch")
			}

			switch record.Type {
			case BackupFull:
				if record.ParentID != "" {
					addBlocked("full_backup_has_parent")
				}
				if v, err := parseVersion(record.ServerVersion); err == nil && minimumFull != nil && compareVersion(v, *minimumFull) < 0 {
					addBlocked("full_backup_predates_required_version")
				}
				currentID = ""
			case BackupIncremental:
				if record.ParentID == "" {
					addUnknown("incremental_parent_missing")
					currentID = ""
					continue
				}
				parent, parentExists := byID[record.ParentID]
				if !parentExists {
					addUnknown("backup_parent_missing")
					currentID = ""
					continue
				}
				if !record.CreatedAt.IsZero() && !parent.CreatedAt.IsZero() && !parent.CreatedAt.Before(record.CreatedAt) {
					addBlocked("backup_parent_not_older")
				}
				childVersion, childErr := parseVersion(record.ServerVersion)
				parentVersion, parentErr := parseVersion(parent.ServerVersion)
				if childErr == nil && parentErr == nil && compareVersion(childVersion, parentVersion) < 0 {
					addBlocked("backup_version_regressed_in_chain")
				}
				if childErr == nil && minimumFull != nil && compareVersion(childVersion, *minimumFull) < 0 {
					addBlocked("incremental_backup_predates_required_version")
				}
				currentID = record.ParentID
			default:
				addBlocked("backup_type_unsupported")
				currentID = ""
			}
		}
		if len(chain) > 0 {
			last := chain[len(chain)-1]
			if last.Type != BackupFull {
				addUnknown("full_backup_root_unresolved")
			} else {
				result.RootFullID = last.ID
			}
			ids := make([]string, len(chain))
			for i := range chain {
				ids[i] = chain[i].ID
			}
			result.ChainBackupIDs = ids
		}
		if candidate.WALArchiveStatus == EvidenceFailed {
			addBlocked("wal_archive_integrity_failed")
		} else if candidate.WALArchiveStatus != EvidenceVerified {
			addUnknown("wal_archive_coverage_unverified")
		}
	}

	switch {
	case blocked:
		result.Verdict = Blocked
	case indeterminate:
		result.Verdict = Indeterminate
	default:
		result.Verdict = Qualified
	}
	for reason := range reasons {
		result.ReasonCodes = append(result.ReasonCodes, reason)
	}
	sort.Strings(result.ReasonCodes)
	return result
}
