package lineage

import (
	"testing"
	"time"
)

func record(id string, kind BackupType, version, parent string, created string) BackupRecord {
	stamp, _ := time.Parse(time.RFC3339, created)
	return BackupRecord{
		ID: id, Type: kind, ServerVersion: version,
		SystemIdentifier: "7438291029384756102", CreatedAt: stamp,
		ParentID: parent, IntegrityStatus: EvidenceVerified, WALArchiveStatus: EvidenceVerified,
	}
}

func policy(candidate string) Policy {
	return Policy{
		CandidateID: candidate, CurrentServerVersion: "18.7",
		ExpectedSystemIdentifier: "7438291029384756102", MinimumFullBackupVersion: "18.7",
	}
}

func has(result Result, code string) bool {
	for _, reason := range result.ReasonCodes {
		if reason == code {
			return true
		}
	}
	return false
}

func TestFreshPostUpgradeFullBackupQualifies(t *testing.T) {
	full := record("full-187", BackupFull, "18.7", "", "2026-11-20T10:00:00Z")
	got := Evaluate(policy("full-187"), []BackupRecord{full})
	if got.Verdict != Qualified || got.RootFullID != "full-187" {
		t.Fatalf("expected qualified full backup, got %+v", got)
	}
}

func TestIncrementalChainFromFreshFullQualifies(t *testing.T) {
	full := record("full-187", BackupFull, "18.7", "", "2026-11-20T10:00:00Z")
	inc1 := record("inc-1", BackupIncremental, "18.7", "full-187", "2026-11-20T11:00:00Z")
	inc2 := record("inc-2", BackupIncremental, "18.7", "inc-1", "2026-11-20T12:00:00Z")
	got := Evaluate(policy("inc-2"), []BackupRecord{inc2, inc1, full})
	if got.Verdict != Qualified || got.RootFullID != "full-187" || len(got.ChainBackupIDs) != 3 {
		t.Fatalf("expected qualified chain, got %+v", got)
	}
}

func TestPre18_7FullBackupBlocksIncrementalChain(t *testing.T) {
	full := record("full-186", BackupFull, "18.6", "", "2026-11-10T10:00:00Z")
	inc := record("inc-187", BackupIncremental, "18.7", "full-186", "2026-11-20T10:00:00Z")
	got := Evaluate(policy("inc-187"), []BackupRecord{full, inc})
	if got.Verdict != Blocked || !has(got, "full_backup_predates_required_version") {
		t.Fatalf("pre-18.7 full base must be rejected, got %+v", got)
	}
}

func TestMissingParentIsIndeterminate(t *testing.T) {
	inc := record("inc-1", BackupIncremental, "18.7", "missing-full", "2026-11-20T11:00:00Z")
	got := Evaluate(policy("inc-1"), []BackupRecord{inc})
	if got.Verdict != Indeterminate || !has(got, "backup_parent_missing") {
		t.Fatalf("missing parent must be indeterminate, got %+v", got)
	}
}

func TestUnverifiedIntegrityIsIndeterminate(t *testing.T) {
	full := record("full-187", BackupFull, "18.7", "", "2026-11-20T10:00:00Z")
	full.IntegrityStatus = EvidenceUnknown
	got := Evaluate(policy("full-187"), []BackupRecord{full})
	if got.Verdict != Indeterminate || !has(got, "backup_integrity_unverified") {
		t.Fatalf("unknown integrity must not qualify, got %+v", got)
	}
}

func TestFailedWALArchiveBlocks(t *testing.T) {
	full := record("full-187", BackupFull, "18.7", "", "2026-11-20T10:00:00Z")
	full.WALArchiveStatus = EvidenceFailed
	got := Evaluate(policy("full-187"), []BackupRecord{full})
	if got.Verdict != Blocked || !has(got, "wal_archive_integrity_failed") {
		t.Fatalf("failed archive evidence must block, got %+v", got)
	}
}

func TestCrossMajorBackupBlocks(t *testing.T) {
	full := record("full-19", BackupFull, "19.0", "", "2026-11-20T10:00:00Z")
	got := Evaluate(policy("full-19"), []BackupRecord{full})
	if got.Verdict != Blocked || !has(got, "backup_major_version_mismatch") {
		t.Fatalf("cross-major physical backup must block, got %+v", got)
	}
}

func TestParentTimestampMustPrecedeChild(t *testing.T) {
	full := record("full-187", BackupFull, "18.7", "", "2026-11-20T12:00:00Z")
	inc := record("inc-1", BackupIncremental, "18.7", "full-187", "2026-11-20T11:00:00Z")
	got := Evaluate(policy("inc-1"), []BackupRecord{full, inc})
	if got.Verdict != Blocked || !has(got, "backup_parent_not_older") {
		t.Fatalf("non-causal backup chain must block, got %+v", got)
	}
}

func TestParentCycleBlocks(t *testing.T) {
	a := record("a", BackupIncremental, "18.7", "b", "2026-11-20T11:00:00Z")
	b := record("b", BackupIncremental, "18.7", "a", "2026-11-20T12:00:00Z")
	got := Evaluate(policy("a"), []BackupRecord{a, b})
	if got.Verdict != Blocked || !has(got, "backup_parent_cycle") {
		t.Fatalf("parent cycle must block, got %+v", got)
	}
}

func TestDuplicateBackupIDBlocks(t *testing.T) {
	full := record("full-187", BackupFull, "18.7", "", "2026-11-20T10:00:00Z")
	got := Evaluate(policy("full-187"), []BackupRecord{full, full})
	if got.Verdict != Blocked || !has(got, "backup_id_duplicated") {
		t.Fatalf("duplicate identity must block, got %+v", got)
	}
}

func TestMissingArchiveEvidenceIsIndeterminate(t *testing.T) {
	full := record("full-187", BackupFull, "18.7", "", "2026-11-20T10:00:00Z")
	full.WALArchiveStatus = EvidenceUnknown
	got := Evaluate(policy("full-187"), []BackupRecord{full})
	if got.Verdict != Indeterminate || !has(got, "wal_archive_coverage_unverified") {
		t.Fatalf("unknown WAL coverage must remain indeterminate, got %+v", got)
	}
}

func TestVersionParserRejectsSuffixes(t *testing.T) {
	if _, err := parseVersion("18.7junk"); err == nil {
		t.Fatal("non-numeric suffix must not be silently accepted")
	}
}
