package main

import (
	"strings"
	"testing"
)

func TestDecodeStrictRejectsUnknownAndTrailingJSON(t *testing.T) {
	good := "{\"profile\":\"mon031-backup-lineage-input-v1\",\"version\":\"1.0.0\",\"policy\":{\"candidate_id\":\"x\",\"current_server_version\":\"18.7\",\"expected_system_identifier\":\"1\",\"minimum_full_backup_version\":\"18.7\"},\"backups\":[{\"id\":\"x\",\"type\":\"full\",\"server_version\":\"18.7\",\"system_identifier\":\"1\",\"created_at\":\"2026-11-20T10:00:00Z\",\"integrity_status\":\"verified\",\"wal_archive_status\":\"verified\"}]}"
	if _, err := decodeStrict(strings.NewReader(good)); err != nil {
		t.Fatalf("valid input rejected: %v", err)
	}
	unknown := strings.TrimSuffix(good, "}") + ",\"unexpected\":true}"
	if _, err := decodeStrict(strings.NewReader(unknown)); err == nil {
		t.Fatal("unknown top-level field accepted")
	}
	if _, err := decodeStrict(strings.NewReader(good + " {}")); err == nil {
		t.Fatal("trailing JSON document accepted")
	}
}
