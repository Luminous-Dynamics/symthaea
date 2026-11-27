package lineage

import (
	"fmt"
	"testing"
	"time"
)

func BenchmarkEvaluateTwentyBackupChain(b *testing.B) {
	base := time.Date(2026, 11, 20, 10, 0, 0, 0, time.UTC)
	records := make([]BackupRecord, 20)
	for i := range records {
		kind := BackupIncremental
		parent := fmt.Sprintf("inc-%02d", i-1)
		if i == 0 {
			kind = BackupFull
			parent = ""
		}
		records[i] = BackupRecord{
			ID: fmt.Sprintf("inc-%02d", i), Type: kind, ServerVersion: "18.7",
			SystemIdentifier: "7438291029384756102", CreatedAt: base.Add(time.Duration(i) * time.Minute),
			ParentID: parent, IntegrityStatus: EvidenceVerified, WALArchiveStatus: EvidenceVerified,
		}
	}
	p := Policy{
		CandidateID: "inc-19", CurrentServerVersion: "18.7",
		ExpectedSystemIdentifier: "7438291029384756102", MinimumFullBackupVersion: "18.7",
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if result := Evaluate(p, records); result.Verdict != Qualified {
			b.Fatalf("unexpected verdict: %+v", result)
		}
	}
}
