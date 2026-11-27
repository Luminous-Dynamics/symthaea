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

func benchmarkBundle(b *testing.B) (string, Manifest, Policy) {
	b.Helper()
	root := b.TempDir()
	data := make([]byte, 256*1024)
	for i := range data {
		data[i] = byte(i % 251)
	}
	if err := os.WriteFile(filepath.Join(root, "payload.bin"), data, 0o600); err != nil {
		b.Fatal(err)
	}
	sum := sha256.Sum256(data)
	now := time.Date(2026, 11, 27, 9, 0, 0, 0, time.UTC)
	pub, private, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		b.Fatal(err)
	}
	m := Manifest{
		Profile: ProfileV1, Version: VersionV1, RunID: "bench-run", Sequence: 1,
		CapturedAt: now, ClockDomain: "bench-clock", ClockUncertaintyNS: int64(time.Millisecond),
		Source: SourceBoundary{
			IncidentID: "incident", SourceClusterID: "cluster", Database: "db",
			SystemIdentifier: "123456789", TimelineID: "1", SnapshotID: "snapshot",
			CaptureLSN: "0/ABCDE", ScopeDigest: hex.EncodeToString(sha256.New().Sum([]byte("scope"))),
			InventoryDigest: hex.EncodeToString(sha256.New().Sum([]byte("inventory"))),
		},
		Artifacts: []Artifact{{
			Path: "payload.bin", MediaType: "application/octet-stream", Role: "payload",
			Producer: "benchmark", CommandID: "bench-command", SizeBytes: int64(len(data)),
			SHA256: hex.EncodeToString(sum[:]), CollectedAt: now,
			ClockDomain: "bench-clock", ClockUncertaintyNS: int64(time.Millisecond),
		}},
	}
	if err := SignManifest(&m, "bench-key", private); err != nil {
		b.Fatal(err)
	}
	return root, m, Policy{
		Now: now.Add(time.Second), MaxAge: time.Hour, ClockDomain: "bench-clock",
		MaxClockUncertaintyNS: int64(10 * time.Millisecond),
		TrustedKeys: map[string]ed25519.PublicKey{"bench-key": pub},
	}
}

func BenchmarkVerifyBundle(b *testing.B) {
	root, manifest, policy := benchmarkBundle(b)
	b.SetBytes(256 * 1024)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		result := VerifyBundle(root, manifest, policy)
		if result.Verdict != VerdictQualified {
			b.Fatalf("unexpected verifier verdict: %+v", result)
		}
	}
}
