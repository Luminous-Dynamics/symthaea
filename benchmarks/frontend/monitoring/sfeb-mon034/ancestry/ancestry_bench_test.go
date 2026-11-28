package ancestry

import (
	"fmt"
	"strings"
	"testing"
)

func BenchmarkParseTimelineHistory1000(b *testing.B) {
	var builder strings.Builder
	for i := 1; i <= 1000; i++ {
		fmt.Fprintf(&builder, "%d\t0/%08X\tpromotion %d\n", i, i*4096, i)
	}
	history := builder.String()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		segments, err := ParseHistory(1001, history)
		if err != nil || len(segments) != 1001 {
			b.Fatalf("history parse failed: %v", err)
		}
	}
}


func benchmarkInput(lines int) Input {
	var builder strings.Builder
	for i := 1; i <= lines; i++ {
		fmt.Fprintf(&builder, "%d\t0/%08X\tpromotion %d\n", i, i*4096, i)
	}
	history := builder.String()
	return Input{
		Profile: Profile,
		Version: Version,
		ExpectedSystemIdentifier: "7438291029384756102",
		BaseBackupSystemIdentifier: "7438291029384756102",
		BaseBackupTimelineID: 1,
		BaseBackupLSN: "0/00000100",
		RecoveredSystemIdentifier: "7438291029384756102",
		RecoveredTimelineID: uint32(lines + 1),
		RecoveryTargetLSN: FormatLSN(uint64(lines+1) * 4096),
		HistoryTimelineID: uint32(lines + 1),
		HistorySHA256: HistorySHA256(history),
		HistoryText: history,
	}
}

func BenchmarkEvaluateTimelineChain1000(b *testing.B) {
	input := benchmarkInput(1000)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		result := Evaluate(input)
		if result.Verdict != Qualified {
			b.Fatalf("1000-entry ancestry did not qualify: %+v", result)
		}
	}
}
