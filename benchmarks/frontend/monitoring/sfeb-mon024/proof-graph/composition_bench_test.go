package composition

import "testing"

func BenchmarkEvaluateQualifiedGraph(b *testing.B) {
	in := fixture()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = Evaluate(in)
	}
}

func BenchmarkEvaluateAdversarialGraph(b *testing.B) {
	in := fixture()
	in.Receipts[2].Checks["ack_within_durable_watermark"] = CheckFailed
	in.UnresolvedMutationIDs = []string{"unknown-command"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = Evaluate(in)
	}
}
