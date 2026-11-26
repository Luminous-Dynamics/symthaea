package assurance

import "testing"

func BenchmarkCompareDigestEqual(b *testing.B) {
	a, c := requestEvidence(hexDigest('a')), requestEvidence(hexDigest('a'))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ { _ = CompareDigest(a, c) }
}
func BenchmarkCompareDigestIncomparable(b *testing.B) {
	a, c := requestEvidence(hexDigest('a')), requestEvidence(hexDigest('a'))
	c.Context.Profile = "urn:other"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ { _ = CompareDigest(a, c) }
}
func BenchmarkQualifyDigitalEffect(b *testing.B) {
	in := validDigitalInput()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ { _ = QualifyDigitalEffect(in) }
}
func BenchmarkProjectOARLifecycle(b *testing.B) {
	root := OARRecord{ReceiptID:"d", AgentID:"a", RunID:"r", ActionType:"refund", ActionTarget:"stripe", Phase:OARDecision, Result:"allowed", Verification:Verified}
	exec := OARRecord{ReceiptID:"e", DecisionRef:"d", AgentID:"a", RunID:"r", ActionType:"refund", ActionTarget:"stripe", Phase:OARExecution, Result:"executed", Verification:Verified}
	conf := OARRecord{ReceiptID:"c", DecisionRef:"e", AgentID:"a", RunID:"r", ActionType:"refund", ActionTarget:"stripe", Phase:OARConfirmation, Result:"confirmed", Verification:Verified}
	records := []OARRecord{root, exec, conf}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ { _ = ProjectOARLifecycle(records) }
}
