package capture

import "testing"

func BenchmarkVerifyBoundedCaptureSmall(b *testing.B) {
	m := baseManifest(0, 4, 4, item(1, "e1", 'a'), item(2, "e2", 'b'), item(3, "e3", 'c'), item(4, "e4", 'd'))
	w := independentWitness(m)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = VerifyBoundedCapture(m, w)
	}
}

func BenchmarkVerifyBoundedCaptureLarge(b *testing.B) {
	items := make([]CapturedItem, 1000)
	for i := range items {
		items[i] = item(uint64(i+1), "event-"+string(rune(i+1)), byte(i%200+1))
	}
	m := baseManifest(0, 1000, 1000, items...)
	w := independentWitness(m)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = VerifyBoundedCapture(m, w)
	}
}
