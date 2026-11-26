package capture

import "testing"

func FuzzGapNeverQualifiesComplete(f *testing.F) {
	f.Add(uint64(5), uint64(9), uint64(9))
	f.Add(uint64(0), uint64(100), uint64(100))
	f.Fuzz(func(t *testing.T, start, end, watermark uint64) {
		if end <= start || watermark < end || end-start < 2 || end-start > 1_000_000 {
			t.Skip()
		}
		m := baseManifest(start, end, watermark, item(end, "last-event", 'c'))
		r := VerifyBoundedCapture(m, nil)
		if r.Coverage == CoverageComplete {
			t.Fatalf("gap was promoted complete: start=%d end=%d report=%+v", start, end, r)
		}
	})
}

func FuzzSharedRootNeverProducesWitnessedNegative(f *testing.F) {
	f.Add("root-A")
	f.Add("different")
	f.Fuzz(func(t *testing.T, root string) {
		if root == "" {
			t.Skip()
		}
		m := baseManifest(1, 1, 1)
		m.Source.ControlRoot = root
		w := independentWitness(m)
		w.ControlRoot = root
		r := VerifyBoundedCapture(m, w)
		if r.NegativeEvidence == NoEventsWitnessed {
			t.Fatalf("shared root witnessed absence: %+v", r)
		}
	})
}

func FuzzQuerySubstitutionNeverQualifies(f *testing.F) {
	f.Add("different-snapshot")
	f.Add("other-stream")
	f.Fuzz(func(t *testing.T, substitute string) {
		if substitute == "" {
			t.Skip()
		}
		m := baseManifest(2, 2, 2)
		m.Source.SnapshotToken = substitute
		if m.Source.SnapshotToken == m.Boundary.SnapshotToken {
			t.Skip()
		}
		r := VerifyBoundedCapture(m, nil)
		if r.Coverage == CoverageComplete {
			t.Fatalf("query/snapshot substitution promoted: %+v", r)
		}
	})
}
