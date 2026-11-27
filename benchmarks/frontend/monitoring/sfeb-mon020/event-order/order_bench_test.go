package eventorder

import (
	"fmt"
	"testing"
)

func benchmarkInput(n int, crossSource bool) (Input, map[string]VerifierPin) {
	sources := []Source{{ID: "agent", ControlRoot: "agent-root", IdentityDigest: dig("agent", "a-key", "source-identity"), ControlRootDigest: dig("agent", "agent-root", "control-root-membership")}}
	if crossSource {
		sources = append(sources,
			Source{ID: "db", ControlRoot: "db-root", IdentityDigest: dig("db", "d-key", "source-identity"), ControlRootDigest: dig("db", "db-root", "control-root-membership")},
			Source{ID: "observer", ControlRoot: "observer-root", IdentityDigest: dig("observer", "o-key", "source-identity"), ControlRootDigest: dig("observer", "observer-root", "control-root-membership")},
		)
	}
	in := Input{Sources: sources, EvaluationAt: 100}
	in.Events = append(in.Events,
		Event{ID: "start-producer", Kind: ActivityStarted, SubjectID: "producer", ActivityID: "producer", SourceID: "agent", Sequence: 1, RecordDigest: dig("start-producer", "sp", "event")},
	)
	if crossSource {
		in.Events = append(in.Events, Event{ID: "start-consumer", Kind: ActivityStarted, SubjectID: "consumer", ActivityID: "consumer", SourceID: "db", Sequence: 1, RecordDigest: dig("start-consumer", "sc", "event")})
	} else {
		in.Events = append(in.Events, Event{ID: "start-consumer", Kind: ActivityStarted, SubjectID: "consumer", ActivityID: "consumer", SourceID: "agent", Sequence: 2, RecordDigest: dig("start-consumer", "sc", "event")})
	}
	for i := 0; i < n; i++ {
		id := fmt.Sprintf("entity-%08d", i)
		genID := "gen-" + id
		useID := "use-" + id
		genSeq := uint64(3 + i)
		useSeq := uint64(3 + i)
		if !crossSource {
			genSeq = uint64(3 + 2*i)
			useSeq = uint64(4 + 2*i)
		}
		gd := dig(genID, "gen-value", "event")
		ud := dig(useID, "use-value", "event")
		in.Events = append(in.Events,
			Event{ID: genID, Kind: EntityGenerated, SubjectID: id, ActivityID: "producer", SourceID: "agent", Sequence: genSeq, RecordDigest: gd},
			Event{ID: useID, Kind: EntityUsed, SubjectID: id, ActivityID: "consumer", SourceID: map[bool]string{true: "db", false: "agent"}[crossSource], Sequence: useSeq, RecordDigest: ud},
		)
		if crossSource {
			wd := dig("witness-"+id, "handoff", "order-witness")
			in.Witnesses = append(in.Witnesses, OrderWitness{ID: "witness-" + id, BeforeEventID: genID, AfterEventID: useID, WitnessSourceID: "observer", EvidenceDigest: wd})
		}
	}
	producerEndSeq := uint64(3 + n*2)
	consumerEndSource := "agent"
	consumerEndSeq := uint64(4 + n*2)
	if crossSource {
		producerEndSeq = uint64(3 + n)
		consumerEndSource = "db"
		consumerEndSeq = uint64(3 + n)
	}
	in.Events = append(in.Events,
		Event{ID: "end-producer", Kind: ActivityEnded, SubjectID: "producer", ActivityID: "producer", SourceID: "agent", Sequence: producerEndSeq, RecordDigest: dig("end-producer", "ep", "event")},
		Event{ID: "end-consumer", Kind: ActivityEnded, SubjectID: "consumer", ActivityID: "consumer", SourceID: consumerEndSource, Sequence: consumerEndSeq, RecordDigest: dig("end-consumer", "ec", "event")},
	)
	for _, s := range sources {
		in.Reports = append(in.Reports, normalizedReport(&in, "source:"+s.ID, s.IdentityDigest))
		in.Reports = append(in.Reports, normalizedReport(&in, "control-root:"+s.ID, s.ControlRootDigest))
	}
	for _, e := range in.Events {
		in.Reports = append(in.Reports, normalizedReport(&in, "event:"+e.ID, e.RecordDigest))
	}
	for _, w := range in.Witnesses {
		in.Reports = append(in.Reports, normalizedReport(&in, "witness:"+w.ID, w.EvidenceDigest))
	}
	return in, map[string]VerifierPin{"v1": {"v1", "1", "b1", "roots1"}}
}

func BenchmarkEvaluateSameSource(b *testing.B) {
	for _, n := range []int{10, 100, 1000, 10000} {
		b.Run(fmt.Sprintf("entities_%d", n), func(b *testing.B) {
			in, pins := benchmarkInput(n, false)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				r := Evaluate(in, pins)
				if r.Decision != Qualified {
					b.Fatalf("got %s: %s", r.Decision, r.Reason)
				}
			}
		})
	}
}

func BenchmarkEvaluateCrossSourceWitness(b *testing.B) {
	for _, n := range []int{10, 100, 1000, 5000} {
		b.Run(fmt.Sprintf("witnesses_%d", n), func(b *testing.B) {
			in, pins := benchmarkInput(n, true)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				r := Evaluate(in, pins)
				if r.Decision != Qualified {
					b.Fatalf("got %s: %s", r.Decision, r.Reason)
				}
			}
		})
	}
}