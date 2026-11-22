package main

import (
	"strings"
	"testing"
)

// Compare the full two-step classification/read ceiling grid against a direct
// lattice oracle, independent of Validate's decision logic.
func TestGeneratedTwoStepReadLatticeAgainstIndependentOracle(t *testing.T) {
	labels := []string{"public", "internal", "confidential", "restricted"}
	for sourceLevel := 0; sourceLevel < len(labels); sourceLevel++ {
		for firstRead := 0; firstRead < len(labels); firstRead++ {
			for intermediateDeclared := 0; intermediateDeclared < len(labels); intermediateDeclared++ {
				for secondRead := 0; secondRead < len(labels); secondRead++ {
					t.Run(labels[sourceLevel]+"/"+labels[firstRead]+"/"+labels[intermediateDeclared]+"/"+labels[secondRead], func(t *testing.T) {
						x := baseRequest()
						x.Context.MaxActionLabel = "restricted"
						x.Sources[0].Label = labels[sourceLevel]
						x.Skills[0].ReadableMaxLabel = labels[firstRead]
						x.Skills[0].MaxTokens = 100
						s2 := x.Skills[0]
						s2.ID = "s2"
						s2.Digest = strings.Repeat("c", 64)
						s2.ReadableMaxLabel = labels[secondRead]
						s2.Conflicts = nil
						x.Skills = append(x.Skills, s2)
						x.Plan.Steps[0].DeclaredOutputLabel = labels[intermediateDeclared]
						x.Plan.Steps = append(x.Plan.Steps, Step{
							ID: "st2", SkillID: "s2", Inputs: []string{"o1"}, OutputID: "o2",
							DeclaredOutputLabel: "public", Target: "case-7", ExposureMode: "summary",
							Audience: "support-team", Tokens: 0,
						})
						x.Plan.Sinks = []Sink{{ArtifactID: "o2", Kind: "internal", Target: "case-7", Audience: "support-team", MaxLabel: "restricted"}}
						intermediate := sourceLevel
						if intermediateDeclared > intermediate {
							intermediate = intermediateDeclared
						}
						want := sourceLevel <= firstRead && intermediate <= secondRead
						got := Validate(x).Accepted
						if got != want {
							t.Fatalf("source=%s first-read=%s declared=%s second-read=%s: got=%v want=%v", labels[sourceLevel], labels[firstRead], labels[intermediateDeclared], labels[secondRead], got, want)
						}
					})
				}
			}
		}
	}
}
