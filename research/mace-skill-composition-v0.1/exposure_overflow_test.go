package main

import (
	"strings"
	"testing"
)

func TestExposureCounterOverflowFailsClosed(t *testing.T) {
	x := fixtureRequest()
	maxInt := int(^uint(0) >> 1)
	x.Context.ExposureBudget = 10
	x.Skills[0].MaxTokens = maxInt
	x.Plan.Steps[0].Tokens = maxInt

	s2 := x.Skills[0]
	s2.ID = "s2"
	s2.Digest = strings.Repeat("c", 64)
	x.Skills = append(x.Skills, s2)
	x.Plan.Steps = append(x.Plan.Steps, Step{
		ID: "st2", SkillID: "s2", Inputs: []string{"o1"}, OutputID: "o2",
		DeclaredOutputLabel: "internal", Target: "case-7",
		ExposureMode: "summary", Audience: "support-team", Tokens: maxInt,
	})
	x.Plan.Sinks = []Sink{{ArtifactID: "o2", Kind: "internal", Target: "case-7", Audience: "support-team", MaxLabel: "internal"}}

	r := Validate(x)
	if r.Accepted || !findingCode(r, "EXPOSURE_COUNTER_OVERFLOW") {
		t.Fatalf("integer overflow must fail closed: accepted=%v findings=%+v exposure=%d", r.Accepted, r.Findings, r.ExposureTokens)
	}
	if r.ExposureTokens != maxInt {
		t.Fatalf("overflow must saturate exposure counter, got %d want %d", r.ExposureTokens, maxInt)
	}
}