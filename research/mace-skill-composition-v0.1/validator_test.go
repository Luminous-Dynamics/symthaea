package main

import (
	"strings"
	"testing"
)

func fixtureRequest() Request {
	return Request{
		Context: Context{
			RequestID: "r1", Principal: "alice", Scope: "tenant-a",
			Purpose: "support", Audience: "support-team", ExpectedTarget: "case-7",
			Capabilities: []string{"case:read", "case:update"},
			EvaluatedAt: "2026-11-11T08:00:00Z", ExposureBudget: 100,
			MaxActionLabel: "internal",
		},
		Skills: []Skill{{
			ID: "s1", Digest: strings.Repeat("a", 64), Status: "active",
			ExpiresAt: "2027-01-01T00:00:00Z", Scopes: []string{"tenant-a"},
			Purposes: []string{"support"}, Audiences: []string{"support-team"},
			ReadableMaxLabel: "internal", ExposureModes: []string{"summary"},
			RequiredCapabilities: []string{"case:read"}, MaxTokens: 80,
		}},
		Sources: []Artifact{{
			ID: "a1", Digest: strings.Repeat("b", 64), Label: "internal",
			Scope: "tenant-a", Purpose: "support", Audience: "support-team",
			Target: "case-7", Current: true,
		}},
		Plan: Plan{
			Steps: []Step{{
				ID: "st1", SkillID: "s1", Inputs: []string{"a1"}, OutputID: "o1",
				DeclaredOutputLabel: "public", Target: "case-7",
				ExposureMode: "summary", Audience: "support-team", Tokens: 20,
			}},
			Sinks: []Sink{{
				ArtifactID: "o1", Kind: "internal", Target: "case-7",
				Audience: "support-team", MaxLabel: "internal",
			}},
		},
	}
}

func findingCode(r Report, code string) bool {
	for _, f := range r.Findings {
		if f.Code == code { return true }
	}
	return false
}

func TestValidComposition(t *testing.T) {
	r := Validate(fixtureRequest())
	if !r.Accepted { t.Fatalf("unexpected rejection: %+v", r.Findings) }
	if r.Derived[0].Label != "internal" { t.Fatalf("classification downgraded: %+v", r.Derived) }
}

func TestClassificationDowngradeBlocked(t *testing.T) {
	x := fixtureRequest()
	x.Sources[0].Label = "confidential"
	r := Validate(x)
	if r.Accepted || !findingCode(r, "READ_LEVEL_EXCEEDED") { t.Fatalf("expected block: %+v", r.Findings) }
}

func TestCapabilityMintingBlocked(t *testing.T) {
	x := fixtureRequest()
	x.Skills[0].CapabilityGrants = []string{"admin"}
	r := Validate(x)
	if r.Accepted || !findingCode(r, "CAPABILITY_MINTING") { t.Fatalf("expected block: %+v", r.Findings) }
}

func TestTargetRebindingBlocked(t *testing.T) {
	x := fixtureRequest()
	x.Plan.Steps[0].Target = "case-8"
	r := Validate(x)
	if r.Accepted || !findingCode(r, "TARGET_REBINDING") { t.Fatalf("expected block: %+v", r.Findings) }
}

func TestExternalActionNeedsAuthorization(t *testing.T) {
	x := fixtureRequest()
	x.Plan.Sinks[0].Kind = "external_action"
	x.Plan.Sinks[0].ActionKind = "case.update"
	x.Plan.Sinks[0].ArgumentsDigest = strings.Repeat("d", 64)
	x.Plan.Sinks[0].RequiredCapability = "case:update"
	r := Validate(x)
	if r.Accepted || !findingCode(r, "ACTION_AUTH_REQUIRED") { t.Fatalf("expected block: %+v", r.Findings) }
}

func TestCrossSkillConflictBlocksComposition(t *testing.T) {
	x := fixtureRequest()
	s2 := x.Skills[0]
	s2.ID = "s2"
	s2.Conflicts = []string{"s1"}
	x.Skills = append(x.Skills, s2)
	x.Plan.Steps = append(x.Plan.Steps, Step{
		ID: "st2", SkillID: "s2", Inputs: []string{"o1"}, OutputID: "o2",
		DeclaredOutputLabel: "internal", Target: "case-7",
		ExposureMode: "summary", Audience: "support-team",
	})
	r := Validate(x)
	if r.Accepted || !findingCode(r, "SKILL_CONFLICT") { t.Fatalf("expected block: %+v", r.Findings) }
}

func TestCanonicalReportIsStableAcrossRepeatedRuns(t *testing.T) {
	x := fixtureRequest()
	s2 := x.Skills[0]
	s2.ID = "s2"
	s2.Conflicts = []string{"s1"}
	x.Skills = append(x.Skills, s2)
	x.Plan.Steps = append(x.Plan.Steps, Step{
		ID: "st2", SkillID: "s2", Inputs: []string{"o1"}, OutputID: "o2",
		DeclaredOutputLabel: "internal", Target: "case-7",
		ExposureMode: "summary", Audience: "support-team",
	})
	a, b := Validate(x), Validate(x)
	if a.ReportDigest != b.ReportDigest { t.Fatalf("unstable report digest: %s != %s", a.ReportDigest, b.ReportDigest) }
}
