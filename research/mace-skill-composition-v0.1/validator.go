package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"regexp"
	"sort"
	"time"
)

type Label uint8

const (
	Public Label = iota
	Internal
	Confidential
	Restricted
)

func (l Label) String() string {
	switch l {
	case Public:
		return "public"
	case Internal:
		return "internal"
	case Confidential:
		return "confidential"
	case Restricted:
		return "restricted"
	}
	return "invalid"
}
func ParseLabel(s string) (Label, bool) {
	switch s {
	case "public":
		return Public, true
	case "internal":
		return Internal, true
	case "confidential":
		return Confidential, true
	case "restricted":
		return Restricted, true
	}
	return 0, false
}

type Request struct {
	Context Context    `json:"context"`
	Skills  []Skill    `json:"skills"`
	Sources []Artifact `json:"sources"`
	Plan    Plan       `json:"plan"`
}
type Context struct {
	RequestID      string   `json:"request_id"`
	Principal      string   `json:"principal"`
	Scope          string   `json:"scope"`
	Purpose        string   `json:"purpose"`
	Audience       string   `json:"audience"`
	ExpectedTarget string   `json:"expected_target"`
	Capabilities   []string `json:"capabilities"`
	EvaluatedAt    string   `json:"evaluated_at"`
	ExposureBudget int      `json:"exposure_budget"`
	MaxActionLabel string   `json:"max_action_label"`
}
type Skill struct {
	ID                   string   `json:"id"`
	Digest               string   `json:"digest"`
	Status               string   `json:"status"`
	ExpiresAt            string   `json:"expires_at"`
	Scopes               []string `json:"scopes"`
	ScopeAgnostic        bool     `json:"scope_agnostic"`
	Purposes             []string `json:"purposes"`
	PurposeAgnostic      bool     `json:"purpose_agnostic"`
	Audiences            []string `json:"audiences"`
	ReadableMaxLabel     string   `json:"readable_max_label"`
	ExposureModes        []string `json:"exposure_modes"`
	RequiredCapabilities []string `json:"required_capabilities"`
	CapabilityGrants     []string `json:"capability_grants"`
	Conflicts            []string `json:"conflicts"`
	MaxTokens            int      `json:"max_tokens"`
}
type Artifact struct {
	ID       string `json:"id"`
	Digest   string `json:"digest"`
	Label    string `json:"label"`
	Scope    string `json:"scope"`
	Purpose  string `json:"purpose"`
	Audience string `json:"audience"`
	Target   string `json:"target"`
	Current  bool   `json:"current"`
}
type Step struct {
	ID                  string   `json:"id"`
	SkillID             string   `json:"skill_id"`
	Inputs              []string `json:"inputs"`
	OutputID            string   `json:"output_id"`
	DeclaredOutputLabel string   `json:"declared_output_label"`
	Target              string   `json:"target"`
	ExposureMode        string   `json:"exposure_mode"`
	Audience            string   `json:"audience"`
	Tokens              int      `json:"tokens"`
}
type Sink struct {
	ArtifactID         string         `json:"artifact_id"`
	Kind               string         `json:"kind"`
	Target             string         `json:"target"`
	Audience           string         `json:"audience"`
	MaxLabel           string         `json:"max_label"`
	ActionKind         string         `json:"action_kind,omitempty"`
	ArgumentsDigest    string         `json:"arguments_digest,omitempty"`
	RequiredCapability string         `json:"required_capability,omitempty"`
	Authorization      *Authorization `json:"authorization,omitempty"`
}
type Authorization struct {
	VerifierStatus     string `json:"verifier_status"`
	Principal          string `json:"principal"`
	Scope              string `json:"scope"`
	ActionKind         string `json:"action_kind"`
	Target             string `json:"target"`
	ArgumentsDigest    string `json:"arguments_digest"`
	RequiredCapability string `json:"required_capability"`
	ExpiresAt          string `json:"expires_at"`
	Nonce              string `json:"nonce"`
}
type Plan struct {
	Steps []Step `json:"steps"`
	Sinks []Sink `json:"sinks"`
}
type Finding struct {
	Code   string `json:"code"`
	Object string `json:"object"`
	Detail string `json:"detail"`
}
type Derived struct {
	ID     string   `json:"id"`
	Label  string   `json:"label"`
	Target string   `json:"target"`
	Inputs []string `json:"inputs"`
}
type Report struct {
	Accepted             bool      `json:"accepted"`
	PlanDigest           string    `json:"plan_digest"`
	ContextDigest        string    `json:"context_digest"`
	SkillRegistryDigest  string    `json:"skill_registry_digest"`
	SourceManifestDigest string    `json:"source_manifest_digest"`
	Derived              []Derived `json:"derived_artifacts"`
	Findings             []Finding `json:"findings"`
	ExposureTokens       int       `json:"exposure_tokens"`
	ReportDigest         string    `json:"report_digest"`
}

var digestRE = regexp.MustCompile(`^[a-f0-9]{64}$`)

func hash(v any) string {
	b, _ := json.Marshal(v)
	s := sha256.Sum256(b)
	return hex.EncodeToString(s[:])
}
func sorted(xs []string) []string { y := append([]string(nil), xs...); sort.Strings(y); return y }
func has(xs []string, v string) bool {
	for _, x := range xs {
		if x == v {
			return true
		}
	}
	return false
}
func unique(xs []string) bool {
	m := map[string]bool{}
	for _, x := range xs {
		if x == "" || m[x] {
			return false
		}
		m[x] = true
	}
	return true
}
func validDigest(s string) bool { return digestRE.MatchString(s) }
func add(r *Report, code, obj, detail string) {
	r.Findings = append(r.Findings, Finding{code, obj, detail})
}
func Validate(in Request) Report {
	r := Report{Accepted: true, Derived: []Derived{}, Findings: []Finding{}}
	c := in.Context
	// Canonicalize set-valued fields before hashing. Plan step/sink order is intentionally preserved.
	cc := c
	cc.Capabilities = sorted(c.Capabilities)
	ss := append([]Skill(nil), in.Skills...)
	for i := range ss {
		ss[i].Scopes = sorted(ss[i].Scopes)
		ss[i].Purposes = sorted(ss[i].Purposes)
		ss[i].Audiences = sorted(ss[i].Audiences)
		ss[i].ExposureModes = sorted(ss[i].ExposureModes)
		ss[i].RequiredCapabilities = sorted(ss[i].RequiredCapabilities)
		ss[i].CapabilityGrants = sorted(ss[i].CapabilityGrants)
		ss[i].Conflicts = sorted(ss[i].Conflicts)
	}
	sort.Slice(ss, func(i, j int) bool { if ss[i].ID != ss[j].ID { return ss[i].ID < ss[j].ID }; return hash(ss[i]) < hash(ss[j]) })
	srcs := append([]Artifact(nil), in.Sources...)
	sort.Slice(srcs, func(i, j int) bool { if srcs[i].ID != srcs[j].ID { return srcs[i].ID < srcs[j].ID }; return hash(srcs[i]) < hash(srcs[j]) })
	r.ContextDigest = hash(cc)
	r.SkillRegistryDigest = hash(ss)
	r.SourceManifestDigest = hash(srcs)
	r.PlanDigest = hash(in.Plan)
	fail := func(code, obj, detail string) { add(&r, code, obj, detail); r.Accepted = false }
	now, nowErr := time.Parse(time.RFC3339, c.EvaluatedAt)
	if c.RequestID == "" || c.Principal == "" || c.Scope == "" || c.Purpose == "" || c.Audience == "" || c.ExpectedTarget == "" || nowErr != nil || c.ExposureBudget < 0 {
		fail("INVALID_CONTEXT", c.RequestID, "required context missing or malformed")
	}
	if !unique(c.Capabilities) {
		fail("INVALID_CAPABILITIES", c.RequestID, "capabilities must be nonempty and unique")
	}
	maxLabel, ok := ParseLabel(c.MaxActionLabel)
	if !ok {
		fail("INVALID_LABEL", c.RequestID, "unknown max_action_label")
	}
	skills := map[string]Skill{}
	selected := map[string]bool{}
	for _, st := range in.Plan.Steps {
		selected[st.SkillID] = true
	}
	for _, s := range in.Skills {
		if s.ID == "" || skills[s.ID].ID != "" {
			fail("DUPLICATE_SKILL", s.ID, "skill identity missing or duplicated")
			continue
		}
		skills[s.ID] = s
		if !selected[s.ID] {
			continue
		}
		if !validDigest(s.Digest) {
			fail("INVALID_SKILL_DIGEST", s.ID, "skill digest must be lowercase SHA-256")
		}
		if s.Status != "active" {
			fail("SKILL_NOT_ACTIVE", s.ID, "skill status is not active")
		}
		ex, e := time.Parse(time.RFC3339, s.ExpiresAt)
		if e != nil || nowErr == nil && !ex.After(now) {
			fail("SKILL_EXPIRED", s.ID, "skill expiry is missing, invalid or not after evaluation time")
		}
		if (!s.ScopeAgnostic && !has(s.Scopes, c.Scope)) || (!s.PurposeAgnostic && !has(s.Purposes, c.Purpose)) || !has(s.Audiences, c.Audience) {
			fail("SKILL_NOT_APPLICABLE", s.ID, "scope, purpose or audience does not match")
		}
		if _, ok := ParseLabel(s.ReadableMaxLabel); !ok {
			fail("INVALID_SKILL_LABEL", s.ID, "unknown readable_max_label")
		}
		if len(s.CapabilityGrants) > 0 {
			fail("CAPABILITY_MINTING", s.ID, "skills cannot grant capabilities")
		}
		if s.MaxTokens < 0 {
			fail("INVALID_TOKEN_BUDGET", s.ID, "negative per-skill token budget")
		}
	}
	skillIDs := make([]string, 0, len(skills))
	for id := range skills { skillIDs = append(skillIDs, id) }
	sort.Strings(skillIDs)
	for _, id := range skillIDs {
		s := skills[id]
		if !selected[id] {
			continue
		}
		for _, other := range s.Conflicts {
			if _, ok := skills[other]; !ok {
				fail("UNKNOWN_CONFLICT", id, "conflict references unknown skill "+other)
			} else if other == id {
				fail("SELF_CONFLICT", id, "skill declares conflict with itself")
			} else if selected[other] && (has(skills[other].Conflicts, id) || has(s.Conflicts, other)) {
				fail("SKILL_CONFLICT", id, "selected skills declare a conflict with "+other)
			}
		}
	}
	arts := map[string]Artifact{}
	for _, a := range in.Sources {
		if a.ID == "" || arts[a.ID].ID != "" {
			fail("DUPLICATE_SOURCE", a.ID, "source identity missing or duplicated")
			continue
		}
		arts[a.ID] = a
		if !validDigest(a.Digest) {
			fail("INVALID_SOURCE_DIGEST", a.ID, "source digest must be lowercase SHA-256")
		}
		lab, ok := ParseLabel(a.Label)
		if !ok {
			fail("INVALID_LABEL", a.ID, "unknown source label")
		}
		if !a.Current {
			fail("STALE_SOURCE", a.ID, "source is not current")
		}
		if ok && lab != Public && a.Scope != c.Scope {
			fail("SCOPE_MISMATCH", a.ID, "non-public source scope differs from request")
		}
		if a.Purpose != c.Purpose {
			fail("PURPOSE_MISMATCH", a.ID, "source purpose differs from request")
		}
		if ok && lab != Public && a.Audience != c.Audience {
			fail("AUDIENCE_MISMATCH", a.ID, "non-public source audience differs from request")
		}
	}
	usedSkills := map[string]bool{}
	stepIDs := map[string]bool{}
	exposure := 0
	for _, st := range in.Plan.Steps {
		if st.ID == "" || stepIDs[st.ID] {
			fail("DUPLICATE_STEP", st.ID, "step identity missing or duplicated")
		}
		stepIDs[st.ID] = true
		sk, ok := skills[st.SkillID]
		if !ok {
			fail("UNKNOWN_SKILL", st.ID, "step references unknown skill")
			continue
		}
		usedSkills[sk.ID] = true
		if !unique(st.Inputs) {
			fail("DUPLICATE_INPUT", st.ID, "input identities must be nonempty and unique")
		}
		if !has(sk.ExposureModes, st.ExposureMode) || st.Audience != c.Audience {
			fail("EXPOSURE_NOT_ALLOWED", st.ID, "exposure mode or audience is not admitted")
		}
		if st.Tokens < 0 || st.Tokens > sk.MaxTokens {
			fail("SKILL_TOKEN_BUDGET", st.ID, "step token use exceeds skill budget")
		}
		// Token counts are untrusted request data. Saturate and reject rather
		// than allowing signed-integer overflow to wrap the aggregate negative.
		maxInt := int(^uint(0) >> 1)
		if st.Tokens > 0 && st.Tokens > maxInt-exposure {
			fail("EXPOSURE_COUNTER_OVERFLOW", st.ID, "aggregate skill exposure counter would overflow")
			exposure = maxInt
		} else if st.Tokens > 0 {
			exposure += st.Tokens
		}
		for _, cap := range sk.RequiredCapabilities {
			if !has(c.Capabilities, cap) {
				fail("MISSING_CAPABILITY", st.ID, "principal lacks required capability "+cap)
			}
		}
		maxRead, _ := ParseLabel(sk.ReadableMaxLabel)
		outLab, outOK := ParseLabel(st.DeclaredOutputLabel)
		if !outOK {
			fail("INVALID_LABEL", st.ID, "unknown declared output label")
		}
		label := outLab
		target := st.Target
		if st.OutputID == "" || arts[st.OutputID].ID != "" {
			fail("DUPLICATE_ARTIFACT", st.OutputID, "output identity missing or duplicated")
		}
		for _, iid := range st.Inputs {
			a, exists := arts[iid]
			if !exists {
				fail("MISSING_OR_FORWARD_INPUT", st.ID, "input not available before this step: "+iid)
				continue
			}
			il, valid := ParseLabel(a.Label)
			if valid && il > label {
				label = il
			}
			if valid && il > maxRead {
				fail("READ_LEVEL_EXCEEDED", st.ID, "skill cannot read input "+iid)
			}
			if valid && il != Public && a.Scope != c.Scope {
				fail("SCOPE_MISMATCH", iid, "intermediate scope differs from request")
			}
			if a.Purpose != c.Purpose {
				fail("PURPOSE_MISMATCH", iid, "intermediate purpose differs from request")
			}
			if valid && il != Public && a.Audience != c.Audience {
				fail("AUDIENCE_MISMATCH", iid, "intermediate audience differs from request")
			}
			if a.Target != "" {
				if target != "" && target != a.Target {
					fail("TARGET_COLLISION", st.ID, "step target conflicts with inherited target")
				}
				if target == "" {
					target = a.Target
				}
			}
		}
		if target == "" {
			target = c.ExpectedTarget
		}
		if target != c.ExpectedTarget {
			fail("TARGET_REBINDING", st.ID, "derived target differs from request target")
		}
		if outOK {
			arts[st.OutputID] = Artifact{ID: st.OutputID, Digest: hash(struct {
				Step   string
				Label  string
				Target string
				Inputs []string
			}{st.ID, label.String(), target, st.Inputs}), Label: label.String(), Scope: c.Scope, Purpose: c.Purpose, Audience: c.Audience, Target: target, Current: true}
			r.Derived = append(r.Derived, Derived{ID: st.OutputID, Label: label.String(), Target: target, Inputs: append([]string(nil), st.Inputs...)})
		}
	}
	for id := range skills {
		if !usedSkills[id] { /* registered but not selected is permitted */
		}
	}
	if exposure > c.ExposureBudget {
		fail("REQUEST_TOKEN_BUDGET", c.RequestID, "aggregate skill exposure exceeds request budget")
	}
	r.ExposureTokens = exposure
	nonces := map[string]bool{}
	for _, sink := range in.Plan.Sinks {
		a, ok := arts[sink.ArtifactID]
		if !ok {
			fail("UNKNOWN_SINK_ARTIFACT", sink.ArtifactID, "sink artifact does not exist")
			continue
		}
		lab, valid := ParseLabel(a.Label)
		limit, lok := ParseLabel(sink.MaxLabel)
		if !valid || !lok {
			fail("INVALID_SINK_LABEL", sink.ArtifactID, "unknown artifact or sink label")
		} else if lab > limit || lab > maxLabel {
			fail("SINK_LABEL_BLOCKED", sink.ArtifactID, "artifact classification exceeds sink/request ceiling")
		}
		if sink.Audience != c.Audience {
			fail("SINK_AUDIENCE_MISMATCH", sink.ArtifactID, "sink audience differs from request")
		}
		target := sink.Target
		if target == "" {
			target = c.ExpectedTarget
		}
		if a.Target != "" && target != a.Target || target != c.ExpectedTarget {
			fail("SINK_TARGET_MISMATCH", sink.ArtifactID, "sink target does not match inherited/request target")
		}
		switch sink.Kind {
		case "internal":
		case "external_action":
			if sink.ActionKind == "" || sink.RequiredCapability == "" {
				fail("ACTION_BINDING_INCOMPLETE", sink.ArtifactID, "external action requires nonempty action kind and capability")
				continue
			}
			auth := sink.Authorization
			if auth == nil {
				fail("ACTION_AUTH_REQUIRED", sink.ArtifactID, "external action lacks independent authorization receipt")
				continue
			}
			expiry, e := time.Parse(time.RFC3339, auth.ExpiresAt)
			if !validDigest(sink.ArgumentsDigest) || auth.VerifierStatus != "verified" || auth.Principal != c.Principal || auth.Scope != c.Scope || auth.ActionKind != sink.ActionKind || auth.Target != target || auth.ArgumentsDigest != sink.ArgumentsDigest || auth.RequiredCapability != sink.RequiredCapability || !has(c.Capabilities, auth.RequiredCapability) || e != nil || nowErr == nil && !expiry.After(now) || auth.Nonce == "" || nonces[auth.Nonce] {
				fail("ACTION_AUTH_INVALID", sink.ArtifactID, "authorization receipt mismatch, expired, replayed or malformed")
			} else {
				nonces[auth.Nonce] = true
			}
		default:
			fail("UNKNOWN_SINK_KIND", sink.ArtifactID, "sink kind is not supported")
		}
	}
	sort.Slice(r.Findings, func(i, j int) bool {
		if r.Findings[i].Code != r.Findings[j].Code { return r.Findings[i].Code < r.Findings[j].Code }
		if r.Findings[i].Object != r.Findings[j].Object { return r.Findings[i].Object < r.Findings[j].Object }
		return r.Findings[i].Detail < r.Findings[j].Detail
	})
	r.ReportDigest = ""
	r.ReportDigest = hash(struct {
		Accepted                       bool
		Plan, Context, Skills, Sources string
		Derived                        []Derived
		Findings                       []Finding
		Exposure                       int
	}{r.Accepted, r.PlanDigest, r.ContextDigest, r.SkillRegistryDigest, r.SourceManifestDigest, r.Derived, r.Findings, r.ExposureTokens})
	return r
}
