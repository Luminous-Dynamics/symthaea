// Package coverage reconciles normalized, pre-authenticated action evidence.
// Cryptographic adapters must validate source signatures, receipts and
// watermarks before constructing these values. This package is not a signer,
// SCITT verifier, or PostgreSQL transaction implementation.
package coverage

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"sort"
	"strings"
)

type Status string

const (
	Verified      Status = "verified"
	Missing       Status = "missing"
	Contradicted  Status = "contradicted"
	Stale         Status = "stale"
	Indeterminate Status = "indeterminate"
)

type Domains struct {
	Identity      string
	Privilege     string
	Process       string
	Administrator string
	KeyCustody    string
	Storage       string
	Network       string
	Operator      string
}

type Manifest struct {
	BoundaryID                    string
	TenantID                      string
	ResourceID                    string
	FromSequence                  uint64 // exclusive lower bound
	ThroughSequence               uint64 // inclusive externally anchored high-water mark
	InitialRevision               uint64 // independently established state immediately before FromSequence
	InitialDigest                 string // canonical SHA-256 digest for InitialRevision
	IssuedAtUnix                  int64
	ExpiresAtUnix                 int64
	AllowedMutationPaths          []string
	ExpectedReadPath              string
	RequiredObserverIndependence  []string
	RequiredWatermarkIndependence []string
	WriterDomains                 Domains
}

type Intent struct {
	OperationID       string
	TenantID          string
	ResourceID        string
	RequestDigest     string
	AuthorityRevision string
	FenceToken        uint64
	MutationPath      string
}

type SinkEvent struct {
	Sequence          uint64
	OperationID       string
	TenantID           string
	ResourceID         string
	RequestDigest      string
	AuthorityRevision  string
	FenceToken         uint64
	Outcome            string // "committed" or "rejected"
	CommittedAtUnix    int64
	PriorRevision      uint64
	PriorDigest        string
	ResultRevision     uint64
	ResultDigest       string
	MutationPath       string
}

type Observation struct {
	SinkSequence     uint64
	OperationID      string
	TenantID         string
	ResourceID       string
	ObservedRevision uint64
	ObservedDigest   string
	ReadPath         string
	ObservedAtUnix   int64
	ReaderDomains    Domains
}

type Watermark struct {
	BoundaryID      string
	TenantID        string
	ResourceID      string
	FromSequence    uint64
	ThroughSequence uint64
	ManifestDigest  string
	Authenticated   bool // set only by a separate cryptographic adapter
	CreatedAtUnix   int64
	SignerDomains   Domains
}

type Input struct {
	Manifest                 Manifest
	Intents                  []Intent
	Events                   []SinkEvent
	Observations             []Observation
	Watermark                *Watermark
	NowUnix                  int64
	MaxWatermarkAgeSeconds   int64 // <= 0 disables freshness evaluation
	MaxObservationAgeSeconds int64 // <= 0 disables freshness evaluation
}

type DimensionFinding struct {
	Dimension string \`json:"dimension"\`
	Status    Status \`json:"status"\`
	Subject   string \`json:"subject,omitempty"\`
	Observer  string \`json:"observer,omitempty"\`
}

type Result struct {
	Manifest              Status             \`json:"manifest"\`
	Watermark             Status             \`json:"watermark"\`
	WatermarkIndependence Status             \`json:"watermark_independence"\`
	WatermarkFreshness    Status             \`json:"watermark_freshness"\`
	SequenceCoverage      Status             \`json:"sequence_coverage"\`
	IntentSinkLinkage     Status             \`json:"intent_sink_linkage"\`
	SinkReadback          Status             \`json:"sink_readback"\`
	SinkReadbackFreshness Status             \`json:"sink_readback_freshness"\`
	ObserverIndependence  Status             \`json:"observer_independence"\`
	EffectQualified       bool               \`json:"effect_qualified"\`
	Findings              []DimensionFinding \`json:"findings,omitempty"\`
	Problems              []string           \`json:"problems,omitempty"\`
}

var domainDimensions = []struct {
	name string
	get  func(Domains) string
}{
	{"identity", func(d Domains) string { return d.Identity }},
	{"privilege", func(d Domains) string { return d.Privilege }},
	{"process", func(d Domains) string { return d.Process }},
	{"administrator", func(d Domains) string { return d.Administrator }},
	{"key_custody", func(d Domains) string { return d.KeyCustody }},
	{"storage", func(d Domains) string { return d.Storage }},
	{"network", func(d Domains) string { return d.Network }},
	{"operator", func(d Domains) string { return d.Operator }},
}

// CanonicalManifestDigest returns a domain-separated, length-prefixed digest
// for this reference manifest profile. It is not a general JSON canonicalizer.
func CanonicalManifestDigest(m Manifest) (string, error) {
	if err := validateManifest(m); err != nil {
		return "", err
	}
	paths := append([]string(nil), m.AllowedMutationPaths...)
	sort.Strings(paths)
	if hasDuplicate(paths) {
		return "", errors.New("allowed mutation paths contain duplicates")
	}
	var b bytes.Buffer
	b.WriteString("SFEB-MON009-BOUNDARY-MANIFEST-V1\x00")
	writeString(&b, m.BoundaryID)
	writeString(&b, m.TenantID)
	writeString(&b, m.ResourceID)
	_ = binary.Write(&b, binary.BigEndian, m.FromSequence)
	_ = binary.Write(&b, binary.BigEndian, m.ThroughSequence)
	_ = binary.Write(&b, binary.BigEndian, m.InitialRevision)
	writeString(&b, m.InitialDigest)
	_ = binary.Write(&b, binary.BigEndian, m.IssuedAtUnix)
	_ = binary.Write(&b, binary.BigEndian, m.ExpiresAtUnix)
	writeString(&b, m.ExpectedReadPath)
	_ = binary.Write(&b, binary.BigEndian, uint32(len(paths)))
	for _, p := range paths {
		writeString(&b, p)
	}
	observerRequired := append([]string(nil), m.RequiredObserverIndependence...)
	watermarkRequired := append([]string(nil), m.RequiredWatermarkIndependence...)
	sort.Strings(observerRequired)
	sort.Strings(watermarkRequired)
	writeStringList(&b, observerRequired)
	writeStringList(&b, watermarkRequired)
	for _, dim := range domainDimensions {
		writeString(&b, dim.get(m.WriterDomains))
	}
	digest := sha256.Sum256(b.Bytes())
	return hex.EncodeToString(digest[:]), nil
}

func Verify(in Input) Result {
	r := Result{
		Manifest:              Indeterminate,
		Watermark:             Indeterminate,
		WatermarkFreshness:    Indeterminate,
		SequenceCoverage:      Indeterminate,
		IntentSinkLinkage:     Indeterminate,
		SinkReadback:          Indeterminate,
		SinkReadbackFreshness: Indeterminate,
		ObserverIndependence:  Indeterminate,
		WatermarkIndependence: Indeterminate,
	}
	if err := validateManifest(in.Manifest); err != nil {
		r.Manifest = Contradicted
		addProblem(&r, "invalid manifest: %v", err)
		return r
	}
	r.Manifest = Verified

	manifestDigest, err := CanonicalManifestDigest(in.Manifest)
	if err != nil {
		r.Manifest = Contradicted
		addProblem(&r, "cannot digest manifest: %v", err)
		return r
	}

	wmStatus, wmIndependence, freshStatus, wmFindings := verifyWatermark(in.Manifest, manifestDigest, in.Watermark, in.NowUnix, in.MaxWatermarkAgeSeconds)
	r.Watermark = wmStatus
	r.WatermarkIndependence = wmIndependence
	r.WatermarkFreshness = freshStatus
	r.Findings = append(r.Findings, prefixFindings("watermark", wmFindings)...)
	if wmStatus != Verified {
		addProblem(&r, "external watermark does not authenticate the declared boundary and sequence range")
	}
	if freshStatus == Stale {
		addProblem(&r, "external watermark is older than the configured freshness bound")
	}

	intents := make(map[string]Intent, len(in.Intents))
	for _, intent := range in.Intents {
		if intent.OperationID == "" || intent.TenantID == "" || intent.ResourceID == "" ||
			!validSHA256Hex(intent.RequestDigest) || intent.AuthorityRevision == "" || intent.MutationPath == "" {
			setWorst(&r.IntentSinkLinkage, Contradicted)
			addProblem(&r, "intent has an empty required binding")
			continue
		}
		if intent.TenantID != in.Manifest.TenantID || intent.ResourceID != in.Manifest.ResourceID ||
			!contains(in.Manifest.AllowedMutationPaths, intent.MutationPath) {
			setWorst(&r.IntentSinkLinkage, Contradicted)
			addProblem(&r, "intent %q is outside the manifest scope or allowed mutation paths", intent.OperationID)
		}
		if _, exists := intents[intent.OperationID]; exists {
			setWorst(&r.IntentSinkLinkage, Contradicted)
			addProblem(&r, "duplicate intent operation ID %q", intent.OperationID)
			continue
		}
		intents[intent.OperationID] = intent
	}

	events := append([]SinkEvent(nil), in.Events...)
	sort.Slice(events, func(i, j int) bool { return events[i].Sequence < events[j].Sequence })
	seqStatus := verifySequenceRange(in.Manifest, events, &r)
	r.SequenceCoverage = seqStatus

	if r.IntentSinkLinkage == Indeterminate {
		r.IntentSinkLinkage = Verified
	}
	seenOps := make(map[string]struct{}, len(events))
	seenEvents := make(map[string]SinkEvent, len(events))
	lastCommittedRevision := in.Manifest.InitialRevision
	lastCommittedDigest := in.Manifest.InitialDigest
	var lastCommittedFence uint64
	hasCommittedPredecessor := false
	for _, event := range events {
		if event.OperationID == "" {
			setWorst(&r.IntentSinkLinkage, Contradicted)
			addProblem(&r, "sink event at sequence %d has an empty operation ID", event.Sequence)
			continue
		}
		if _, exists := seenOps[event.OperationID]; exists {
			setWorst(&r.IntentSinkLinkage, Contradicted)
			addProblem(&r, "multiple sink events use operation ID %q", event.OperationID)
			continue
		}
		seenOps[event.OperationID] = struct{}{}
		seenEvents[event.OperationID] = event
		intent, ok := intents[event.OperationID]
		if !ok {
			setWorst(&r.IntentSinkLinkage, Contradicted)
			addProblem(&r, "sink event %q has no matching caller intent", event.OperationID)
			continue
		}
		if !sameIntentEvent(intent, event) {
			setWorst(&r.IntentSinkLinkage, Contradicted)
			addProblem(&r, "intent/sink binding mismatch for operation %q", event.OperationID)
			continue
		}
		if event.TenantID != in.Manifest.TenantID || event.ResourceID != in.Manifest.ResourceID {
			setWorst(&r.IntentSinkLinkage, Contradicted)
			addProblem(&r, "event %q is outside the manifest tenant/resource scope", event.OperationID)
		}
		if !contains(in.Manifest.AllowedMutationPaths, event.MutationPath) {
			setWorst(&r.IntentSinkLinkage, Contradicted)
			addProblem(&r, "event %q used undeclared mutation path %q", event.OperationID, event.MutationPath)
		}
		if event.Outcome != "committed" && event.Outcome != "rejected" {
			setWorst(&r.IntentSinkLinkage, Indeterminate)
			addProblem(&r, "event %q has unknown outcome %q", event.OperationID, event.Outcome)
		}
		if event.Outcome == "committed" {
			if !validSHA256Hex(event.RequestDigest) || !validSHA256Hex(event.PriorDigest) || !validSHA256Hex(event.ResultDigest) || event.ResultRevision <= event.PriorRevision || event.CommittedAtUnix <= 0 {
				setWorst(&r.IntentSinkLinkage, Contradicted)
				addProblem(&r, "committed event %q lacks a valid state transition, canonical digest or commit time", event.OperationID)
			}
			if event.PriorRevision != lastCommittedRevision || !equalDigest(event.PriorDigest, lastCommittedDigest) {
				setWorst(&r.IntentSinkLinkage, Contradicted)
				addProblem(&r, "committed event %q does not continue the independently established prior revision/digest", event.OperationID)
			}
			if hasCommittedPredecessor && event.FenceToken < lastCommittedFence {
				setWorst(&r.IntentSinkLinkage, Contradicted)
				addProblem(&r, "committed event %q regresses the accepted fencing token", event.OperationID)
			}
			lastCommittedRevision = event.ResultRevision
			lastCommittedDigest = event.ResultDigest
			lastCommittedFence = event.FenceToken
			hasCommittedPredecessor = true
			if event.CommittedAtUnix < in.Manifest.IssuedAtUnix || event.CommittedAtUnix > in.Manifest.ExpiresAtUnix {
				setWorst(&r.IntentSinkLinkage, Contradicted)
				addProblem(&r, "committed event %q falls outside the boundary manifest validity interval", event.OperationID)
			}
		}
	}
	for opID := range intents {
		if _, ok := seenEvents[opID]; !ok {
			setWorst(&r.IntentSinkLinkage, Missing)
			addProblem(&r, "caller intent %q has no matching sink outcome", opID)
		}
	}

	observations := make(map[string]Observation, len(in.Observations))
	observationInputStatus := Verified
	for _, obs := range in.Observations {
		if obs.OperationID == "" {
			setWorst(&observationInputStatus, Contradicted)
			addProblem(&r, "observation has an empty operation ID")
			continue
		}
		if _, exists := observations[obs.OperationID]; exists {
			setWorst(&observationInputStatus, Contradicted)
			addProblem(&r, "multiple readback observations supplied for operation %q", obs.OperationID)
			continue
		}
		observations[obs.OperationID] = obs
	}

	committed := 0
	readbackStatus := observationInputStatus
	independenceStatus := Verified
	freshnessStatus := Verified
	for _, event := range events {
		obs, hasObs := observations[event.OperationID]
		if event.Outcome == "rejected" {
			if hasObs {
				setWorst(&readbackStatus, Contradicted)
				addProblem(&r, "rejected operation %q unexpectedly has a committed-effect readback", event.OperationID)
			}
			continue
		}
		if event.Outcome != "committed" {
			setWorst(&readbackStatus, Indeterminate)
			independenceStatus = Indeterminate
			continue
		}
		committed++
		if !hasObs {
			if readbackStatus != Contradicted {
				setWorst(&readbackStatus, Missing)
			}
			independenceStatus = Missing
			addProblem(&r, "committed operation %q has no independent readback", event.OperationID)
			continue
		}
		if obs.SinkSequence != event.Sequence || obs.TenantID != event.TenantID || obs.ResourceID != event.ResourceID || obs.ObservedRevision != event.ResultRevision || !equalDigest(obs.ObservedDigest, event.ResultDigest) {
			setWorst(&readbackStatus, Contradicted)
			addProblem(&r, "readback does not match exact committed revision/digest for operation %q", event.OperationID)
		}
		if obs.ReadPath == "" || obs.ReadPath != in.Manifest.ExpectedReadPath {
			setWorst(&readbackStatus, Contradicted)
			addProblem(&r, "readback for operation %q came from unexpected path %q", event.OperationID, obs.ReadPath)
		}
		if obs.ObservedAtUnix < event.CommittedAtUnix || obs.ObservedAtUnix <= 0 || (in.NowUnix > 0 && obs.ObservedAtUnix > in.NowUnix) {
			setWorst(&freshnessStatus, Indeterminate)
			addProblem(&r, "readback time for operation %q is inconsistent with commit/audit time", event.OperationID)
		} else if in.MaxObservationAgeSeconds > 0 && in.NowUnix <= 0 {
			setWorst(&freshnessStatus, Indeterminate)
			addProblem(&r, "current time is required when a readback freshness bound is configured")
		} else if in.MaxObservationAgeSeconds > 0 && in.NowUnix-obs.ObservedAtUnix > in.MaxObservationAgeSeconds {
			setWorst(&freshnessStatus, Stale)
			addProblem(&r, "readback for operation %q is older than the configured freshness bound", event.OperationID)
		}
		findings, status := compareDomains(in.Manifest.WriterDomains, obs.ReaderDomains, in.Manifest.RequiredObserverIndependence)
		r.Findings = append(r.Findings, findings...)
		if status == Contradicted {
			independenceStatus = Contradicted
		} else if status == Indeterminate && independenceStatus != Contradicted {
			independenceStatus = Indeterminate
		}
	}
	for opID := range observations {
		event, ok := seenEvents[opID]
		if !ok || event.Outcome != "committed" {
			setWorst(&readbackStatus, Contradicted)
			addProblem(&r, "observation %q has no matching committed sink event", opID)
		}
	}
	if committed == 0 && readbackStatus == Verified {
		setWorst(&readbackStatus, Missing)
		independenceStatus = Missing
	}
	r.SinkReadback = readbackStatus
	r.SinkReadbackFreshness = freshnessStatus
	r.ObserverIndependence = independenceStatus

	// Coverage is a conjunction of event-range completeness and successful
	// intent/sink linkage. Watermark authenticity is a separate claim.
	r.SequenceCoverage = seqStatus
	if seqStatus == Verified && r.IntentSinkLinkage == Verified {
		// Keep coverage separate from observer independence; the latter is
		// exposed as its own claim and gates effect qualification below.
	} else if seqStatus == Contradicted || r.IntentSinkLinkage == Contradicted {
		r.SequenceCoverage = Contradicted
	} else if seqStatus == Missing || r.IntentSinkLinkage == Missing {
		r.SequenceCoverage = Missing
	} else {
		r.SequenceCoverage = Indeterminate
	}

	r.EffectQualified = r.Manifest == Verified && r.Watermark == Verified && r.WatermarkIndependence == Verified &&
		r.SequenceCoverage == Verified && r.WatermarkFreshness == Verified &&
		r.IntentSinkLinkage == Verified && r.SinkReadback == Verified && r.SinkReadbackFreshness == Verified &&
		r.ObserverIndependence == Verified && committed > 0
	return r
}

func validateManifest(m Manifest) error {
	if strings.TrimSpace(m.BoundaryID) == "" || strings.TrimSpace(m.TenantID) == "" || strings.TrimSpace(m.ResourceID) == "" {
		return errors.New("boundary, tenant and resource IDs are required")
	}
	if m.ThroughSequence < m.FromSequence {
		return errors.New("through sequence precedes from sequence")
	}
	if m.ThroughSequence == ^uint64(0) {
		return errors.New("through sequence cannot be the maximum uint64 in this profile")
	}
	if !validSHA256Hex(m.InitialDigest) {
		return errors.New("initial state digest must be canonical lowercase SHA-256 hex")
	}
	if m.IssuedAtUnix <= 0 || m.ExpiresAtUnix <= m.IssuedAtUnix {
		return errors.New("manifest issue time must be positive and expiry must follow it")
	}
	if len(m.AllowedMutationPaths) == 0 || strings.TrimSpace(m.ExpectedReadPath) == "" {
		return errors.New("allowed mutation paths and expected read path are required")
	}
	if hasDuplicate(m.AllowedMutationPaths) {
		return errors.New("allowed mutation paths contain duplicates")
	}
	for _, path := range m.AllowedMutationPaths {
		if strings.TrimSpace(path) == "" {
			return errors.New("allowed mutation path is empty")
		}
	}
	if err := validateRequiredDimensions(m.RequiredObserverIndependence); err != nil {
		return fmt.Errorf("observer independence policy: %w", err)
	}
	if err := validateRequiredDimensions(m.RequiredWatermarkIndependence); err != nil {
		return fmt.Errorf("watermark independence policy: %w", err)
	}
	for _, d := range domainDimensions {
		if strings.TrimSpace(d.get(m.WriterDomains)) == "" {
			return fmt.Errorf("writer %s domain is required", d.name)
		}
	}
	return nil
}

func verifyWatermark(m Manifest, digest string, w *Watermark, now, maxAge int64) (Status, Status, Status, []DimensionFinding) {
	if w == nil {
		return Missing, Missing, Missing, nil
	}
	if !w.Authenticated {
		return Missing, Indeterminate, Indeterminate, nil
	}
	if w.CreatedAtUnix <= 0 || w.CreatedAtUnix < m.IssuedAtUnix || w.CreatedAtUnix > m.ExpiresAtUnix {
		return Contradicted, Indeterminate, Indeterminate, nil
	}
	if w.BoundaryID != m.BoundaryID || w.TenantID != m.TenantID || w.ResourceID != m.ResourceID ||
		w.FromSequence != m.FromSequence || w.ThroughSequence != m.ThroughSequence ||
		!validSHA256Hex(w.ManifestDigest) || w.ManifestDigest != digest {
		return Contradicted, Indeterminate, Indeterminate, nil
	}
	findings, signerStatus := compareDomains(m.WriterDomains, w.SignerDomains, m.RequiredWatermarkIndependence)
	freshness := Verified
	if maxAge > 0 {
		if now < w.CreatedAtUnix {
			freshness = Indeterminate
		} else if now-w.CreatedAtUnix > maxAge {
			freshness = Stale
		}
	}
	return Verified, signerStatus, freshness, findings
}

func verifySequenceRange(m Manifest, events []SinkEvent, r *Result) Status {
	if uint64(len(events)) > m.ThroughSequence-m.FromSequence {
		addProblem(r, "more events supplied than the declared sequence range permits")
		return Contradicted
	}
	expected := m.FromSequence + 1
	for _, event := range events {
		if event.Sequence < expected {
			addProblem(r, "duplicate or out-of-order sink sequence %d", event.Sequence)
			return Contradicted
		}
		if event.Sequence > m.ThroughSequence {
			addProblem(r, "sink sequence %d exceeds anchored high-water mark %d", event.Sequence, m.ThroughSequence)
			return Contradicted
		}
		if event.Sequence != expected {
			addProblem(r, "missing sink sequence range %d..%d", expected, event.Sequence-1)
			return Missing
		}
		expected++
	}
	if expected != m.ThroughSequence+1 {
		addProblem(r, "missing sink sequence range %d..%d", expected, m.ThroughSequence)
		return Missing
	}
	return Verified
}

func sameIntentEvent(i Intent, e SinkEvent) bool {
	return i.OperationID == e.OperationID && i.TenantID == e.TenantID && i.ResourceID == e.ResourceID &&
		equalDigest(i.RequestDigest, e.RequestDigest) && i.AuthorityRevision == e.AuthorityRevision &&
		i.FenceToken == e.FenceToken && i.MutationPath == e.MutationPath
}

func compareDomains(subject, observer Domains, required []string) ([]DimensionFinding, Status) {
	findings := make([]DimensionFinding, 0, len(domainDimensions))
	requiredSet := make(map[string]struct{}, len(required))
	for _, name := range required {
		requiredSet[name] = struct{}{}
	}
	overall := Verified
	for _, d := range domainDimensions {
		s, o := d.get(subject), d.get(observer)
		f := DimensionFinding{Dimension: d.name, Subject: s, Observer: o}
		switch {
		case s == "" || o == "":
			f.Status = Indeterminate
		case s == o:
			f.Status = Contradicted
		default:
			f.Status = Verified
		}
		if _, isRequired := requiredSet[d.name]; isRequired {
			switch f.Status {
			case Contradicted:
				overall = Contradicted
			case Indeterminate:
				if overall != Contradicted {
					overall = Indeterminate
				}
			}
		}
		findings = append(findings, f)
	}
	return findings, overall
}

func validateRequiredDimensions(required []string) error {
	if len(required) == 0 {
		return errors.New("at least one independent domain dimension is required")
	}
	allowed := make(map[string]struct{}, len(domainDimensions))
	for _, d := range domainDimensions {
		allowed[d.name] = struct{}{}
	}
	seen := make(map[string]struct{}, len(required))
	for _, name := range required {
		if _, ok := allowed[name]; !ok {
			return fmt.Errorf("unknown dimension %q", name)
		}
		if _, ok := seen[name]; ok {
			return fmt.Errorf("duplicate dimension %q", name)
		}
		seen[name] = struct{}{}
	}
	return nil
}

func prefixFindings(prefix string, findings []DimensionFinding) []DimensionFinding {
	out := make([]DimensionFinding, len(findings))
	for i, f := range findings {
		f.Dimension = prefix + "." + f.Dimension
		out[i] = f
	}
	return out
}

func writeStringList(b *bytes.Buffer, values []string) {
	_ = binary.Write(b, binary.BigEndian, uint32(len(values)))
	for _, value := range values {
		writeString(b, value)
	}
}

func writeString(b *bytes.Buffer, s string) {
	_ = binary.Write(b, binary.BigEndian, uint32(len(s)))
	b.WriteString(s)
}

func hasDuplicate(values []string) bool {
	seen := make(map[string]struct{}, len(values))
	for _, value := range values {
		if _, ok := seen[value]; ok {
			return true
		}
		seen[value] = struct{}{}
	}
	return false
}

func contains(values []string, want string) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
}

func validSHA256Hex(value string) bool {
	if len(value) != sha256.Size*2 || strings.TrimSpace(value) != value || value != strings.ToLower(value) {
		return false
	}
	decoded, err := hex.DecodeString(value)
	return err == nil && len(decoded) == sha256.Size
}

func equalDigest(a, b string) bool {
	return validSHA256Hex(a) && validSHA256Hex(b) && a == b
}

func setWorst(current *Status, candidate Status) {
	rank := map[Status]int{Verified: 0, Missing: 1, Stale: 2, Indeterminate: 3, Contradicted: 4}
	if rank[candidate] > rank[*current] {
		*current = candidate
	}
}

func addProblem(r *Result, format string, args ...any) {
	r.Problems = append(r.Problems, fmt.Sprintf(format, args...))
}
