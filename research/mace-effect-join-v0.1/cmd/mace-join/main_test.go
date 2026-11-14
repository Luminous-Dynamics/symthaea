package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/Luminous-Dynamics/symthaea/research/mace-effect-join-v0.1/join"
)

func writeTemp(t *testing.T, content string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "input.json")
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestRunRequiresOutOfBandTrustAndPolicy(t *testing.T) {
	var out, errOut bytes.Buffer
	code := run([]string{"--bundle", "-"}, strings.NewReader(`{}`), &out, &errOut)
	if code != 2 || !strings.Contains(errOut.String(), "--trust-store and --policy are required") {
		t.Fatalf("code=%d stderr=%q", code, errOut.String())
	}
}

func TestRunRejectsAgentSuppliedTrustAndPolicyFields(t *testing.T) {
	bundle := writeTemp(t, `{"authorization":{},"trust_store":{},"policy_state":{}}`)
	trust := writeTemp(t, `{}`)
	policy := writeTemp(t, `{}`)
	var out, errOut bytes.Buffer
	code := run([]string{"--bundle", bundle, "--trust-store", trust, "--policy", policy}, strings.NewReader(""), &out, &errOut)
	if code != 1 || !strings.Contains(errOut.String(), "decode evidence bundle") {
		t.Fatalf("code=%d stderr=%q", code, errOut.String())
	}
}

func TestRunRejectsDuplicateBundleJSONKeys(t *testing.T) {
	bundle := writeTemp(t, `{"authorization":{},"authorization":{}}`)
	trust := writeTemp(t, `{"keys":[]}`)
	policy := writeTemp(t, `{"policy_state":{},"expected_intent":{}}`)
	var out, errOut bytes.Buffer
	code := run([]string{"--bundle", bundle, "--trust-store", trust, "--policy", policy}, strings.NewReader(""), &out, &errOut)
	if code != 1 || !strings.Contains(errOut.String(), "duplicate-key scan") {
		t.Fatalf("code=%d stderr=%q", code, errOut.String())
	}
}

func TestRunRejectsMalformedPinnedTrustStore(t *testing.T) {
	bundle := writeTemp(t, `{}`)
	trust := writeTemp(t, `{"keys":[],"unknown":true}`)
	policy := writeTemp(t, `{"policy_state":{},"expected_intent":{}}`)
	var out, errOut bytes.Buffer
	code := run([]string{"--bundle", bundle, "--trust-store", trust, "--policy", policy}, strings.NewReader(""), &out, &errOut)
	if code != 1 || !strings.Contains(errOut.String(), "decode trust store") {
		t.Fatalf("code=%d stderr=%q", code, errOut.String())
	}
}

func TestRunReportsMissingPinnedTrustFile(t *testing.T) {
	bundle := writeTemp(t, `{}`)
	policy := writeTemp(t, `{"policy_state":{},"expected_intent":{}}`)
	var out, errOut bytes.Buffer
	code := run([]string{"--bundle", bundle, "--trust-store", filepath.Join(t.TempDir(), "missing.json"), "--policy", policy}, strings.NewReader(""), &out, &errOut)
	if code != 1 || !strings.Contains(errOut.String(), "read trusted root configuration") {
		t.Fatalf("code=%d stderr=%q", code, errOut.String())
	}
}

func TestReadInputRejectsOversizedData(t *testing.T) {
	_, err := readInput("-", strings.NewReader(strings.Repeat("x", join.MaximumRequestBytes+1)))
	if err == nil {
		t.Fatal("oversized evidence bundle accepted")
	}
}

func TestReadFileBoundedRejectsEmptyFile(t *testing.T) {
	path := writeTemp(t, "")
	if _, err := readFileBounded(path); err == nil {
		t.Fatal("empty pinned file accepted")
	}
}
