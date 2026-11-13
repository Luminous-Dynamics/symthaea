// mace-join verifies an untrusted evidence bundle against a separate,
// operator-pinned trust store and policy/intent file. Never keep these files in
// the same agent-writable location as the evidence bundle.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/Luminous-Dynamics/symthaea/research/mace-effect-join-v0.1/join"
)

type policyFile struct {
	State    join.PolicyState       `json:"policy_state"`
	Expected join.IntentExpectation `json:"expected_intent"`
}

func main() {
	os.Exit(run(os.Args[1:], os.Stdin, os.Stdout, os.Stderr))
}

func run(args []string, stdin io.Reader, stdout, stderr io.Writer) int {
	flags := flag.NewFlagSet("mace-join", flag.ContinueOnError)
	flags.SetOutput(stderr)
	bundlePath := flags.String("bundle", "-", "signed evidence bundle JSON; use - for stdin")
	trustPath := flags.String("trust-store", "", "operator-pinned trust store JSON (required)")
	policyPath := flags.String("policy", "", "operator-pinned policy and task-intent JSON (required)")
	if err := flags.Parse(args); err != nil {
		return 2
	}
	if flags.NArg() != 0 {
		fmt.Fprintln(stderr, "mace-join: unexpected positional arguments")
		return 2
	}
	if *trustPath == "" || *policyPath == "" {
		fmt.Fprintln(stderr, "mace-join: --trust-store and --policy are required and must be provisioned out of band")
		return 2
	}
	bundleBytes, err := readInput(*bundlePath, stdin)
	if err != nil {
		fmt.Fprintln(stderr, "mace-join:", err)
		return 1
	}
	trustBytes, err := readFileBounded(*trustPath)
	if err != nil {
		fmt.Fprintln(stderr, "mace-join: read trusted root configuration:", err)
		return 1
	}
	policyBytes, err := readFileBounded(*policyPath)
	if err != nil {
		fmt.Fprintln(stderr, "mace-join: read pinned policy:", err)
		return 1
	}
	var bundle join.Bundle
	var trust join.TrustStore
	var policy policyFile
	if err := join.DecodeStrict(bundleBytes, &bundle); err != nil {
		fmt.Fprintln(stderr, "mace-join: decode evidence bundle:", err)
		return 1
	}
	if err := join.DecodeStrict(trustBytes, &trust); err != nil {
		fmt.Fprintln(stderr, "mace-join: decode trust store:", err)
		return 1
	}
	if err := join.DecodeStrict(policyBytes, &policy); err != nil {
		fmt.Fprintln(stderr, "mace-join: decode policy/intent:", err)
		return 1
	}
	result, err := join.Verify(bundle, trust, policy.State, policy.Expected)
	if err != nil {
		fmt.Fprintln(stderr, "mace-join: verify:", err)
		return 1
	}
	encoder := json.NewEncoder(stdout)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(result); err != nil {
		fmt.Fprintln(stderr, "mace-join: encode result:", err)
		return 1
	}
	return 0
}

func readInput(path string, stdin io.Reader) ([]byte, error) {
	var reader io.Reader = stdin
	if path != "-" {
		f, err := os.Open(path)
		if err != nil {
			return nil, err
		}
		defer f.Close()
		reader = f
	}
	data, err := io.ReadAll(io.LimitReader(reader, join.MaximumRequestBytes+1))
	if err != nil {
		return nil, err
	}
	if len(data) == 0 || len(data) > join.MaximumRequestBytes {
		return nil, fmt.Errorf("evidence bundle is empty or exceeds %d bytes", join.MaximumRequestBytes)
	}
	return data, nil
}

func readFileBounded(path string) ([]byte, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	data, err := io.ReadAll(io.LimitReader(f, join.MaximumRequestBytes+1))
	if err != nil {
		return nil, err
	}
	if len(data) == 0 || len(data) > join.MaximumRequestBytes {
		return nil, fmt.Errorf("file %q is empty or exceeds %d bytes", path, join.MaximumRequestBytes)
	}
	return data, nil
}
