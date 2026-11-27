package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"

	capture "luminousdynamics/mon038-kafka-partition-capture"
)

func main() {
	input := flag.String("input", "", "MON-038 partition-vector capture input")
	derive := flag.Bool("derive-fixture-proof-digests", false, "derive digest bindings for offline_fixture inputs only")
	flag.Parse()
	if *input == "" {
		fmt.Fprintln(os.Stderr, "--input is required")
		os.Exit(64)
	}
	file, err := os.Open(*input)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(65)
	}
	defer file.Close()

	raw, err := io.ReadAll(io.LimitReader(file, (32<<20)+1))
	if err != nil || len(raw) > 32<<20 {
		fmt.Fprintln(os.Stderr, "input read error or input exceeds 32 MiB")
		os.Exit(65)
	}
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.DisallowUnknownFields()
	var in capture.Input
	if err := decoder.Decode(&in); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(65)
	}
	var trailing any
	if err := decoder.Decode(&trailing); err != io.EOF {
		fmt.Fprintln(os.Stderr, "trailing or malformed JSON")
		os.Exit(65)
	}

	if *derive {
		if !capture.DeriveFixtureProofDigests(&in) {
			fmt.Fprintln(os.Stderr, "digest derivation is restricted to complete offline_fixture inputs")
			os.Exit(65)
		}
		encoded, err := json.MarshalIndent(in, "", "  ")
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(70)
		}
		fmt.Println(string(encoded))
		return
	}

	out := capture.Evaluate(in)
	encoded, err := json.MarshalIndent(out, "", "  ")
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(70)
	}
	fmt.Println(string(encoded))
	switch out.CaptureVerdict {
	case capture.Blocked:
		os.Exit(2)
	case capture.Indeterminate:
		os.Exit(3)
	}
}
