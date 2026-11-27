package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"

	replay "luminousdynamics/mon040-replay-reproducibility"
)

func main() {
	path := flag.String("input", "", "MON-040 replay reproducibility input")
	flag.Parse()
	if *path == "" {
		fmt.Fprintln(os.Stderr, "--input is required")
		os.Exit(64)
	}
	file, err := os.Open(*path)
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
	var in replay.Input
	if err := decoder.Decode(&in); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(65)
	}
	var trailing any
	if err := decoder.Decode(&trailing); err != io.EOF {
		fmt.Fprintln(os.Stderr, "trailing or malformed JSON")
		os.Exit(65)
	}
	out := replay.Evaluate(in)
	encoded, err := json.MarshalIndent(out, "", "  ")
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(70)
	}
	fmt.Println(string(encoded))
	switch out.CampaignVerdict {
	case replay.Blocked:
		os.Exit(2)
	case replay.Indeterminate:
		os.Exit(3)
	}
}
