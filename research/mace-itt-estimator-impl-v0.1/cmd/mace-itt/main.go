// mace-itt reads paired randomized task records from JSONL and emits a
// task-clustered ITT report. It does not adjudicate task or control truth.
package main

import (
	"bufio"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"

	"mace-itt-estimator-impl-v0.1/itt"
)

type rosterInput struct {
	SchemaVersion string   `json:"schema_version"`
	PairIDs       []string `json:"pair_ids"`
}
type input struct {
	SchemaVersion string   `json:"schema_version"`
	Pair          itt.Pair `json:"pair"`
}

func main() {
	bootstrap := flag.Int("bootstrap", 2000, "task-cluster bootstrap replicates (>=200)")
	seed := flag.Int64("seed", 1, "deterministic bootstrap RNG seed")
	rosterPath := flag.String("roster", "", "JSON assignment roster; required for completeness checking")
	flag.Parse()
	cfg := itt.Config{BootstrapReplicates: *bootstrap, BootstrapSeed: *seed}
	if *rosterPath != "" {
		b, err := os.ReadFile(*rosterPath)
		if err != nil { fmt.Fprintln(os.Stderr, "mace-itt: roster:", err); os.Exit(1) }
		var r rosterInput
		d := json.NewDecoder(bytesReader(b))
		d.DisallowUnknownFields()
		if err := d.Decode(&r); err != nil || r.SchemaVersion != "mace-assignment-roster-v1" {
			fmt.Fprintln(os.Stderr, "mace-itt: invalid assignment roster"); os.Exit(1)
		}
		var trailing any
		if err := d.Decode(&trailing); !errors.Is(err, io.EOF) {
			fmt.Fprintln(os.Stderr, "mace-itt: trailing data in assignment roster"); os.Exit(1)
		}
		cfg.ExpectedPairIDs = r.PairIDs
		cfg.RequireAssignmentRoster = true
	}
	if err := run(os.Stdin, os.Stdout, cfg); err != nil {
		fmt.Fprintln(os.Stderr, "mace-itt:", err); os.Exit(1)
	}
}

func run(in io.Reader, out io.Writer, cfg itt.Config) error {
	s := bufio.NewScanner(in)
	s.Buffer(make([]byte, 64*1024), 4*1024*1024)
	pairs := []itt.Pair{}
	line := 0
	for s.Scan() {
		line++
		if len(s.Bytes()) == 0 { return fmt.Errorf("line %d: blank lines are not allowed", line) }
		var rec input
		d := json.NewDecoder(newSliceReader(s.Bytes()))
		d.DisallowUnknownFields()
		if err := d.Decode(&rec); err != nil { return fmt.Errorf("line %d: %w", line, err) }
		var extra any
		if err := d.Decode(&extra); !errors.Is(err, io.EOF) {
			if err == nil { return fmt.Errorf("line %d: multiple JSON values", line) }
			return fmt.Errorf("line %d: trailing data: %w", line, err)
		}
		if rec.SchemaVersion != "mace-itt-pair-v1" { return fmt.Errorf("line %d: unsupported schema %q", line, rec.SchemaVersion) }
		pairs = append(pairs, rec.Pair)
	}
	if err := s.Err(); err != nil { return err }
	report, err := itt.EstimateITT(pairs, cfg)
	if err != nil { return err }
	enc := json.NewEncoder(out)
	enc.SetIndent("", "  ")
	return enc.Encode(report)
}

func bytesReader(b []byte) *sliceReader { return newSliceReader(b) }
type sliceReader struct { b []byte; i int }
func newSliceReader(b []byte) *sliceReader { return &sliceReader{b: b} }
func (r *sliceReader) Read(p []byte) (int, error) {
	if r.i >= len(r.b) { return 0, io.EOF }
	n := copy(p, r.b[r.i:])
	r.i += n
	return n, nil
}
