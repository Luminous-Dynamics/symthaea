package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"

	"luminousdynamics/mon041-custody"
)

func main() {
	inputPath := flag.String("input", "", "path to MON-041 custody input JSON")
	flag.Parse()
	if *inputPath == "" { fail("--input is required", 64) }
	f, err := os.Open(*inputPath)
	if err != nil { fail("cannot open input: "+err.Error(), 66) }
	defer f.Close()
	dec := json.NewDecoder(f)
	dec.DisallowUnknownFields()
	var in custody.Input
	if err := dec.Decode(&in); err != nil { fail("invalid input JSON: "+err.Error(), 65) }
	var extra any
	if err := dec.Decode(&extra); !errors.Is(err, io.EOF) { fail("input must contain exactly one JSON document", 65) }
	out := custody.Evaluate(in)
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(out); err != nil { fail("cannot encode result: "+err.Error(), 74) }
	switch out.CustodyVerdict {
	case custody.Qualified: os.Exit(0)
	case custody.Blocked: os.Exit(2)
	default: os.Exit(3)
	}
}

func fail(message string, code int) { fmt.Fprintln(os.Stderr, message); os.Exit(code) }
