package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"
)

func main() {
	dec := json.NewDecoder(os.Stdin)
	dec.DisallowUnknownFields()
	var in Request
	if err := dec.Decode(&in); err != nil { fail(err) }
	var extra any
	if err := dec.Decode(&extra); err != io.EOF { fail(errors.New("trailing JSON content")) }
	report := Validate(in)
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(report); err != nil { fail(err) }
	if !report.Accepted { os.Exit(1) }
}
func fail(err error) { fmt.Fprintln(os.Stderr, strings.TrimSpace(err.Error())); os.Exit(2) }
