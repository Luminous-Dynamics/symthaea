package main

import (
 "bytes"
 "encoding/json"
 "errors"
 "flag"
 "fmt"
 "io"
 "os"
 invalidation "luminousdynamics/mon045-evidence-invalidation"
)

func main() {
 input:=flag.String("input","","MON-045 dependency graph input JSON")
 flag.Parse()
 if *input=="" { fail("--input is required",64) }
 f,err:=os.Open(*input); if err!=nil { fail("cannot open input: "+err.Error(),66) }; defer f.Close()
 raw,err:=io.ReadAll(io.LimitReader(f,(16<<20)+1))
 if err!=nil||len(raw)>16<<20 { fail("input read error or input exceeds 16 MiB",65) }
 dec:=json.NewDecoder(bytes.NewReader(raw)); dec.DisallowUnknownFields()
 var in invalidation.Input
 if err:=dec.Decode(&in); err!=nil { fail("invalid JSON: "+err.Error(),65) }
 var extra any
 if err:=dec.Decode(&extra); !errors.Is(err,io.EOF) { fail("input must contain exactly one JSON document",65) }
 out:=invalidation.Evaluate(in)
 enc:=json.NewEncoder(os.Stdout); enc.SetIndent("","  ")
 if err:=enc.Encode(out); err!=nil { fail("cannot encode result: "+err.Error(),74) }
 switch out.GraphVerdict { case invalidation.Qualified: os.Exit(0); case invalidation.Blocked: os.Exit(2); default: os.Exit(3) }
}
func fail(message string,code int){fmt.Fprintln(os.Stderr,message);os.Exit(code)}
