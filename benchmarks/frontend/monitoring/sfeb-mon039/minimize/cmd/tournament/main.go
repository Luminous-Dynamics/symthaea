package main

import (
 "bytes"
 "encoding/json"
 "errors"
 "flag"
 "fmt"
 "io"
 "os"
 minimize "luminousdynamics/mon039-counterfactual-fault-reduction"
)

func main() {
 inputPath:=flag.String("input","","MON-039 tournament input JSON")
 flag.Parse()
 if *inputPath=="" { fail("--input is required",64) }
 f,err:=os.Open(*inputPath);if err!=nil{fail("cannot open input: "+err.Error(),66)};defer f.Close()
 raw,err:=io.ReadAll(io.LimitReader(f,(32<<20)+1))
 if err!=nil||len(raw)>32<<20{fail("input read error or input exceeds 32 MiB",65)}
 dec:=json.NewDecoder(bytes.NewReader(raw));dec.DisallowUnknownFields()
 var in minimize.TournamentInput
 if err:=dec.Decode(&in);err!=nil{fail("invalid JSON: "+err.Error(),65)}
 var extra any;if err:=dec.Decode(&extra);!errors.Is(err,io.EOF){fail("input must contain exactly one JSON document",65)}
 if in.Profile!=minimize.TournamentProfile||in.Version!=minimize.TournamentVersion{fail("unsupported MON-039 tournament profile/version",65)}
 result:=minimize.MinimizeTournament(in.Input,in.Orders)
 enc:=json.NewEncoder(os.Stdout);enc.SetIndent("","  ")
 if err:=enc.Encode(result);err!=nil{fail("cannot encode result: "+err.Error(),74)}
 switch result.TournamentVerdict{case minimize.Qualified:os.Exit(0);case minimize.Blocked:os.Exit(2);default:os.Exit(3)}
}
func fail(message string,code int){fmt.Fprintln(os.Stderr,message);os.Exit(code)}
