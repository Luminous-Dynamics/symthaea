package main

import (
 "bytes"
 "encoding/json"
 "flag"
 "fmt"
 "io"
 "os"
 campaign "luminousdynamics/mon037-unified-recovery-campaign"
)

func main(){
 path:=flag.String("input","","MON-037 campaign input");derive:=flag.Bool("derive-fixture-record-digests",false,"derive canonical record-set digests for offline fixtures only");flag.Parse()
 if *path==""{fmt.Fprintln(os.Stderr,"--input is required");os.Exit(64)}
 file,err:=os.Open(*path);if err!=nil{fmt.Fprintln(os.Stderr,err);os.Exit(65)};defer file.Close()
 raw,err:=io.ReadAll(io.LimitReader(file,(32<<20)+1));if err!=nil||len(raw)>32<<20{fmt.Fprintln(os.Stderr,"input read or size error");os.Exit(65)}
 dec:=json.NewDecoder(bytes.NewReader(raw));dec.DisallowUnknownFields()
 var in campaign.Input
 if err=dec.Decode(&in);err!=nil{fmt.Fprintln(os.Stderr,err);os.Exit(65)}
 var trailing any
 if err=dec.Decode(&trailing);err!=io.EOF{fmt.Fprintln(os.Stderr,"trailing or malformed JSON");os.Exit(65)}
 if *derive{if !campaign.DeriveFixtureRecordSetDigests(&in){fmt.Fprintln(os.Stderr,"digest derivation is restricted to offline_fixture mode");os.Exit(65)};derived,err:=json.MarshalIndent(in,"","  ");if err!=nil{fmt.Fprintln(os.Stderr,err);os.Exit(70)};fmt.Println(string(derived));return};out:=campaign.Evaluate(in);encoded,err:=json.MarshalIndent(out,"","  ");if err!=nil{fmt.Fprintln(os.Stderr,err);os.Exit(70)}
 fmt.Println(string(encoded))
 if out.CampaignVerdict==campaign.Blocked{os.Exit(2)}
 if out.CampaignVerdict==campaign.Indeterminate{os.Exit(3)}
}
