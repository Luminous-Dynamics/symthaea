package capability

import (
 "testing"
 "time"
)

var evaluated=time.Date(2026,11,28,12,0,0,0,time.UTC)
func sha(s string)string{return digest(s)}
func testIdentity()ConnectorIdentity{return ConnectorIdentity{
 ConnectorID:"debezium-postgres-01",ConnectorVersion:"3.7.1.Final",DistributionSHA256:sha("distribution"),
 DatabaseEngine:"postgresql",DatabaseVersion:"18.7",LogicalDecodingPlugin:"pgoutput",BrokerVersion:"4.3.1",ClientVersion:"4.3.1",
 ConfigurationSHA256:sha("config"),PublicationSHA256:sha("publication"),SchemaHistorySHA256:sha("schema-history"),
 SnapshotMode:"initial",TransactionMetadataEnabled:true,TransactionMetadataEpoch:"connector-deployment-epoch-001",
 OperationFiltersSHA256:sha("operation-filters"),ColumnPolicySHA256:sha("column-policy"),TopicRoutingSHA256:sha("topic-routing"),
}}
func testRequired(id,domain,obj,op string,required bool)RequiredItem{return RequiredItem{ItemID:id,StateDomain:domain,ObjectID:obj,Operation:op,Required:required,ScopeSHA256:sha("scope-"+id)}}
func supportedItem(id,scope,identity string)CapabilityItem{return CapabilityItem{ItemID:id,ScopeSHA256:scope,Status:Supported,Conformance:&ConformanceEvidence{CapabilityIdentitySHA256:identity,TestProfileSHA256:sha("test-profile-"+id),ReportSHA256:sha("report-"+id),Status:EvidencePassed}}}
func validInput()Input{
 identity:=testIdentity();idSHA:=IdentityDigest(identity)
 req:=[]RequiredItem{
  testRequired("ddl-schema-history","schema","public.ledger","schema_snapshot",true),
  testRequired("transaction-metadata-epoch","transaction","connector-epoch-001","begin_end",true),
 }
 items:=[]CapabilityItem{
  supportedItem(req[0].ItemID,req[0].ScopeSHA256,idSHA),
  supportedItem(req[1].ItemID,req[1].ScopeSHA256,idSHA),
 }
 manifest:=Manifest{Identity:identity,Items:items}
 return Input{Profile:Profile,Version:Version,Mode:OfflineFixture,EvaluatedAt:evaluated,TrialID:"mon044-trial-001",
  CapabilityIdentitySHA256:idSHA,CapabilityManifestSHA256:ManifestDigest(manifest),RequiredScopeSHA256:RequiredScopeDigest(req),
  RequiredScopeSourceID:"independent-db-catalog-export",RequiredScopeProvenanceSHA256:sha("source-inventory-receipt"),
  RequiredItems:req,Manifest:manifest}
}
func TestFullyCoveredRequiredScopeQualifiesOfflineCoverageOnly(t *testing.T){
 got:=Evaluate(validInput())
 if got.CapabilityVerdict!=Qualified||got.CoverageVerdict!=Qualified{t.Fatalf("expected offline scope coverage: %+v",got)}
 if got.CoveredRequiredCount!=2||got.CampaignVerdict!=Indeterminate||got.ProductionActionPermitted{t.Fatalf("offline capability proof must not authorize recovery: %+v",got)}
}
func TestPostgresDDLBlindSpotBlocksRequiredScope(t *testing.T){
 in:=validInput();in.Manifest.Items[0].Status=Unsupported
 in.Manifest.Items[0].Conformance=nil
 in.Manifest.Items[0].AuthorityReference="Debezium PostgreSQL connector docs"
 in.Manifest.Items[0].AuthorityDocumentSHA256=sha("document-ddl")
 in.CapabilityManifestSHA256=ManifestDigest(in.Manifest)
 got:=Evaluate(in)
 if got.CoverageVerdict!=Blocked||len(got.UnsupportedRequired)!=1{t.Fatalf("DDL is unsupported by logical decoding and cannot count as recovered: %+v",got)}
}
func TestTransactionMetadataBeforeActivationEpochIsNotClaimed(t *testing.T){
 in:=validInput();in.Manifest.Identity.TransactionMetadataEpoch="after-required-history"
 in.CapabilityIdentitySHA256=IdentityDigest(in.Manifest.Identity)
 for i:=range in.Manifest.Items{if in.Manifest.Items[i].Conformance!=nil{in.Manifest.Items[i].Conformance.CapabilityIdentitySHA256=in.CapabilityIdentitySHA256}}
 in.Manifest.Items[1].Status=Conditional
 in.Manifest.Items[1].RequiredPrerequisites=[]string{"transaction_metadata_enabled_before_scope"}
 in.Manifest.Items[1].PrerequisiteEvidence=[]PrerequisiteEvidence{{PrerequisiteID:"transaction_metadata_enabled_before_scope",EvidenceSHA256:sha("activation-proof"),Status:EvidenceFailed}}
 in.CapabilityManifestSHA256=ManifestDigest(in.Manifest)
 got:=Evaluate(in)
 if got.CoverageVerdict!=Blocked{t.Fatalf("missing historical transaction metadata must not be inferred: %+v",got)}
}
func TestGeneratedColumnUnsupportedRequiredBlocks(t *testing.T){
 in:=validInput();in.Manifest.Items[0].Status=Unsupported;in.Manifest.Items[0].Conformance=nil
 in.Manifest.Items[0].AuthorityReference="PostgreSQL pgoutput connector documentation"
 in.Manifest.Items[0].AuthorityDocumentSHA256=sha("generated-column-doc")
 in.CapabilityManifestSHA256=ManifestDigest(in.Manifest)
 got:=Evaluate(in)
 if got.CoverageVerdict!=Blocked{t.Fatalf("generated value requirement must not pass without a capture/recompute lane: %+v",got)}
}
func TestConditionalCapabilityRequiresEveryPrerequisite(t *testing.T){
 in:=validInput();it:=&in.Manifest.Items[0];it.Status=Conditional
 it.RequiredPrerequisites=[]string{"schema-history-present","publication-includes-table"}
 it.PrerequisiteEvidence=[]PrerequisiteEvidence{{PrerequisiteID:"schema-history-present",EvidenceSHA256:sha("history"),Status:EvidencePassed}}
 in.CapabilityManifestSHA256=ManifestDigest(in.Manifest)
 got:=Evaluate(in)
 if got.CoverageVerdict!=Indeterminate{t.Fatalf("missing conditional prerequisite must hold: %+v",got)}
}
func TestConditionalCapabilityWithPassedPrerequisitesQualifies(t *testing.T){
 in:=validInput();it:=&in.Manifest.Items[0];it.Status=Conditional
 it.RequiredPrerequisites=[]string{"schema-history-present","publication-includes-table"}
 it.PrerequisiteEvidence=[]PrerequisiteEvidence{
  {PrerequisiteID:"schema-history-present",EvidenceSHA256:sha("history"),Status:EvidencePassed},
  {PrerequisiteID:"publication-includes-table",EvidenceSHA256:sha("publication"),Status:EvidencePassed},
 }
 in.CapabilityManifestSHA256=ManifestDigest(in.Manifest)
 if got:=Evaluate(in);got.CoverageVerdict!=Qualified{t.Fatalf("all prerequisites pass: %+v",got)}
}
func TestExcludedRequiredOperationBlocks(t *testing.T){
 in:=validInput();in.Manifest.Items[0].Status=Excluded;in.Manifest.Items[0].Conformance=nil
 in.Manifest.Items[0].AuthorityReference="connector skipped-operation configuration"
 in.Manifest.Items[0].AuthorityDocumentSHA256=sha("filters")
 in.CapabilityManifestSHA256=ManifestDigest(in.Manifest)
 got:=Evaluate(in)
 if got.CoverageVerdict!=Blocked||len(got.ExcludedRequired)!=1{t.Fatalf("required skipped operation must block: %+v",got)}
}
func TestNotApplicableRequiresJustification(t *testing.T){
 in:=validInput();in.Manifest.Items[0].Status=NotApplicable;in.Manifest.Items[0].Conformance=nil
 in.Manifest.Items[0].JustificationSHA256=""
 in.CapabilityManifestSHA256=ManifestDigest(in.Manifest)
 if got:=Evaluate(in);got.CoverageVerdict!=Indeterminate{t.Fatalf("unjustified N/A must hold: %+v",got)}
}
func TestNotApplicableWithJustificationIsExplicit(t *testing.T){
 in:=validInput();in.Manifest.Items[0].Status=NotApplicable;in.Manifest.Items[0].Conformance=nil
 in.Manifest.Items[0].JustificationSHA256=sha("not-applicable-proof")
 in.CapabilityManifestSHA256=ManifestDigest(in.Manifest)
 if got:=Evaluate(in);got.CoverageVerdict!=Qualified{t.Fatalf("justified N/A can close its scope item: %+v",got)}
}
func TestUnknownRequiredCapabilityHolds(t *testing.T){
 in:=validInput();in.Manifest.Items[0].Status=Unknown;in.Manifest.Items[0].Conformance=nil
 in.CapabilityManifestSHA256=ManifestDigest(in.Manifest)
 if got:=Evaluate(in);got.CoverageVerdict!=Indeterminate||len(got.UnknownRequired)!=1{t.Fatalf("unknown capability must hold: %+v",got)}
}
func TestMissingCapabilityItemHolds(t *testing.T){
 in:=validInput();in.Manifest.Items=in.Manifest.Items[:1]
 in.CapabilityManifestSHA256=ManifestDigest(in.Manifest)
 if got:=Evaluate(in);got.CoverageVerdict!=Indeterminate||len(got.MissingRequired)!=1{t.Fatalf("missing declaration must hold: %+v",got)}
}
func TestCapabilityScopeMismatchBlocks(t *testing.T){
 in:=validInput();in.Manifest.Items[0].ScopeSHA256=sha("different-scope")
 in.CapabilityManifestSHA256=ManifestDigest(in.Manifest)
 if got:=Evaluate(in);got.CoverageVerdict!=Blocked{t.Fatalf("scope mismatch must block: %+v",got)}
}
func TestDuplicateCapabilityItemBlocks(t *testing.T){
 in:=validInput();in.Manifest.Items=append(in.Manifest.Items,in.Manifest.Items[0])
 in.CapabilityManifestSHA256=ManifestDigest(in.Manifest)
 if got:=Evaluate(in);got.CoverageVerdict!=Blocked{t.Fatalf("duplicate capability entry must block: %+v",got)}
}
func TestManifestAndIdentityTamperingBlock(t *testing.T){
 in:=validInput();in.Manifest.Identity.ConfigurationSHA256=sha("mutated")
 if got:=Evaluate(in);got.CoverageVerdict!=Blocked{t.Fatalf("identity/config drift without digest update must block: %+v",got)}
}
func TestRequiredInventoryNeedsIndependentSourceProvenance(t *testing.T){
 in:=validInput();in.RequiredScopeSourceID="";if got:=Evaluate(in);got.CoverageVerdict!=Blocked{t.Fatalf("missing scope source identity must block: %+v",got)}
 in=validInput();in.RequiredScopeProvenanceSHA256="";if got:=Evaluate(in);got.CoverageVerdict!=Indeterminate{t.Fatalf("missing scope source receipt must hold: %+v",got)}
}
func TestOutOfScopeCapabilityDeclarationsRemainVisible(t *testing.T){
 in:=validInput();extra:=supportedItem("optional-sequence-capture",sha("optional-scope"),in.CapabilityIdentitySHA256)
 in.Manifest.Items=append(in.Manifest.Items,extra);in.CapabilityManifestSHA256=ManifestDigest(in.Manifest)
 got:=Evaluate(in)
 if got.CoverageVerdict!=Qualified||len(got.OutOfScopeDeclarations)!=1{t.Fatalf("extra declaration must be visible without expanding required scope: %+v",got)}
}
func TestRuntimeCandidateNeverAuthorizesCampaign(t *testing.T){
 in:=validInput();in.Mode=RuntimeCandidate;got:=Evaluate(in)
 if got.CampaignVerdict!=Indeterminate||got.ProductionActionPermitted{t.Fatalf("runtime candidate cannot self-authorize: %+v",got)}
}
func TestClonedConformanceReportCannotCrossCapabilityIdentity(t *testing.T){
 in:=validInput();in.Manifest.Identity.ConfigurationSHA256=sha("new-config")
 in.CapabilityIdentitySHA256=IdentityDigest(in.Manifest.Identity)
 in.Manifest.Items[0].Conformance.CapabilityIdentitySHA256=sha("old-config-identity")
 in.CapabilityManifestSHA256=ManifestDigest(in.Manifest)
 if got:=Evaluate(in);got.CoverageVerdict!=Blocked{t.Fatalf("conformance from old config must not qualify new config: %+v",got)}
}
