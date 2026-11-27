package ancestry

import "testing"

const history3 = "1\t0/00000064\tpromotion from timeline 1\n2\t0/000000C8\tpromotion from timeline 2\n"
const systemID = "7438291029384756102"

func baseInput() Input {
	return Input{
		Profile: Profile, Version: Version,
		ExpectedSystemIdentifier: systemID,
		BaseBackupSystemIdentifier: systemID,
		BaseBackupTimelineID: 1, BaseBackupLSN: "0/00000050",
		RecoveredSystemIdentifier: systemID,
		RecoveredTimelineID: 3, RecoveryTargetLSN: "0/000000F0",
		HistoryTimelineID: 3, HistoryText: history3,
		HistorySHA256: HistorySHA256(history3),
	}
}

func has(r Result, reason string) bool {
	for _, item := range r.ReasonCodes { if item == reason { return true } }
	return false
}

func TestAncestorBaseAndDescendantTargetQualify(t *testing.T) {
	got := Evaluate(baseInput())
	if got.Verdict != Qualified || got.SourceTimelineInterval == nil || got.TargetTimelineInterval == nil {
		t.Fatalf("expected valid ancestry proof, got %+v", got)
	}
	if got.SourceTimelineInterval.EndLSN != "0/00000064" || got.TargetTimelineInterval.StartLSN != "0/000000C8" {
		t.Fatalf("unexpected timeline intervals: %+v %+v", got.SourceTimelineInterval, got.TargetTimelineInterval)
	}
}

func TestIntermediateTimelineBackupQualifies(t *testing.T) {
	in := baseInput()
	in.BaseBackupTimelineID = 2
	in.BaseBackupLSN = "0/00000080"
	got := Evaluate(in)
	if got.Verdict != Qualified {
		t.Fatalf("backup on intermediate ancestor should qualify: %+v", got)
	}
}

func TestBaseBackupAtOrPastItsForkDoesNotQualify(t *testing.T) {
	in := baseInput()
	in.BaseBackupLSN = "0/00000064"
	got := Evaluate(in)
	if got.Verdict != Blocked || !has(got, "base_backup_lsn_outside_source_timeline_interval") {
		t.Fatalf("backup at parent forkpoint belongs to next timeline and must block: %+v", got)
	}
	in.BaseBackupLSN = "0/00000065"
	got = Evaluate(in)
	if got.Verdict != Blocked || !has(got, "base_backup_lsn_outside_source_timeline_interval") {
		t.Fatalf("base backup past parent fork must block: %+v", got)
	}
}

func TestTargetBeforeDescendantForkBlocks(t *testing.T) {
	in := baseInput()
	in.RecoveryTargetLSN = "0/000000C7"
	got := Evaluate(in)
	if got.Verdict != Blocked || !has(got, "recovery_target_precedes_target_timeline_fork") {
		t.Fatalf("target before child timeline start must block: %+v", got)
	}
}

func TestTargetAtForkBoundaryIsIndeterminate(t *testing.T) {
	in := baseInput()
	in.RecoveryTargetLSN = "0/000000C8"
	got := Evaluate(in)
	if got.Verdict != Indeterminate || !has(got, "recovery_target_exactly_at_timeline_fork_boundary") {
		t.Fatalf("fork equality must remain indeterminate: %+v", got)
	}
}

func TestBackupTimelineNotInRequestedBranchBlocks(t *testing.T) {
	in := baseInput()
	in.BaseBackupTimelineID = 2
	in.BaseBackupLSN = "0/00000080"
	in.RecoveredTimelineID = 4
	in.HistoryTimelineID = 4
	in.HistoryText = "1\t0/00000064\tpromotion from timeline 1\n3\t0/00000080\tpromotion from timeline 3\n"
	in.HistorySHA256 = HistorySHA256(in.HistoryText)
	got := Evaluate(in)
	if got.Verdict != Blocked || !has(got, "base_backup_timeline_not_in_target_ancestry") {
		t.Fatalf("backup from sibling branch must block: %+v", got)
	}
}

func TestSystemIdentifierMismatchBlocks(t *testing.T) {
	in := baseInput()
	in.RecoveredSystemIdentifier = "999"
	got := Evaluate(in)
	if got.Verdict != Blocked || !has(got, "system_identifier_mismatch") {
		t.Fatalf("different physical cluster must block: %+v", got)
	}
}

func TestHistoryDigestMutationBlocks(t *testing.T) {
	in := baseInput()
	in.HistoryText += "# altered\n"
	got := Evaluate(in)
	if got.Verdict != Blocked || !has(got, "timeline_history_digest_mismatch") {
		t.Fatalf("history byte mutation must block: %+v", got)
	}
}

func TestMalformedHistoryBlocks(t *testing.T) {
	for _, invalid := range []string{
		"2\t0/00000064\tmissing root timeline\n",
		"1\tbad-lsn\treason\n",
		"1\t0/00000064\n",
		"1\t0/00000064\troot\n1\t0/000000C8\tduplicate parent\n",
		"1\t0/000000C8\troot\n2\t0/00000064\tbackwards fork\n",
		"1\t0/00000064\troot\n3\t0/000000C8\tparent id cannot equal target\n",
	} {
		if _, err := ParseHistory(3, invalid); err == nil {
			t.Fatalf("invalid history accepted: %q", invalid)
		}
	}
}

func TestSameTimelineRecoveryUsesComparableLSNs(t *testing.T) {
	in := baseInput()
	in.BaseBackupTimelineID = 3
	in.BaseBackupLSN = "0/000000D0"
	in.RecoveryTargetLSN = "0/000000E0"
	got := Evaluate(in)
	if got.Verdict != Qualified {
		t.Fatalf("same-timeline ordered recovery should qualify: %+v", got)
	}
	in.RecoveryTargetLSN = "0/000000CF"
	got = Evaluate(in)
	if got.Verdict != Blocked || !has(got, "recovery_target_precedes_base_backup_on_same_timeline") {
		t.Fatalf("same-timeline target before backup must block: %+v", got)
	}
}

func TestTimelineOneNeedsNoHistory(t *testing.T) {
	in := Input{
		Profile: Profile, Version: Version,
		ExpectedSystemIdentifier: systemID,
		BaseBackupSystemIdentifier: systemID, BaseBackupTimelineID: 1, BaseBackupLSN: "0/10",
		RecoveredSystemIdentifier: systemID, RecoveredTimelineID: 1, RecoveryTargetLSN: "0/20",
		HistoryTimelineID: 1,
	}
	got := Evaluate(in)
	if got.Verdict != Qualified { t.Fatalf("timeline 1 recovery should qualify: %+v", got) }
}

func TestMissingHistoryIsIndeterminate(t *testing.T) {
	in := baseInput()
	in.HistoryText = ""
	in.HistorySHA256 = ""
	got := Evaluate(in)
	if got.Verdict != Indeterminate || !has(got, "target_timeline_history_missing") {
		t.Fatalf("missing target history must be indeterminate: %+v", got)
	}
}

func TestHistoryCommentsAndReasonsWithSpacesAreAccepted(t *testing.T) {
	h := "# archived from independent WAL store\n1\t0/00000064\tpromotion during planned switchover\n2\t0/000000C8\toperator failover after partition\n"
	segments, err := ParseHistory(3, h)
	if err != nil || len(segments) != 3 {
		t.Fatalf("valid comments/reason text rejected: %v %+v", err, segments)
	}
}

func TestLSNParserIsStrict(t *testing.T) {
	for _, v := range []string{"", "0", "0/0/0", "GG/1", "100000000/1", "0/100000000"} {
		if _, err := ParseLSN(v); err == nil {
			t.Fatalf("invalid LSN accepted: %q", v)
		}
	}
	if got, err := ParseLSN("A/BC"); err != nil || FormatLSN(got) != "A/000000BC" {
		t.Fatalf("valid LSN parse/format failed: %X %v", got, err)
	}
}


func TestBackupLSNBeforeTimelineStartBlocks(t *testing.T) {
	in := baseInput()
	in.BaseBackupTimelineID = 2
	in.BaseBackupLSN = "0/00000063"
	got := Evaluate(in)
	if got.Verdict != Blocked || !has(got, "base_backup_lsn_before_timeline_start") {
		t.Fatalf("base backup before source timeline start must block: %+v", got)
	}
}

func TestBackupAtTimelineStartIsIndeterminate(t *testing.T) {
	in := baseInput()
	in.BaseBackupTimelineID = 2
	in.BaseBackupLSN = "0/00000064"
	got := Evaluate(in)
	if got.Verdict != Indeterminate || !has(got, "base_backup_lsn_at_timeline_start_boundary") {
		t.Fatalf("exact source timeline start boundary must remain indeterminate: %+v", got)
	}
}

func TestZeroLengthIntermediateTimelineIsRepresentable(t *testing.T) {
	history := "1\t0/00000064\tpromotion from timeline 1\n2\t0/00000064\timmediate promotion from timeline 2\n"
	segments, err := ParseHistory(3, history)
	if err != nil || len(segments) != 3 || segments[1].Start != segments[1].End {
		t.Fatalf("equal consecutive forkpoints should preserve a zero-length timeline interval: %v %+v", err, segments)
	}
}

func TestSystemIdentifierMustBeCanonicalUint64(t *testing.T) {
	for _, invalid := range []string{"not-a-system-id", "0", "18446744073709551616", "07438291029384756102"} {
		in := baseInput()
		in.ExpectedSystemIdentifier = invalid
		in.BaseBackupSystemIdentifier = invalid
		in.RecoveredSystemIdentifier = invalid
		got := Evaluate(in)
		if got.Verdict != Blocked || !has(got, "system_identifier_invalid") {
			t.Fatalf("invalid/non-canonical system identifier %q must block: %+v", invalid, got)
		}
	}
}
