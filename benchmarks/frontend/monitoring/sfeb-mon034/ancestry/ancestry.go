package ancestry

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"sort"
	"strconv"
	"strings"
)

const (
	Profile = "mon034-postgres-timeline-ancestry-input-v1"
	Version = "1.0.0"
)

type Input struct {
	Profile string `json:"profile"`
	Version string `json:"version"`
	ExpectedSystemIdentifier string `json:"expected_system_identifier"`
	BaseBackupSystemIdentifier string `json:"base_backup_system_identifier"`
	BaseBackupTimelineID uint32 `json:"base_backup_timeline_id"`
	BaseBackupLSN string `json:"base_backup_lsn"`
	RecoveredSystemIdentifier string `json:"recovered_system_identifier"`
	RecoveredTimelineID uint32 `json:"recovered_timeline_id"`
	RecoveryTargetLSN string `json:"recovery_target_lsn"`
	HistoryTimelineID uint32 `json:"history_timeline_id"`
	HistorySHA256 string `json:"history_sha256"`
	HistoryText string `json:"history_text"`
}

type Verdict string

const (
	Qualified Verdict = "qualified"
	Blocked Verdict = "blocked"
	Indeterminate Verdict = "indeterminate"
)

type Interval struct {
	TimelineID uint32 `json:"timeline_id"`
	StartLSN string `json:"start_lsn"`
	EndLSN string `json:"end_lsn"`
}

type Result struct {
	Verdict Verdict `json:"verdict"`
	ReasonCodes []string `json:"reason_codes"`
	SourceTimelineInterval *Interval `json:"source_timeline_interval"`
	TargetTimelineInterval *Interval `json:"target_timeline_interval"`
	ComputedHistorySHA256 string `json:"computed_history_sha256"`
	BaseBackupLSN string `json:"base_backup_lsn"`
	RecoveryTargetLSN string `json:"recovery_target_lsn"`
	BaseBackupTimelineID uint32 `json:"base_backup_timeline_id"`
	RecoveredTimelineID uint32 `json:"recovered_timeline_id"`
}

var (
	ErrInvalidLSN = errors.New("invalid PostgreSQL LSN")
	ErrInvalidHistory = errors.New("invalid PostgreSQL timeline history")
)

type segment struct {
	TimelineID uint32
	Start uint64
	End uint64
	HasEnd bool
}

func ParseLSN(value string) (uint64, error) {
	parts := strings.Split(value, "/")
	if len(parts) != 2 || parts[0] == "" || parts[1] == "" ||
		len(parts[0]) > 8 || len(parts[1]) > 8 {
		return 0, fmt.Errorf("%w: %q", ErrInvalidLSN, value)
	}
	hi, err := strconv.ParseUint(parts[0], 16, 32)
	if err != nil {
		return 0, fmt.Errorf("%w: %q", ErrInvalidLSN, value)
	}
	lo, err := strconv.ParseUint(parts[1], 16, 32)
	if err != nil {
		return 0, fmt.Errorf("%w: %q", ErrInvalidLSN, value)
	}
	return (hi << 32) | lo, nil
}

func FormatLSN(value uint64) string {
	return fmt.Sprintf("%X/%08X", value>>32, value&0xffffffff)
}

func HistorySHA256(history string) string {
	sum := sha256.Sum256([]byte(history))
	return hex.EncodeToString(sum[:])
}

// ParseHistory interprets PostgreSQL's target-TLI.history format. Each parent
// timeline has the interval [previous switchpoint, current switchpoint).
// The recovered target timeline begins at the final switchpoint.
func ParseHistory(targetTLI uint32, history string) ([]segment, error) {
	if targetTLI == 0 {
		return nil, fmt.Errorf("%w: timeline zero", ErrInvalidHistory)
	}
	if targetTLI == 1 {
		if strings.TrimSpace(history) != "" {
			return nil, fmt.Errorf("%w: timeline 1 must not have a history file", ErrInvalidHistory)
		}
		return []segment{{TimelineID: 1, Start: 0}}, nil
	}
	if strings.TrimSpace(history) == "" {
		return nil, fmt.Errorf("%w: timeline history is missing", ErrInvalidHistory)
	}

	segments := make([]segment, 0, 8)
	var previousTLI uint32
	var previousSwitch uint64
	seen := false
	for lineNo, raw := range strings.Split(history, "\n") {
		line := strings.TrimSpace(raw)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) < 3 {
			return nil, fmt.Errorf("%w: line %d has fewer than three fields", ErrInvalidHistory, lineNo+1)
		}
		tli64, err := strconv.ParseUint(fields[0], 10, 32)
		if err != nil || tli64 == 0 {
			return nil, fmt.Errorf("%w: invalid parent TLI on line %d", ErrInvalidHistory, lineNo+1)
		}
		tli := uint32(tli64)
		if tli >= targetTLI || (seen && tli <= previousTLI) {
			return nil, fmt.Errorf("%w: parent timeline IDs must increase and precede target TLI", ErrInvalidHistory)
		}
		if !seen && tli != 1 {
			return nil, fmt.Errorf("%w: history must include root timeline 1", ErrInvalidHistory)
		}
		switchPoint, err := ParseLSN(fields[1])
		if err != nil {
			return nil, fmt.Errorf("%w: invalid switchpoint on line %d", ErrInvalidHistory, lineNo+1)
		}
		// Adjacent switchpoints bound one and the same parent timeline.
		// Do not compare LSNs from unrelated timeline IDs.
		if switchPoint < previousSwitch {
			return nil, fmt.Errorf("%w: switchpoints move backwards within the parent lineage", ErrInvalidHistory)
		}
		segments = append(segments, segment{
			TimelineID: tli,
			Start: previousSwitch,
			End: switchPoint,
			HasEnd: true,
		})
		previousTLI = tli
		previousSwitch = switchPoint
		seen = true
	}
	if !seen {
		return nil, fmt.Errorf("%w: no parent timeline entries", ErrInvalidHistory)
	}
	segments = append(segments, segment{TimelineID: targetTLI, Start: previousSwitch})
	return segments, nil
}

func intervalFor(s segment) *Interval {
	value := &Interval{TimelineID: s.TimelineID, StartLSN: FormatLSN(s.Start)}
	if s.HasEnd {
		value.EndLSN = FormatLSN(s.End)
	}
	return value
}

func Evaluate(in Input) Result {
	result := Result{
		Verdict: Qualified,
		ReasonCodes: []string{},
		BaseBackupLSN: in.BaseBackupLSN,
		RecoveryTargetLSN: in.RecoveryTargetLSN,
		BaseBackupTimelineID: in.BaseBackupTimelineID,
		RecoveredTimelineID: in.RecoveredTimelineID,
	}
	reasons := map[string]struct{}{}
	blocked, unknown := false, false
	addBlocked := func(code string) { blocked = true; reasons[code] = struct{}{} }
	addUnknown := func(code string) { unknown = true; reasons[code] = struct{}{} }

	if in.Profile != Profile || in.Version != Version {
		addBlocked("unsupported_profile_or_version")
	}
	parseSystemIdentifier := func(value string) (uint64, bool) {
		if value == "" {
			addUnknown("system_identifier_missing")
			return 0, false
		}
		parsed, err := strconv.ParseUint(value, 10, 64)
		if err != nil || parsed == 0 || strconv.FormatUint(parsed, 10) != value {
			addBlocked("system_identifier_invalid")
			return 0, false
		}
		return parsed, true
	}
	expectedSystemID, expectedOK := parseSystemIdentifier(in.ExpectedSystemIdentifier)
	baseSystemID, baseOK := parseSystemIdentifier(in.BaseBackupSystemIdentifier)
	recoveredSystemID, recoveredOK := parseSystemIdentifier(in.RecoveredSystemIdentifier)
	if expectedOK && baseOK && recoveredOK &&
		(expectedSystemID != baseSystemID || expectedSystemID != recoveredSystemID) {
		addBlocked("system_identifier_mismatch")
	}
	if in.BaseBackupTimelineID == 0 || in.RecoveredTimelineID == 0 {
		addBlocked("timeline_id_zero")
	}
	if in.HistoryTimelineID != in.RecoveredTimelineID {
		addBlocked("history_file_timeline_mismatch")
	}

	baseLSN, baseErr := ParseLSN(in.BaseBackupLSN)
	targetLSN, targetErr := ParseLSN(in.RecoveryTargetLSN)
	if baseErr != nil { addBlocked("base_backup_lsn_invalid") }
	if targetErr != nil { addBlocked("recovery_target_lsn_invalid") }

	if in.RecoveredTimelineID > 1 {
		if in.HistoryText == "" || in.HistorySHA256 == "" {
			addUnknown("target_timeline_history_missing")
		} else {
			computed := HistorySHA256(in.HistoryText)
			result.ComputedHistorySHA256 = computed
			if !strings.EqualFold(computed, in.HistorySHA256) {
				addBlocked("timeline_history_digest_mismatch")
			}
		}
	} else {
		if in.HistoryText != "" || in.HistorySHA256 != "" {
			addBlocked("timeline_1_has_unexpected_history")
		}
		result.ComputedHistorySHA256 = HistorySHA256("")
	}

	segments, historyErr := ParseHistory(in.RecoveredTimelineID, in.HistoryText)
	if historyErr != nil {
		if in.RecoveredTimelineID > 1 && in.HistoryText == "" {
			addUnknown("target_timeline_history_unavailable")
		} else {
			addBlocked("timeline_history_invalid")
		}
	}

	if historyErr == nil && baseErr == nil && targetErr == nil {
		var sourceSegment *segment
		targetSegment := segments[len(segments)-1]
		for i := range segments {
			if segments[i].TimelineID == in.BaseBackupTimelineID {
				copy := segments[i]
				sourceSegment = &copy
				break
			}
		}
		result.TargetTimelineInterval = intervalFor(targetSegment)
		if sourceSegment == nil {
			addBlocked("base_backup_timeline_not_in_target_ancestry")
		} else {
			result.SourceTimelineInterval = intervalFor(*sourceSegment)
			if baseLSN < sourceSegment.Start {
				addBlocked("base_backup_lsn_before_timeline_start")
			} else if baseLSN == sourceSegment.Start {
				addUnknown("base_backup_lsn_at_timeline_start_boundary")
			} else if sourceSegment.HasEnd && baseLSN >= sourceSegment.End {
				addBlocked("base_backup_lsn_outside_source_timeline_interval")
			}
			if in.BaseBackupTimelineID == in.RecoveredTimelineID {
				if targetLSN < baseLSN {
					addBlocked("recovery_target_precedes_base_backup_on_same_timeline")
				}
			} else {
				if targetLSN < targetSegment.Start {
					addBlocked("recovery_target_precedes_target_timeline_fork")
				} else if targetLSN == targetSegment.Start {
					addUnknown("recovery_target_exactly_at_timeline_fork_boundary")
				}
			}
		}
	}
	switch {
	case blocked:
		result.Verdict = Blocked
	case unknown:
		result.Verdict = Indeterminate
	default:
		result.Verdict = Qualified
	}
	for reason := range reasons { result.ReasonCodes = append(result.ReasonCodes, reason) }
	sort.Strings(result.ReasonCodes)
	return result
}
