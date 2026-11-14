package effectbridge

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"time"
)

type DispatchStatus string

const (
	DispatchCompleted DispatchStatus = "completed"
	DispatchRejected  DispatchStatus = "rejected"
	DispatchUnknown   DispatchStatus = "unknown"
)

type DispatchResult struct {
	Status  DispatchStatus `json:"status"`
	Receipt *Receipt       `json:"receipt,omitempty"`
	Error   string         `json:"error,omitempty"`
}

type ReconcileStatus string

const (
	ReconcileCompleted     ReconcileStatus = "completed"
	ReconcileNotExecuted   ReconcileStatus = "not_executed_in_complete_service_journal"
	ReconcileIndeterminate ReconcileStatus = "indeterminate"
)

type ReconcileResult struct {
	Status   ReconcileStatus `json:"status"`
	Evidence Reconciliation  `json:"evidence"`
	Reason   string          `json:"reason,omitempty"`
}

type Client struct {
	baseURL            string
	http               *http.Client
	injectFenceAckLoss bool
}

func NewClient(baseURL string, timeout time.Duration) (*Client, error) {
	parsed, err := url.Parse(baseURL)
	if err != nil || parsed.Scheme != "http" || parsed.Host == "" || strings.Contains(parsed.Host, " ") {
		return nil, errors.New("baseURL must be an absolute http URL")
	}
	if timeout <= 0 {
		return nil, errors.New("timeout must be positive")
	}
	return &Client{baseURL: strings.TrimRight(baseURL, "/"), http: &http.Client{Timeout: timeout}}, nil
}

// WithFenceAckLossForTest enables the destination's deterministic fault hook.
// It is intentionally not an environment or production default.
func (c *Client) WithFenceAckLossForTest(enabled bool) { c.injectFenceAckLoss = enabled }

func (c *Client) Dispatch(ctx context.Context, req Request, injectDropAfterCommit bool) DispatchResult {
	if _, err := IntentDigest(req); err != nil {
		return DispatchResult{Status: DispatchRejected, Error: err.Error()}
	}
	body, _ := json.Marshal(req)
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/v1/effects", bytes.NewReader(body))
	if err != nil {
		return DispatchResult{Status: DispatchRejected, Error: err.Error()}
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if injectDropAfterCommit {
		httpReq.Header.Set("X-MACE-Test-Drop-Ack", "after-commit")
	}
	resp, err := c.http.Do(httpReq)
	if err != nil {
		// A transport error cannot distinguish pre-commit failure from remote success.
		return DispatchResult{Status: DispatchUnknown, Error: err.Error()}
	}
	defer resp.Body.Close()
	if resp.StatusCode == http.StatusConflict || resp.StatusCode == http.StatusBadRequest || resp.StatusCode == http.StatusForbidden {
		return DispatchResult{Status: DispatchRejected, Error: fmt.Sprintf("destination rejected dispatch: HTTP %d", resp.StatusCode)}
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return DispatchResult{Status: DispatchUnknown, Error: fmt.Sprintf("destination response is not a final effect result: HTTP %d", resp.StatusCode)}
	}
	var receipt Receipt
	if err := json.NewDecoder(resp.Body).Decode(&receipt); err != nil {
		return DispatchResult{Status: DispatchUnknown, Error: "successful HTTP response lacked a verifiable receipt: " + err.Error()}
	}
	intent, err := IntentDigest(req)
	if err != nil || VerifyReceiptForRequest(receipt, req) != nil || receipt.RecordKind != "effect_committed" || receipt.IntentSHA256 != intent {
		return DispatchResult{Status: DispatchUnknown, Error: "destination receipt does not verify and bind exact request intent"}
	}
	return DispatchResult{Status: DispatchCompleted, Receipt: &receipt}
}

func (c *Client) Reconcile(ctx context.Context, req Request) ReconcileResult {
	intent, err := IntentDigest(req)
	if err != nil {
		return ReconcileResult{Status: ReconcileIndeterminate, Reason: err.Error()}
	}
	endpoint := c.baseURL + "/v1/reconcile/" + url.PathEscape(req.IdempotencyKey)
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return ReconcileResult{Status: ReconcileIndeterminate, Reason: err.Error()}
	}
	resp, err := c.http.Do(httpReq)
	if err != nil {
		return ReconcileResult{Status: ReconcileIndeterminate, Reason: "authoritative reconciliation unavailable: " + err.Error()}
	}
	defer resp.Body.Close()
	var evidence Reconciliation
	if err := json.NewDecoder(resp.Body).Decode(&evidence); err != nil {
		return ReconcileResult{Status: ReconcileIndeterminate, Reason: "invalid reconciliation response: " + err.Error()}
	}
	if resp.StatusCode != http.StatusOK || !evidence.Journal.Complete {
		return ReconcileResult{Status: ReconcileIndeterminate, Evidence: evidence, Reason: "service journal is not currently authoritative/complete"}
	}
	switch evidence.Status {
	case "found":
		if evidence.Receipt == nil || VerifyReceiptForRequest(*evidence.Receipt, req) != nil || evidence.Receipt.IntentSHA256 != intent || evidence.Receipt.RecordKind != "effect_committed" {
			return ReconcileResult{Status: ReconcileIndeterminate, Evidence: evidence, Reason: "service receipt conflicts with exact request identity"}
		}
		return ReconcileResult{Status: ReconcileCompleted, Evidence: evidence}
	case "not_executed_fenced":
		if evidence.Receipt == nil || VerifyReceiptForRequest(*evidence.Receipt, req) != nil || evidence.Receipt.IntentSHA256 != intent || evidence.Receipt.RecordKind != "fence_committed" {
			return ReconcileResult{Status: ReconcileIndeterminate, Evidence: evidence, Reason: "fence receipt conflicts with exact request identity"}
		}
		return ReconcileResult{Status: ReconcileNotExecuted, Evidence: evidence, Reason: "the destination durably fenced this exact intent; later arrivals with this idempotency key are rejected"}
	case "not_found_unfenced":
		return ReconcileResult{Status: ReconcileIndeterminate, Evidence: evidence, Reason: "absence is not proof of non-execution while an earlier request may still be in flight; obtain a destination-side atomic fence"}
	default:
		return ReconcileResult{Status: ReconcileIndeterminate, Evidence: evidence, Reason: "unknown service reconciliation state"}
	}
}

// FenceAndReconcile asks the destination to linearize this exact intent against
// any delayed dispatch. The destination either returns the already-committed
// effect or durably records a tombstone that rejects all later arrivals.
// Repeating this request is idempotent. A transport error remains indeterminate.
func (c *Client) FenceAndReconcile(ctx context.Context, req Request) ReconcileResult {
	intent, err := IntentDigest(req)
	if err != nil {
		return ReconcileResult{Status: ReconcileIndeterminate, Reason: err.Error()}
	}
	body, _ := json.Marshal(req)
	endpoint := c.baseURL + "/v1/fence/" + url.PathEscape(req.IdempotencyKey)
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPut, endpoint, bytes.NewReader(body))
	if err != nil {
		return ReconcileResult{Status: ReconcileIndeterminate, Reason: err.Error()}
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if c.injectFenceAckLoss {
		httpReq.Header.Set("X-MACE-Test-Drop-Fence-Ack", "after-commit")
	}
	resp, err := c.http.Do(httpReq)
	if err != nil {
		return ReconcileResult{Status: ReconcileIndeterminate, Reason: "destination fence acknowledgement unknown: " + err.Error()}
	}
	defer resp.Body.Close()
	var evidence Reconciliation
	if err := json.NewDecoder(resp.Body).Decode(&evidence); err != nil {
		return ReconcileResult{Status: ReconcileIndeterminate, Reason: "invalid fence/reconciliation response: " + err.Error()}
	}
	if (resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusCreated) || !evidence.Journal.Complete || evidence.IdempotencyKey != req.IdempotencyKey {
		return ReconcileResult{Status: ReconcileIndeterminate, Evidence: evidence, Reason: "destination did not establish a complete exact-intent fence"}
	}
	switch evidence.Status {
	case "found":
		if evidence.Receipt == nil || VerifyReceiptForRequest(*evidence.Receipt, req) != nil || evidence.Receipt.RecordKind != "effect_committed" || evidence.Receipt.IntentSHA256 != intent {
			return ReconcileResult{Status: ReconcileIndeterminate, Evidence: evidence, Reason: "existing effect does not bind the requested intent"}
		}
		return ReconcileResult{Status: ReconcileCompleted, Evidence: evidence}
	case "not_executed_fenced":
		if evidence.Receipt == nil || VerifyReceiptForRequest(*evidence.Receipt, req) != nil || evidence.Receipt.RecordKind != "fence_committed" || evidence.Receipt.IntentSHA256 != intent {
			return ReconcileResult{Status: ReconcileIndeterminate, Evidence: evidence, Reason: "destination fence does not bind the requested intent"}
		}
		return ReconcileResult{Status: ReconcileNotExecuted, Evidence: evidence, Reason: "destination durably fenced this exact intent against delayed execution"}
	default:
		return ReconcileResult{Status: ReconcileIndeterminate, Evidence: evidence, Reason: "destination returned no terminal effect-or-fence state"}
	}
}
