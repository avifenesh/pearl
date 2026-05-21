package jsonrpccache

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/caddyserver/caddy/v2"
	"github.com/caddyserver/caddy/v2/caddyconfig/caddyfile"
	"github.com/caddyserver/caddy/v2/modules/caddyhttp"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.uber.org/zap"
)

func newTestHandler(rules ...CacheRule) *Handler {
	return &Handler{
		Rules:  rules,
		logger: zap.NewNop(),
		cache: &cache{
			entries: make(map[string]*cacheEntry),
		},
	}
}

type staticBackend struct {
	calls atomic.Int64
}

func (b *staticBackend) ServeHTTP(w http.ResponseWriter, r *http.Request) error {
	b.calls.Add(1)
	body, _ := io.ReadAll(r.Body)
	var req jsonrpcRequest
	json.Unmarshal(body, &req)

	resp := jsonrpcResponse{
		JSONRPC: "2.0",
		ID:      req.ID,
		Result:  json.RawMessage(`{"capabilities":{}}`),
	}
	out, _ := json.Marshal(resp)
	w.Header().Set("Content-Type", "application/json")
	w.Write(out)
	return nil
}

func makeJSONRPCBody(method string, id any) []byte {
	body, _ := json.Marshal(map[string]any{
		"jsonrpc": "1.0",
		"method":  method,
		"params":  []any{},
		"id":      id,
	})
	return body
}

func doRequest(t *testing.T, h *Handler, next caddyhttp.Handler, method string, id any) *httptest.ResponseRecorder {
	t.Helper()
	body := makeJSONRPCBody(method, id)
	req := httptest.NewRequest(http.MethodPost, "/", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	err := h.ServeHTTP(w, req, next)
	require.NoError(t, err)
	return w
}

func TestCacheMissThenHit(t *testing.T) {
	backend := &staticBackend{}
	h := newTestHandler(CacheRule{Method: "getblocktemplate", TTL: caddy.Duration(5 * time.Second)})

	w1 := doRequest(t, h, backend, "getblocktemplate", 1)
	assert.Equal(t, http.StatusOK, w1.Code)
	assert.Equal(t, int64(1), backend.calls.Load(), "first request should reach backend")
	assert.Empty(t, w1.Header().Get("X-Jsonrpc-Cache"), "first request should be a MISS")

	w2 := doRequest(t, h, backend, "getblocktemplate", 2)
	assert.Equal(t, http.StatusOK, w2.Code)
	assert.Equal(t, int64(1), backend.calls.Load(), "second request should be served from cache")
	assert.Equal(t, "HIT", w2.Header().Get("X-Jsonrpc-Cache"))
}

func TestTTLExpiry(t *testing.T) {
	backend := &staticBackend{}
	h := newTestHandler(CacheRule{Method: "getblocktemplate", TTL: caddy.Duration(50 * time.Millisecond)})

	doRequest(t, h, backend, "getblocktemplate", 1)
	assert.Equal(t, int64(1), backend.calls.Load())

	time.Sleep(80 * time.Millisecond)

	doRequest(t, h, backend, "getblocktemplate", 2)
	assert.Equal(t, int64(2), backend.calls.Load(), "expired cache should forward to backend")
}

func TestIDRewriting(t *testing.T) {
	backend := &staticBackend{}
	h := newTestHandler(CacheRule{Method: "getblocktemplate", TTL: caddy.Duration(5 * time.Second)})

	doRequest(t, h, backend, "getblocktemplate", 100)

	w := doRequest(t, h, backend, "getblocktemplate", 999)
	var resp jsonrpcResponse
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &resp))

	var gotID int
	require.NoError(t, json.Unmarshal(resp.ID, &gotID))
	assert.Equal(t, 999, gotID, "cached response should have the caller's request id")
}

func TestNonCachedMethodsPassthrough(t *testing.T) {
	backend := &staticBackend{}
	h := newTestHandler(CacheRule{Method: "getblocktemplate", TTL: caddy.Duration(5 * time.Second)})

	methods := []string{"getbestblockhash", "sendrawtransaction", "getmempoolinfo"}
	for _, m := range methods {
		doRequest(t, h, backend, m, 1)
	}
	assert.Equal(t, int64(3), backend.calls.Load(), "non-cached methods should always reach backend")
}

func TestSingleFlightCoalescing(t *testing.T) {
	var backendCalls atomic.Int64
	slowBackend := caddyhttp.HandlerFunc(func(w http.ResponseWriter, r *http.Request) error {
		backendCalls.Add(1)
		time.Sleep(100 * time.Millisecond)
		body, _ := io.ReadAll(r.Body)
		var req jsonrpcRequest
		json.Unmarshal(body, &req)
		resp := jsonrpcResponse{
			JSONRPC: "2.0",
			ID:      req.ID,
			Result:  json.RawMessage(`{"capabilities":{}}`),
		}
		out, _ := json.Marshal(resp)
		w.Write(out)
		return nil
	})

	h := newTestHandler(CacheRule{Method: "getblocktemplate", TTL: caddy.Duration(5 * time.Second)})

	var wg sync.WaitGroup
	for i := range 10 {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			doRequest(t, h, slowBackend, "getblocktemplate", id)
		}(i)
	}
	wg.Wait()

	assert.Equal(t, int64(1), backendCalls.Load(), "10 concurrent requests should coalesce into 1 backend call")
}

func TestMalformedRequestsPassthrough(t *testing.T) {
	var backendCalls atomic.Int64
	backend := caddyhttp.HandlerFunc(func(w http.ResponseWriter, r *http.Request) error {
		backendCalls.Add(1)
		w.WriteHeader(http.StatusOK)
		return nil
	})

	h := newTestHandler(CacheRule{Method: "getblocktemplate", TTL: caddy.Duration(5 * time.Second)})

	tests := []struct {
		name string
		body string
	}{
		{"not json", "this is not json"},
		{"missing method", `{"jsonrpc":"1.0","id":1}`},
		{"empty body", ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			backendCalls.Store(0)
			req := httptest.NewRequest(http.MethodPost, "/", bytes.NewReader([]byte(tt.body)))
			w := httptest.NewRecorder()
			err := h.ServeHTTP(w, req, backend)
			require.NoError(t, err)
			assert.Equal(t, int64(1), backendCalls.Load(), "malformed request should pass through to backend")
		})
	}
}

// TestStuckUpstreamReleasesCoalescedWaiters asserts that a slow or unresponsive
// upstream does not block every concurrent caller for the full duration of the
// primary in-flight request. Concurrent callers for the same JSON-RPC method
// share a singleflight in-flight call so the upstream is not hammered, but
// each caller must still be able to abort independently when its own request
// context is canceled (for example, when an HTTP client times out or
// disconnects).
//
// The invariants under test:
//
//   - Concurrent callers for the same cached method are coalesced into a single
//     upstream request.
//   - A coalesced caller whose own request context is canceled must return
//     promptly, regardless of whether the primary upstream call has completed.
//   - Once the primary upstream call eventually completes (or is otherwise
//     released), the cache state must not be wedged: a subsequent request
//     against a healthy backend must succeed.
func TestStuckUpstreamReleasesCoalescedWaiters(t *testing.T) {
	const (
		clientCancelTimeout = 200 * time.Millisecond
		callerLatencyBudget = 600 * time.Millisecond
		coalescedCallers    = 50
	)

	var backendCalls atomic.Int64
	upstreamEntered := make(chan struct{}, 1)
	upstreamUnblock := make(chan struct{})

	stuckBackend := caddyhttp.HandlerFunc(func(w http.ResponseWriter, r *http.Request) error {
		backendCalls.Add(1)
		select {
		case upstreamEntered <- struct{}{}:
		default:
		}
		select {
		case <-upstreamUnblock:
		case <-r.Context().Done():
			return r.Context().Err()
		}
		body, _ := io.ReadAll(r.Body)
		var req jsonrpcRequest
		_ = json.Unmarshal(body, &req)
		resp := jsonrpcResponse{
			JSONRPC: "2.0",
			ID:      req.ID,
			Result:  json.RawMessage(`{"capabilities":{}}`),
		}
		out, _ := json.Marshal(resp)
		_, _ = w.Write(out)
		return nil
	})

	h := newTestHandler(CacheRule{Method: "getblocktemplate", TTL: caddy.Duration(5 * time.Second)})

	// Primary caller holds the singleflight in-flight call open until we close
	// upstreamUnblock. Its context never expires inside this test, so any
	// timing observed on the coalesced callers must come from their own
	// context cancellation, not from the primary finishing.
	primaryDone := make(chan struct{})
	go func() {
		defer close(primaryDone)
		body := makeJSONRPCBody("getblocktemplate", 0)
		req := httptest.NewRequest(http.MethodPost, "/", bytes.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()
		_ = h.ServeHTTP(w, req, stuckBackend)
	}()

	// Wait until the upstream call has actually entered the stuck backend so
	// the next batch of callers genuinely coalesce on it.
	select {
	case <-upstreamEntered:
	case <-time.After(time.Second):
		t.Fatal("primary upstream call never entered stuck backend")
	}

	type callerResult struct {
		duration time.Duration
		err      error
	}
	results := make(chan callerResult, coalescedCallers)
	for i := 1; i <= coalescedCallers; i++ {
		go func(id int) {
			ctx, cancel := context.WithTimeout(context.Background(), clientCancelTimeout)
			defer cancel()
			body := makeJSONRPCBody("getblocktemplate", id)
			req := httptest.NewRequest(http.MethodPost, "/", bytes.NewReader(body)).WithContext(ctx)
			req.Header.Set("Content-Type", "application/json")
			w := httptest.NewRecorder()

			start := time.Now()
			err := h.ServeHTTP(w, req, stuckBackend)
			results <- callerResult{duration: time.Since(start), err: err}
		}(i)
	}

	maxObserved := time.Duration(0)
	for i := 0; i < coalescedCallers; i++ {
		select {
		case r := <-results:
			if r.duration > maxObserved {
				maxObserved = r.duration
			}
			if r.duration > callerLatencyBudget {
				t.Errorf("coalesced caller waited %s (budget %s); singleflight is parking waiters past client cancellation",
					r.duration, callerLatencyBudget)
			}
		case <-time.After(2 * time.Second):
			t.Fatalf("a coalesced caller did not return within 2s; singleflight is blocking concurrent callers on a stuck upstream (max observed so far %s)", maxObserved)
		}
	}

	// The cache must still coalesce these into one upstream call. The primary
	// is in flight; coalesced callers should not have triggered additional
	// upstream attempts.
	assert.Equal(t, int64(1), backendCalls.Load(),
		"singleflight must still coalesce concurrent callers into a single upstream call")

	// Release the primary so the goroutine exits cleanly and the cache state
	// is no longer wedged on the abandoned miss.
	close(upstreamUnblock)
	select {
	case <-primaryDone:
	case <-time.After(time.Second):
		t.Fatal("primary caller never returned after upstream was unblocked")
	}

	// Recovery: a fresh caller against a healthy backend must succeed. This
	// guards against the cache state remaining wedged on a forgotten
	// in-flight key after a timed-out miss.
	healthyBackend := &staticBackend{}
	w := doRequest(t, h, healthyBackend, "getblocktemplate", 9999)
	require.Equal(t, http.StatusOK, w.Code,
		"after the stuck miss is abandoned, the next request must succeed against a healthy backend")
}

// TestUpstreamRequestPreservesCallerContextValues asserts that the request
// handed to the next handler on a cache miss carries the same context values
// as the inbound request. Caddy stashes per-request state (most importantly
// *caddy.Replacer) on the request context and downstream handlers such as
// reverse_proxy retrieve it via type assertion; a request whose context lacks
// those values will panic the upstream chain and crash the server.
//
// The coalesced upstream call runs in a separate goroutine spawned by
// singleflight, so the upstream context cannot simply be the caller's
// context. It must inherit the caller's context VALUES while detaching
// from its cancellation, since the in-flight call is shared across many
// callers and must outlive any single caller.
func TestUpstreamRequestPreservesCallerContextValues(t *testing.T) {
	type ctxKey struct{}

	const sentinelValue = "v1"

	var (
		seenValue   string
		seenOK      bool
		callerKey   = ctxKey{}
		nextEntered = make(chan struct{}, 1)
	)

	probeBackend := caddyhttp.HandlerFunc(func(w http.ResponseWriter, r *http.Request) error {
		v, ok := r.Context().Value(callerKey).(string)
		seenValue, seenOK = v, ok
		select {
		case nextEntered <- struct{}{}:
		default:
		}
		body, _ := io.ReadAll(r.Body)
		var req jsonrpcRequest
		_ = json.Unmarshal(body, &req)
		resp := jsonrpcResponse{
			JSONRPC: "2.0",
			ID:      req.ID,
			Result:  json.RawMessage(`{"capabilities":{}}`),
		}
		out, _ := json.Marshal(resp)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(out)
		return nil
	})

	h := newTestHandler(CacheRule{Method: "getblocktemplate", TTL: caddy.Duration(5 * time.Second)})

	body := makeJSONRPCBody("getblocktemplate", 1)
	ctx := context.WithValue(context.Background(), callerKey, sentinelValue)
	req := httptest.NewRequest(http.MethodPost, "/", bytes.NewReader(body)).WithContext(ctx)
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	require.NoError(t, h.ServeHTTP(w, req, probeBackend),
		"a request whose next handler reads a context value must not error")

	select {
	case <-nextEntered:
	case <-time.After(time.Second):
		t.Fatal("next handler was never invoked")
	}

	assert.True(t, seenOK, "next handler must see caller-side context values from singleflight goroutine")
	assert.Equal(t, sentinelValue, seenValue, "next handler must observe the value the caller stored on its context")
	assert.Equal(t, http.StatusOK, w.Code, "successful upstream call must surface a 200 to the caller")
}

// TestUpstreamRequestSurvivesCallerCancellation asserts that the next handler
// is invoked with a context that is NOT canceled even after the caller's
// request context has been canceled. The shared upstream call must outlive
// any single caller's lifecycle so coalesced waiters and the cache entry are
// produced from a clean run, not aborted because one caller hung up.
func TestUpstreamRequestSurvivesCallerCancellation(t *testing.T) {
	upstreamCtxErr := make(chan error, 1)

	probeBackend := caddyhttp.HandlerFunc(func(w http.ResponseWriter, r *http.Request) error {
		// Give the caller time to cancel its own context. If the upstream
		// context were chained to the caller's it would be canceled here.
		time.Sleep(50 * time.Millisecond)
		upstreamCtxErr <- r.Context().Err()
		body, _ := io.ReadAll(r.Body)
		var req jsonrpcRequest
		_ = json.Unmarshal(body, &req)
		resp := jsonrpcResponse{
			JSONRPC: "2.0",
			ID:      req.ID,
			Result:  json.RawMessage(`{"capabilities":{}}`),
		}
		out, _ := json.Marshal(resp)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(out)
		return nil
	})

	h := newTestHandler(CacheRule{Method: "getblocktemplate", TTL: caddy.Duration(5 * time.Second)})

	callerCtx, cancelCaller := context.WithCancel(context.Background())
	body := makeJSONRPCBody("getblocktemplate", 1)
	req := httptest.NewRequest(http.MethodPost, "/", bytes.NewReader(body)).WithContext(callerCtx)
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	done := make(chan struct{})
	go func() {
		_ = h.ServeHTTP(w, req, probeBackend)
		close(done)
	}()

	// Cancel the caller while the upstream is in flight. The shared
	// upstream call must not see this cancellation.
	time.Sleep(10 * time.Millisecond)
	cancelCaller()

	select {
	case err := <-upstreamCtxErr:
		assert.NoError(t, err, "upstream context must not propagate the caller's cancellation")
	case <-time.After(time.Second):
		t.Fatal("upstream backend never observed its context state")
	}

	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("caller goroutine never returned after cancellation")
	}
}

func TestGETRequestsPassthrough(t *testing.T) {
	var backendCalls atomic.Int64
	backend := caddyhttp.HandlerFunc(func(w http.ResponseWriter, r *http.Request) error {
		backendCalls.Add(1)
		w.WriteHeader(http.StatusOK)
		return nil
	})

	h := newTestHandler(CacheRule{Method: "getblocktemplate", TTL: caddy.Duration(5 * time.Second)})

	req := httptest.NewRequest(http.MethodGet, "/", nil)
	w := httptest.NewRecorder()
	err := h.ServeHTTP(w, req, backend)
	require.NoError(t, err)
	assert.Equal(t, int64(1), backendCalls.Load(), "GET requests should pass through")
}

// decodeRPCError extracts the JSON-RPC error code and message from a response
// body. It returns the empty values if the body does not look like a JSON-RPC
// error response.
func decodeRPCError(t *testing.T, body []byte) (int, string) {
	t.Helper()
	var resp jsonrpcResponse
	require.NoError(t, json.Unmarshal(body, &resp))
	if len(resp.Error) == 0 {
		return 0, ""
	}
	var rpcErr struct {
		Code    int    `json:"code"`
		Message string `json:"message"`
	}
	require.NoError(t, json.Unmarshal(resp.Error, &rpcErr))
	return rpcErr.Code, rpcErr.Message
}

// TestMissTimeoutReturnsTimeoutEnvelope asserts that when an upstream call
// exceeds the configured miss budget the caller receives an HTTP 504 with a
// well-formed JSON-RPC error envelope whose id field matches the request id.
func TestMissTimeoutReturnsTimeoutEnvelope(t *testing.T) {
	const (
		missTimeout = 80 * time.Millisecond
		requestID   = 4242
	)

	release := make(chan struct{})
	defer close(release)

	stuckBackend := caddyhttp.HandlerFunc(func(w http.ResponseWriter, r *http.Request) error {
		select {
		case <-release:
			return nil
		case <-r.Context().Done():
			return r.Context().Err()
		}
	})

	h := newTestHandler(CacheRule{Method: "getblocktemplate", TTL: caddy.Duration(5 * time.Second)})
	h.MissTimeout = caddy.Duration(missTimeout)

	body := makeJSONRPCBody("getblocktemplate", requestID)
	req := httptest.NewRequest(http.MethodPost, "/", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	start := time.Now()
	require.NoError(t, h.ServeHTTP(w, req, stuckBackend))
	elapsed := time.Since(start)

	assert.Equal(t, http.StatusGatewayTimeout, w.Code)
	assert.Equal(t, "application/json", w.Header().Get("Content-Type"))
	assert.LessOrEqual(t, elapsed, missTimeout+500*time.Millisecond,
		"timeout should fire within the configured budget plus reasonable slack")

	var resp jsonrpcResponse
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &resp))
	assert.Equal(t, "2.0", resp.JSONRPC)

	var gotID int
	require.NoError(t, json.Unmarshal(resp.ID, &gotID))
	assert.Equal(t, requestID, gotID, "timeout response must preserve the request id")

	code, msg := decodeRPCError(t, w.Body.Bytes())
	assert.Equal(t, -32603, code, "timeout response should use the JSON-RPC internal error code")
	assert.Contains(t, msg, "timeout")
}

// TestUpstreamErrorNormalizesToTimeoutEnvelope asserts that when the upstream
// itself returns context.DeadlineExceeded (or context.Canceled), the result is
// surfaced as the same HTTP 504 JSON-RPC envelope as a wait-side timeout.
// This keeps the externally visible behavior deterministic regardless of
// which timer wins the internal race.
func TestUpstreamErrorNormalizesToTimeoutEnvelope(t *testing.T) {
	const requestID = 7

	backend := caddyhttp.HandlerFunc(func(w http.ResponseWriter, r *http.Request) error {
		return context.DeadlineExceeded
	})

	h := newTestHandler(CacheRule{Method: "getblocktemplate", TTL: caddy.Duration(5 * time.Second)})
	// Long miss timeout so the wait-side timer cannot fire first.
	h.MissTimeout = caddy.Duration(5 * time.Second)

	body := makeJSONRPCBody("getblocktemplate", requestID)
	req := httptest.NewRequest(http.MethodPost, "/", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	require.NoError(t, h.ServeHTTP(w, req, backend))

	assert.Equal(t, http.StatusGatewayTimeout, w.Code)

	var resp jsonrpcResponse
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &resp))
	var gotID int
	require.NoError(t, json.Unmarshal(resp.ID, &gotID))
	assert.Equal(t, requestID, gotID)

	code, _ := decodeRPCError(t, w.Body.Bytes())
	assert.Equal(t, -32603, code)
}

// TestForgetEnablesFreshUpstreamAfterTimeout asserts that once a stalled
// in-flight call has timed out, a subsequent caller drives a fresh upstream
// attempt against the new backend instead of joining the abandoned call or
// receiving a stale cached entry.
func TestForgetEnablesFreshUpstreamAfterTimeout(t *testing.T) {
	const missTimeout = 80 * time.Millisecond

	stuckBackend := caddyhttp.HandlerFunc(func(w http.ResponseWriter, r *http.Request) error {
		<-r.Context().Done()
		return r.Context().Err()
	})

	h := newTestHandler(CacheRule{Method: "getblocktemplate", TTL: caddy.Duration(5 * time.Second)})
	h.MissTimeout = caddy.Duration(missTimeout)

	w1 := doRequest(t, h, stuckBackend, "getblocktemplate", 1)
	require.Equal(t, http.StatusGatewayTimeout, w1.Code, "first call should time out")

	// Give the abandoned upstream goroutine a moment to exit so subsequent
	// tests don't race on the singleflight bookkeeping.
	time.Sleep(missTimeout + 50*time.Millisecond)

	healthy := &staticBackend{}
	w2 := doRequest(t, h, healthy, "getblocktemplate", 2)
	assert.Equal(t, http.StatusOK, w2.Code)
	assert.Equal(t, int64(1), healthy.calls.Load(),
		"healthy backend must be called after the stalled call is forgotten")
}

// TestJSONRPCErrorResponseNotCached asserts that an HTTP 200 response carrying
// a JSON-RPC error field is surfaced to the caller but not cached, so a
// transient upstream failure does not get propagated to every coalesced caller
// for the duration of the TTL.
func TestJSONRPCErrorResponseNotCached(t *testing.T) {
	var backendCalls atomic.Int64
	errorBackend := caddyhttp.HandlerFunc(func(w http.ResponseWriter, r *http.Request) error {
		backendCalls.Add(1)
		body, _ := io.ReadAll(r.Body)
		var req jsonrpcRequest
		_ = json.Unmarshal(body, &req)
		resp := jsonrpcResponse{
			JSONRPC: "2.0",
			ID:      req.ID,
			Error:   json.RawMessage(`{"code":-32000,"message":"transient upstream failure"}`),
		}
		out, _ := json.Marshal(resp)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(out)
		return nil
	})

	h := newTestHandler(CacheRule{Method: "getblocktemplate", TTL: caddy.Duration(5 * time.Second)})

	w1 := doRequest(t, h, errorBackend, "getblocktemplate", 1)
	assert.Equal(t, http.StatusOK, w1.Code)
	code, _ := decodeRPCError(t, w1.Body.Bytes())
	assert.Equal(t, -32000, code, "RPC error from upstream must be surfaced to the caller")
	assert.Equal(t, int64(1), backendCalls.Load())

	w2 := doRequest(t, h, errorBackend, "getblocktemplate", 2)
	assert.Equal(t, http.StatusOK, w2.Code)
	assert.Empty(t, w2.Header().Get("X-Jsonrpc-Cache"), "RPC error response must not be served as a cache hit")
	assert.Equal(t, int64(2), backendCalls.Load(), "JSON-RPC error response must not be cached")
}

// TestCaddyfileParsesMissTimeout covers parser acceptance and rejection of
// the miss_timeout subdirective.
func TestCaddyfileParsesMissTimeout(t *testing.T) {
	t.Run("valid duration", func(t *testing.T) {
		h := &Handler{}
		input := `jsonrpc_cache {
			cache getblocktemplate 1s
			miss_timeout 2500ms
		}`
		require.NoError(t, h.UnmarshalCaddyfile(caddyfile.NewTestDispenser(input)))
		assert.Equal(t, caddy.Duration(2500*time.Millisecond), h.MissTimeout)
		require.Len(t, h.Rules, 1)
		assert.Equal(t, "getblocktemplate", h.Rules[0].Method)
	})

	t.Run("default when omitted", func(t *testing.T) {
		h := &Handler{}
		input := `jsonrpc_cache {
			cache getblocktemplate 1s
		}`
		require.NoError(t, h.UnmarshalCaddyfile(caddyfile.NewTestDispenser(input)))
		assert.EqualValues(t, 0, h.MissTimeout, "parser should not set MissTimeout when omitted")
		assert.Equal(t, defaultMissTimeout, h.missTimeout(), "handler should fall back to the default")
	})

	t.Run("rejects non-duration", func(t *testing.T) {
		h := &Handler{}
		input := `jsonrpc_cache {
			miss_timeout not-a-duration
		}`
		assert.Error(t, h.UnmarshalCaddyfile(caddyfile.NewTestDispenser(input)))
	})

	t.Run("rejects negative", func(t *testing.T) {
		h := &Handler{}
		input := `jsonrpc_cache {
			miss_timeout -1s
		}`
		assert.Error(t, h.UnmarshalCaddyfile(caddyfile.NewTestDispenser(input)))
	})
}

// TestValidateRejectsNegativeMissTimeout guards against an invalid handler
// configuration when MissTimeout is set programmatically (e.g., via JSON
// config) rather than through the Caddyfile parser.
func TestValidateRejectsNegativeMissTimeout(t *testing.T) {
	h := &Handler{
		Rules:       []CacheRule{{Method: "getblocktemplate", TTL: caddy.Duration(time.Second)}},
		MissTimeout: caddy.Duration(-1),
	}
	assert.Error(t, h.Validate())
}

// TestCaddyfileExamplesParse asserts the example Caddyfiles shipped alongside
// this plugin contain a jsonrpc_cache block whose syntax is accepted by the
// plugin's UnmarshalCaddyfile. This guards against the examples drifting from
// the parser as new directives are added.
func TestCaddyfileExamplesParse(t *testing.T) {
	examples := []string{
		filepath.Join("..", "Caddyfile"),
		filepath.Join("..", "Caddyfile.domain.example"),
	}

	for _, path := range examples {
		t.Run(filepath.Base(path), func(t *testing.T) {
			content, err := os.ReadFile(path)
			require.NoError(t, err)

			block := extractCaddyfileBlock(t, string(content), "jsonrpc_cache")

			h := &Handler{}
			require.NoError(t, h.UnmarshalCaddyfile(caddyfile.NewTestDispenser(block)))
			require.NotEmpty(t, h.Rules, "example should declare at least one cache rule")
		})
	}
}

// extractCaddyfileBlock returns the source span of the named top-level block,
// from `name {` through its matching closing brace, suitable for feeding to a
// caddyfile.Dispenser.
func extractCaddyfileBlock(t *testing.T, content, name string) string {
	t.Helper()
	start := strings.Index(content, name+" {")
	require.GreaterOrEqual(t, start, 0, "block %q not found", name)

	depth := 0
	for i := start; i < len(content); i++ {
		switch content[i] {
		case '{':
			depth++
		case '}':
			depth--
			if depth == 0 {
				return content[start : i+1]
			}
		}
	}
	t.Fatalf("unterminated block %q", name)
	return ""
}
