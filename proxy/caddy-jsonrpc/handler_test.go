package jsonrpc

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/caddyserver/caddy/v2"
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

func TestAllowlistPermitsCachedMethod(t *testing.T) {
	backend := &staticBackend{}
	h := newTestHandler(CacheRule{Method: "getblocktemplate", TTL: caddy.Duration(5 * time.Second)})
	h.Allow = []string{"getblocktemplate", "submitblock"}

	w := doRequest(t, h, backend, "getblocktemplate", 1)
	assert.Equal(t, http.StatusOK, w.Code)
	assert.Equal(t, int64(1), backend.calls.Load(), "allowed method should reach backend")
}

func TestAllowlistPermitsUncachedMethod(t *testing.T) {
	backend := &staticBackend{}
	h := newTestHandler(CacheRule{Method: "getblocktemplate", TTL: caddy.Duration(5 * time.Second)})
	h.Allow = []string{"getblocktemplate", "submitblock"}

	// submitblock is allowed but has no cache rule — it must pass through.
	w := doRequest(t, h, backend, "submitblock", 1)
	assert.Equal(t, http.StatusOK, w.Code)
	assert.Equal(t, int64(1), backend.calls.Load(), "allowed uncached method should reach backend")
}

func TestAllowlistRejectsUnlistedMethod(t *testing.T) {
	backend := &staticBackend{}
	h := newTestHandler(CacheRule{Method: "getblocktemplate", TTL: caddy.Duration(5 * time.Second)})
	h.Allow = []string{"getblocktemplate", "submitblock"}

	w := doRequest(t, h, backend, "getblockcount", 7)
	assert.Equal(t, http.StatusForbidden, w.Code, "disallowed method should be rejected")
	assert.Equal(t, int64(0), backend.calls.Load(), "disallowed method must not reach backend")

	var resp jsonrpcResponse
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &resp))
	assert.NotEmpty(t, resp.Error, "rejection should carry a JSON-RPC error")

	var gotID int
	require.NoError(t, json.Unmarshal(resp.ID, &gotID))
	assert.Equal(t, 7, gotID, "rejection should echo the caller's request id")
}

func TestAllowlistRejectsNonPOST(t *testing.T) {
	backend := &staticBackend{}
	h := newTestHandler(CacheRule{Method: "getblocktemplate", TTL: caddy.Duration(5 * time.Second)})
	h.Allow = []string{"getblocktemplate"}

	req := httptest.NewRequest(http.MethodGet, "/", nil)
	w := httptest.NewRecorder()
	require.NoError(t, h.ServeHTTP(w, req, backend))
	assert.Equal(t, http.StatusForbidden, w.Code)
	assert.Equal(t, int64(0), backend.calls.Load(), "non-POST must not reach backend when an allowlist is set")
}

func TestAllowlistRejectsUnparseableBody(t *testing.T) {
	backend := &staticBackend{}
	h := newTestHandler(CacheRule{Method: "getblocktemplate", TTL: caddy.Duration(5 * time.Second)})
	h.Allow = []string{"getblocktemplate"}

	req := httptest.NewRequest(http.MethodPost, "/", bytes.NewReader([]byte("this is not json")))
	w := httptest.NewRecorder()
	require.NoError(t, h.ServeHTTP(w, req, backend))
	assert.Equal(t, http.StatusForbidden, w.Code)
	assert.Equal(t, int64(0), backend.calls.Load(), "unparseable body must not reach backend when an allowlist is set")
}

func TestNoAllowlistPermitsEveryMethod(t *testing.T) {
	backend := &staticBackend{}
	h := newTestHandler(CacheRule{Method: "getblocktemplate", TTL: caddy.Duration(5 * time.Second)})
	// No Allow configured: methods outside the cache rules still pass through.

	w := doRequest(t, h, backend, "getblockcount", 1)
	assert.Equal(t, http.StatusOK, w.Code)
	assert.Equal(t, int64(1), backend.calls.Load(), "without an allowlist every method should reach backend")
}

func TestResolveAllowList(t *testing.T) {
	t.Setenv("TEST_ALLOWED_METHODS", "getblocktemplate submitblock")

	tests := []struct {
		name string
		in   []string
		want []string
	}{
		{"literal tokens", []string{"getblocktemplate", "submitblock"}, []string{"getblocktemplate", "submitblock"}},
		{"space-separated entry", []string{"getblocktemplate submitblock"}, []string{"getblocktemplate", "submitblock"}},
		{"comma-separated entry", []string{"getblocktemplate,submitblock"}, []string{"getblocktemplate", "submitblock"}},
		{"env placeholder", []string{"{env.TEST_ALLOWED_METHODS}"}, []string{"getblocktemplate", "submitblock"}},
		{"env placeholder plus literal", []string{"{env.TEST_ALLOWED_METHODS}", "getblockcount"}, []string{"getblocktemplate", "submitblock", "getblockcount"}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, resolveAllowList(tt.in))
		})
	}
}

func TestProvisionResolvesAllowFromEnv(t *testing.T) {
	t.Setenv("TEST_ALLOWED_METHODS", "getblocktemplate, submitblock")

	h := &Handler{Allow: []string{"{env.TEST_ALLOWED_METHODS}"}}
	require.NoError(t, h.Provision(caddy.Context{}))
	assert.Equal(t, []string{"getblocktemplate", "submitblock"}, h.Allow)
}

func TestProvisionEmptyAllowPermitsAll(t *testing.T) {
	// Env var unset/empty: an allow directive that resolves to nothing must NOT
	// fail; the allowlist is inactive and every method is permitted.
	t.Setenv("TEST_ALLOWED_METHODS", "")

	h := &Handler{Allow: []string{"{env.TEST_ALLOWED_METHODS}"}}
	require.NoError(t, h.Provision(caddy.Context{}))
	assert.Empty(t, h.Allow, "allow resolving to nothing should leave the allowlist empty (permit all)")

	// Behaves as allow-all: an arbitrary method still reaches the backend.
	backend := &staticBackend{}
	w := doRequest(t, h, backend, "getblockcount", 1)
	assert.Equal(t, http.StatusOK, w.Code)
	assert.Equal(t, int64(1), backend.calls.Load(), "with an empty allowlist every method should reach backend")
}

func TestProvisionNoAllowSucceeds(t *testing.T) {
	h := &Handler{}
	require.NoError(t, h.Provision(caddy.Context{}))
	assert.Empty(t, h.Allow, "no allow directive should leave the allowlist empty (permit all)")
}
