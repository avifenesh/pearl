package jsonrpccache

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"sync"
	"time"

	"github.com/caddyserver/caddy/v2"
	"github.com/caddyserver/caddy/v2/modules/caddyhttp"
	"go.uber.org/zap"
	"golang.org/x/sync/singleflight"
)

// defaultMissTimeout bounds how long a cache miss may wait for the upstream to
// respond before the in-flight call is abandoned. Without a bound, a single
// slow upstream call can park every concurrent caller for the duration of that
// call, regardless of whether the callers themselves still want the result.
const defaultMissTimeout = 5 * time.Second

func init() {
	caddy.RegisterModule(Handler{})
}

// Handler is a Caddy middleware that caches JSON-RPC responses by method name.
// It parses incoming POST bodies, checks the JSON-RPC method field, and serves
// cached responses for configured methods within their TTL. Concurrent requests
// for the same method are coalesced into a single upstream call via
// singleflight; each caller waits independently and can abandon the wait when
// its own request context is canceled, so a slow or unresponsive upstream
// cannot block all concurrent callers.
type Handler struct {
	// Rules defines which JSON-RPC methods to cache and for how long.
	Rules []CacheRule `json:"rules,omitempty"`

	// MissTimeout bounds how long a single cache miss may wait for the
	// upstream. When this elapses, the in-flight call is forgotten so a
	// later request can start a fresh attempt, and the caller is returned
	// a gateway timeout. If zero, defaultMissTimeout is used.
	MissTimeout caddy.Duration `json:"miss_timeout,omitempty"`

	cache  *cache
	logger *zap.Logger
}

// CacheRule maps a JSON-RPC method name to a TTL.
type CacheRule struct {
	Method string         `json:"method"`
	TTL    caddy.Duration `json:"ttl"`
}

// CaddyModule returns the Caddy module information.
func (Handler) CaddyModule() caddy.ModuleInfo {
	return caddy.ModuleInfo{
		ID:  "http.handlers.jsonrpc_cache",
		New: func() caddy.Module { return new(Handler) },
	}
}

// Provision sets up the handler.
func (h *Handler) Provision(ctx caddy.Context) error {
	h.logger = ctx.Logger()
	h.cache = &cache{
		entries: make(map[string]*cacheEntry),
	}
	return nil
}

// Validate ensures the handler configuration is valid.
func (h *Handler) Validate() error {
	for _, r := range h.Rules {
		if r.Method == "" {
			return fmt.Errorf("cache rule has empty method name")
		}
		if r.TTL <= 0 {
			return fmt.Errorf("cache rule for %q has invalid TTL", r.Method)
		}
	}
	if h.MissTimeout < 0 {
		return fmt.Errorf("miss_timeout must be non-negative")
	}
	return nil
}

// missTimeout returns the configured cache-miss timeout, or the default if
// none was set.
func (h Handler) missTimeout() time.Duration {
	if d := time.Duration(h.MissTimeout); d > 0 {
		return d
	}
	return defaultMissTimeout
}

// Cleanup releases resources held by the handler.
func (h *Handler) Cleanup() error {
	if h.cache != nil {
		h.cache.clear()
	}
	return nil
}

func (h Handler) ServeHTTP(w http.ResponseWriter, r *http.Request, next caddyhttp.Handler) error {
	if r.Method != http.MethodPost {
		return next.ServeHTTP(w, r)
	}

	body, err := io.ReadAll(r.Body)
	r.Body.Close()
	if err != nil {
		return next.ServeHTTP(w, r)
	}

	var req jsonrpcRequest
	if err := json.Unmarshal(body, &req); err != nil {
		r.Body = io.NopCloser(bytes.NewReader(body))
		return next.ServeHTTP(w, r)
	}

	rule := h.findRule(req.Method)
	if rule == nil {
		r.Body = io.NopCloser(bytes.NewReader(body))
		return next.ServeHTTP(w, r)
	}

	if entry, ok := h.cache.get(req.Method); ok && entry.fresh() {
		return writeCachedResponse(w, entry, req.ID, true)
	}

	ttl := time.Duration(rule.TTL)
	missTimeout := h.missTimeout()

	// DoChan coalesces concurrent callers into a single upstream call but
	// lets each caller wait independently, so a slow upstream cannot park
	// callers whose own request context has been canceled.
	resultCh := h.cache.group.DoChan(req.Method, func() (any, error) {
		// The upstream call uses its own deadline. context.Background is
		// the parent because this in-flight call is shared across many
		// callers and must not be tied to any single caller's lifecycle.
		upstreamCtx, cancel := context.WithTimeout(context.Background(), missTimeout)
		defer cancel()
		return h.fetchFromUpstream(upstreamCtx, r, body, next, req.Method, ttl)
	})

	select {
	case res := <-resultCh:
		if res.Err != nil {
			// Normalize upstream context expiry to the same envelope as
			// the wait-side timeout. The in-flight call has completed
			// (with this error) so singleflight will let the next caller
			// drive a fresh attempt without us having to Forget.
			if errors.Is(res.Err, context.DeadlineExceeded) || errors.Is(res.Err, context.Canceled) {
				return writeTimeoutResponse(w, req.ID)
			}
			return res.Err
		}
		return writeCachedResponse(w, res.Val.(*cacheEntry), req.ID, false)

	case <-r.Context().Done():
		// The caller gave up; do not block waiting for the upstream. The
		// in-flight call continues for other coalesced waiters and the
		// next caller, but this caller exits immediately.
		return r.Context().Err()

	case <-time.After(missTimeout):
		// The upstream is taking too long. Forget the key so a later
		// caller can drive a fresh attempt instead of joining this
		// stalled in-flight call.
		h.cache.group.Forget(req.Method)
		h.logger.Warn("cache miss timeout, abandoning in-flight upstream call",
			zap.String("method", req.Method),
			zap.Duration("miss_timeout", missTimeout),
		)
		return writeTimeoutResponse(w, req.ID)
	}
}

func (h Handler) findRule(method string) *CacheRule {
	for i := range h.Rules {
		if h.Rules[i].Method == method {
			return &h.Rules[i]
		}
	}
	return nil
}

// fetchFromUpstream issues the upstream call against a request bound to ctx,
// so the upstream can be canceled when the cache-miss budget elapses. The
// cache entry is only stored if the call completed successfully (HTTP 200,
// no JSON-RPC error) and the upstream context was not canceled or timed out,
// to avoid repopulating the cache with a partial, stale, or error response.
func (h Handler) fetchFromUpstream(ctx context.Context, r *http.Request, body []byte, next caddyhttp.Handler, method string, ttl time.Duration) (*cacheEntry, error) {
	// r is shared with the goroutine that originated this in-flight call;
	// WithContext copies only the request struct, so we must not mutate
	// any of r's other fields here.
	upstreamReq := r.WithContext(ctx)
	upstreamReq.Body = io.NopCloser(bytes.NewReader(body))

	rec := &responseRecorder{body: &bytes.Buffer{}, statusCode: http.StatusOK}
	if err := next.ServeHTTP(rec, upstreamReq); err != nil {
		return nil, err
	}

	if err := ctx.Err(); err != nil {
		return nil, err
	}

	entry := &cacheEntry{
		body:     rec.body.Bytes(),
		storedAt: time.Now(),
		ttl:      ttl,
	}

	// Only cache responses that are both transport-successful and
	// application-successful. JSON-RPC error responses (HTTP 200 with a
	// non-empty error field) typically indicate a transient upstream
	// problem and must not be cached, otherwise every coalesced caller
	// would see the same failure for the duration of the TTL.
	if rec.statusCode == http.StatusOK && !responseHasError(entry.body) {
		h.cache.set(method, entry)

		h.logger.Debug("cached JSON-RPC response",
			zap.String("method", method),
			zap.Duration("ttl", ttl),
			zap.Int("size", len(entry.body)),
		)
	}

	return entry, nil
}

// responseHasError reports whether body parses as a JSON-RPC response with a
// non-empty error field. Non-JSON-RPC bodies return false so that pass-through
// behavior is unchanged for them.
func responseHasError(body []byte) bool {
	var resp jsonrpcResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		return false
	}
	return len(resp.Error) != 0
}

// writeTimeoutResponse writes an HTTP 504 with a JSON-RPC 2.0 envelope. The
// request id is preserved so JSON-RPC clients can correlate the failure with
// the originating request.
func writeTimeoutResponse(w http.ResponseWriter, requestID json.RawMessage) error {
	body, err := json.Marshal(jsonrpcResponse{
		JSONRPC: "2.0",
		ID:      requestID,
		Error:   json.RawMessage(`{"code":-32603,"message":"upstream timeout"}`),
	})
	if err != nil {
		// Fallback for the (practically impossible) case where the id
		// cannot be re-encoded. Surface a well-formed envelope with a
		// null id rather than a half-written response.
		body = []byte(`{"jsonrpc":"2.0","error":{"code":-32603,"message":"upstream timeout"},"id":null}`)
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusGatewayTimeout)
	_, writeErr := w.Write(body)
	return writeErr
}

// cache holds the per-method cached responses behind a mutex.
type cache struct {
	mu      sync.RWMutex
	entries map[string]*cacheEntry
	group   singleflight.Group
}

func (c *cache) get(method string) (*cacheEntry, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	e, ok := c.entries[method]
	return e, ok
}

func (c *cache) set(method string, entry *cacheEntry) {
	c.mu.Lock()
	c.entries[method] = entry
	c.mu.Unlock()
}

func (c *cache) clear() {
	c.mu.Lock()
	clear(c.entries)
	c.mu.Unlock()
}

type cacheEntry struct {
	body     []byte
	storedAt time.Time
	ttl      time.Duration
}

func (e *cacheEntry) fresh() bool {
	return time.Since(e.storedAt) < e.ttl
}

type jsonrpcRequest struct {
	ID     json.RawMessage `json:"id"`
	Method string          `json:"method"`
}

type jsonrpcResponse struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id"`
	Result  json.RawMessage `json:"result,omitempty"`
	Error   json.RawMessage `json:"error,omitempty"`
}

func writeCachedResponse(w http.ResponseWriter, entry *cacheEntry, requestID json.RawMessage, hit bool) error {
	var resp jsonrpcResponse
	if err := json.Unmarshal(entry.body, &resp); err != nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, writeErr := w.Write(entry.body)
		return writeErr
	}

	resp.ID = requestID
	rewritten, err := json.Marshal(resp)
	if err != nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, writeErr := w.Write(entry.body)
		return writeErr
	}

	w.Header().Set("Content-Type", "application/json")
	if hit {
		w.Header().Set("X-Jsonrpc-Cache", "HIT")
	}
	w.WriteHeader(http.StatusOK)
	_, writeErr := w.Write(rewritten)
	return writeErr
}

// responseRecorder captures the response body and status code from the upstream handler.
type responseRecorder struct {
	body       *bytes.Buffer
	header     http.Header
	statusCode int
}

func (rec *responseRecorder) Header() http.Header {
	if rec.header == nil {
		rec.header = make(http.Header)
	}
	return rec.header
}

func (rec *responseRecorder) Write(b []byte) (int, error) {
	return rec.body.Write(b)
}

func (rec *responseRecorder) WriteHeader(code int) {
	rec.statusCode = code
}

// Interface guards
var (
	_ caddy.Provisioner           = (*Handler)(nil)
	_ caddy.Validator             = (*Handler)(nil)
	_ caddy.CleanerUpper          = (*Handler)(nil)
	_ caddyhttp.MiddlewareHandler = (*Handler)(nil)
)
