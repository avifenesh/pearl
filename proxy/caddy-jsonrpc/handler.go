package jsonrpc

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/caddyserver/caddy/v2"
	"github.com/caddyserver/caddy/v2/modules/caddyhttp"
	"go.uber.org/zap"
	"golang.org/x/sync/singleflight"
)

func init() {
	caddy.RegisterModule(Handler{})
}

// Handler is a Caddy middleware for JSON-RPC traffic. It parses incoming POST
// bodies to read the JSON-RPC method, then enforces an optional per-method
// allowlist (rejecting other methods with HTTP 403) and caches responses for
// configured methods within their TTL. Concurrent requests for the same method
// are coalesced via singleflight.
type Handler struct {
	// Rules defines which JSON-RPC methods to cache and for how long.
	Rules []CacheRule `json:"rules,omitempty"`

	// Allow, when non-empty, is the set of JSON-RPC methods permitted
	// through the proxy. Requests for any other method — and any request
	// whose method cannot be determined (non-POST, unreadable, or
	// unparseable bodies) — are rejected with HTTP 403. When empty, every
	// method is allowed (the cache still applies to configured methods).
	//
	// Entries may contain Caddy placeholders (e.g. "{env.RPC_ALLOWED_METHODS}")
	// and whitespace- or comma-separated lists; both are resolved and flattened
	// in Provision. If an allow directive resolves to nothing (e.g. the
	// referenced env var is unset), the allowlist is inactive and every method
	// is permitted.
	Allow []string `json:"allow,omitempty"`

	cache  *cache
	logger *zap.Logger
}

// rpcErrMethodNotAllowed is the JSON-RPC error code returned for methods the
// allowlist rejects. -32601 is the standard "method not found" code; from a
// client's perspective an allowlist-blocked method is effectively unavailable.
const rpcErrMethodNotAllowed = -32601

// CacheRule maps a JSON-RPC method name to a TTL.
type CacheRule struct {
	Method string         `json:"method"`
	TTL    caddy.Duration `json:"ttl"`
}

// CaddyModule returns the Caddy module information.
func (Handler) CaddyModule() caddy.ModuleInfo {
	return caddy.ModuleInfo{
		ID:  "http.handlers.jsonrpc",
		New: func() caddy.Module { return new(Handler) },
	}
}

// Provision sets up the handler.
func (h *Handler) Provision(ctx caddy.Context) error {
	h.logger = ctx.Logger()
	h.cache = &cache{
		entries: make(map[string]*cacheEntry),
	}

	// Resolve the allowlist: expand placeholders (e.g. {env.VAR}) and flatten
	// whitespace/comma-separated lists. If it resolves to nothing — typically an
	// unset environment variable — the allowlist is simply inactive and every
	// method is permitted (a warning is logged so the case is visible).
	if len(h.Allow) > 0 {
		h.Allow = resolveAllowList(h.Allow)
		if len(h.Allow) > 0 {
			h.logger.Info("JSON-RPC method allowlist active", zap.Strings("methods", h.Allow))
		} else {
			h.logger.Warn("JSON-RPC allow directive resolved to no methods; all methods are permitted")
		}
	}

	return nil
}

// resolveAllowList expands Caddy placeholders in each entry and splits the
// results on whitespace and commas, so a single "{env.RPC_ALLOWED_METHODS}"
// holding "getblocktemplate submitblock" (or "getblocktemplate,submitblock")
// becomes two distinct method names. Literal entries pass through unchanged.
func resolveAllowList(entries []string) []string {
	repl := caddy.NewReplacer()
	var out []string
	for _, e := range entries {
		expanded := repl.ReplaceAll(e, "")
		out = append(out, strings.Fields(strings.ReplaceAll(expanded, ",", " "))...)
	}
	return out
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
	for _, m := range h.Allow {
		if m == "" {
			return fmt.Errorf("allow list contains an empty method name")
		}
	}
	return nil
}

// Cleanup releases resources held by the handler.
func (h *Handler) Cleanup() error {
	if h.cache != nil {
		h.cache.clear()
	}
	return nil
}

func (h Handler) ServeHTTP(w http.ResponseWriter, r *http.Request, next caddyhttp.Handler) error {
	enforce := len(h.Allow) > 0

	// JSON-RPC calls are POST. A non-POST request carries no method to match
	// against the allowlist, so deny it when one is configured; otherwise
	// preserve the original passthrough behavior.
	if r.Method != http.MethodPost {
		if enforce {
			return h.deny(w, nil, "non-POST request")
		}
		return next.ServeHTTP(w, r)
	}

	body, err := io.ReadAll(r.Body)
	r.Body.Close()
	if err != nil {
		if enforce {
			return h.deny(w, nil, "unreadable request body")
		}
		return next.ServeHTTP(w, r)
	}

	var req jsonrpcRequest
	if err := json.Unmarshal(body, &req); err != nil {
		if enforce {
			return h.deny(w, nil, "unparseable JSON-RPC request")
		}
		r.Body = io.NopCloser(bytes.NewReader(body))
		return next.ServeHTTP(w, r)
	}

	// Enforce the method allowlist before any caching or upstream call.
	if enforce && !h.isAllowed(req.Method) {
		return h.deny(w, req.ID, fmt.Sprintf("method %q is not permitted", req.Method))
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
	result, err, _ := h.cache.group.Do(req.Method, func() (any, error) {
		return h.fetchFromUpstream(r, body, next, req.Method, ttl)
	})
	if err != nil {
		return err
	}

	return writeCachedResponse(w, result.(*cacheEntry), req.ID, false)
}

func (h Handler) findRule(method string) *CacheRule {
	for i := range h.Rules {
		if h.Rules[i].Method == method {
			return &h.Rules[i]
		}
	}
	return nil
}

// isAllowed reports whether method is permitted by the allowlist. The list is
// small (a handful of methods), so a linear scan is cheaper than a map and
// needs no setup in Provision.
func (h Handler) isAllowed(method string) bool {
	for _, m := range h.Allow {
		if m == method {
			return true
		}
	}
	return false
}

// deny rejects a request the allowlist does not permit. It logs the reason and
// writes HTTP 403 with a JSON-RPC error body so both HTTP-aware and
// JSON-RPC-aware clients get a clear signal. id echoes the caller's request id
// when known (nil becomes JSON null).
func (h Handler) deny(w http.ResponseWriter, id json.RawMessage, reason string) error {
	if h.logger != nil {
		h.logger.Debug("rejected disallowed JSON-RPC request", zap.String("reason", reason))
	}
	if len(id) == 0 {
		id = json.RawMessage("null")
	}

	body, err := json.Marshal(struct {
		Result any             `json:"result"`
		Error  jsonrpcError    `json:"error"`
		ID     json.RawMessage `json:"id"`
	}{
		Error: jsonrpcError{Code: rpcErrMethodNotAllowed, Message: reason},
		ID:    id,
	})
	if err != nil {
		return err
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusForbidden)
	_, werr := w.Write(body)
	return werr
}

func (h Handler) fetchFromUpstream(r *http.Request, body []byte, next caddyhttp.Handler, method string, ttl time.Duration) (*cacheEntry, error) {
	r.Body = io.NopCloser(bytes.NewReader(body))

	rec := &responseRecorder{body: &bytes.Buffer{}, statusCode: http.StatusOK}
	if err := next.ServeHTTP(rec, r); err != nil {
		return nil, err
	}

	entry := &cacheEntry{
		body:     rec.body.Bytes(),
		storedAt: time.Now(),
		ttl:      ttl,
	}

	if rec.statusCode == http.StatusOK {
		h.cache.set(method, entry)

		h.logger.Debug("cached JSON-RPC response",
			zap.String("method", method),
			zap.Duration("ttl", ttl),
			zap.Int("size", len(entry.body)),
		)
	}

	return entry, nil
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

// jsonrpcError is the error object returned when the allowlist rejects a method.
type jsonrpcError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
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
