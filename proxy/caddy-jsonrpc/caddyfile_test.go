package jsonrpc

import (
	"testing"
	"time"

	"github.com/caddyserver/caddy/v2"
	"github.com/caddyserver/caddy/v2/caddyconfig/caddyfile"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestUnmarshalCaddyfileAllowAndCache(t *testing.T) {
	d := caddyfile.NewTestDispenser(`
jsonrpc {
	allow getblocktemplate submitblock
	cache getblocktemplate 1s
}`)

	var h Handler
	require.NoError(t, h.UnmarshalCaddyfile(d))

	assert.Equal(t, []string{"getblocktemplate", "submitblock"}, h.Allow)
	require.Len(t, h.Rules, 1)
	assert.Equal(t, "getblocktemplate", h.Rules[0].Method)
	assert.Equal(t, caddy.Duration(time.Second), h.Rules[0].TTL)
}

func TestUnmarshalCaddyfileAllowRepeated(t *testing.T) {
	d := caddyfile.NewTestDispenser(`
jsonrpc {
	allow getblocktemplate
	allow submitblock getblockcount
}`)

	var h Handler
	require.NoError(t, h.UnmarshalCaddyfile(d))
	assert.Equal(t, []string{"getblocktemplate", "submitblock", "getblockcount"}, h.Allow)
}

func TestUnmarshalCaddyfileAllowRequiresArg(t *testing.T) {
	d := caddyfile.NewTestDispenser(`
jsonrpc {
	allow
}`)

	var h Handler
	assert.Error(t, h.UnmarshalCaddyfile(d), "allow with no methods should be a parse error")
}

func TestUnmarshalCaddyfileNoAllowIsEmpty(t *testing.T) {
	d := caddyfile.NewTestDispenser(`
jsonrpc {
	cache getblocktemplate 1s
}`)

	var h Handler
	require.NoError(t, h.UnmarshalCaddyfile(d))
	assert.Empty(t, h.Allow, "omitting allow should leave the allowlist empty (permit all)")
}

// TestAllowFromEnvEndToEnd mirrors the shipped Caddyfile: parse the directive
// with an {env.VAR} placeholder, then provision it with the variable set.
func TestAllowFromEnvEndToEnd(t *testing.T) {
	t.Setenv("RPC_ALLOWED_METHODS", "getblocktemplate submitblock")

	d := caddyfile.NewTestDispenser(`
jsonrpc {
	allow {env.RPC_ALLOWED_METHODS}
	cache getblocktemplate 1s
}`)

	var h Handler
	require.NoError(t, h.UnmarshalCaddyfile(d))
	// The placeholder is stored verbatim until Provision resolves it.
	assert.Equal(t, []string{"{env.RPC_ALLOWED_METHODS}"}, h.Allow)

	require.NoError(t, h.Provision(caddy.Context{}))
	assert.Equal(t, []string{"getblocktemplate", "submitblock"}, h.Allow)
}

// TestAllowFromEnvUnsetPermitsAll documents what happens when the operator
// leaves RPC_ALLOWED_METHODS unset: parsing and provisioning both succeed, the
// allowlist is inactive, and every method is permitted.
func TestAllowFromEnvUnsetPermitsAll(t *testing.T) {
	t.Setenv("RPC_ALLOWED_METHODS", "") // empty == unset for os.Getenv

	d := caddyfile.NewTestDispenser(`
jsonrpc {
	allow {env.RPC_ALLOWED_METHODS}
	cache getblocktemplate 1s
}`)

	var h Handler
	require.NoError(t, h.UnmarshalCaddyfile(d), "parsing should succeed even with the env var unset")
	assert.Equal(t, []string{"{env.RPC_ALLOWED_METHODS}"}, h.Allow)

	require.NoError(t, h.Provision(caddy.Context{}), "an unset RPC_ALLOWED_METHODS must not fail provisioning")
	assert.Empty(t, h.Allow, "an unset allowlist should permit all methods")
}
